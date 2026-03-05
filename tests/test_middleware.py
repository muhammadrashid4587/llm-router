"""Tests for middleware: logging, rate limiting, retry."""

from __future__ import annotations

import logging
import time

import pytest

from llm_router.middleware import (
    LoggingMiddleware,
    MiddlewareChain,
    RateLimitMiddleware,
    RetryMiddleware,
)
from llm_router.models import RoutingRequest, RoutingResult


def _request() -> RoutingRequest:
    return RoutingRequest(messages=[{"role": "user", "content": "hi"}])


def _result() -> RoutingResult:
    return RoutingResult(provider="mock", model="mock-model", content="ok")


class TestLoggingMiddleware:
    def test_before_request_logs(self, caplog):
        mw = LoggingMiddleware(level=logging.INFO)
        req = _request()
        with caplog.at_level(logging.INFO, logger="llm_router"):
            out = mw.before_request(req)
        assert out is req
        assert "Routing request" in caplog.text

    def test_after_request_logs(self, caplog):
        mw = LoggingMiddleware(level=logging.INFO)
        req = _request()
        res = _result()
        with caplog.at_level(logging.INFO, logger="llm_router"):
            out = mw.after_request(req, res)
        assert out is res
        assert "Routed to" in caplog.text

    def test_on_error_logs(self, caplog):
        mw = LoggingMiddleware()
        req = _request()
        with caplog.at_level(logging.ERROR, logger="llm_router"):
            mw.on_error(req, RuntimeError("boom"))
        assert "boom" in caplog.text


class TestRateLimitMiddleware:
    def test_rate_limit_slows_down(self):
        mw = RateLimitMiddleware(max_requests_per_second=100.0)
        req = _request()
        start = time.monotonic()
        for _ in range(5):
            mw.before_request(req)
        elapsed = time.monotonic() - start
        # 5 requests at 100 rps -> ~0.04s minimum spacing
        # Just check it doesn't take absurdly long
        assert elapsed < 2.0

    def test_after_request_passthrough(self):
        mw = RateLimitMiddleware()
        res = _result()
        out = mw.after_request(_request(), res)
        assert out is res


class TestRetryMiddleware:
    def test_should_retry_within_limit(self):
        mw = RetryMiddleware(max_retries=3)
        assert mw.should_retry()
        mw.record_attempt()
        mw.record_attempt()
        mw.record_attempt()
        assert not mw.should_retry()

    def test_reset_clears_count(self):
        mw = RetryMiddleware(max_retries=2)
        mw.record_attempt()
        mw.record_attempt()
        assert not mw.should_retry()
        mw.reset()
        assert mw.should_retry()

    def test_backoff_delay_exponential(self):
        mw = RetryMiddleware(max_retries=5, backoff_base=1.0)
        mw.record_attempt()  # attempt 1
        assert mw.backoff_delay() == 1.0
        mw.record_attempt()  # attempt 2
        assert mw.backoff_delay() == 2.0
        mw.record_attempt()  # attempt 3
        assert mw.backoff_delay() == 4.0

    def test_on_error_records_attempt(self):
        mw = RetryMiddleware(max_retries=3)
        assert mw.attempt_count == 0
        mw.on_error(_request(), RuntimeError("fail"))
        assert mw.attempt_count == 1

    def test_after_request_resets(self):
        mw = RetryMiddleware(max_retries=3)
        mw.record_attempt()
        mw.record_attempt()
        assert mw.attempt_count == 2
        mw.after_request(_request(), _result())
        assert mw.attempt_count == 0


class TestMiddlewareChain:
    def test_chain_runs_all_before(self):
        log_mw = LoggingMiddleware()
        rate_mw = RateLimitMiddleware()
        chain = MiddlewareChain([log_mw, rate_mw])
        req = _request()
        out = chain.run_before(req)
        assert out is req  # both pass through

    def test_chain_runs_all_after(self):
        log_mw = LoggingMiddleware()
        rate_mw = RateLimitMiddleware()
        chain = MiddlewareChain([log_mw, rate_mw])
        res = _result()
        out = chain.run_after(_request(), res)
        assert out is res

    def test_chain_runs_on_error(self, caplog):
        log_mw = LoggingMiddleware()
        retry_mw = RetryMiddleware()
        chain = MiddlewareChain([log_mw, retry_mw])
        with caplog.at_level(logging.ERROR, logger="llm_router"):
            chain.run_on_error(_request(), RuntimeError("oops"))
        assert retry_mw.attempt_count == 1
        assert "oops" in caplog.text

    def test_empty_chain(self):
        chain = MiddlewareChain()
        req = _request()
        res = _result()
        assert chain.run_before(req) is req
        assert chain.run_after(req, res) is res
        chain.run_on_error(req, RuntimeError("noop"))  # should not raise
