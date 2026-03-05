"""Pre/post request hooks: logging, rate limiting, retry."""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from llm_router.models import RoutingRequest, RoutingResult

logger = logging.getLogger("llm_router")


class Middleware(ABC):
    """Base class for request/response middleware."""

    @abstractmethod
    def before_request(self, request: RoutingRequest) -> RoutingRequest:
        """Called before routing. May modify or reject the request."""
        ...

    @abstractmethod
    def after_request(
        self, request: RoutingRequest, result: RoutingResult
    ) -> RoutingResult:
        """Called after a successful response. May modify the result."""
        ...

    def on_error(self, request: RoutingRequest, error: Exception) -> None:
        """Called when a routing attempt raises an exception."""


# ---------------------------------------------------------------------------
# LoggingMiddleware
# ---------------------------------------------------------------------------

class LoggingMiddleware(Middleware):
    """Logs requests and responses at configurable level."""

    def __init__(self, level: int = logging.INFO) -> None:
        self.level = level

    def before_request(self, request: RoutingRequest) -> RoutingRequest:
        logger.log(
            self.level,
            "Routing request: capabilities=%s, preferred=%s",
            request.capabilities,
            request.preferred_providers,
        )
        return request

    def after_request(
        self, request: RoutingRequest, result: RoutingResult
    ) -> RoutingResult:
        logger.log(
            self.level,
            "Routed to %s/%s -- latency=%.1fms cost=%.6f success=%s",
            result.provider,
            result.model,
            result.latency_ms,
            result.cost,
            result.success,
        )
        return result

    def on_error(self, request: RoutingRequest, error: Exception) -> None:
        logger.error("Routing error: %s", error)


# ---------------------------------------------------------------------------
# RateLimitMiddleware
# ---------------------------------------------------------------------------

class RateLimitMiddleware(Middleware):
    """Simple token-bucket rate limiter (per-router, not per-provider)."""

    def __init__(self, max_requests_per_second: float = 10.0) -> None:
        if max_requests_per_second <= 0:
            raise ValueError("max_requests_per_second must be positive")
        self.max_rps = max_requests_per_second
        self._interval = 1.0 / max_requests_per_second
        self._lock = threading.Lock()
        self._last_request_time: float = 0.0

    def before_request(self, request: RoutingRequest) -> RoutingRequest:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._interval:
                wait = self._interval - elapsed
                time.sleep(wait)
            self._last_request_time = time.monotonic()
        return request

    def after_request(
        self, request: RoutingRequest, result: RoutingResult
    ) -> RoutingResult:
        return result


# ---------------------------------------------------------------------------
# RetryMiddleware
# ---------------------------------------------------------------------------

class RetryMiddleware(Middleware):
    """Tracks retry metadata.  Actual retry logic lives in the router, but this
    middleware records attempts and can enforce a maximum."""

    def __init__(self, max_retries: int = 3, backoff_base: float = 0.5) -> None:
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._attempt_count: int = 0
        self._lock = threading.Lock()

    @property
    def attempt_count(self) -> int:
        return self._attempt_count

    def reset(self) -> None:
        with self._lock:
            self._attempt_count = 0

    def should_retry(self) -> bool:
        with self._lock:
            return self._attempt_count < self.max_retries

    def record_attempt(self) -> None:
        with self._lock:
            self._attempt_count += 1

    def backoff_delay(self) -> float:
        """Exponential backoff delay for the current attempt number."""
        with self._lock:
            return self.backoff_base * (2 ** (self._attempt_count - 1))

    def before_request(self, request: RoutingRequest) -> RoutingRequest:
        return request

    def after_request(
        self, request: RoutingRequest, result: RoutingResult
    ) -> RoutingResult:
        self.reset()
        return result

    def on_error(self, request: RoutingRequest, error: Exception) -> None:
        self.record_attempt()


# ---------------------------------------------------------------------------
# MiddlewareChain helper
# ---------------------------------------------------------------------------

class MiddlewareChain:
    """Runs a list of middleware in order."""

    def __init__(self, middlewares: list[Middleware] | None = None) -> None:
        self.middlewares: list[Middleware] = middlewares or []

    def run_before(self, request: RoutingRequest) -> RoutingRequest:
        for mw in self.middlewares:
            request = mw.before_request(request)
        return request

    def run_after(
        self, request: RoutingRequest, result: RoutingResult
    ) -> RoutingResult:
        for mw in self.middlewares:
            result = mw.after_request(request, result)
        return result

    def run_on_error(self, request: RoutingRequest, error: Exception) -> None:
        for mw in self.middlewares:
            mw.on_error(request, error)
