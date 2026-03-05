"""Tests for the health checker and circuit breaker."""

from __future__ import annotations

import time

import pytest

from llm_router.health import CircuitBreaker, HealthChecker


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.state == "closed"
        assert not cb.is_open

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"
        cb.record_failure()
        assert cb.is_open
        assert cb.state == "open"

    def test_success_resets_counter(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert cb.state == "closed"  # only 1 consecutive failure

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        assert cb.is_open
        time.sleep(0.06)
        # After timeout, is_open returns False (half-open probe allowed)
        assert not cb.is_open
        assert cb.state == "half_open"

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        time.sleep(0.06)
        _ = cb.is_open  # triggers half_open
        cb.record_success()
        assert cb.state == "closed"

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        time.sleep(0.06)
        _ = cb.is_open  # triggers half_open
        cb.record_failure()
        assert cb.is_open

    def test_reset(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.is_open
        cb.reset()
        assert cb.state == "closed"
        assert not cb.is_open


class TestHealthChecker:
    def test_record_success_updates_stats(self):
        hc = HealthChecker()
        hc.record_success("p1", 100.0)
        hc.record_success("p1", 200.0)
        status = hc.get_status("p1")
        assert status.total_requests == 2
        assert status.total_failures == 0
        assert status.avg_latency_ms == 150.0
        assert status.is_healthy

    def test_record_failure_updates_stats(self):
        hc = HealthChecker()
        hc.record_success("p1", 50.0)
        hc.record_failure("p1")
        status = hc.get_status("p1")
        assert status.total_requests == 2
        assert status.total_failures == 1
        assert status.error_rate == 0.5

    def test_circuit_opens_on_failures(self):
        hc = HealthChecker(failure_threshold=2)
        hc.record_failure("p1")
        assert hc.is_healthy("p1")
        hc.record_failure("p1")
        assert not hc.is_healthy("p1")

    def test_avg_latency_no_data(self):
        hc = HealthChecker()
        assert hc.get_avg_latency("unknown") == float("inf")

    def test_reset_single_provider(self):
        hc = HealthChecker()
        hc.record_success("p1", 100.0)
        hc.record_success("p2", 200.0)
        hc.reset("p1")
        assert hc.get_avg_latency("p1") == float("inf")
        assert hc.get_avg_latency("p2") == 200.0

    def test_reset_all(self):
        hc = HealthChecker()
        hc.record_success("p1", 100.0)
        hc.record_success("p2", 200.0)
        hc.reset()
        assert hc.get_avg_latency("p1") == float("inf")
        assert hc.get_avg_latency("p2") == float("inf")

    def test_is_healthy_unknown_provider(self):
        hc = HealthChecker()
        assert hc.is_healthy("never_seen")

    def test_get_status_unknown_provider(self):
        hc = HealthChecker()
        status = hc.get_status("new")
        assert status.is_healthy
        assert status.total_requests == 0
