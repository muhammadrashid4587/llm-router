"""Tests for routing policies."""

from __future__ import annotations

import pytest

from llm_router.health import HealthChecker
from llm_router.models import ModelCapability, ProviderConfig, RoutingRequest
from llm_router.policies import (
    CostOptimizedPolicy,
    FallbackPolicy,
    LatencyOptimizedPolicy,
    RoundRobinPolicy,
)
from llm_router.providers import MockProvider


def _make_providers() -> dict[str, MockProvider]:
    cheap = MockProvider(
        config=ProviderConfig(
            name="cheap",
            models=["mock-cheap"],
            capabilities=[ModelCapability.CHAT, ModelCapability.FUNCTION_CALLING],
            priority=1,
        ),
    )
    expensive = MockProvider(
        config=ProviderConfig(
            name="expensive",
            models=["mock-expensive"],
            capabilities=[ModelCapability.CHAT, ModelCapability.VISION],
            priority=0,
        ),
    )
    return {"cheap": cheap, "expensive": expensive}


def _request(**kwargs) -> RoutingRequest:
    return RoutingRequest(
        messages=[{"role": "user", "content": "test"}],
        **kwargs,
    )


class TestCostOptimizedPolicy:
    def test_selects_cheapest(self):
        policy = CostOptimizedPolicy()
        providers = _make_providers()
        health = HealthChecker()
        result = policy.select(providers, _request(), health)
        # mock-cheap is cheaper than mock-expensive
        assert result[0][0] == "cheap"

    def test_max_cost_filter(self):
        policy = CostOptimizedPolicy(max_cost_per_1k=0.0002)
        providers = _make_providers()
        health = HealthChecker()
        result = policy.select(providers, _request(), health)
        # Only mock-cheap (0.00015 avg) passes, mock-expensive (0.075 avg) does not
        assert len(result) == 1
        assert result[0][0] == "cheap"

    def test_request_level_cost_override(self):
        policy = CostOptimizedPolicy()  # no policy-level cap
        providers = _make_providers()
        health = HealthChecker()
        req = _request(max_cost_per_1k_tokens=0.0002)
        result = policy.select(providers, req, health)
        assert len(result) == 1
        assert result[0][0] == "cheap"

    def test_capability_filter(self):
        policy = CostOptimizedPolicy()
        providers = _make_providers()
        health = HealthChecker()
        req = _request(capabilities=["vision"])
        result = policy.select(providers, req, health)
        assert len(result) == 1
        assert result[0][0] == "expensive"

    def test_excluded_providers(self):
        policy = CostOptimizedPolicy()
        providers = _make_providers()
        health = HealthChecker()
        req = _request(excluded_providers=["cheap"])
        result = policy.select(providers, req, health)
        assert all(name != "cheap" for name, _ in result)


class TestLatencyOptimizedPolicy:
    def test_selects_lowest_latency(self):
        policy = LatencyOptimizedPolicy()
        providers = _make_providers()
        health = HealthChecker()
        # Simulate latency data
        health.record_success("cheap", 200.0)
        health.record_success("expensive", 50.0)
        result = policy.select(providers, _request(), health)
        assert result[0][0] == "expensive"

    def test_max_latency_filter(self):
        policy = LatencyOptimizedPolicy(max_latency_ms=100.0)
        providers = _make_providers()
        health = HealthChecker()
        health.record_success("cheap", 200.0)
        health.record_success("expensive", 50.0)
        result = policy.select(providers, _request(), health)
        assert len(result) == 1
        assert result[0][0] == "expensive"

    def test_no_latency_data_returns_all(self):
        policy = LatencyOptimizedPolicy()
        providers = _make_providers()
        health = HealthChecker()
        result = policy.select(providers, _request(), health)
        # Both should appear (both have inf latency, which is equal)
        assert len(result) == 2


class TestRoundRobinPolicy:
    def test_distributes_evenly(self):
        policy = RoundRobinPolicy()
        providers = _make_providers()
        health = HealthChecker()
        req = _request()

        first_choices = []
        for _ in range(4):
            selection = policy.select(providers, req, health)
            first_choices.append(selection[0][0])

        # Should alternate between the two providers
        assert "cheap" in first_choices
        assert "expensive" in first_choices

    def test_single_provider(self):
        policy = RoundRobinPolicy()
        providers = {"only": _make_providers()["cheap"]}
        health = HealthChecker()
        for _ in range(3):
            result = policy.select(providers, _request(), health)
            assert result[0][0] == "cheap"


class TestFallbackPolicy:
    def test_respects_priority(self):
        policy = FallbackPolicy()
        providers = _make_providers()
        health = HealthChecker()
        result = policy.select(providers, _request(), health)
        # expensive has priority 0 (higher), cheap has priority 1
        assert result[0][0] == "expensive"

    def test_preferred_providers_promoted(self):
        policy = FallbackPolicy()
        providers = _make_providers()
        health = HealthChecker()
        req = _request(preferred_providers=["cheap"])
        result = policy.select(providers, req, health)
        assert result[0][0] == "cheap"

    def test_unhealthy_provider_skipped(self):
        policy = FallbackPolicy()
        providers = _make_providers()
        health = HealthChecker(failure_threshold=2)
        # Trip the circuit breaker for expensive
        health.record_failure("expensive")
        health.record_failure("expensive")
        result = policy.select(providers, _request(), health)
        assert all(name != "expensive" for name, _ in result)

    def test_empty_when_all_unhealthy(self):
        policy = FallbackPolicy()
        providers = _make_providers()
        health = HealthChecker(failure_threshold=1)
        health.record_failure("cheap")
        health.record_failure("expensive")
        result = policy.select(providers, _request(), health)
        assert result == []
