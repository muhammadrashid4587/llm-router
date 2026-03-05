"""Tests for the LLMRouter main class."""

from __future__ import annotations

import pytest

from llm_router.models import ModelCapability, ProviderConfig
from llm_router.policies import (
    CostOptimizedPolicy,
    FallbackPolicy,
    LatencyOptimizedPolicy,
)
from llm_router.providers import MockProvider, ProviderRegistry
from llm_router.router import LLMRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(*providers: MockProvider) -> ProviderRegistry:
    reg = ProviderRegistry()
    for p in providers:
        reg.register(p)
    return reg


def _cheap_provider(name: str = "cheap") -> MockProvider:
    return MockProvider(
        config=ProviderConfig(
            name=name,
            models=["mock-cheap"],
            capabilities=[ModelCapability.CHAT, ModelCapability.FUNCTION_CALLING],
            priority=1,
        ),
        response_content="cheap response",
    )


def _expensive_provider(name: str = "expensive") -> MockProvider:
    return MockProvider(
        config=ProviderConfig(
            name=name,
            models=["mock-expensive"],
            capabilities=[ModelCapability.CHAT, ModelCapability.VISION],
            priority=2,
        ),
        response_content="expensive response",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLLMRouterBasic:
    def test_route_single_provider(self):
        prov = _cheap_provider()
        router = LLMRouter(providers=_make_registry(prov), policy=FallbackPolicy())
        result = router.route(messages=[{"role": "user", "content": "hi"}])
        assert result.success
        assert result.provider == "cheap"
        assert result.content == "cheap response"

    def test_route_selects_cheapest(self):
        cheap = _cheap_provider()
        expensive = _expensive_provider()
        router = LLMRouter(
            providers=_make_registry(cheap, expensive),
            policy=CostOptimizedPolicy(),
        )
        result = router.route(messages=[{"role": "user", "content": "hi"}])
        assert result.provider == "cheap"

    def test_route_with_capability_filter(self):
        cheap = _cheap_provider()  # has function_calling, no vision
        expensive = _expensive_provider()  # has vision, no function_calling
        router = LLMRouter(
            providers=_make_registry(cheap, expensive),
            policy=CostOptimizedPolicy(),
        )
        result = router.route(
            messages=[{"role": "user", "content": "describe this image"}],
            capabilities=["vision"],
        )
        assert result.provider == "expensive"

    def test_route_no_matching_provider_raises(self):
        cheap = _cheap_provider()
        router = LLMRouter(
            providers=_make_registry(cheap),
            policy=CostOptimizedPolicy(),
        )
        with pytest.raises(RuntimeError, match="No providers available"):
            router.route(
                messages=[{"role": "user", "content": "hi"}],
                capabilities=["embeddings"],
            )

    def test_route_excluded_provider(self):
        cheap = _cheap_provider()
        expensive = _expensive_provider()
        router = LLMRouter(
            providers=_make_registry(cheap, expensive),
            policy=CostOptimizedPolicy(),
        )
        result = router.route(
            messages=[{"role": "user", "content": "hi"}],
            excluded_providers=["cheap"],
        )
        assert result.provider == "expensive"


class TestLLMRouterFallback:
    def test_fallback_on_failure(self):
        failing = MockProvider(
            config=ProviderConfig(
                name="failing",
                models=["mock-model"],
                capabilities=[ModelCapability.CHAT],
                priority=0,
            ),
            fail=True,
        )
        healthy = MockProvider(
            config=ProviderConfig(
                name="healthy",
                models=["mock-model"],
                capabilities=[ModelCapability.CHAT],
                priority=1,
            ),
            response_content="fallback response",
        )
        router = LLMRouter(
            providers=_make_registry(failing, healthy),
            policy=FallbackPolicy(),
            fallback=True,
        )
        result = router.route(messages=[{"role": "user", "content": "hi"}])
        assert result.provider == "healthy"
        assert result.content == "fallback response"

    def test_no_fallback_raises_immediately(self):
        failing = MockProvider(
            config=ProviderConfig(
                name="failing",
                models=["mock-model"],
                capabilities=[ModelCapability.CHAT],
                priority=0,
            ),
            fail=True,
        )
        healthy = MockProvider(
            config=ProviderConfig(
                name="healthy",
                models=["mock-model"],
                capabilities=[ModelCapability.CHAT],
                priority=1,
            ),
        )
        router = LLMRouter(
            providers=_make_registry(failing, healthy),
            policy=FallbackPolicy(),
            fallback=False,
        )
        with pytest.raises(RuntimeError, match="All providers failed"):
            router.route(messages=[{"role": "user", "content": "hi"}])

    def test_all_providers_fail(self):
        f1 = MockProvider(
            config=ProviderConfig(
                name="f1", models=["mock-model"], capabilities=[ModelCapability.CHAT]
            ),
            fail=True,
        )
        f2 = MockProvider(
            config=ProviderConfig(
                name="f2", models=["mock-model"], capabilities=[ModelCapability.CHAT]
            ),
            fail=True,
        )
        router = LLMRouter(
            providers=_make_registry(f1, f2),
            policy=FallbackPolicy(),
            fallback=True,
        )
        with pytest.raises(RuntimeError, match="All providers failed"):
            router.route(messages=[{"role": "user", "content": "hi"}])


class TestLLMRouterHealth:
    def test_health_report(self):
        prov = _cheap_provider()
        router = LLMRouter(providers=_make_registry(prov), policy=FallbackPolicy())
        router.route(messages=[{"role": "user", "content": "hi"}])
        health = router.get_health("cheap")
        assert health["is_healthy"] is True
        assert health["total_requests"] == 1

    def test_add_remove_provider(self):
        router = LLMRouter(
            providers=_make_registry(_cheap_provider()),
            policy=FallbackPolicy(),
        )
        assert "cheap" in router.provider_names
        router.remove_provider("cheap")
        assert "cheap" not in router.provider_names

        new_prov = _expensive_provider()
        router.add_provider(new_prov)
        assert "expensive" in router.provider_names


class TestLLMRouterFromDict:
    def test_from_config_dict(self):
        router = LLMRouter(
            providers={
                "test_mock": {
                    "provider_type": "mock",
                    "models": ["mock-model"],
                    "capabilities": ["chat"],
                }
            },
            policy=FallbackPolicy(),
        )
        result = router.route(messages=[{"role": "user", "content": "hi"}])
        assert result.success
        assert result.provider == "test_mock"
