"""Tests for the async LLM router and providers."""

from __future__ import annotations

import asyncio

import pytest

from llm_router.async_providers import AsyncLLMProvider, AsyncMockProvider
from llm_router.async_router import AsyncLLMRouter
from llm_router.models import ModelCapability, ProviderConfig
from llm_router.policies import CostOptimizedPolicy, FallbackPolicy


@pytest.fixture
def mock_provider() -> AsyncMockProvider:
    return AsyncMockProvider(latency_ms=10.0)


@pytest.fixture
def fast_provider() -> AsyncMockProvider:
    return AsyncMockProvider(
        config=ProviderConfig(
            name="fast",
            api_key="key",
            models=["fast-model"],
            capabilities=[ModelCapability.CHAT],
            priority=1,
        ),
        latency_ms=5.0,
        response_content="Fast response",
    )


@pytest.fixture
def slow_provider() -> AsyncMockProvider:
    return AsyncMockProvider(
        config=ProviderConfig(
            name="slow",
            api_key="key",
            models=["slow-model"],
            capabilities=[ModelCapability.CHAT],
            priority=2,
        ),
        latency_ms=50.0,
        response_content="Slow response",
    )


@pytest.fixture
def failing_provider() -> AsyncMockProvider:
    return AsyncMockProvider(
        config=ProviderConfig(
            name="failing",
            api_key="key",
            models=["fail-model"],
            capabilities=[ModelCapability.CHAT],
            priority=1,
        ),
        fail=True,
    )


class TestAsyncMockProvider:
    """Tests for the async mock provider."""

    @pytest.mark.asyncio
    async def test_basic_completion(self, mock_provider: AsyncMockProvider) -> None:
        result = await mock_provider.complete(
            [{"role": "user", "content": "Hello"}]
        )
        assert result.success
        assert result.content == "Mock async response"
        assert result.provider == "async-mock"

    @pytest.mark.asyncio
    async def test_failure_mode(self, failing_provider: AsyncMockProvider) -> None:
        with pytest.raises(RuntimeError, match="configured to fail"):
            await failing_provider.complete([{"role": "user", "content": "Hi"}])

    def test_provider_properties(self, mock_provider: AsyncMockProvider) -> None:
        assert mock_provider.name == "async-mock"
        assert "mock-model" in mock_provider.models
        assert mock_provider.default_model == "mock-model"

    def test_supports_capabilities(self, mock_provider: AsyncMockProvider) -> None:
        assert mock_provider.supports({"chat"})
        assert not mock_provider.supports({"vision"})


class TestAsyncLLMRouter:
    """Tests for the async router."""

    @pytest.mark.asyncio
    async def test_basic_routing(self, mock_provider: AsyncMockProvider) -> None:
        router = AsyncLLMRouter(
            providers={"async-mock": mock_provider},
            policy=FallbackPolicy(),
        )
        result = await router.route(
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert result.success
        assert result.content == "Mock async response"

    @pytest.mark.asyncio
    async def test_fallback_on_failure(
        self,
        failing_provider: AsyncMockProvider,
        slow_provider: AsyncMockProvider,
    ) -> None:
        router = AsyncLLMRouter(
            providers={
                "failing": failing_provider,
                "slow": slow_provider,
            },
            policy=FallbackPolicy(),
            fallback=True,
        )
        result = await router.route(
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result.success
        assert result.provider == "slow"

    @pytest.mark.asyncio
    async def test_no_fallback_raises(
        self,
        failing_provider: AsyncMockProvider,
    ) -> None:
        router = AsyncLLMRouter(
            providers={"failing": failing_provider},
            policy=FallbackPolicy(),
            fallback=False,
        )
        with pytest.raises(RuntimeError):
            await router.route(messages=[{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_all_providers_fail(
        self,
        failing_provider: AsyncMockProvider,
    ) -> None:
        fail2 = AsyncMockProvider(
            config=ProviderConfig(
                name="fail2", api_key="k", models=["m"],
                capabilities=[ModelCapability.CHAT], priority=2,
            ),
            fail=True,
        )
        router = AsyncLLMRouter(
            providers={"failing": failing_provider, "fail2": fail2},
            policy=FallbackPolicy(),
            fallback=True,
        )
        with pytest.raises(RuntimeError, match="All providers failed"):
            await router.route(messages=[{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_concurrent_routing(self, mock_provider: AsyncMockProvider) -> None:
        router = AsyncLLMRouter(
            providers={"async-mock": mock_provider},
            policy=FallbackPolicy(),
        )
        requests = [
            {"messages": [{"role": "user", "content": f"Request {i}"}]}
            for i in range(5)
        ]
        results = await router.route_concurrent(requests)
        assert len(results) == 5
        assert all(r.success for r in results)

    def test_provider_management(self, mock_provider: AsyncMockProvider) -> None:
        router = AsyncLLMRouter(
            providers={"async-mock": mock_provider},
            policy=FallbackPolicy(),
        )
        assert "async-mock" in router.provider_names

        router.remove_provider("async-mock")
        assert "async-mock" not in router.provider_names

        router.add_provider(mock_provider)
        assert "async-mock" in router.provider_names

    def test_health_reporting(self, mock_provider: AsyncMockProvider) -> None:
        router = AsyncLLMRouter(
            providers={"async-mock": mock_provider},
            policy=FallbackPolicy(),
        )
        health = router.get_health("async-mock")
        assert "provider" in health
        assert "is_healthy" in health

    @pytest.mark.asyncio
    async def test_no_providers_raises(self) -> None:
        router = AsyncLLMRouter(providers={}, policy=FallbackPolicy())
        with pytest.raises(RuntimeError, match="No providers available"):
            await router.route(messages=[{"role": "user", "content": "Hi"}])
