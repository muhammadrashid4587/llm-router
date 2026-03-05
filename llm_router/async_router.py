"""Async LLM router for non-blocking concurrent request handling.

Mirrors :class:`LLMRouter` but operates entirely with ``async/await``,
enabling high-throughput routing in production services.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from llm_router.async_providers import AsyncLLMProvider, AsyncMockProvider
from llm_router.health import HealthChecker
from llm_router.models import (
    ModelCapability,
    ProviderConfig,
    RoutingRequest,
    RoutingResult,
)
from llm_router.policies import FallbackPolicy, RoutingPolicy

logger = logging.getLogger(__name__)


class AsyncLLMRouter:
    """Async-first LLM router that picks the best provider per request.

    Usage::

        router = AsyncLLMRouter(providers={"mock": mock_provider}, policy=policy)
        result = await router.route(messages=[{"role": "user", "content": "Hi"}])
    """

    def __init__(
        self,
        providers: dict[str, AsyncLLMProvider],
        policy: RoutingPolicy,
        *,
        fallback: bool = False,
        health: HealthChecker | None = None,
    ) -> None:
        self._providers = dict(providers)
        self._policy = policy
        self._fallback = fallback
        self._health = health or HealthChecker()

    @property
    def provider_names(self) -> list[str]:
        return list(self._providers.keys())

    def add_provider(self, provider: AsyncLLMProvider) -> None:
        self._providers[provider.name] = provider

    def remove_provider(self, name: str) -> None:
        self._providers.pop(name, None)

    def get_health(self, provider: str) -> dict[str, Any]:
        status = self._health.get_status(provider)
        return {
            "provider": status.provider_name,
            "is_healthy": status.is_healthy,
            "avg_latency_ms": status.avg_latency_ms,
            "error_rate": status.error_rate,
            "circuit_open": status.circuit_open,
        }

    async def route(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        capabilities: list[str] | None = None,
        **kwargs: Any,
    ) -> RoutingResult:
        """Route a request to the best available provider.

        Args:
            messages: Chat messages.
            model: Optional specific model override.
            capabilities: Required capabilities (e.g. ``["vision"]``).
            **kwargs: Forwarded to the provider's ``complete()`` method.

        Returns:
            A :class:`RoutingResult` from the selected provider.

        Raises:
            RuntimeError: If no provider is available or all providers fail.
        """
        cap_set = set(capabilities or [])

        # Build candidate list based on policy.
        request = RoutingRequest(
            messages=messages,
            model=model,
            capabilities=list(cap_set) if cap_set else [],
        )

        candidates = self._policy.select(
            self._providers,
            request,
            self._health,
        )

        if not candidates:
            raise RuntimeError("No providers available for this request")

        last_error: Exception | None = None
        for provider_name, chosen_model in candidates:
            provider = self._providers.get(provider_name)
            if provider is None:
                continue

            try:
                result = await provider.complete(
                    messages, model=chosen_model or model, **kwargs,
                )
                self._health.record_success(provider_name, result.latency_ms)
                return result
            except Exception as exc:
                logger.warning("Provider %s failed: %s", provider_name, exc)
                self._health.record_failure(provider_name)
                last_error = exc
                if not self._fallback:
                    raise RuntimeError(
                        f"Provider {provider_name} failed: {exc}"
                    ) from exc

        raise RuntimeError(
            f"All providers failed. Last error: {last_error}"
        ) from last_error

    async def route_concurrent(
        self,
        requests: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[RoutingResult]:
        """Route multiple requests concurrently using asyncio.gather.

        Args:
            requests: List of dicts, each with at least a ``messages`` key.
            **kwargs: Forwarded to each ``route()`` call.

        Returns:
            A list of :class:`RoutingResult` in the same order as the input.
        """
        tasks = [
            self.route(
                messages=req["messages"],
                model=req.get("model"),
                capabilities=req.get("capabilities"),
                **kwargs,
            )
            for req in requests
        ]
        return list(await asyncio.gather(*tasks))
