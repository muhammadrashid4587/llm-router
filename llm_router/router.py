"""LLMRouter -- the main entry point that routes requests to providers."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from llm_router.health import HealthChecker
from llm_router.middleware import MiddlewareChain, Middleware, RetryMiddleware
from llm_router.models import ProviderConfig, RoutingRequest, RoutingResult
from llm_router.policies import FallbackPolicy, RoutingPolicy
from llm_router.pricing import estimate_cost
from llm_router.providers import LLMProvider, MockProvider, ProviderRegistry

logger = logging.getLogger("llm_router")


class LLMRouter:
    """Smart router that dispatches LLM requests to the best provider.

    Parameters
    ----------
    providers:
        Either a :class:`ProviderRegistry` or a dict of
        ``{name: config_dict}`` that will be turned into one.
    policy:
        The :class:`RoutingPolicy` used to rank providers.
    fallback:
        If *True* (default) the router tries the next candidate when the
        selected provider fails.  Equivalent to wrapping the policy output
        in a fallback loop.
    health_checker:
        Optional pre-built :class:`HealthChecker`.  One is created
        automatically if not supplied.
    middlewares:
        Optional list of :class:`Middleware` hooks.
    failure_threshold:
        Number of consecutive failures before the circuit breaker opens.
    recovery_timeout:
        Seconds before a tripped circuit breaker allows a probe request.
    """

    def __init__(
        self,
        providers: ProviderRegistry | dict[str, dict[str, Any]],
        policy: RoutingPolicy | None = None,
        fallback: bool = True,
        health_checker: HealthChecker | None = None,
        middlewares: list[Middleware] | None = None,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        # Build registry
        if isinstance(providers, dict):
            self._registry = ProviderRegistry.from_configs(providers)
        else:
            self._registry = providers

        self.policy = policy or FallbackPolicy()
        self.fallback = fallback
        self.health = health_checker or HealthChecker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
        self._mw_chain = MiddlewareChain(middlewares)
        self._lock = threading.Lock()

    # -- convenience properties ------------------------------------------------

    @property
    def providers(self) -> dict[str, LLMProvider]:
        return self._registry.all()

    @property
    def provider_names(self) -> list[str]:
        return self._registry.names()

    # -- main routing method ---------------------------------------------------

    def route(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        capabilities: list[str] | None = None,
        max_cost_per_1k_tokens: float | None = None,
        max_latency_ms: float | None = None,
        preferred_providers: list[str] | None = None,
        excluded_providers: list[str] | None = None,
        **kwargs: Any,
    ) -> RoutingResult:
        """Route a chat-completion request to the best available provider.

        Raises
        ------
        RuntimeError
            If no provider can serve the request (all failed or none matched).
        """
        request = RoutingRequest(
            messages=messages,
            model=model,
            capabilities=capabilities or [],
            max_cost_per_1k_tokens=max_cost_per_1k_tokens,
            max_latency_ms=max_latency_ms,
            preferred_providers=preferred_providers or [],
            excluded_providers=excluded_providers or [],
            extra=kwargs,
        )

        # Run pre-request middleware
        request = self._mw_chain.run_before(request)

        # Get ordered candidate list from policy
        candidates = self.policy.select(self.providers, request, self.health)

        if not candidates:
            raise RuntimeError(
                "No providers available for the request "
                f"(capabilities={request.capabilities}, "
                f"excluded={request.excluded_providers})"
            )

        last_error: Exception | None = None

        attempts = candidates if self.fallback else candidates[:1]

        for provider_name, model_name in attempts:
            provider = self._registry.get(provider_name)
            if provider is None:
                continue

            try:
                start = time.perf_counter()
                result = provider.complete(messages=request.messages, model=model_name, **kwargs)
                latency_ms = (time.perf_counter() - start) * 1000

                # Update health stats
                self.health.record_success(provider_name, latency_ms)

                # Ensure latency is recorded on the result
                if result.latency_ms == 0:
                    result.latency_ms = latency_ms

                # Run post-request middleware
                result = self._mw_chain.run_after(request, result)
                return result

            except Exception as exc:
                last_error = exc
                self.health.record_failure(provider_name)
                self._mw_chain.run_on_error(request, exc)
                logger.warning(
                    "Provider %s failed: %s -- trying next", provider_name, exc
                )
                continue

        raise RuntimeError(
            f"All providers failed. Last error: {last_error}"
        )

    # -- admin helpers ---------------------------------------------------------

    def add_provider(self, provider: LLMProvider) -> None:
        """Register a new provider at runtime."""
        self._registry.register(provider)

    def remove_provider(self, name: str) -> None:
        """Remove a provider at runtime."""
        self._registry.remove(name)

    def get_health(self, provider: str | None = None) -> dict[str, Any]:
        """Return health status for one or all providers."""
        if provider:
            return self.health.get_status(provider).model_dump()
        return {
            name: self.health.get_status(name).model_dump()
            for name in self.provider_names
        }
