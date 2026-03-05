"""Routing policies: CostOptimized, LatencyOptimized, RoundRobin, Fallback.

Each policy implements ``select`` which receives the available providers, the
routing request, and a health checker, and returns an ordered list of
(provider_name, model) tuples to try.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from llm_router.models import ProviderScore, RoutingRequest
from llm_router.pricing import cost_per_1k

if TYPE_CHECKING:
    from llm_router.health import HealthChecker
    from llm_router.providers import LLMProvider


class RoutingPolicy(ABC):
    """Abstract routing policy."""

    @abstractmethod
    def select(
        self,
        providers: dict[str, LLMProvider],
        request: RoutingRequest,
        health: HealthChecker,
    ) -> list[tuple[str, str]]:
        """Return an ordered list of ``(provider_name, model)`` to try."""


def _candidate_models(
    providers: dict[str, LLMProvider],
    request: RoutingRequest,
    health: HealthChecker,
) -> list[tuple[str, str]]:
    """Return provider/model pairs that satisfy the request constraints."""
    required = request.required_capabilities
    candidates: list[tuple[str, str]] = []

    for name, provider in providers.items():
        # Exclude explicitly excluded providers
        if name in request.excluded_providers:
            continue
        # Skip unhealthy providers
        if not health.is_healthy(name):
            continue
        # Capability check
        if required and not provider.supports(required):
            continue
        # Use requested model or the default
        model = request.model or provider.default_model()
        candidates.append((name, model))

    return candidates


# ---------------------------------------------------------------------------
# CostOptimizedPolicy
# ---------------------------------------------------------------------------

class CostOptimizedPolicy(RoutingPolicy):
    """Pick the cheapest provider/model that meets the constraints."""

    def __init__(self, max_cost_per_1k: float | None = None) -> None:
        self.max_cost_per_1k = max_cost_per_1k

    def select(
        self,
        providers: dict[str, LLMProvider],
        request: RoutingRequest,
        health: HealthChecker,
    ) -> list[tuple[str, str]]:
        candidates = _candidate_models(providers, request, health)

        scored: list[tuple[float, str, str]] = []
        for pname, model in candidates:
            cpk = cost_per_1k(model)
            # Apply max_cost constraint from policy or request
            max_cost = request.max_cost_per_1k_tokens or self.max_cost_per_1k
            if max_cost is not None and cpk > max_cost:
                continue
            scored.append((cpk, pname, model))

        scored.sort(key=lambda t: t[0])
        return [(pname, model) for _, pname, model in scored]


# ---------------------------------------------------------------------------
# LatencyOptimizedPolicy
# ---------------------------------------------------------------------------

class LatencyOptimizedPolicy(RoutingPolicy):
    """Pick the provider with the lowest observed latency."""

    def __init__(self, max_latency_ms: float | None = None) -> None:
        self.max_latency_ms = max_latency_ms

    def select(
        self,
        providers: dict[str, LLMProvider],
        request: RoutingRequest,
        health: HealthChecker,
    ) -> list[tuple[str, str]]:
        candidates = _candidate_models(providers, request, health)

        scored: list[tuple[float, str, str]] = []
        for pname, model in candidates:
            avg_lat = health.get_avg_latency(pname)
            max_lat = request.max_latency_ms or self.max_latency_ms
            if max_lat is not None and avg_lat > max_lat and avg_lat != float("inf"):
                continue
            scored.append((avg_lat, pname, model))

        scored.sort(key=lambda t: t[0])
        return [(pname, model) for _, pname, model in scored]


# ---------------------------------------------------------------------------
# RoundRobinPolicy
# ---------------------------------------------------------------------------

class RoundRobinPolicy(RoutingPolicy):
    """Distribute requests evenly across healthy providers."""

    def __init__(self) -> None:
        self._index = 0
        self._lock = threading.Lock()

    def select(
        self,
        providers: dict[str, LLMProvider],
        request: RoutingRequest,
        health: HealthChecker,
    ) -> list[tuple[str, str]]:
        candidates = _candidate_models(providers, request, health)
        if not candidates:
            return []

        with self._lock:
            idx = self._index % len(candidates)
            self._index += 1

        # Rotate so the selected candidate is first, rest follow in order
        ordered = candidates[idx:] + candidates[:idx]
        return ordered


# ---------------------------------------------------------------------------
# FallbackPolicy
# ---------------------------------------------------------------------------

class FallbackPolicy(RoutingPolicy):
    """Try providers in priority order (lower ProviderConfig.priority first)."""

    def select(
        self,
        providers: dict[str, LLMProvider],
        request: RoutingRequest,
        health: HealthChecker,
    ) -> list[tuple[str, str]]:
        candidates = _candidate_models(providers, request, health)

        # Sort by priority on the ProviderConfig
        def _priority(pair: tuple[str, str]) -> int:
            prov = providers[pair[0]]
            return prov.config.priority

        candidates.sort(key=_priority)

        # Respect preferred_providers: move them to the front
        if request.preferred_providers:
            preferred = []
            rest = []
            for c in candidates:
                if c[0] in request.preferred_providers:
                    preferred.append(c)
                else:
                    rest.append(c)
            candidates = preferred + rest

        return candidates
