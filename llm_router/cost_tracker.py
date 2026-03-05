"""Cost tracking and budget management for LLM routing.

Aggregates token costs across requests with per-provider and per-model
breakdowns.  Thread-safe for use in concurrent applications.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CostRecord:
    """A single cost entry."""

    provider: str
    model: str
    cost: float
    tokens: int
    timestamp: float


@dataclass
class CostSummary:
    """Aggregated cost statistics."""

    total_cost: float = 0.0
    total_tokens: int = 0
    request_count: int = 0
    avg_cost_per_request: float = 0.0
    per_provider: Dict[str, float] = field(default_factory=dict)
    per_model: Dict[str, float] = field(default_factory=dict)


class CostTracker:
    """Thread-safe cost tracker with budget management.

    Usage::

        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", cost=0.003, tokens=150)
        summary = tracker.get_summary()
        print(f"Total spend: ${summary.total_cost:.4f}")
    """

    def __init__(self) -> None:
        self._records: List[CostRecord] = []
        self._lock = threading.Lock()

    def record(
        self,
        provider: str,
        model: str,
        cost: float,
        tokens: int = 0,
    ) -> None:
        """Record a request's cost.

        Args:
            provider: Provider name (e.g. ``"openai"``).
            model: Model identifier (e.g. ``"gpt-4o"``).
            cost: Total cost in USD.
            tokens: Total tokens consumed.
        """
        entry = CostRecord(
            provider=provider,
            model=model,
            cost=cost,
            tokens=tokens,
            timestamp=time.time(),
        )
        with self._lock:
            self._records.append(entry)

    def get_summary(self) -> CostSummary:
        """Return an aggregated cost summary across all recorded requests."""
        with self._lock:
            records = list(self._records)

        if not records:
            return CostSummary()

        total_cost = sum(r.cost for r in records)
        total_tokens = sum(r.tokens for r in records)
        count = len(records)

        per_provider: Dict[str, float] = {}
        per_model: Dict[str, float] = {}
        for r in records:
            per_provider[r.provider] = per_provider.get(r.provider, 0.0) + r.cost
            per_model[r.model] = per_model.get(r.model, 0.0) + r.cost

        return CostSummary(
            total_cost=total_cost,
            total_tokens=total_tokens,
            request_count=count,
            avg_cost_per_request=total_cost / count,
            per_provider=per_provider,
            per_model=per_model,
        )

    def get_provider_costs(self) -> Dict[str, float]:
        """Return per-provider cost totals."""
        return self.get_summary().per_provider

    def get_model_costs(self) -> Dict[str, float]:
        """Return per-model cost totals."""
        return self.get_summary().per_model

    def budget_remaining(self, budget: float) -> float:
        """Return how much of the budget remains.

        Args:
            budget: Total budget in USD.

        Returns:
            Remaining budget (may be negative if over budget).
        """
        return budget - self.get_summary().total_cost

    def is_over_budget(self, budget: float) -> bool:
        """Check if spending has exceeded the budget.

        Args:
            budget: Total budget in USD.
        """
        return self.get_summary().total_cost > budget

    def get_windowed_cost(self, window_seconds: float) -> float:
        """Return the total cost within a recent time window.

        Args:
            window_seconds: Number of seconds to look back.

        Returns:
            Total cost within the window.
        """
        cutoff = time.time() - window_seconds
        with self._lock:
            return sum(r.cost for r in self._records if r.timestamp >= cutoff)

    def reset(self) -> None:
        """Clear all tracked records."""
        with self._lock:
            self._records.clear()
