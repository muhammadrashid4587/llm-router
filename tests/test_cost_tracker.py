"""Tests for the cost tracker module."""

from __future__ import annotations

import threading
import time

import pytest

from llm_router.cost_tracker import CostTracker, CostSummary


class TestCostTracker:
    """Tests for CostTracker."""

    def test_empty_tracker(self) -> None:
        tracker = CostTracker()
        summary = tracker.get_summary()
        assert summary.total_cost == 0.0
        assert summary.request_count == 0
        assert summary.per_provider == {}
        assert summary.per_model == {}

    def test_record_single(self) -> None:
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", cost=0.003, tokens=150)
        summary = tracker.get_summary()
        assert summary.total_cost == pytest.approx(0.003)
        assert summary.total_tokens == 150
        assert summary.request_count == 1
        assert summary.avg_cost_per_request == pytest.approx(0.003)

    def test_record_multiple(self) -> None:
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", cost=0.003, tokens=100)
        tracker.record("anthropic", "claude-sonnet", cost=0.005, tokens=200)
        tracker.record("openai", "gpt-4o-mini", cost=0.001, tokens=50)
        summary = tracker.get_summary()
        assert summary.total_cost == pytest.approx(0.009)
        assert summary.total_tokens == 350
        assert summary.request_count == 3
        assert summary.avg_cost_per_request == pytest.approx(0.003)

    def test_per_provider_breakdown(self) -> None:
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", cost=0.01, tokens=100)
        tracker.record("openai", "gpt-4o", cost=0.02, tokens=200)
        tracker.record("anthropic", "claude", cost=0.05, tokens=300)
        costs = tracker.get_provider_costs()
        assert costs["openai"] == pytest.approx(0.03)
        assert costs["anthropic"] == pytest.approx(0.05)

    def test_per_model_breakdown(self) -> None:
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", cost=0.01, tokens=100)
        tracker.record("openai", "gpt-4o-mini", cost=0.002, tokens=100)
        tracker.record("openai", "gpt-4o", cost=0.01, tokens=100)
        costs = tracker.get_model_costs()
        assert costs["gpt-4o"] == pytest.approx(0.02)
        assert costs["gpt-4o-mini"] == pytest.approx(0.002)

    def test_budget_remaining(self) -> None:
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", cost=0.30, tokens=100)
        assert tracker.budget_remaining(1.00) == pytest.approx(0.70)
        assert tracker.budget_remaining(0.20) == pytest.approx(-0.10)

    def test_is_over_budget(self) -> None:
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", cost=0.50, tokens=100)
        assert not tracker.is_over_budget(1.00)
        assert tracker.is_over_budget(0.40)

    def test_windowed_cost(self) -> None:
        tracker = CostTracker()
        # Record and immediately check — should be within window.
        tracker.record("openai", "gpt-4o", cost=0.01, tokens=100)
        assert tracker.get_windowed_cost(60.0) == pytest.approx(0.01)
        # A tiny window should still catch very recent records.
        assert tracker.get_windowed_cost(0.001) >= 0.0

    def test_reset(self) -> None:
        tracker = CostTracker()
        tracker.record("openai", "gpt-4o", cost=0.01, tokens=100)
        tracker.record("openai", "gpt-4o", cost=0.02, tokens=200)
        tracker.reset()
        summary = tracker.get_summary()
        assert summary.total_cost == 0.0
        assert summary.request_count == 0

    def test_thread_safety(self) -> None:
        """Concurrent recording should not lose data."""
        tracker = CostTracker()
        num_threads = 10
        records_per_thread = 100

        def worker() -> None:
            for _ in range(records_per_thread):
                tracker.record("p", "m", cost=0.001, tokens=1)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        summary = tracker.get_summary()
        assert summary.request_count == num_threads * records_per_thread
        assert summary.total_cost == pytest.approx(
            num_threads * records_per_thread * 0.001
        )
