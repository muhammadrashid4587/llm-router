"""Health checker -- tracks provider latency, error rates, and implements a circuit breaker."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from llm_router.models import HealthStatus


@dataclass
class _ProviderStats:
    """Mutable statistics for one provider."""

    total_requests: int = 0
    total_failures: int = 0
    consecutive_failures: int = 0
    latencies: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    last_failure_time: float = 0.0
    last_success_time: float = 0.0


class CircuitBreaker:
    """Per-provider circuit breaker.

    * **Closed** (normal): requests flow through.
    * **Open**: after *failure_threshold* consecutive failures the circuit opens
      and requests are blocked for *recovery_timeout* seconds.
    * **Half-open**: after the timeout, one probe request is allowed.  If it
      succeeds the circuit closes; if it fails the circuit re-opens.
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._lock = threading.Lock()
        self._consecutive_failures: int = 0
        self._opened_at: float = 0.0
        self._state: str = "closed"  # closed | open | half_open

    # -- public API -----------------------------------------------------------

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._state == "open":
                if time.monotonic() - self._opened_at >= self.recovery_timeout:
                    self._state = "half_open"
                    return False  # allow a probe
                return True
            return False

    @property
    def state(self) -> str:
        # trigger timeout check
        _ = self.is_open
        with self._lock:
            return self._state

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._state = "closed"

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._state = "open"
                self._opened_at = time.monotonic()

    def reset(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._state = "closed"
            self._opened_at = 0.0


class HealthChecker:
    """Central health tracker for all providers.

    Thread-safe: each method acquires a lock before mutating state.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._lock = threading.Lock()
        self._stats: dict[str, _ProviderStats] = {}
        self._breakers: dict[str, CircuitBreaker] = {}

    # -- internal helpers -----------------------------------------------------

    def _ensure_provider(self, name: str) -> None:
        """Lazily create stats / breaker for a provider."""
        if name not in self._stats:
            self._stats[name] = _ProviderStats()
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                failure_threshold=self._failure_threshold,
                recovery_timeout=self._recovery_timeout,
            )

    # -- public API -----------------------------------------------------------

    def record_success(self, provider: str, latency_ms: float) -> None:
        with self._lock:
            self._ensure_provider(provider)
            stats = self._stats[provider]
            stats.total_requests += 1
            stats.consecutive_failures = 0
            stats.latencies.append(latency_ms)
            stats.last_success_time = time.time()
            self._breakers[provider].record_success()

    def record_failure(self, provider: str) -> None:
        with self._lock:
            self._ensure_provider(provider)
            stats = self._stats[provider]
            stats.total_requests += 1
            stats.total_failures += 1
            stats.consecutive_failures += 1
            stats.last_failure_time = time.time()
            self._breakers[provider].record_failure()

    def is_healthy(self, provider: str) -> bool:
        with self._lock:
            self._ensure_provider(provider)
            return not self._breakers[provider].is_open

    def get_status(self, provider: str) -> HealthStatus:
        with self._lock:
            self._ensure_provider(provider)
            stats = self._stats[provider]
            breaker = self._breakers[provider]
            avg_lat = (
                sum(stats.latencies) / len(stats.latencies) if stats.latencies else 0.0
            )
            err_rate = (
                stats.total_failures / stats.total_requests
                if stats.total_requests > 0
                else 0.0
            )
            return HealthStatus(
                provider_name=provider,
                is_healthy=not breaker.is_open,
                consecutive_failures=stats.consecutive_failures,
                total_requests=stats.total_requests,
                total_failures=stats.total_failures,
                avg_latency_ms=avg_lat,
                error_rate=err_rate,
                circuit_open=breaker.is_open,
                last_failure_time=stats.last_failure_time,
                last_success_time=stats.last_success_time,
            )

    def get_avg_latency(self, provider: str) -> float:
        with self._lock:
            self._ensure_provider(provider)
            lats = self._stats[provider].latencies
            return sum(lats) / len(lats) if lats else float("inf")

    def reset(self, provider: str | None = None) -> None:
        with self._lock:
            if provider is None:
                self._stats.clear()
                self._breakers.clear()
            else:
                self._stats.pop(provider, None)
                self._breakers.pop(provider, None)
