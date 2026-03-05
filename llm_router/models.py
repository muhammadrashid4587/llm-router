"""Pydantic models for routing requests, results, provider config, and capabilities."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModelCapability(str, Enum):
    """Capabilities a model may support."""

    CHAT = "chat"
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"
    JSON_MODE = "json_mode"
    EMBEDDINGS = "embeddings"
    CODE = "code"


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    name: str
    api_key: str = ""
    models: list[str] = Field(default_factory=list)
    capabilities: list[ModelCapability] = Field(default_factory=list)
    base_url: str | None = None
    priority: int = Field(default=0, description="Lower number = higher priority for fallback")
    max_retries: int = 3
    timeout_seconds: float = 30.0
    extra: dict[str, Any] = Field(default_factory=dict)


class RoutingRequest(BaseModel):
    """An incoming request to be routed to a provider."""

    messages: list[dict[str, Any]]
    model: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    max_cost_per_1k_tokens: float | None = None
    max_latency_ms: float | None = None
    preferred_providers: list[str] = Field(default_factory=list)
    excluded_providers: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def required_capabilities(self) -> set[str]:
        return set(self.capabilities)


class RoutingResult(BaseModel):
    """The result returned after routing a request."""

    provider: str
    model: str
    content: str = ""
    cost: float = 0.0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderScore(BaseModel):
    """Internal scoring of a provider for routing decisions."""

    provider_name: str
    model: str
    score: float = 0.0
    estimated_cost_per_1k: float = 0.0
    estimated_latency_ms: float = 0.0
    meets_constraints: bool = True
    reason: str = ""


class HealthStatus(BaseModel):
    """Health status snapshot for a provider."""

    provider_name: str
    is_healthy: bool = True
    consecutive_failures: int = 0
    total_requests: int = 0
    total_failures: int = 0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    circuit_open: bool = False
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
