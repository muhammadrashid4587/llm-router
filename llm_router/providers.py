"""Provider registry and abstract base class.

Concrete implementations for OpenAI, Anthropic, and a mock provider used in
tests and local development.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from llm_router.models import ModelCapability, ProviderConfig, RoutingResult


class LLMProvider(ABC):
    """Abstract base for an LLM provider."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def models(self) -> list[str]:
        return self.config.models

    @property
    def capabilities(self) -> set[str]:
        return {c.value for c in self.config.capabilities}

    def supports(self, required: set[str]) -> bool:
        """Return True if this provider supports all *required* capabilities."""
        return required.issubset(self.capabilities)

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> RoutingResult:
        """Send a chat-completion request and return a RoutingResult."""

    def default_model(self) -> str:
        """Return the first model in the config list, or a sensible fallback."""
        return self.models[0] if self.models else "unknown"


# ---------------------------------------------------------------------------
# Mock provider (for tests / offline work)
# ---------------------------------------------------------------------------

class MockProvider(LLMProvider):
    """A provider that returns canned responses -- useful for testing."""

    def __init__(
        self,
        config: ProviderConfig | None = None,
        *,
        name: str = "mock",
        latency_ms: float = 10.0,
        response_content: str = "Mock response",
        fail: bool = False,
        fail_message: str = "mock failure",
    ) -> None:
        if config is None:
            config = ProviderConfig(
                name=name,
                models=["mock-model"],
                capabilities=[
                    ModelCapability.CHAT,
                    ModelCapability.FUNCTION_CALLING,
                ],
            )
        super().__init__(config)
        self._latency_ms = latency_ms
        self._response_content = response_content
        self._fail = fail
        self._fail_message = fail_message
        self.call_count = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> RoutingResult:
        self.call_count += 1
        used_model = model or self.default_model()

        if self._fail:
            raise RuntimeError(self._fail_message)

        # Simulate latency
        time.sleep(self._latency_ms / 1000)

        prompt_tokens = sum(len(m.get("content", "").split()) for m in messages) * 2
        completion_tokens = len(self._response_content.split()) * 2

        return RoutingResult(
            provider=self.name,
            model=used_model,
            content=self._response_content,
            cost=0.0,
            latency_ms=self._latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            success=True,
        )


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """Provider backed by the OpenAI Python SDK."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "The openai package is required for OpenAIProvider. "
                    "Install it with: pip install llm-router[openai]"
                ) from exc
            kwargs: dict[str, Any] = {"api_key": self.config.api_key}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> RoutingResult:
        client = self._get_client()
        used_model = model or self.default_model()

        start = time.perf_counter()
        response = client.chat.completions.create(
            model=used_model,
            messages=messages,
            **kwargs,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        usage = response.usage

        from llm_router.pricing import estimate_cost

        return RoutingResult(
            provider=self.name,
            model=used_model,
            content=choice.message.content or "",
            cost=estimate_cost(used_model, usage.prompt_tokens, usage.completion_tokens),
            latency_ms=latency_ms,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            success=True,
        )


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """Provider backed by the Anthropic Python SDK."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import Anthropic  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "The anthropic package is required for AnthropicProvider. "
                    "Install it with: pip install llm-router[anthropic]"
                ) from exc
            kwargs: dict[str, Any] = {"api_key": self.config.api_key}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = Anthropic(**kwargs)
        return self._client

    def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> RoutingResult:
        client = self._get_client()
        used_model = model or self.default_model()

        # Anthropic API uses a different message format: extract system if present
        system_msg = ""
        api_messages = []
        for m in messages:
            if m.get("role") == "system":
                system_msg = m.get("content", "")
            else:
                api_messages.append(m)

        create_kwargs: dict[str, Any] = {
            "model": used_model,
            "messages": api_messages,
            "max_tokens": kwargs.pop("max_tokens", 1024),
        }
        if system_msg:
            create_kwargs["system"] = system_msg
        create_kwargs.update(kwargs)

        start = time.perf_counter()
        response = client.messages.create(**create_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        from llm_router.pricing import estimate_cost

        return RoutingResult(
            provider=self.name,
            model=used_model,
            content=content,
            cost=estimate_cost(
                used_model, response.usage.input_tokens, response.usage.output_tokens
            ),
            latency_ms=latency_ms,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            success=True,
        )


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_BUILTIN_PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "mock": MockProvider,
}


class ProviderRegistry:
    """Registry that maps provider names to LLMProvider instances."""

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}

    # -- public API -----------------------------------------------------------

    def register(self, provider: LLMProvider) -> None:
        self._providers[provider.name] = provider

    def get(self, name: str) -> LLMProvider | None:
        return self._providers.get(name)

    def all(self) -> dict[str, LLMProvider]:
        return dict(self._providers)

    def names(self) -> list[str]:
        return list(self._providers.keys())

    def remove(self, name: str) -> None:
        self._providers.pop(name, None)

    def __len__(self) -> int:
        return len(self._providers)

    def __contains__(self, name: str) -> bool:
        return name in self._providers

    # -- factory helpers ------------------------------------------------------

    @classmethod
    def from_configs(cls, configs: dict[str, dict[str, Any]]) -> ProviderRegistry:
        """Build a registry from a dict of ``{name: config_dict}``."""
        registry = cls()
        for name, raw in configs.items():
            raw = dict(raw)  # avoid mutating the caller's dict
            provider_type = raw.pop("provider_type", name)
            provider_cls = _BUILTIN_PROVIDERS.get(provider_type)
            if provider_cls is None:
                raise ValueError(
                    f"Unknown provider type {provider_type!r}. "
                    f"Available: {list(_BUILTIN_PROVIDERS)}"
                )
            config = ProviderConfig(name=name, **raw)
            registry.register(provider_cls(config))
        return registry
