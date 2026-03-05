"""Async LLM provider implementations.

Mirrors the synchronous providers in ``providers.py`` but uses ``async/await``
for non-blocking I/O, critical for production LLM applications that must
handle many concurrent requests.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

from llm_router.models import ModelCapability, ProviderConfig, RoutingResult
from llm_router.pricing import estimate_cost


class AsyncLLMProvider(ABC):
    """Abstract base class for async LLM providers."""

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
    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> RoutingResult:
        """Send a chat-completion request and return a RoutingResult."""
        ...

    @property
    def default_model(self) -> str:
        return self.models[0] if self.models else "unknown"


class AsyncMockProvider(AsyncLLMProvider):
    """Async mock provider for testing."""

    def __init__(
        self,
        config: ProviderConfig | None = None,
        latency_ms: float = 50.0,
        response_content: str = "Mock async response",
        fail: bool = False,
    ) -> None:
        if config is None:
            config = ProviderConfig(
                name="async-mock",
                api_key="mock-key",
                models=["mock-model"],
                capabilities=[ModelCapability.CHAT, ModelCapability.FUNCTION_CALLING],
            )
        super().__init__(config)
        self._latency_ms = latency_ms
        self._response_content = response_content
        self._fail = fail

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> RoutingResult:
        await asyncio.sleep(self._latency_ms / 1000.0)

        if self._fail:
            raise RuntimeError("AsyncMockProvider configured to fail")

        return RoutingResult(
            provider=self.name,
            model=model or self.default_model,
            content=self._response_content,
            cost=0.0001,
            latency_ms=self._latency_ms,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            success=True,
        )


class AsyncOpenAIProvider(AsyncLLMProvider):
    """Async OpenAI provider using ``openai.AsyncOpenAI``."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        try:
            import openai
            self._client = openai.AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
            )
        except ImportError as exc:
            raise ImportError(
                "The openai package is required for AsyncOpenAIProvider. "
                "Install it with: pip install openai"
            ) from exc

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> RoutingResult:
        model = model or self.default_model
        start = time.perf_counter()

        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=self.config.timeout_seconds,
        )

        latency_ms = (time.perf_counter() - start) * 1000.0
        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice and choice.message else ""

        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        cost = estimate_cost(model, prompt_tokens, completion_tokens)

        return RoutingResult(
            provider=self.name,
            model=model,
            content=content or "",
            cost=cost,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            success=True,
        )


class AsyncAnthropicProvider(AsyncLLMProvider):
    """Async Anthropic provider using ``anthropic.AsyncAnthropic``."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=config.api_key)
        except ImportError as exc:
            raise ImportError(
                "The anthropic package is required for AsyncAnthropicProvider. "
                "Install it with: pip install anthropic"
            ) from exc

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> RoutingResult:
        model = model or self.default_model
        start = time.perf_counter()

        # Anthropic uses system as a separate parameter.
        system_msg = ""
        api_messages = []
        for m in messages:
            if m.get("role") == "system":
                system_msg = m.get("content", "")
            else:
                api_messages.append(m)

        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "messages": api_messages,
            "timeout": self.config.timeout_seconds,
        }
        if system_msg:
            create_kwargs["system"] = system_msg

        response = await self._client.messages.create(**create_kwargs)

        latency_ms = (time.perf_counter() - start) * 1000.0
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        prompt_tokens = response.usage.input_tokens if response.usage else 0
        completion_tokens = response.usage.output_tokens if response.usage else 0
        cost = estimate_cost(model, prompt_tokens, completion_tokens)

        return RoutingResult(
            provider=self.name,
            model=model,
            content=content,
            cost=cost,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            success=True,
        )
