"""Cost-per-token pricing data for routing decisions.

Prices are in USD per 1,000 tokens (prompt / completion).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Pricing for a single model."""

    prompt_per_1k: float
    completion_per_1k: float

    @property
    def avg_per_1k(self) -> float:
        """Average cost per 1k tokens (simple mean of prompt + completion)."""
        return (self.prompt_per_1k + self.completion_per_1k) / 2


# ---- Pricing catalogue (USD per 1,000 tokens) ----
# Updated with representative pricing; real prices change frequently.

PRICING: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o": ModelPricing(prompt_per_1k=0.0025, completion_per_1k=0.01),
    "gpt-4o-mini": ModelPricing(prompt_per_1k=0.00015, completion_per_1k=0.0006),
    "gpt-4-turbo": ModelPricing(prompt_per_1k=0.01, completion_per_1k=0.03),
    "gpt-4": ModelPricing(prompt_per_1k=0.03, completion_per_1k=0.06),
    "gpt-3.5-turbo": ModelPricing(prompt_per_1k=0.0005, completion_per_1k=0.0015),
    # Anthropic
    "claude-sonnet-4-20250514": ModelPricing(prompt_per_1k=0.003, completion_per_1k=0.015),
    "claude-3-5-haiku-20241022": ModelPricing(prompt_per_1k=0.001, completion_per_1k=0.005),
    "claude-3-opus-20240229": ModelPricing(prompt_per_1k=0.015, completion_per_1k=0.075),
    # Mock / testing
    "mock-model": ModelPricing(prompt_per_1k=0.0, completion_per_1k=0.0),
    "mock-cheap": ModelPricing(prompt_per_1k=0.0001, completion_per_1k=0.0002),
    "mock-expensive": ModelPricing(prompt_per_1k=0.05, completion_per_1k=0.10),
}


def get_pricing(model: str) -> ModelPricing:
    """Return pricing for *model*, falling back to a conservative default."""
    return PRICING.get(model, ModelPricing(prompt_per_1k=0.01, completion_per_1k=0.03))


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in USD for a given token count."""
    p = get_pricing(model)
    return (prompt_tokens / 1000) * p.prompt_per_1k + (completion_tokens / 1000) * p.completion_per_1k


def cost_per_1k(model: str) -> float:
    """Return the average cost per 1k tokens (used for quick comparisons)."""
    return get_pricing(model).avg_per_1k
