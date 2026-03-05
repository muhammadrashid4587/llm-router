# llm-router

A smart LLM request router that picks the best provider/model for each request based on configurable policies: cost optimization, latency targets, capability matching, and automatic fallbacks.

## Features

- **Multiple routing policies** -- cost-optimized, latency-optimized, round-robin, and fallback
- **Capability matching** -- route to providers that support vision, function calling, etc.
- **Circuit breaker** -- automatically stops sending requests to failing providers
- **Health tracking** -- monitors latency and error rates per provider
- **Middleware hooks** -- logging, rate limiting, retry logic
- **Thread-safe** -- safe for concurrent use

## Installation

```bash
pip install llm-router
```

With provider SDKs:

```bash
pip install llm-router[all]       # OpenAI + Anthropic
pip install llm-router[openai]    # OpenAI only
pip install llm-router[anthropic] # Anthropic only
```

## Quick Start

```python
from llm_router import LLMRouter, CostOptimizedPolicy

router = LLMRouter(
    providers={
        "openai": {
            "api_key": "sk-...",
            "models": ["gpt-4o", "gpt-4o-mini"],
        },
        "anthropic": {
            "api_key": "sk-ant-...",
            "models": ["claude-sonnet-4-20250514"],
        },
    },
    policy=CostOptimizedPolicy(max_cost_per_1k=0.01),
    fallback=True,
)

result = router.route(
    messages=[{"role": "user", "content": "Hello!"}],
    capabilities=["function_calling"],
)
print(result.provider)   # provider that was selected
print(result.model)      # model that was used
print(result.content)    # response content
print(result.cost)       # estimated cost
print(result.latency_ms) # round-trip latency
```

## Routing Policies

| Policy | Description |
|---|---|
| `CostOptimizedPolicy` | Picks the cheapest provider that meets constraints |
| `LatencyOptimizedPolicy` | Picks the provider with the lowest observed latency |
| `RoundRobinPolicy` | Distributes requests evenly across providers |
| `FallbackPolicy` | Tries providers in priority order until one succeeds |

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
