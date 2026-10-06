"""Sol calls with fixed settings and Standard pricing."""

from math import isfinite
from time import perf_counter
from typing import Any

from openai import OpenAI

from models.common import Budget, Completion, ProviderError, TokenUsage

MODEL = "gpt-6.1-sol"
MAX_INPUT_TOKENS = 922_000
MAX_OUTPUT_TOKENS = 128_000
LONG_CONTEXT_THRESHOLD = 272_000
PRICING_URL = "https://developers.openai.com/api/docs/pricing"
# Standard rates per million tokens, checked on 2026-10-06.
SHORT_RATES = {"input": 2.0, "cache_hit": 0.10, "cache_write": 2.50, "output": 10.0}
LONG_RATES = {"input": 4.0, "cache_hit": 0.20, "cache_write": 5.0, "output": 15.0}


def _count(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Token counts must be nonnegative integers.")
    return value


def parse_usage(raw: dict[str, Any]) -> TokenUsage:
    """Keep cache reads and writes separate; reasoning is part of output."""
    input_tokens = _count(raw.get("prompt_tokens"))
    output_tokens = _count(raw.get("completion_tokens"))
    total = raw.get("total_tokens")
    if total is not None and _count(total) != input_tokens + output_tokens:
        raise ValueError("Total tokens do not match input and output.")
    details = raw.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    hits = 0 if cached is None else _count(cached)
    misses = input_tokens - hits
    if misses < 0:
        raise ValueError("Cache hits exceed the input token count.")
    writes = details.get("cache_write_tokens")
    if writes is not None:
        writes = _count(writes)
        if writes > misses:
            raise ValueError("Cache writes exceed input tokens that missed the cache.")
    reasoning = (raw.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if reasoning is not None:
        reasoning = _count(reasoning)
        if reasoning > output_tokens:
            raise ValueError("Reasoning tokens exceed the output token count.")
    if input_tokens > MAX_INPUT_TOKENS or output_tokens > MAX_OUTPUT_TOKENS:
        raise ValueError("Provider usage exceeds the model limits.")
    return TokenUsage(input_tokens, output_tokens, hits, misses, reasoning, writes)


def estimate_cost(usage: TokenUsage) -> float:
    """Charge all cache misses as writes when their split is not reported."""
    rates = LONG_RATES if usage.input_tokens > LONG_CONTEXT_THRESHOLD else SHORT_RATES
    writes = usage.cache_write_tokens
    if writes is None:
        writes = usage.cache_miss_tokens
    ordinary = usage.cache_miss_tokens - writes
    return (
        ordinary * rates["input"]
        + usage.cache_hit_tokens * rates["cache_hit"]
        + writes * rates["cache_write"]
        + usage.output_tokens * rates["output"]
    ) / 1_000_000


class SolClient:
    def __init__(
        self,
        api_key: str,
        budget: Budget,
        max_output_tokens: int = 8192,
        timeout_seconds: float = 120,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("OPENAI_API_KEY is required.")
        if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= MAX_OUTPUT_TOKENS:
            raise ValueError(f"Output limit must be between 1 and {MAX_OUTPUT_TOKENS}.")
        if not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Timeout must be finite and positive.")
        self.budget = budget
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            timeout=timeout_seconds,
            max_retries=0,
        )

    @property
    def request_reserve_usd(self) -> float:
        # Reserve the full model limits at the highest Standard token rates.
        return (
            MAX_INPUT_TOKENS * LONG_RATES["cache_write"] + MAX_OUTPUT_TOKENS * LONG_RATES["output"]
        ) / 1_000_000

    @property
    def config(self) -> dict[str, Any]:
        return {
            "model": MODEL,
            "base_url": "https://api.openai.com/v1",
            "reasoning_effort": "high",
            "max_output_tokens": self.max_output_tokens,
            "timeout_seconds": self.timeout_seconds,
            "response_format": "json_object",
            "service_tier": "default",
            "store": False,
            "max_retries": 0,
            "pricing_basis": "Standard; if cache-write counts are unknown, price all cache misses as writes",
            "pricing_checked": "2026-10-06",
            "pricing_url": PRICING_URL,
            "short_context_rates_per_million": SHORT_RATES.copy(),
            "long_context_rates_per_million": LONG_RATES.copy(),
            "long_context_threshold": LONG_CONTEXT_THRESHOLD,
            "request_reserve_usd": self.request_reserve_usd,
        }

    def complete(self, messages: list[dict[str, str]]) -> Completion:
        if not messages or any(
            set(message) != {"role", "content"}
            or message["role"] not in {"system", "user", "assistant"}
            or not isinstance(message["content"], str)
            for message in messages
        ):
            raise ValueError("Messages must contain only a text content and a valid role.")
        reserve = self.request_reserve_usd
        self.budget.reserve(reserve)
        start = perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_completion_tokens=self.max_output_tokens,
                reasoning_effort="high",
                response_format={"type": "json_object"},
                service_tier="default",
                store=False,
                stream=False,
            )
        except Exception as error:  # noqa: BLE001 - Never expose a provider error body.
            raise ProviderError(
                {
                    "error_type": type(error).__name__,
                    "status_code": getattr(error, "status_code", None),
                    "cost_usd": reserve,
                    "cost_uncertain": True,
                    "latency_seconds": perf_counter() - start,
                }
            ) from None

        latency = perf_counter() - start
        raw: dict[str, Any] = {}
        cost = reserve
        cost_uncertain = True
        try:
            raw = response.model_dump(mode="json")
            if raw.get("service_tier") != "default":
                raise ValueError("Response does not confirm Standard processing.")
            usage = parse_usage(raw["usage"])
            cost = estimate_cost(usage)
            self.budget.settle(reserve, cost)
            cost_uncertain = False
            choice = raw["choices"][0]
            content = choice["message"].get("content") or ""
            finish_reason = choice["finish_reason"]
            if not isinstance(content, str) or not isinstance(finish_reason, str):
                raise TypeError("Response has invalid content or finish reason.")
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            raise ProviderError(
                {
                    "error_type": "invalid_response",
                    "cost_usd": cost,
                    "cost_uncertain": cost_uncertain,
                    "latency_seconds": latency,
                    "raw_response": raw,
                }
            ) from None
        return Completion(content, usage, cost, latency, finish_reason, raw)

    def close(self) -> None:
        self.client.close()
