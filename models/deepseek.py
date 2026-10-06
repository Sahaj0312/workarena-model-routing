"""DeepSeek calls with fixed settings and conservative cost accounting."""

from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Any

from openai import OpenAI

MODEL = "deepseek-flash"
CONTEXT_TOKENS = 1_048_576
MAX_OUTPUT_TOKENS = 393_216
# Peak rates from the official pricing page, checked on 2026-10-05.
INPUT_USD_PER_MILLION = 0.30
CACHED_USD_PER_MILLION = 0.006
OUTPUT_USD_PER_MILLION = 1.20
PRICING_URL = "https://api-docs.deepseek.com/quick_start/pricing/"


class BudgetExceeded(RuntimeError):
    """The remaining budget cannot cover another request."""


class ProviderError(RuntimeError):
    """A failed call with a safe message and its cost record."""

    def __init__(self, record: dict[str, Any]):
        self.record = record
        super().__init__(f"DeepSeek request failed: {record['error_type']}.")


class Budget:
    """Track peak-price estimates and retain charges for uncertain calls."""

    def __init__(self, limit_usd: float):
        if not isfinite(limit_usd) or limit_usd <= 0:
            raise ValueError("Budget must be a finite positive amount.")
        self.limit_usd = limit_usd
        self.spent_usd = 0.0
        self.uncertain_usd = 0.0

    def reserve(self, amount: float) -> None:
        if not isfinite(amount) or amount <= 0:
            raise ValueError("Reservation must be a finite positive amount.")
        if self.spent_usd + amount > self.limit_usd:
            raise BudgetExceeded("Remaining budget cannot cover the request reserve.")
        # A timeout or interruption must not make a possibly billed call free.
        self.spent_usd += amount
        self.uncertain_usd += amount

    def settle(self, reserved: float, actual: float) -> None:
        self.spent_usd += actual - reserved
        self.uncertain_usd = max(0.0, self.uncertain_usd - reserved)


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cache_hit_tokens: int
    cache_miss_tokens: int
    reasoning_tokens: int | None


@dataclass(frozen=True)
class Completion:
    content: str
    usage: TokenUsage
    cost_usd: float
    latency_seconds: float
    finish_reason: str
    raw_response: dict[str, Any]


def _count(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Token counts must be nonnegative integers.")
    return value


def parse_usage(raw: dict[str, Any]) -> TokenUsage:
    """Read provider counts without adding reasoning tokens a second time."""
    input_tokens = _count(raw.get("prompt_tokens"))
    output_tokens = _count(raw.get("completion_tokens"))
    input_details = raw.get("prompt_tokens_details") or {}
    output_details = raw.get("completion_tokens_details") or {}
    cached = raw.get("prompt_cache_hit_tokens", input_details.get("cached_tokens", 0))
    cache_hit_tokens = _count(cached)
    cache_miss_tokens = _count(raw.get("prompt_cache_miss_tokens", input_tokens - cache_hit_tokens))
    if cache_hit_tokens + cache_miss_tokens != input_tokens:
        raise ValueError("Cache counts do not match the input token count.")
    reasoning = output_details.get("reasoning_tokens")
    if reasoning is not None:
        reasoning = _count(reasoning)
        if reasoning > output_tokens:
            raise ValueError("Reasoning token count exceeds the output token count.")
    return TokenUsage(input_tokens, output_tokens, cache_hit_tokens, cache_miss_tokens, reasoning)


def estimate_cost(usage: TokenUsage) -> float:
    """Use peak rates; the account's actual charge can be lower."""
    return (
        usage.cache_miss_tokens * INPUT_USD_PER_MILLION
        + usage.cache_hit_tokens * CACHED_USD_PER_MILLION
        + usage.output_tokens * OUTPUT_USD_PER_MILLION
    ) / 1_000_000


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        budget: Budget,
        max_output_tokens: int = 8192,
        timeout_seconds: float = 120,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("DEEPSEEK_API_KEY is required.")
        if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= MAX_OUTPUT_TOKENS:
            raise ValueError(f"Output limit must be between 1 and {MAX_OUTPUT_TOKENS}.")
        if not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Timeout must be finite and positive.")
        self.budget = budget
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=timeout_seconds,
            max_retries=0,
        )

    @property
    def request_reserve_usd(self) -> float:
        # Billed output can exceed the requested token limit. Reserve the full
        # model limits so no tokenizer estimate or small overrun can undercount.
        return (
            CONTEXT_TOKENS * INPUT_USD_PER_MILLION + MAX_OUTPUT_TOKENS * OUTPUT_USD_PER_MILLION
        ) / 1_000_000

    @property
    def config(self) -> dict[str, Any]:
        return {
            "model": MODEL,
            "base_url": "https://api.deepseek.com",
            "thinking": "enabled",
            "reasoning_effort": "high",
            "top_p": 1.0,
            "temperature": "ignored by provider in thinking mode",
            "max_output_tokens": self.max_output_tokens,
            "timeout_seconds": self.timeout_seconds,
            "response_format": "json_object",
            "max_retries": 0,
            "pricing_basis": "peak rates; conservative estimate",
            "pricing_checked": "2026-10-05",
            "pricing_url": PRICING_URL,
            "input_usd_per_million": INPUT_USD_PER_MILLION,
            "cached_input_usd_per_million": CACHED_USD_PER_MILLION,
            "output_usd_per_million": OUTPUT_USD_PER_MILLION,
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
                max_tokens=self.max_output_tokens,
                reasoning_effort="high",
                top_p=1.0,
                extra_body={"thinking": {"type": "enabled"}},
                response_format={"type": "json_object"},
                stream=False,
            )
        except Exception as error:  # noqa: BLE001 - Never expose a provider error body.
            # Error bodies can contain request data. Store only type and status.
            record = {
                "error_type": type(error).__name__,
                "status_code": getattr(error, "status_code", None),
                "cost_usd": reserve,
                "cost_uncertain": True,
                "latency_seconds": perf_counter() - start,
            }
            raise ProviderError(record) from None

        latency = perf_counter() - start
        raw: dict[str, Any] = {}
        cost = reserve
        cost_uncertain = True
        try:
            raw = response.model_dump(mode="json")
            usage = parse_usage(raw["usage"])
            cost = estimate_cost(usage)
            self.budget.settle(reserve, cost)
            cost_uncertain = False
            if usage.input_tokens > CONTEXT_TOKENS or usage.output_tokens > MAX_OUTPUT_TOKENS:
                raise ValueError("Provider usage exceeds the model limits.")
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
