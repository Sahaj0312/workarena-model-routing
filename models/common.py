"""Shared response records and request budget."""

from dataclasses import dataclass
from math import isfinite
from typing import Any


class BudgetExceeded(RuntimeError):
    """The remaining budget cannot cover another request."""


class ProviderError(RuntimeError):
    """A failed call with a safe message and its cost record."""

    def __init__(self, record: dict[str, Any]):
        self.record = record
        super().__init__(f"Model request failed: {record['error_type']}.")


class Budget:
    """Track estimated charges and retain reserves for uncertain calls."""

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
    cache_write_tokens: int | None = None


@dataclass(frozen=True)
class Completion:
    content: str
    usage: TokenUsage
    cost_usd: float
    latency_seconds: float
    finish_reason: str
    raw_response: dict[str, Any]
