from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from models.common import Budget, BudgetExceeded, ProviderError
from models.sol import SolClient, estimate_cost, parse_usage


@pytest.fixture
def provider(monkeypatch):
    create = Mock()
    sdk = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=Mock()
    )
    constructor = Mock(return_value=sdk)
    monkeypatch.setattr("models.sol.OpenAI", constructor)
    return SolClient("test-key", Budget(50)), create, constructor


def usage(*, input_tokens=10000, cached=8000, writes=1500, output_tokens=1000):
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "prompt_tokens_details": {"cached_tokens": cached, "cache_write_tokens": writes},
        "completion_tokens_details": {"reasoning_tokens": 800},
    }


def response(*, counts=None, tier="default", finish="stop"):
    raw = {
        "service_tier": tier,
        "usage": usage() if counts is None else counts,
        "choices": [
            {"message": {"content": '{"action":"stop","args":[]}'}, "finish_reason": finish}
        ],
    }
    return SimpleNamespace(model_dump=lambda **kwargs: raw)


def test_cache_writes_replace_input_charge_and_reasoning_is_not_added(provider):
    client, create, _ = provider
    create.return_value = response()

    result = client.complete([{"role": "user", "content": "Return an action as JSON."}])

    assert result.usage.cache_miss_tokens == 2000
    assert result.usage.cache_write_tokens == 1500
    assert result.usage.output_tokens == 1000
    assert result.usage.reasoning_tokens == 800
    assert result.cost_usd == pytest.approx(0.01555)
    assert client.budget.spent_usd == pytest.approx(result.cost_usd)
    assert client.budget.uncertain_usd == 0


@pytest.mark.parametrize("writes,expected", [(None, 0.0158), (0, 0.0148)])
def test_unknown_cache_writes_are_conservative(writes, expected):
    parsed = parse_usage(usage(writes=writes))
    assert parsed.cache_write_tokens is writes
    assert estimate_cost(parsed) == pytest.approx(expected)


@pytest.mark.parametrize("input_tokens,expected", [(272000, 0.04175), (272001, 0.078504)])
def test_long_context_boundary_reprices_the_whole_request(input_tokens, expected):
    parsed = parse_usage(usage(input_tokens=input_tokens, cached=270000))
    assert estimate_cost(parsed) == pytest.approx(expected)


def test_fixed_request_preserves_history_without_tools_or_extra_instructions(provider):
    client, create, constructor = provider
    create.return_value = response()
    messages = [
        {"role": "system", "content": "Shared prompt JSON"},
        {"role": "user", "content": "Unchanged observation"},
    ]

    client.complete(messages)

    assert create.call_args.kwargs == {
        "model": "gpt-6.1-sol",
        "messages": messages,
        "max_completion_tokens": 8192,
        "reasoning_effort": "high",
        "response_format": {"type": "json_object"},
        "service_tier": "default",
        "store": False,
        "stream": False,
    }
    assert create.call_args.kwargs["messages"] is messages
    assert constructor.call_args.kwargs["max_retries"] == 0
    assert client.request_reserve_usd == pytest.approx(6.53)


def test_budget_stops_before_call(provider):
    client, create, _ = provider
    client.budget = Budget(6.52)
    with pytest.raises(BudgetExceeded):
        client.complete([{"role": "user", "content": "JSON"}])
    create.assert_not_called()
    assert client.budget.spent_usd == 0


def test_timeout_retains_reserve_without_retry_or_error_body(provider):
    client, create, _ = provider
    create.side_effect = TimeoutError("private provider data")
    with pytest.raises(ProviderError) as caught:
        client.complete([{"role": "user", "content": "JSON"}])
    create.assert_called_once()
    assert client.budget.spent_usd == pytest.approx(6.53)
    assert client.budget.uncertain_usd == pytest.approx(6.53)
    assert "private provider data" not in str(caught.value.record)


@pytest.mark.parametrize("tier", [None, "priority"])
def test_unknown_processing_price_retains_reserve(provider, tier):
    client, create, _ = provider
    create.return_value = response(tier=tier)
    with pytest.raises(ProviderError):
        client.complete([{"role": "user", "content": "JSON"}])
    assert client.budget.spent_usd == pytest.approx(6.53)
    assert client.budget.uncertain_usd == pytest.approx(6.53)


@pytest.mark.parametrize(
    "counts",
    [
        {},
        {"prompt_tokens": True, "completion_tokens": 1},
        {"prompt_tokens": 1, "completion_tokens": -1},
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 1},
        usage(cached=10001),
        usage(writes=2001),
        usage(input_tokens=922001),
        usage(output_tokens=128001),
    ],
)
def test_invalid_usage_cannot_release_the_reserve(provider, counts):
    client, create, _ = provider
    create.return_value = response(counts=counts)
    with pytest.raises(ProviderError) as caught:
        client.complete([{"role": "user", "content": "JSON"}])
    assert caught.value.record["cost_uncertain"] is True
    assert client.budget.spent_usd == pytest.approx(6.53)
    assert client.budget.uncertain_usd == pytest.approx(6.53)


def test_length_response_keeps_full_billed_usage(provider):
    client, create, _ = provider
    create.return_value = response(counts=usage(output_tokens=8194), finish="length")
    result = client.complete([{"role": "user", "content": "JSON"}])
    assert result.finish_reason == "length"
    assert result.usage.output_tokens == 8194
    assert result.cost_usd == pytest.approx(0.08749)
