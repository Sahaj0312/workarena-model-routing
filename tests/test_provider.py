from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from models.deepseek import Budget, BudgetExceeded, DeepSeekClient, ProviderError

MESSAGES = [{"role": "user", "content": "Return one browser action as JSON."}]


@pytest.fixture
def provider(monkeypatch):
    create = Mock()
    sdk = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=Mock()
    )
    constructor = Mock(return_value=sdk)
    monkeypatch.setattr("models.deepseek.OpenAI", constructor)
    client = DeepSeekClient("test-key", Budget(1), max_output_tokens=1024)
    return client, create, constructor


def response(*, usage=None, finish_reason="stop"):
    raw = {
        "choices": [
            {"message": {"content": '{"action":"stop","args":[]}'}, "finish_reason": finish_reason}
        ],
        "usage": usage,
    }
    return SimpleNamespace(model_dump=lambda **kwargs: raw)


def test_cached_input_and_reasoning_are_charged_once(provider):
    client, create, constructor = provider
    create.return_value = response(
        usage={
            "prompt_tokens": 10000,
            "completion_tokens": 1000,
            "prompt_cache_hit_tokens": 8000,
            "prompt_cache_miss_tokens": 2000,
            "completion_tokens_details": {"reasoning_tokens": 800},
        }
    )

    result = client.complete(MESSAGES)

    assert result.cost_usd == pytest.approx(0.001848)
    assert client.budget.spent_usd == pytest.approx(result.cost_usd)
    assert client.budget.uncertain_usd == 0
    assert result.usage.output_tokens == 1000
    assert result.usage.reasoning_tokens == 800
    assert constructor.call_args.kwargs["max_retries"] == 0
    create.assert_called_once()
    request = create.call_args.kwargs
    assert request["max_tokens"] == client.config["max_output_tokens"]
    assert request["reasoning_effort"] == client.config["reasoning_effort"]
    assert request["extra_body"] == {"thinking": {"type": "enabled"}}


def test_insufficient_budget_stops_before_api_call(provider):
    client, create, _ = provider
    client.budget = Budget(0.01)

    with pytest.raises(BudgetExceeded):
        client.complete(MESSAGES)

    create.assert_not_called()
    assert client.budget.spent_usd == 0


def test_timeout_keeps_reserve_and_does_not_leak_error_body(provider):
    client, create, _ = provider
    create.side_effect = TimeoutError("secret-from-provider")

    with pytest.raises(ProviderError) as caught:
        client.complete(MESSAGES)

    create.assert_called_once()
    assert client.budget.spent_usd == pytest.approx(client.request_reserve_usd)
    assert client.budget.uncertain_usd == pytest.approx(client.request_reserve_usd)
    assert caught.value.record["cost_uncertain"] is True
    assert "secret-from-provider" not in str(caught.value.record)


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"prompt_tokens": 10, "completion_tokens": -1},
        {"prompt_tokens": 10, "completion_tokens": True},
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "prompt_cache_hit_tokens": 8,
            "prompt_cache_miss_tokens": 8,
        },
    ],
)
def test_unknown_or_invalid_usage_keeps_reserve(provider, usage):
    client, create, _ = provider
    create.return_value = response(usage=usage)

    with pytest.raises(ProviderError) as caught:
        client.complete(MESSAGES)

    assert caught.value.record["error_type"] == "invalid_response"
    assert client.budget.spent_usd == pytest.approx(client.request_reserve_usd)
    assert client.budget.uncertain_usd == pytest.approx(client.request_reserve_usd)


def test_truncated_response_still_records_cost(provider):
    client, create, _ = provider
    create.return_value = response(
        usage={"prompt_tokens": 1000, "completion_tokens": 1024},
        finish_reason="length",
    )

    result = client.complete(MESSAGES)

    assert result.finish_reason == "length"
    assert result.usage.cache_miss_tokens == 1000
    assert result.cost_usd == pytest.approx(0.0015288)
    assert client.budget.spent_usd == pytest.approx(result.cost_usd)


def test_broken_choices_with_valid_usage_keep_known_cost(provider):
    client, create, _ = provider
    raw = {"usage": {"prompt_tokens": 1000, "completion_tokens": 100}, "choices": []}
    create.return_value = SimpleNamespace(model_dump=lambda **kwargs: raw)

    with pytest.raises(ProviderError) as caught:
        client.complete(MESSAGES)

    assert caught.value.record["cost_uncertain"] is False
    assert caught.value.record["cost_usd"] == pytest.approx(0.00042)
    assert client.budget.spent_usd == pytest.approx(0.00042)
    assert client.budget.uncertain_usd == 0


def test_length_response_can_bill_more_tokens_than_requested(provider):
    _, create, _ = provider
    client = DeepSeekClient("test-key", Budget(1), max_output_tokens=8192)
    raw = {
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        "usage": {
            "prompt_tokens": 173722,
            "completion_tokens": 8194,
            "prompt_cache_hit_tokens": 158208,
            "prompt_cache_miss_tokens": 15514,
            "completion_tokens_details": {"reasoning_tokens": 8194},
        },
    }
    create.return_value = SimpleNamespace(model_dump=lambda **kwargs: raw)

    result = client.complete(MESSAGES)

    assert create.call_args.kwargs["max_tokens"] == 8192
    assert result.finish_reason == "length"
    assert result.content == ""
    assert result.usage.output_tokens == result.usage.reasoning_tokens == 8194
    assert result.cost_usd == pytest.approx(0.015436248)
    assert client.budget.spent_usd == pytest.approx(result.cost_usd)
    assert client.budget.uncertain_usd == 0


def test_reserve_covers_model_limit_even_with_small_requested_output(provider):
    client, _, _ = provider
    assert client.max_output_tokens == 1024
    assert client.request_reserve_usd == pytest.approx(0.786432)


def test_usage_above_model_limit_is_rejected_but_charged(provider):
    client, create, _ = provider
    create.return_value = response(
        usage={"prompt_tokens": 1000, "completion_tokens": 393217}, finish_reason="length"
    )

    with pytest.raises(ProviderError) as caught:
        client.complete(MESSAGES)

    assert caught.value.record["error_type"] == "invalid_response"
    assert caught.value.record["cost_uncertain"] is False
    assert client.budget.spent_usd == pytest.approx(0.4721604)
    assert client.budget.uncertain_usd == 0


@pytest.mark.parametrize("amount", [0, -1, float("inf"), float("nan")])
def test_invalid_budget_is_rejected(amount):
    with pytest.raises(ValueError):
        Budget(amount)
