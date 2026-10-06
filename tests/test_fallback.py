import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent import runner
from models.common import Budget, Completion, ProviderError, TokenUsage

DEEPSEEK = "deepseek-flash"
SOL = "gpt-6.1-sol"
TASK = {"task_id": "workarena.synthetic", "seed": 42}
INVALID = "This is not an action."


def reply(content='{"action":"click","args":["42"]}', **changes):
    return replace(
        Completion(content, TokenUsage(100, 20, 10, 90, 5, 2), 0.01, 0.2, "stop", {}),
        **changes,
    )


class Client:
    def __init__(self, model, budget, *responses, reserve=0.1, after_call=None):
        self.config = {"model": model}
        self.budget = budget
        self.responses = iter(responses)
        self.reserve = reserve
        self.after_call = after_call
        self.messages = []

    def complete(self, messages):
        self.budget.reserve(self.reserve)
        self.messages.append([dict(message) for message in messages])
        response = next(self.responses)
        if self.after_call:
            self.after_call()
        if isinstance(response, ProviderError):
            if not response.record.get("cost_uncertain"):
                self.budget.settle(self.reserve, response.record["cost_usd"])
            raise response
        if isinstance(response, BaseException):
            raise response
        self.budget.settle(self.reserve, response.cost_usd)
        return response


class Env:
    def __init__(self, succeed_after=1, step_error=None):
        self.task = SimpleNamespace(instance=SimpleNamespace(snow_url="https://synthetic.invalid"))
        self.succeed_after = succeed_after
        self.step_error = step_error
        self.resets = 0
        self.actions = []
        self.closed = False

    def observation(self):
        return {"goal": "Synthetic goal", "accessibility_tree": f"state {len(self.actions)}"}

    def reset(self, *, seed):
        self.resets += 1
        return self.observation(), {"grader_secret": "hidden"}

    def step(self, action):
        self.actions.append(action)
        if self.step_error:
            raise self.step_error
        done = len(self.actions) >= self.succeed_after
        return self.observation(), int(done), done, False, {}

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def synthetic_observations(monkeypatch):
    monkeypatch.setattr(runner, "observation_text", lambda obs: dict(obs))


def run(tmp_path, cheap, sol, env, **kwargs):
    result = runner.run_task(
        TASK, cheap, tmp_path / "run", fallback_client=sol, env_factory=lambda task: env, **kwargs
    )
    events = [json.loads(line) for line in (tmp_path / "run/trace.jsonl").read_text().splitlines()]
    assert json.loads((tmp_path / "run/run.json").read_text()) == result
    assert env.closed
    return result, events


def assert_accounted(result):
    for field in (
        "input_tokens",
        "output_tokens",
        "cache_hit_tokens",
        "cache_miss_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "cost_usd",
        "uncertain_cost_usd",
        "model_latency_seconds",
        "model_calls",
        "browser_actions",
    ):
        values = [model[field] for model in result["by_model"].values()]
        if None in values:
            assert result[field] is None
        else:
            assert result[field] == pytest.approx(sum(values)), field


def test_invalid_reply_hands_off_without_reset_or_added_instructions(tmp_path):
    budget = Budget(10)
    cheap = Client(DEEPSEEK, budget, reply(INVALID))
    sol = Client(SOL, budget, reply())
    env = Env()

    result, events = run(tmp_path, cheap, sol, env)

    assert result["success"] is True
    assert result["model_calls"] == 2
    assert result["browser_actions"] == 1
    assert env.resets == 1
    assert sol.messages[0] == cheap.messages[0] + [
        {"role": "assistant", "content": INVALID},
        cheap.messages[0][-1],
    ]
    assert sol.messages[0][0] == {"role": "system", "content": runner.SYSTEM_PROMPT}
    assert "grader_secret" not in json.dumps(sol.messages)
    assert result["fallback_triggered"] is True
    assert result["fallback_used"] is True
    assert result["fallback_trigger_turn"] == 0
    assert result["fallback_call_turn"] == 1
    handoffs = [event for event in events if event["event"] == "handoff"]
    assert len(handoffs) == 1
    assert handoffs[0]["source_model"] == DEEPSEEK
    assert handoffs[0]["target_model"] == SOL
    assert handoffs[0]["reason"] == "invalid_action"
    assert_accounted(result)


def test_progress_and_all_history_survive_permanent_handoff(tmp_path):
    budget = Budget(10)
    cheap = Client(DEEPSEEK, budget, reply(), reply(INVALID))
    sol = Client(SOL, budget, reply(), reply())
    env = Env(succeed_after=3)

    result, _ = run(tmp_path, cheap, sol, env)

    assert result["success"] is True
    assert len(cheap.messages) == 2
    assert len(sol.messages) == 2
    assert env.resets == 1
    assert env.actions == ["click('42')"] * 3
    assert sol.messages[0][:-2] == cheap.messages[1]
    assert sol.messages[0][-2]["content"] == INVALID
    assert sol.messages[0][-1] == cheap.messages[1][-1]
    assert json.loads(sol.messages[0][-1]["content"])["accessibility_tree"] == "state 1"
    assert json.loads(sol.messages[1][-1]["content"])["accessibility_tree"] == "state 2"
    assert result["model_calls"] == 4
    assert result["cost_usd"] == pytest.approx(0.04)
    assert result["by_model"][DEEPSEEK]["model_calls"] == 2
    assert result["by_model"][DEEPSEEK]["browser_actions"] == 1
    assert result["by_model"][SOL]["model_calls"] == 2
    assert_accounted(result)


def test_invalid_reply_on_last_turn_cannot_add_a_sol_call(tmp_path):
    budget = Budget(10)
    cheap = Client(DEEPSEEK, budget, *([reply()] * 29), reply(INVALID))
    sol = Client(SOL, budget)

    result, _ = run(tmp_path, cheap, sol, Env(succeed_after=99))

    assert result["max_actions"] == 30
    assert result["timeout_seconds"] == 600
    assert result["model_calls"] == 30
    assert result["browser_actions"] == 29
    assert result["cost_usd"] == pytest.approx(0.30)
    assert result["fallback_triggered"] is True
    assert result["fallback_used"] is False
    assert sol.messages == []
    assert_accounted(result)


def test_handoff_does_not_reset_timer(tmp_path, monkeypatch):
    clock = [0]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    budget = Budget(10)
    cheap = Client(DEEPSEEK, budget, reply(INVALID), after_call=lambda: clock.__setitem__(0, 600))
    sol = Client(SOL, budget)

    result, _ = run(tmp_path, cheap, sol, Env())

    assert result["latency_seconds"] == 600
    assert result["model_calls"] == 1
    assert result["fallback_used"] is False
    assert sol.messages == []
    assert_accounted(result)


def test_time_expiring_after_handoff_blocks_sol_call(tmp_path, monkeypatch):
    clock = iter([0, 0, 599, 600, 600])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    budget = Budget(10)
    cheap = Client(DEEPSEEK, budget, reply(INVALID))
    sol = Client(SOL, budget)
    result, events = run(tmp_path, cheap, sol, Env())
    assert result["stop_reason"] == "timeout"
    assert any(event["event"] == "handoff" for event in events)
    assert result["fallback_used"] is False
    assert result["model_calls"] == 1
    assert sol.messages == []


def test_insufficient_shared_budget_is_not_an_actual_sol_call(tmp_path):
    budget = Budget(0.10)
    cheap = Client(DEEPSEEK, budget, reply(INVALID))
    sol = Client(SOL, budget, reserve=0.10)

    result, _ = run(tmp_path, cheap, sol, Env())

    assert result["status"] == "budget_stopped"
    assert result["success"] is None
    assert result["model_calls"] == 1
    assert result["cost_usd"] == pytest.approx(0.01)
    assert result["fallback_triggered"] is True
    assert result["fallback_used"] is False
    assert result["by_model"][SOL]["model_calls"] == 0
    assert sol.messages == []
    assert_accounted(result)


def test_sol_invalid_action_is_terminal(tmp_path):
    budget = Budget(10)
    cheap = Client(DEEPSEEK, budget, reply(INVALID))
    sol = Client(SOL, budget, reply(INVALID))
    result, _ = run(tmp_path, cheap, sol, Env())
    assert result["status"] == "task_failure"
    assert result["stop_reason"] == "invalid_action"
    assert result["model_calls"] == 2
    assert result["browser_actions"] == 0
    assert_accounted(result)


def test_sol_uses_only_the_remaining_turns(tmp_path):
    budget = Budget(10)
    cheap = Client(DEEPSEEK, budget, reply(INVALID))
    sol = Client(SOL, budget, reply())
    result, _ = run(tmp_path, cheap, sol, Env(succeed_after=2), max_actions=2)
    assert result["status"] == "task_failure"
    assert result["stop_reason"] == "action_limit"
    assert result["model_calls"] == 2
    assert result["browser_actions"] == 1
    assert len(sol.messages) == 1
    assert_accounted(result)


def test_interrupt_after_sol_reservation_keeps_uncertain_charge(tmp_path):
    budget = Budget(10)
    cheap = Client(DEEPSEEK, budget, reply(INVALID))
    sol = Client(SOL, budget, KeyboardInterrupt(), reserve=0.1)
    result, _ = run(tmp_path, cheap, sol, Env())
    assert result["status"] == "infrastructure_error"
    assert result["success"] is None
    assert result["stop_reason"] == "interrupted"
    assert result["fallback_used"] is True
    assert result["fallback_call_turn"] == 1
    assert result["model_calls"] == 2
    assert result["cost_usd"] == pytest.approx(0.11)
    assert result["uncertain_cost_usd"] == pytest.approx(0.1)
    assert result["by_model"][SOL]["model_calls"] == 1
    assert result["by_model"][SOL]["cost_usd"] == pytest.approx(0.1)
    assert result["by_model"][SOL]["uncertain_cost_usd"] == pytest.approx(0.1)
    assert_accounted(result)


def test_each_task_starts_with_deepseek_and_separate_history(tmp_path):
    budget = Budget(10)
    cheap = Client(DEEPSEEK, budget, reply(INVALID), reply(INVALID))
    sol = Client(SOL, budget, reply(), reply())
    for index in range(2):
        result, _ = run(tmp_path / str(index), cheap, sol, Env())
        assert result["success"] is True
        assert result["model_calls"] == 2
        assert result["cost_usd"] == pytest.approx(0.02)
        assert_accounted(result)
    assert len(cheap.messages) == 2
    assert cheap.messages[0] == cheap.messages[1]
    assert len(cheap.messages[1]) == 2
    assert budget.spent_usd == pytest.approx(0.04)


@pytest.mark.parametrize(
    "response,reason,status",
    [
        (reply('{"action":"stop","args":[]}'), "agent_stop", "task_failure"),
        (reply(INVALID, finish_reason="length"), "incomplete_response", "task_failure"),
        (reply(INVALID, finish_reason="content_filter"), "incomplete_response", "provider_error"),
    ],
)
def test_non_format_stops_do_not_handoff(tmp_path, response, reason, status):
    budget = Budget(10)
    cheap = Client(DEEPSEEK, budget, response)
    sol = Client(SOL, budget)
    result, _ = run(tmp_path, cheap, sol, Env())
    assert result["status"] == status
    assert result["stop_reason"] == reason
    assert result["fallback_triggered"] is False
    assert result["fallback_used"] is False
    assert sol.messages == []
    assert_accounted(result)


@pytest.mark.parametrize("phase", ["provider", "budget", "environment"])
def test_errors_do_not_handoff(tmp_path, phase):
    budget = Budget(0.01 if phase == "budget" else 10)
    response = (
        ProviderError(
            {
                "error_type": "timeout",
                "cost_usd": 0.1,
                "cost_uncertain": True,
                "latency_seconds": 0.2,
            }
        )
        if phase == "provider"
        else reply()
    )
    cheap = Client(DEEPSEEK, budget, response)
    sol = Client(SOL, budget)
    env = Env(step_error=RuntimeError("synthetic") if phase == "environment" else None)

    result, _ = run(tmp_path, cheap, sol, env)

    expected = {
        "provider": "provider_error",
        "budget": "budget_stopped",
        "environment": "infrastructure_error",
    }
    assert result["status"] == expected[phase]
    assert result["success"] is None
    assert result["fallback_triggered"] is False
    assert result["fallback_used"] is False
    assert sol.messages == []
    assert_accounted(result)


@pytest.mark.parametrize("uncertain", [False, True])
def test_sol_provider_error_keeps_cost_and_call_attribution(tmp_path, uncertain):
    budget = Budget(10)
    cheap = Client(DEEPSEEK, budget, reply(INVALID))
    sol = Client(
        SOL,
        budget,
        ProviderError(
            {
                "error_type": "timeout",
                "cost_usd": 0.1 if uncertain else 0.03,
                "cost_uncertain": uncertain,
                "latency_seconds": 0.5,
            }
        ),
    )

    result, _ = run(tmp_path, cheap, sol, Env())

    assert result["status"] == "provider_error"
    assert result["success"] is None
    assert result["fallback_used"] is True
    assert result["model_calls"] == 2
    assert result["by_model"][SOL]["model_calls"] == 1
    assert result["by_model"][SOL]["cost_usd"] == pytest.approx(0.1 if uncertain else 0.03)
    assert result["uncertain_cost_usd"] == pytest.approx(0.1 if uncertain else 0)
    assert_accounted(result)


def test_unknown_token_details_remain_unknown_across_handoff(tmp_path):
    budget = Budget(10)
    usage = TokenUsage(100, 20, 10, 90, None, None)
    cheap = Client(DEEPSEEK, budget, reply(INVALID, usage=usage))
    sol = Client(SOL, budget, reply())
    result, _ = run(tmp_path, cheap, sol, Env())
    assert result["reasoning_tokens"] is None
    assert result["cache_write_tokens"] is None
    assert result["by_model"][SOL]["reasoning_tokens"] == 5
    assert result["by_model"][SOL]["cache_write_tokens"] == 2
    assert_accounted(result)


def test_baseline_invalid_action_still_stops_without_handoff(tmp_path):
    cheap = Client(DEEPSEEK, Budget(10), reply(INVALID))
    result, _ = run(tmp_path, cheap, None, Env())
    assert result["model"] == DEEPSEEK
    assert result["status"] == "task_failure"
    assert result["stop_reason"] == "invalid_action"
    assert result["model_calls"] == 1
    assert result["browser_actions"] == 0


def test_fallback_requires_one_shared_budget(tmp_path):
    cheap = Client(DEEPSEEK, Budget(10), reply(INVALID))
    sol = Client(SOL, Budget(10), reply())
    env = Env()
    with pytest.raises(ValueError):
        run(tmp_path, cheap, sol, env)
    assert env.resets == 0
    assert not (tmp_path / "run").exists()
