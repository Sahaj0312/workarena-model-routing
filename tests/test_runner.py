import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from requests import HTTPError, Response

from agent.runner import run_task
from models.deepseek import Budget, BudgetExceeded, Completion, ProviderError, TokenUsage

TASK = {"task_id": "workarena.test", "seed": 42, "level": "l1", "category": "test"}


def observation(error=""):
    return {
        "goal": "Find the requested record",
        "axtree_txt": "[42] button Search",
        "last_action_error": error,
    }


def completion(action="click", args=None):
    return Completion(
        json.dumps({"action": action, "args": ["42"] if args is None else args}),
        TokenUsage(100, 20, 0, 100, 5),
        0.001,
        0.1,
        "stop",
        {"id": "test-response"},
    )


class FakeClient:
    def __init__(self, *responses):
        self.config = {"model": "test-model"}
        self.responses = iter(responses)
        self.messages = []

    def complete(self, messages):
        self.messages.append(list(messages))
        item = next(self.responses)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeEnv:
    def __init__(self, *steps, reset_error=None, close_error=None):
        self.steps = iter(steps)
        self.reset_error = reset_error
        self.close_error = close_error
        self.actions = []
        self.closed = False
        self.task = SimpleNamespace(instance=SimpleNamespace(snow_url="https://test.invalid"))

    def reset(self, *, seed):
        self.seed = seed
        if self.reset_error:
            raise self.reset_error
        return observation(), {"grader_secret": "do not show to model"}

    def step(self, action):
        self.actions.append(action)
        item = next(self.steps)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.closed = True
        if self.close_error:
            raise self.close_error


def run(tmp_path, client, env, **kwargs):
    result = run_task(TASK, client, tmp_path / "run", env_factory=lambda task: env, **kwargs)
    saved = json.loads((tmp_path / "run" / "run.json").read_text())
    events = [
        json.loads(line) for line in (tmp_path / "run" / "trace.jsonl").read_text().splitlines()
    ]
    assert saved == result
    assert events[-1] == {"event": "end", "result": result}
    assert env.closed
    assert env.seed == TASK["seed"]
    return result, events


def test_bad_browser_action_is_feedback_then_grader_can_pass(tmp_path):
    env = FakeEnv(
        (observation("Element not found"), 0, False, False, {}),
        (observation(), 1, True, False, {}),
    )
    client = FakeClient(completion(), completion())

    result, events = run(tmp_path, client, env)

    assert result["status"] == "success"
    assert result["success"] is True
    assert result["browser_actions"] == 2
    assert result["cost_usd"] == pytest.approx(0.002)
    assert result["input_tokens"] == 200
    assert "Element not found" in client.messages[1][-1]["content"]
    assert "grader_secret" not in json.dumps(client.messages)
    assert len([event for event in events if event["event"] == "observation"]) == 3


def test_model_stop_is_not_grader_success(tmp_path):
    result, _ = run(tmp_path, FakeClient(completion("stop", [])), FakeEnv())
    assert result["status"] == "task_failure"
    assert result["success"] is False
    assert result["browser_actions"] == 0


def test_partial_reward_needs_more_work(tmp_path):
    env = FakeEnv(
        (observation(), 0.5, False, False, {}),
        (observation(), 1, True, False, {}),
    )
    result, _ = run(tmp_path, FakeClient(completion(), completion()), env)
    assert result["success"] is True
    assert result["browser_actions"] == 2


def test_action_limit_saves_last_observation(tmp_path):
    env = FakeEnv((observation("Still on search page"), 0, False, False, {}))
    result, events = run(tmp_path, FakeClient(completion()), env, max_actions=1)
    assert result["success"] is False
    assert result["stop_reason"] == "action_limit"
    assert events[-2]["observation"]["last_action_error"] == "Still on search page"


@pytest.mark.parametrize("phase", ["reset", "step"])
def test_environment_error_is_not_model_failure(tmp_path, phase):
    error = RuntimeError("private server data")
    env = FakeEnv(error) if phase == "step" else FakeEnv(reset_error=error)
    result, events = run(tmp_path, FakeClient(completion()), env)
    assert result["status"] == "infrastructure_error"
    assert result["success"] is None
    assert "private server data" not in json.dumps(events)


@pytest.mark.parametrize(
    "error,status",
    [
        (BudgetExceeded(), "budget_stopped"),
        (
            ProviderError({"error_type": "timeout", "cost_usd": 0.3, "latency_seconds": 2}),
            "provider_error",
        ),
    ],
)
def test_api_or_budget_stop_is_not_model_failure(tmp_path, error, status):
    result, _ = run(tmp_path, FakeClient(error), FakeEnv())
    assert result["status"] == status
    assert result["success"] is None
    if isinstance(error, ProviderError):
        assert result["cost_usd"] == 0.3
        assert result["model_calls"] == 1
        assert result["model_latency_seconds"] == 2
    else:
        assert result["model_calls"] == 0


def test_missing_reasoning_usage_is_unknown(tmp_path):
    response = completion("stop", [])
    response = replace(response, usage=replace(response.usage, reasoning_tokens=None))
    result, _ = run(tmp_path, FakeClient(response), FakeEnv())
    assert result["reasoning_tokens"] is None


def test_length_response_is_task_failure_with_all_usage_saved(tmp_path):
    response = replace(
        completion(),
        content="",
        finish_reason="length",
        usage=TokenUsage(173722, 8194, 158208, 15514, 8194),
        cost_usd=0.015436248,
    )
    result, _ = run(tmp_path, FakeClient(response), FakeEnv())
    assert result["status"] == "task_failure"
    assert result["stop_reason"] == "incomplete_response"
    assert result["output_tokens"] == result["reasoning_tokens"] == 8194
    assert result["cost_usd"] == pytest.approx(0.015436248)
    assert result["browser_actions"] == 0


def test_interrupted_request_retains_budget_and_finalizes_trace(tmp_path):
    client = FakeClient()
    client.budget = Budget(1)

    def interrupt(messages):
        client.budget.reserve(0.3)
        raise KeyboardInterrupt()

    client.complete = interrupt
    result, _ = run(tmp_path, client, FakeEnv())
    assert result["success"] is None
    assert result["stop_reason"] == "interrupted"
    assert result["cost_usd"] == pytest.approx(0.3)
    assert result["uncertain_cost_usd"] == pytest.approx(0.3)


def test_cleanup_error_does_not_erase_result(tmp_path):
    env = FakeEnv((observation(), 1, True, False, {}), close_error=RuntimeError())
    result, _ = run(tmp_path, FakeClient(completion()), env)
    assert result["success"] is True
    assert result["cleanup_error"] == "RuntimeError"


def test_instance_mismatch_stops_before_model_call(tmp_path):
    env = FakeEnv()
    nested = SimpleNamespace(instance=SimpleNamespace(snow_url="https://other.invalid"))
    env.task.subtasks = [SimpleNamespace(instance=env.task.instance, task=nested)]
    client = FakeClient()

    result, events = run(tmp_path, client, env)

    assert result["status"] == "infrastructure_error"
    assert result["stop_reason"] == "instance_mismatch"
    assert result["instance_hosts_match"] is False
    assert result["success"] is None
    assert result["model_calls"] == 0
    assert client.messages == []
    assert not any(event["event"] == "model_response" for event in events)


def test_teardown_http_error_keeps_status_and_closes_both_browsers(tmp_path):
    response = Response()
    response.status_code = 404
    error = HTTPError("private-server-and-password", response=response)
    env = FakeEnv((observation(), 1, True, False, {}), close_error=error)
    env.chat = SimpleNamespace(browser=SimpleNamespace(close=Mock(side_effect=RuntimeError())))
    env.browser = SimpleNamespace(close=Mock())

    result, events = run(tmp_path, FakeClient(completion()), env)

    assert result["cleanup_http_status"] == 404
    assert result["success"] is True
    env.chat.browser.close.assert_called_once()
    env.browser.close.assert_called_once()
    assert "private-server-and-password" not in json.dumps(events)


def test_known_credentials_are_redacted_from_input_and_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "private-api-key")
    env = FakeEnv()
    env.task = SimpleNamespace(
        instance=SimpleNamespace(
            snow_url="https://private-instance.test", snow_credentials=("admin", "private-password")
        )
    )
    env.reset = lambda **kwargs: (
        {**observation(), "goal": "private-api-key private-password https://private-instance.test"},
        {},
    )
    client = FakeClient(completion("stop", []))

    run_task(TASK, client, tmp_path / "run", env_factory=lambda task: env)

    text = (tmp_path / "run" / "trace.jsonl").read_text() + json.dumps(client.messages)
    for secret in ("private-api-key", "private-password", "https://private-instance.test"):
        assert secret not in text
    assert "[REDACTED]" in text
