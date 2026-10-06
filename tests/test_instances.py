from types import SimpleNamespace
from unittest.mock import Mock

from agent.runner import instances_match, make_env


def test_composite_task_and_child_receive_one_instance(monkeypatch):
    from browsergym.workarena.tasks.compositional.navigate_and_do import (
        NavigateAndOrderStandardLaptopTask,
    )

    shared = SimpleNamespace(snow_url="https://test.invalid")
    choose_instance = Mock(return_value=shared)
    unexpected_selection = Mock(side_effect=AssertionError("A child selected another instance"))
    monkeypatch.setattr("browsergym.workarena.instance.SNowInstance", choose_instance)
    monkeypatch.setattr("browsergym.workarena.tasks.base.SNowInstance", unexpected_selection)
    monkeypatch.setattr("requests.sessions.Session.request", unexpected_selection)

    def create_env(env_id, **kwargs):
        task = NavigateAndOrderStandardLaptopTask(seed=42, **kwargs["task_kwargs"])
        return SimpleNamespace(task=task)

    monkeypatch.setattr("gymnasium.make", create_env)
    env = make_env({"task_id": "workarena.servicenow.navigate-and-order-standard-laptop-l2"})

    choose_instance.assert_called_once()
    unexpected_selection.assert_not_called()
    assert env.task.instance is shared
    assert env.task.task.instance is shared
    assert instances_match(env)
