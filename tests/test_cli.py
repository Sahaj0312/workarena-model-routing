import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from experiments import cli


@pytest.mark.parametrize("provider", ["deepseek", "sol"])
def test_provider_switch_keeps_task_order_and_limits(tmp_path, monkeypatch, provider):
    tasks = [{"task_id": "workarena.first", "seed": 42}, {"task_id": "workarena.second", "seed": 7}]
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps({"versions": {"test": "1"}, "tasks": tasks}))
    monkeypatch.setattr(cli, "versions", lambda: {"test": "1"})
    monkeypatch.setattr(cli, "provenance", lambda: {})
    monkeypatch.setattr(cli, "summarize", lambda path: {"evaluated": 2})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sol-test-key")
    clients = {
        name: SimpleNamespace(config={"model": name}, close=Mock()) for name in ("deepseek", "sol")
    }
    constructors = {name: Mock(return_value=client) for name, client in clients.items()}
    monkeypatch.setattr("models.deepseek.DeepSeekClient", constructors["deepseek"])
    monkeypatch.setattr("models.sol.SolClient", constructors["sol"])
    run_task = Mock(return_value={"status": "task_failure", "cost_usd": 0})
    monkeypatch.setattr("agent.runner.run_task", run_task)
    args = SimpleNamespace(
        provider=provider,
        manifest=manifest,
        output=tmp_path / provider,
        limit=None,
        budget_usd=50,
        max_actions=30,
        timeout_seconds=600,
    )

    assert cli.run(args) == 0

    selected = constructors[provider]
    assert selected.call_args.args[0] == provider + "-test-key"
    assert selected.call_args.args[1].limit_usd == 50
    constructors["sol" if provider == "deepseek" else "deepseek"].assert_not_called()
    assert [call.args[0] for call in run_task.call_args_list] == tasks
    assert all(call.args[1] is clients[provider] for call in run_task.call_args_list)
    assert all(
        call.kwargs == {"max_actions": 30, "timeout_seconds": 600}
        for call in run_task.call_args_list
    )
    saved = json.loads((args.output / "batch.json").read_text())
    assert saved["provider"] == provider
    assert saved["tasks"] == tasks
    clients[provider].close.assert_called_once()


def test_cli_default_provider_stays_deepseek(monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda path: None)
    run = Mock(return_value=0)
    monkeypatch.setattr(cli, "run", run)
    monkeypatch.setattr(
        "sys.argv", ["workarena-baseline", "run", "--output", "unused", "--budget-usd", "1"]
    )
    assert cli.main() == 0
    assert run.call_args.args[0].provider == "deepseek"


def test_preflight_needs_only_selected_provider_key(monkeypatch, capsys):
    playwright = MagicMock()
    browser = playwright.__enter__.return_value.chromium.launch.return_value
    browser.new_page.return_value.get_by_role.return_value.inner_text.return_value = "Ready"
    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: playwright)
    monkeypatch.setattr(cli, "versions", lambda: {"test": "1"})
    monkeypatch.setenv("HF_TOKEN", "test-hf")
    monkeypatch.setenv("OPENAI_API_KEY", "sol-test-key")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert cli.preflight(False, "sol") == 0
    result = json.loads(capsys.readouterr().out)
    assert result["OPENAI_API_KEY"] is True
    assert "DEEPSEEK_API_KEY" not in result
    assert cli.preflight(False) == 1


@pytest.fixture
def fallback_cli(tmp_path, monkeypatch):
    tasks = [{"task_id": "workarena.synthetic", "seed": 42}]
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps({"versions": {"test": "1"}, "tasks": tasks}))
    monkeypatch.setattr(cli, "versions", lambda: {"test": "1"})
    monkeypatch.setattr(cli, "provenance", lambda: {})
    monkeypatch.setattr(cli, "summarize", lambda path: {"evaluated": 1})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sol-test-key")
    clients = {
        name: SimpleNamespace(config={"model": model}, complete=Mock(), close=Mock())
        for name, model in [("deepseek", "deepseek-flash"), ("sol", "gpt-6.1-sol")]
    }
    constructors = {name: Mock(return_value=client) for name, client in clients.items()}
    monkeypatch.setattr("models.deepseek.DeepSeekClient", constructors["deepseek"])
    monkeypatch.setattr("models.sol.SolClient", constructors["sol"])
    run_task = Mock(return_value={"status": "success", "cost_usd": 0})
    monkeypatch.setattr("agent.runner.run_task", run_task)
    args = SimpleNamespace(
        provider="deepseek",
        fallback_sol=True,
        manifest=manifest,
        output=tmp_path / "output",
        limit=None,
        budget_usd=10,
        max_actions=30,
        timeout_seconds=600,
    )
    return args, clients, constructors, run_task


def test_fallback_cli_shares_budget_closes_clients_and_records_policy(fallback_cli):
    args, clients, constructors, run_task = fallback_cli
    assert cli.run(args) == 0
    budget = constructors["deepseek"].call_args.args[1]
    assert constructors["sol"].call_args.args[1] is budget
    assert budget.limit_usd == 10
    assert constructors["deepseek"].call_args.args[0] == "deepseek-test-key"
    assert constructors["sol"].call_args.args[0] == "sol-test-key"
    assert run_task.call_args.args[1] is clients["deepseek"]
    assert run_task.call_args.kwargs == {
        "max_actions": 30,
        "timeout_seconds": 600,
        "fallback_client": clients["sol"],
    }
    for client in clients.values():
        client.complete.assert_not_called()
        client.close.assert_called_once()
    batch = json.loads((args.output / "batch.json").read_text())
    assert batch["strategy_id"] == "format-fallback-v1"
    assert batch["fallback_config"] == clients["sol"].config
    assert batch["fallback_policy"] == "experiments/format-fallback-v1.json"
    assert len(batch["fallback_policy_sha256"]) == 64


@pytest.mark.parametrize("key", ["DEEPSEEK_API_KEY", "OPENAI_API_KEY"])
@pytest.mark.parametrize("value", [None, "", " \t"])
def test_fallback_requires_both_keys_before_client_or_environment(
    fallback_cli, monkeypatch, key, value
):
    args, _, constructors, run_task = fallback_cli
    if value is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        cli.run(args)
    assert not args.output.exists()
    run_task.assert_not_called()
    for constructor in constructors.values():
        constructor.assert_not_called()


def test_fallback_rejects_sol_as_starting_provider(fallback_cli):
    args, _, constructors, run_task = fallback_cli
    args.provider = "sol"
    with pytest.raises(ValueError):
        cli.run(args)
    run_task.assert_not_called()
    for constructor in constructors.values():
        constructor.assert_not_called()


@pytest.mark.parametrize("phase", ["constructor", "setup", "run"])
def test_fallback_cli_closes_created_clients_on_error(fallback_cli, monkeypatch, phase):
    args, clients, constructors, run_task = fallback_cli
    if phase == "constructor":
        constructors["sol"].side_effect = RuntimeError("synthetic")
    elif phase == "setup":
        monkeypatch.setattr(cli, "provenance", Mock(side_effect=RuntimeError("synthetic")))
    else:
        run_task.side_effect = RuntimeError("synthetic")
    with pytest.raises(RuntimeError, match="synthetic"):
        cli.run(args)
    clients["deepseek"].close.assert_called_once()
    if phase != "constructor":
        clients["sol"].close.assert_called_once()
    for client in clients.values():
        client.complete.assert_not_called()


def test_run_cli_parses_explicit_fallback_flag(monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda path: None)
    run = Mock(return_value=0)
    monkeypatch.setattr(cli, "run", run)
    monkeypatch.setattr(
        "sys.argv",
        ["workarena-baseline", "run", "--output", "unused", "--budget-usd", "1", "--fallback-sol"],
    )
    assert cli.main() == 0
    assert run.call_args.args[0].provider == "deepseek"
    assert run.call_args.args[0].fallback_sol is True


def test_preflight_cli_passes_fallback_without_model_call(monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda path: None)
    preflight = Mock(return_value=0)
    monkeypatch.setattr(cli, "preflight", preflight)
    monkeypatch.setattr("sys.argv", ["workarena-baseline", "preflight", "--fallback-sol"])
    assert cli.main() == 0
    assert preflight.call_args.args == (False, "deepseek", True)


def test_fallback_preflight_checks_both_keys_with_stub_browser(monkeypatch, capsys):
    playwright = MagicMock()
    browser = playwright.__enter__.return_value.chromium.launch.return_value
    browser.new_page.return_value.get_by_role.return_value.inner_text.return_value = "Ready"
    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: playwright)
    monkeypatch.setattr(cli, "versions", lambda: {"test": "1"})
    monkeypatch.setenv("HF_TOKEN", "synthetic-hf")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic-deepseek")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        cli.preflight(False, "deepseek", True)
    playwright.__enter__.assert_not_called()
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-sol")
    assert cli.preflight(False, "deepseek", True) == 0
    checks = json.loads(capsys.readouterr().out)
    assert checks["DEEPSEEK_API_KEY"] is True
    assert checks["OPENAI_API_KEY"] is True
