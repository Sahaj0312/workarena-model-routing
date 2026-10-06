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
