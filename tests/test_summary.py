import json

import pytest

from experiments.cli import summarize


def test_infrastructure_errors_do_not_enter_success_denominator(tmp_path):
    runs = [
        {"status": "success", "success": True, "cost_usd": 0.1},
        {"status": "task_failure", "success": False, "cost_usd": 0.1},
        {"status": "provider_error", "success": None, "cost_usd": 0.3},
    ]
    for index, result in enumerate(runs):
        result.update(
            input_tokens=100,
            output_tokens=20,
            browser_actions=1,
            model_latency_seconds=0.1,
            latency_seconds=1,
            uncertain_cost_usd=0.3 if result["status"] == "provider_error" else 0,
        )
        directory = tmp_path / str(index)
        directory.mkdir()
        (directory / "run.json").write_text(json.dumps(result))

    summary = summarize(tmp_path)

    assert summary["runs"] == 3
    assert summary["evaluated"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["cost_usd"] == pytest.approx(0.5)
    assert summary["statuses"]["provider_error"] == 1
    assert summary["uncertain_cost_usd"] == 0.3


def test_empty_batch_has_no_success_rate(tmp_path):
    assert summarize(tmp_path)["success_rate"] is None
