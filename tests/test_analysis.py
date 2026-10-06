import hashlib
import itertools
import json

import pytest

from analysis.routing import frontier, main, make_plan, score
from routers import heuristic


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def saved_pairs(tmp_path):
    tasks = [{"task_id": f"synthetic-{i}", "seed": i} for i in range(3)]
    roots = {p: tmp_path / p for p in ("deepseek", "sol")}
    pairs = []
    for p, root in roots.items():
        write(root / "batch.json", {"tasks": tasks})
        for i, task in enumerate(tasks):
            goal = "short goal" if i != 1 else "word " * 101
            if p == "sol" and i == 2:
                goal = "different synthetic goal"
            run = dict(
                **task,
                success=(i == 0 if p == "deepseek" else i < 2),
                cost_usd=1 if p == "deepseek" else 3,
                latency_seconds=2,
                browser_actions=1,
                input_tokens=10,
                output_tokens=2,
            )
            run["status"] = "success" if run["success"] else "task_failure"
            run.update(
                model="deepseek-flash" if p == "deepseek" else "gpt-6.1-sol",
                instance_hosts_match=True,
                uncertain_cost_usd=0,
            )
            write(root / f"{i:03}" / "run.json", run)
            write(
                root / f"{i:03}" / "trace.jsonl",
                {"event": "observation", "observation": {"goal": goal}},
            )
            if p == "deepseek":
                pairs.append(dict(**task, index=i, exact_initial_goal_match=i != 2))
            pairs[i][f"{p}_success"] = run["success"]
            pairs[i][f"{p}_cost_usd"] = run["cost_usd"]
            pairs[i][f"{p}_goal_sha256"] = hashlib.sha256(goal.encode()).hexdigest()
    path = tmp_path / "paired.json"
    write(
        path,
        dict(
            pairs=pairs,
            planned_pairs=3,
            valid_paired_outcomes=3,
            source_directories={"deepseek": str(roots["deepseek"]), "sol_main": str(roots["sol"])},
        ),
    )
    return path


def test_frontier_matches_exhaustive_choices():
    pairs = [
        {"deepseek": {"success": False, "cost_usd": 8}, "sol": {"success": True, "cost_usd": 1}},
        {"deepseek": {"success": True, "cost_usd": 2}, "sol": {"success": False, "cost_usd": 1}},
        {"deepseek": {"success": False, "cost_usd": 4}, "sol": {"success": False, "cost_usd": 3}},
    ]
    curve = frontier(pairs)
    brute = {}
    for choices in itertools.product(("deepseek", "sol"), repeat=len(pairs)):
        wins = sum(p[c]["success"] for p, c in zip(pairs, choices))
        cost = sum(p[c]["cost_usd"] for p, c in zip(pairs, choices))
        brute[wins] = min(brute.get(wins, float("inf")), cost)
    assert {r["successes"]: r["cost_usd"] for r in curve["exact"]} == brute
    assert curve["at_least"][0]["successes"] == 1
    assert curve["at_least"][2]["cost_usd"] == 6
    assert curve["at_least"][3]["choices"] is None
    assert not curve["exact"][0]["pareto"]


def test_all_fail_still_pays_and_empty_subset_is_defined():
    pairs = [{p: {"success": False, "cost_usd": c} for p, c in (("deepseek", 2), ("sol", 1))}]
    assert frontier(pairs)["at_least"][0]["cost_usd"] == 1
    assert frontier(pairs)["at_least"][1]["cost_usd"] is None
    assert frontier([])["at_least"] == [
        {"target_successes": 0, "successes": 0, "cost_usd": 0, "choices": []}
    ]


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True, "1"])
def test_frontier_rejects_invalid_cost(bad):
    pair = {p: {"success": False, "cost_usd": bad} for p in ("deepseek", "sol")}
    with pytest.raises(ValueError):
        frontier([pair])


def test_plan_then_score_uses_only_saved_choices(saved_pairs, tmp_path, monkeypatch):
    plan_dir, output = tmp_path / "plan", tmp_path / "score"
    main(["plan", "--paired-summary", str(saved_pairs), "--output", str(plan_dir)])
    plan = json.loads((plan_dir / "plan.json").read_text())
    assert [e["provider"] for e in plan["entries"]] == ["deepseek", "sol", "deepseek"]
    assert [e["goal_words"] for e in plan["entries"]] == [2, 101, 2]
    assert [e["exact_goal_match"] for e in plan["entries"]] == [True, True, False]
    assert "short goal" not in (plan_dir / "plan.json").read_text()
    assert (plan_dir / "plan.json").stat().st_mode & 0o222 == 0

    def forbidden(_):
        raise AssertionError("Scoring must not ask the rule for choices")

    monkeypatch.setattr(heuristic, "choose_provider", forbidden)
    main(
        [
            "score",
            "--paired-summary",
            str(saved_pairs),
            "--plan",
            str(plan_dir / "plan.json"),
            "--output",
            str(output),
        ]
    )
    report = json.loads((output / "metrics.json").read_text())
    result = report["metrics"]["all"]["heuristic"]
    assert (result["successes"], result["cost_usd"], result["mean_latency_seconds"]) == (2, 5, 2)
    assert result["route_fraction"] == {"deepseek": 2 / 3, "sol": 1 / 3}
    assert report["metrics"]["exact_goal_match"]["indices"] == [0, 1]
    assert (output / "frontier.csv").exists()
    with pytest.raises(SystemExit):
        main(["plan", "--paired-summary", str(saved_pairs), "--output", str(plan_dir)])


def test_changed_source_and_plan_rejected(saved_pairs, tmp_path):
    directory = tmp_path / "plan"
    main(["plan", "--paired-summary", str(saved_pairs), "--output", str(directory)])
    plan_path = directory / "plan.json"
    trace = tmp_path / "sol" / "000" / "trace.jsonl"
    original = trace.read_text()
    trace.write_text(original + "\n")
    with pytest.raises(ValueError, match="Source changed"):
        score(saved_pairs, plan_path)
    trace.write_text(original)
    plan_path.chmod(0o644)
    plan_path.write_text(plan_path.read_text() + "\n")
    with pytest.raises(ValueError, match="Plan changed"):
        score(saved_pairs, plan_path)


def test_manifest_order_and_missing_runs_rejected(saved_pairs, tmp_path):
    data = json.loads(saved_pairs.read_text())
    data["pairs"][1]["task_id"] = data["pairs"][0]["task_id"]
    data["pairs"][1]["seed"] = data["pairs"][0]["seed"]
    write(saved_pairs, data)
    with pytest.raises(ValueError, match="unique"):
        make_plan(saved_pairs)


def test_unknown_outcomes_are_not_failures(saved_pairs, tmp_path):
    run_path = tmp_path / "sol" / "000" / "run.json"
    run = json.loads(run_path.read_text())
    run.update(success=None, status="provider_error")
    write(run_path, run)
    directory = tmp_path / "plan"
    main(["plan", "--paired-summary", str(saved_pairs), "--output", str(directory)])
    with pytest.raises(ValueError, match="unfinished"):
        score(saved_pairs, directory / "plan.json")


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_tokens", -1),
        ("latency_seconds", float("inf")),
        ("browser_actions", 1.5),
        ("cost_usd", float("nan")),
        ("uncertain_cost_usd", 1),
        ("instance_hosts_match", False),
    ],
)
def test_invalid_run_metrics_rejected(saved_pairs, tmp_path, field, value):
    run_path = tmp_path / "sol" / "000" / "run.json"
    run = json.loads(run_path.read_text())
    run[field] = value
    write(run_path, run)
    directory = tmp_path / "plan"
    main(["plan", "--paired-summary", str(saved_pairs), "--output", str(directory)])
    with pytest.raises(ValueError):
        score(saved_pairs, directory / "plan.json")


@pytest.mark.parametrize("change", ["choice", "source_coverage"])
def test_invalid_plan_rejected_even_with_replaced_checksum(saved_pairs, tmp_path, change):
    directory = tmp_path / "plan"
    main(["plan", "--paired-summary", str(saved_pairs), "--output", str(directory)])
    path = directory / "plan.json"
    plan = json.loads(path.read_text())
    if change == "choice":
        plan["entries"][0]["provider"] = "unknown"
    else:
        trace_path = next(p for p in plan["source_hashes"] if p.endswith("trace.jsonl"))
        del plan["source_hashes"][trace_path]
    path.chmod(0o644)
    write(path, plan)
    checksum = path.with_suffix(".sha256")
    checksum.chmod(0o644)
    checksum.write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        score(saved_pairs, path)
