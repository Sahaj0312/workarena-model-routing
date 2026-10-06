"""Freeze a static route plan, then score saved outcomes without model calls."""

import argparse
import csv
import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path

PROVIDERS = ("deepseek", "sol")
WORKSPACE = Path(__file__).resolve().parents[1]
NOTE = (
    "Offline selection of observed runs, not escalation or a live router test. "
    "Goal text and live state can differ across providers. The exact-goal subset is descriptive, "
    "not an unbiased sample. Costs exclude routing overhead and do not reproduce a new cache schedule. "
    "Interrupted attempts and smoke costs are excluded from baseline metrics."
)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def key(row):
    task, seed = row.get("task_id"), row.get("seed")
    if not isinstance(task, str) or not task or type(seed) is not int:
        raise ValueError("Invalid task ID or seed")
    return task, seed


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Expected a finite, nonnegative number")
    return value


def sources(summary_path):
    summary = read(summary_path)
    pairs = summary["pairs"]
    keys = [key(p) for p in pairs]
    if (
        not keys
        or len(set(keys)) != len(keys)
        or [p["index"] for p in pairs] != list(range(len(keys)))
    ):
        raise ValueError("Pairs must have unique IDs and consecutive manifest indices")
    if summary["valid_paired_outcomes"] != len(keys) or summary["planned_pairs"] != len(keys):
        raise ValueError("A complete paired manifest is required")
    roots = summary["source_directories"]
    paths, files = {}, [Path(summary_path).resolve()]
    for provider, names in (("deepseek", ["deepseek"]), ("sol", ["sol_main", "sol_final"])):
        paths[provider] = {}
        for name in names:
            if name not in roots:
                continue
            root = Path(roots[name])
            root = root.resolve() if root.is_absolute() else (WORKSPACE / root).resolve()
            batch_path = root / "batch.json"
            tasks = read(batch_path)["tasks"]
            batch_keys = [key(t) for t in tasks]
            if len(set(batch_keys)) != len(batch_keys) or any(k not in keys for k in batch_keys):
                raise ValueError("Source batch has duplicate or unknown pairs")
            if name in ("deepseek", "sol_main") and batch_keys != keys:
                raise ValueError("Source manifest order differs from paired summary")
            files.append(batch_path)
            for i, task_key in enumerate(batch_keys):
                paths[provider][task_key] = root / f"{i:03}"
        if set(paths[provider]) != set(keys):
            raise ValueError("Source runs do not cover every pair")
    return summary, paths, files


def initial_goal(path):
    with Path(path).open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] == "observation":
                goal = event["observation"]["goal"]
                if not isinstance(goal, str) or not goal.strip():
                    break
                return goal
    raise ValueError("Missing initial goal")


def make_plan(summary_path):
    from routers import heuristic

    summary, paths, files = sources(summary_path)
    rule_path = Path(heuristic.__file__).resolve()
    files.append(Path(__file__).resolve())
    entries = []
    for pair in summary["pairs"]:
        task_key = key(pair)
        runs = {p: paths[p][task_key] for p in PROVIDERS}
        goals = {p: initial_goal(runs[p] / "trace.jsonl") for p in PROVIDERS}
        goal_hashes = {p: hashlib.sha256(g.encode()).hexdigest() for p, g in goals.items()}
        if any(goal_hashes[p] != pair[f"{p}_goal_sha256"] for p in PROVIDERS):
            raise ValueError("Goal hash differs from paired summary")
        choice = heuristic.choose_provider(goals["deepseek"])
        if choice not in PROVIDERS:
            raise ValueError("Invalid provider choice")
        entries.append(
            dict(
                index=pair["index"],
                task_id=task_key[0],
                seed=task_key[1],
                provider=choice,
                goal_sha256=goal_hashes["deepseek"],
                goal_words=len(goals["deepseek"].split()),
                exact_goal_match=goals["deepseek"] == goals["sol"],
                runs={p: str(runs[p] / "run.json") for p in PROVIDERS},
            )
        )
        files.extend(r / name for r in runs.values() for name in ("run.json", "trace.jsonl"))
    return dict(
        schema_version=1,
        rule_id=heuristic.RULE_ID,
        rule_path=str(rule_path),
        rule_sha256=digest(rule_path),
        canonical_goal_source="deepseek_initial_observation",
        source_hashes={str(p): digest(p) for p in files},
        entries=entries,
        note=NOTE,
    )


def frontier(pairs):
    """Keep the cheapest route for every reachable exact success count."""
    states = {0: (Decimal(0), [])}
    for pair in pairs:
        following = {}
        for successes, (cost, choices) in states.items():
            for provider in PROVIDERS:
                result = pair[provider]
                if type(result["success"]) is not bool:
                    raise ValueError("Outcome must be a completed success or failure")
                amount = Decimal(str(number(result["cost_usd"])))
                count = successes + result["success"]
                candidate = (cost + amount, choices + [provider])
                if count not in following or candidate[0] < following[count][0]:
                    following[count] = candidate
        states = following
    exact = [
        dict(successes=k, cost_usd=float(v[0]), choices=v[1]) for k, v in sorted(states.items())
    ]
    for row in exact:
        row["pareto"] = not any(
            k > row["successes"] and cost <= states[row["successes"]][0]
            for k, (cost, _) in states.items()
        )
    targets = []
    for target in range(len(pairs) + 1):
        feasible = [k for k in states if k >= target]
        best = min(feasible, key=lambda k: (states[k][0], -k)) if feasible else None
        targets.append(
            dict(
                target_successes=target,
                successes=best,
                cost_usd=float(states[best][0]) if best is not None else None,
                choices=states[best][1] if best is not None else None,
            )
        )
    return dict(
        exact=exact,
        at_least=targets,
        note="Hindsight only. Each pair pays for one run, including both-fail pairs.",
    )


def metrics(pairs, choices):
    selected = [pair[choice] for pair, choice in zip(pairs, choices, strict=True)]
    count = len(selected)
    wins = sum(r["success"] for r in selected)
    cost = sum(r["cost_usd"] for r in selected)
    result = dict(
        tasks=count,
        successes=wins,
        success_rate=wins / count if count else None,
        cost_usd=cost,
        mean_cost_usd=cost / count if count else None,
        cost_per_success_usd=cost / wins if wins else None,
        route_fraction={p: choices.count(p) / count if count else None for p in PROVIDERS},
    )
    for field in ("latency_seconds", "browser_actions", "input_tokens", "output_tokens"):
        total = sum(r[field] for r in selected)
        result[field] = total
        result[f"mean_{field}"] = total / count if count else None
    return result


def score(summary_path, plan_path):
    if Path(plan_path).with_suffix(".sha256").read_text().strip() != digest(plan_path):
        raise ValueError("Plan changed after it was frozen")
    plan = read(plan_path)
    summary, paths, files = sources(summary_path)
    required = {str(p) for p in files} | {str(Path(__file__).resolve())}
    required.update(
        str(root / name)
        for runs in paths.values()
        for root in runs.values()
        for name in ("run.json", "trace.jsonl")
    )
    if set(plan["source_hashes"]) != required:
        raise ValueError("Plan must cover every source file")
    if plan["source_hashes"].get(str(Path(summary_path).resolve())) != digest(summary_path):
        raise ValueError("Paired summary changed after planning")
    if digest(plan["rule_path"]) != plan["rule_sha256"]:
        raise ValueError("Router changed after planning")
    for path, expected in plan["source_hashes"].items():
        if digest(path) != expected:
            raise ValueError("Source changed after planning")
    if [key(e) for e in plan["entries"]] != [key(p) for p in summary["pairs"]]:
        raise ValueError("Plan order differs from paired manifest")
    paired = []
    for entry, pair in zip(plan["entries"], summary["pairs"], strict=True):
        if entry["provider"] not in PROVIDERS or entry["index"] != pair["index"]:
            raise ValueError("Invalid plan selection")
        if entry["goal_sha256"] != pair["deepseek_goal_sha256"]:
            raise ValueError("Plan canonical goal hash differs")
        if (
            type(entry["exact_goal_match"]) is not bool
            or entry["exact_goal_match"] != pair["exact_initial_goal_match"]
        ):
            raise ValueError("Plan goal-match flag differs")
        row = {}
        for provider in PROVIDERS:
            path = paths[provider][key(pair)] / "run.json"
            if str(path) != entry["runs"][provider] or str(path) not in plan["source_hashes"]:
                raise ValueError("Plan run path differs")
            run = read(path)
            if key(run) != key(pair) or type(run["success"]) is not bool:
                raise ValueError("Invalid or unfinished run")
            if run["status"] != ("success" if run["success"] else "task_failure"):
                raise ValueError("Run status differs from outcome")
            expected_model = "deepseek-flash" if provider == "deepseek" else "gpt-6.1-sol"
            if run["model"] != expected_model or run["instance_hosts_match"] is not True:
                raise ValueError("Run provider or environment check differs")
            if number(run["uncertain_cost_usd"]) != 0:
                raise ValueError("Completed runs must have no uncertain charge")
            for field in (
                "cost_usd",
                "latency_seconds",
                "browser_actions",
                "input_tokens",
                "output_tokens",
            ):
                number(run[field])
            if any(
                type(run[f]) is not int
                for f in ("browser_actions", "input_tokens", "output_tokens")
            ):
                raise ValueError("Token and action counts must be integers")
            if (
                run["success"] != pair[f"{provider}_success"]
                or run["cost_usd"] != pair[f"{provider}_cost_usd"]
            ):
                raise ValueError("Run differs from paired outcome or cost")
            row[provider] = run
        paired.append(row)
    output, curves = {}, {}
    for scope in ("all", "exact_goal_match"):
        indices = [
            i for i, e in enumerate(plan["entries"]) if scope == "all" or e["exact_goal_match"]
        ]
        rows = [paired[i] for i in indices]
        choices = [plan["entries"][i]["provider"] for i in indices]
        output[scope] = dict(
            indices=indices,
            heuristic=metrics(rows, choices),
            **{f"always_{p}": metrics(rows, [p] * len(rows)) for p in PROVIDERS},
        )
        curves[scope] = dict(indices=indices, **frontier(rows))
    return dict(
        rule_id=plan["rule_id"],
        plan_sha256=digest(plan_path),
        source_hashes=plan["source_hashes"],
        note=NOTE,
        metrics=output,
    ), curves


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "score"))
    parser.add_argument("--paired-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output directory already exists; choose a new path")
    if (args.stage == "score") != (args.plan is not None):
        parser.error("Only score requires --plan")
    try:
        if args.stage == "plan":
            files = {"plan.json": make_plan(args.paired_summary)}
        else:
            report, curves = score(args.paired_summary, args.plan)
            files = {"metrics.json": report, "frontier.json": curves}
        args.output.mkdir(parents=True, exist_ok=False)
        for name, value in files.items():
            (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        if args.stage == "plan":
            plan_path = args.output / "plan.json"
            checksum = plan_path.with_suffix(".sha256")
            checksum.write_text(digest(plan_path) + "\n")
            plan_path.chmod(0o444)
            checksum.chmod(0o444)
        if args.stage == "score":
            with (args.output / "frontier.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["scope", "target_successes", "successes", "cost_usd"])
                for scope, curve in curves.items():
                    for row in curve["at_least"]:
                        writer.writerow(
                            [scope, row["target_successes"], row["successes"], row["cost_usd"]]
                        )
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
