"""Commands for a serial, fixed-task baseline."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
API_KEY_NAMES = {"deepseek": "DEEPSEEK_API_KEY", "sol": "OPENAI_API_KEY"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def versions():
    return {
        name: importlib.metadata.version(name)
        for name in ("browsergym-workarena", "browsergym-core", "playwright", "openai")
    }


def provenance():
    digest = hashlib.sha256()
    for path in sorted(
        [
            *ROOT.glob("agent/*.py"),
            *ROOT.glob("models/*.py"),
            *ROOT.glob("experiments/*.py"),
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
        ]
    ):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return {
        "git_revision": revision.stdout.strip() or None,
        "source_sha256": digest.hexdigest(),
        "versions": versions(),
        "python": sys.version.split()[0],
    }


def create_manifest(seed):
    from browsergym.workarena import AGENT_CURRICULUM_L2, TASK_CATEGORY_MAP, get_all_tasks_agents

    rng = random.Random(seed)
    tasks = []
    groups = {}
    for task_id, category in sorted(TASK_CATEGORY_MAP.items()):
        groups.setdefault(category, []).append(task_id)
    for group in groups.values():
        rng.shuffle(group)
    while len(tasks) < 25:
        for category, group in sorted(groups.items()):
            if group and len(tasks) < 25:
                tasks.append(
                    {
                        "task_id": group.pop(),
                        "seed": rng.randrange(1000),
                        "level": "l1",
                        "category": category,
                    }
                )
    categories = {
        task.get_task_id(): category
        for category, curriculum in AGENT_CURRICULUM_L2.items()
        for bucket in curriculum["buckets"]
        for task in bucket
    }
    groups = {}
    for task, task_seed in get_all_tasks_agents(filter="l2", meta_seed=seed):
        task_id = task.get_task_id()
        groups.setdefault(categories[task_id], []).append((task_id, task_seed))
    for category, group in sorted(groups.items()):
        for task_id, task_seed in rng.sample(sorted(set(group)), 5):
            tasks.append(
                {"task_id": task_id, "seed": task_seed, "level": "l2", "category": category}
            )
    return {
        "schema_version": 1,
        "selection_seed": seed,
        "description": "Pilot: 25 L1 and 25 L2 tasks. Not the canonical benchmark distribution.",
        "versions": versions(),
        "tasks": tasks,
    }


def preflight(check_access, provider="deepseek"):
    import browsergym.workarena  # noqa: F401
    from playwright.sync_api import sync_playwright

    checks = {
        "versions": versions(),
        "HF_TOKEN": bool(os.getenv("HF_TOKEN")),
        API_KEY_NAMES[provider]: bool(os.getenv(API_KEY_NAMES[provider])),
    }
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content("<button>Ready</button>")
        checks["chromium"] = page.get_by_role("button").inner_text() == "Ready"
        browser.close()
    if check_access:
        from browsergym.workarena.instance import fetch_instances

        checks["workarena_access"] = bool(fetch_instances())
    print(json.dumps(checks, indent=2))
    return 0 if all(value for value in checks.values()) else 1


def summarize(path):
    runs = [json.loads(p.read_text()) for p in sorted(Path(path).glob("*/run.json"))]
    evaluated = [r for r in runs if r["status"] in {"success", "task_failure"}]
    successes = sum(r["success"] is True for r in evaluated)
    cost = sum(r["cost_usd"] for r in runs)
    batch_path = Path(path) / "batch.json"
    planned = len(json.loads(batch_path.read_text())["tasks"]) if batch_path.exists() else None
    return {
        "planned": planned,
        "runs": len(runs),
        "evaluated": len(evaluated),
        "successes": successes,
        "success_rate": successes / len(evaluated) if evaluated else None,
        "cost_usd": cost,
        "average_cost_usd": cost / len(runs) if runs else None,
        "cost_per_success_usd": cost / successes if successes else None,
        "cache_write_tokens": sum(r["cache_write_tokens"] for r in runs)
        if all(r.get("cache_write_tokens") is not None for r in runs)
        else None,
        **{
            key: sum(r[key] for r in runs)
            for key in (
                "input_tokens",
                "output_tokens",
                "browser_actions",
                "model_latency_seconds",
                "latency_seconds",
                "uncertain_cost_usd",
            )
        },
        "statuses": {
            status: sum(r["status"] == status for r in runs)
            for status in sorted({r["status"] for r in runs})
        },
    }


def run(args):
    from agent.runner import INSTANCE_POLICY, run_task
    from models.common import Budget

    if args.provider == "sol":
        from models.sol import SolClient

        client_type = SolClient
    else:
        from models.deepseek import DeepSeekClient

        client_type = DeepSeekClient

    if args.max_actions < 1 or not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError("Action and time limits must be positive.")
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    tasks = manifest["tasks"][: args.limit] if args.limit else manifest["tasks"]
    if not tasks:
        raise ValueError("The manifest has no tasks.")
    if manifest["versions"] != versions():
        raise ValueError("Installed versions do not match the task manifest.")
    if len({(t["task_id"], t["seed"]) for t in tasks}) != len(tasks):
        raise ValueError("The manifest has duplicate task and seed pairs.")
    budget = Budget(args.budget_usd)
    client = client_type(os.environ.get(API_KEY_NAMES[args.provider], ""), budget)
    args.output.mkdir(parents=True, exist_ok=False)
    batch = {
        "started_at": datetime.now(UTC).isoformat(),
        "provenance": provenance(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "tasks": tasks,
        "model_config": client.config,
        "provider": args.provider,
        "budget_usd": args.budget_usd,
        "max_actions": args.max_actions,
        "timeout_seconds": args.timeout_seconds,
        "instance_policy": INSTANCE_POLICY,
    }
    write_json(args.output / "batch.json", batch)
    try:
        for index, task in enumerate(tasks):
            result = run_task(
                task,
                client,
                args.output / f"{index:03d}",
                max_actions=args.max_actions,
                timeout_seconds=args.timeout_seconds,
            )
            print(
                f"{index + 1}/{len(tasks)} {task['task_id']}: {result['status']} "
                f"(${result['cost_usd']:.4f})",
                flush=True,
            )
            if result["status"] not in {"success", "task_failure"} or result.get("cleanup_error"):
                break
    finally:
        client.close()
        batch.update(
            spent_usd=budget.spent_usd,
            uncertain_usd=budget.uncertain_usd,
            finished_at=datetime.now(UTC).isoformat(),
        )
        write_json(args.output / "batch.json", batch)
        summary = summarize(args.output)
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    return 0 if summary["evaluated"] == len(tasks) else 1


def main():
    load_dotenv(ROOT / ".env")
    # Upstream logs can include managed instance URLs. Save only our own events.
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight", help="Check local setup without a model call")
    check.add_argument("--check-access", action="store_true")
    check.add_argument("--provider", choices=API_KEY_NAMES, default="deepseek")
    manifest = commands.add_parser("manifest", help="Write a fixed 50-task pilot")
    manifest.add_argument("--output", type=Path, default=Path("experiments/tasks.json"))
    manifest.add_argument("--seed", type=int, default=42)
    baseline = commands.add_parser("run", help="Run tasks serially; model calls cost money")
    baseline.add_argument("--provider", choices=API_KEY_NAMES, default="deepseek")
    baseline.add_argument("--manifest", type=Path, default=Path("experiments/smoke.json"))
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--budget-usd", type=float, required=True)
    baseline.add_argument("--limit", type=int)
    baseline.add_argument("--max-actions", type=int, default=30)
    baseline.add_argument("--timeout-seconds", type=float, default=600)
    summary = commands.add_parser("summary", help="Summarize saved runs")
    summary.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            return preflight(args.check_access, args.provider)
        if args.command == "manifest":
            if args.output.exists():
                raise ValueError("Manifest already exists. Use a new output path.")
            write_json(args.output, create_manifest(args.seed))
        elif args.command == "run":
            if args.limit is not None and args.limit < 1:
                raise ValueError("Task limit must be positive.")
            return run(args)
        else:
            print(json.dumps(summarize(args.path), indent=2))
    except Exception as exc:
        # Do not print provider or environment exception details with secrets.
        print(
            f"Command failed ({type(exc).__name__}). Check setup and saved traces.", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
