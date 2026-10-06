# WorkArena model routing

Measure whether a cheap browser agent can complete enterprise tasks at low cost.
The first step is a DeepSeek baseline on official WorkArena instances. A later
step will run a stronger model on the same task IDs and seeds, then compare
outcomes and costs. There is no router or training pipeline yet.

## Setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
uv sync --locked
uv run playwright install chromium
```

Get access to
[ServiceNow/WorkArena-Instances](https://huggingface.co/datasets/ServiceNow/WorkArena-Instances)
with your Hugging Face account. Add these values to `.env` in this directory:

```dotenv
HF_TOKEN=your_read_token
DEEPSEEK_API_KEY=your_api_key
```

The DeepSeek account needs credit. An OpenAI key is not needed for this baseline.
Remove old `SNOW_INSTANCE_URL`, `SNOW_INSTANCE_UNAME`, `SNOW_INSTANCE_PWD`, and
`SNOW_INSTANCE_POOL` settings when using the official managed pool. See the
[WorkArena setup guide](https://github.com/ServiceNow/WorkArena#getting-started).

## Run

Check local setup first, then check access to the managed instances:

```sh
uv run workarena-baseline preflight
uv run workarena-baseline preflight --check-access
```

Run a small smoke test before a larger batch. This command permits up to $1 in
estimated DeepSeek cost:

```sh
uv run workarena-baseline run --manifest experiments/smoke.json --output results/smoke --budget-usd 1
uv run workarena-baseline summary results/smoke
```

After you inspect the smoke results and cost, run the pilot with a chosen budget:

```sh
uv run workarena-baseline run --manifest experiments/tasks.json --output results/deepseek --budget-usd 5
uv run workarena-baseline summary results/deepseek
```

Use a new output directory for each batch. Keep the checked-in manifests fixed
for later model comparisons. The pilot has 50 task/seed pairs: 25 L1 tasks and
25 L2 tasks, sampled across categories. It is a pilot, not a full WorkArena
benchmark or a leaderboard result.

## Method and results

The first pilot completed 13 of 50 tasks. Of its 37 failures, 30 were invalid
actions under the fixed JSON interface. The shared-instance setup issue is
fixed, and the full L2 subset was repeated. Action-format compatibility remains
unresolved, so this is not yet a clean baseline for model capability or routing.

The agent receives text observations from the browser accessibility tree and
keeps the full text history. It does not send screenshots. It selects one
allowed browser action per model call. WorkArena grades success;
the model cannot mark its own task as successful. Action and time limits stop
long runs. Task failures and infrastructure errors have separate statuses.

Defaults are 30 model turns, at most 30 browser actions, 8,192 output tokens per
call, and 600 seconds per task. The task timeout is a soft limit, checked between
calls. It cannot cut off stalled setup or a browser or model call already in
progress. The model request has a separate 120-second timeout.

The model ID is `deepseek-flash`, which names DeepSeek V4.1 Flash as of
2026-10-05. Thinking is enabled with high reasoning effort. The provider can
change this alias. Raw responses preserve the model ID and fingerprint when
the provider supplies them; they do not guarantee a fixed model snapshot.

Each task saves a `trace.jsonl` file and a `run.json` result. The trace includes
observations, model responses, actions, usage, and errors. The batch also saves
its settings. API cost is an estimate, not an invoice. The runner uses peak
rates and reserves $0.786432 before each call, based on the full context and
model output limits. This covers billed tokens beyond the requested output
cap. If a call might have been
billed but its usage is unknown, that reserve stays charged to the local budget.
The runner can therefore stop with unused credit in the provider account.
The budget applies to one batch; it does not include other processes or earlier
batches. See [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing/).

Matching task IDs and seeds fixes the sampled task configuration. It does not
freeze the managed service: hosts, generated record IDs, and live data can
change. Review saved goals and versions before treating later runs as pairs.
Keep the same agent prompt, actions, observations, and limits across models.

Each task and its subtasks must use one shared ServiceNow instance. The runner
passes that instance into WorkArena and checks all subtask hosts after setup,
before any model call. A mismatch is an infrastructure error. L2 results from
the earlier runner without this fix are invalid for model comparisons; exclude
all of them, including reported successes. Keep their traces and costs, then
run the full fixed L2 subset again after the fix. A cleanup error also stops
the batch and remains visible in the result.

## Data and development

WorkArena instances are for evaluation and research. Do not train models or
tune routers on their outcomes. Do not use task cheat methods. Select tasks
before looking at model outcomes.

Keep `.env`, managed instance details, and `results/` private. These paths are
ignored by Git. Traces can contain service data and browser details. Review and
redact them before any publication. Do not publish the gated instance files.

Run checks without keys, browsers, or network access:

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Read [AGENTS.md](AGENTS.md) before changing the code or experiment design.
