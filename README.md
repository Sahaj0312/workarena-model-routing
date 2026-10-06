# WorkArena model routing

Measure whether a cheap browser agent can complete enterprise tasks at low cost.
Run DeepSeek Flash and GPT-6.1 Sol on official WorkArena instances with the same
agent, task IDs, and seeds, then compare outcomes and costs. Offline analysis
includes a fixed description-only routing rule and a hindsight cost frontier.
There is no training pipeline.

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
OPENAI_API_KEY=your_api_key
```

The selected provider account needs credit. Only that provider's API key is required.
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

Use `--provider sol` for GPT-6.1 Sol. For example, these separate caps keep smoke
and pilot estimates within $50 in total:

```sh
uv run workarena-baseline preflight --provider sol --check-access
uv run workarena-baseline run --provider sol --manifest experiments/smoke.json --output results/sol-smoke --budget-usd 10
uv run workarena-baseline run --provider sol --manifest experiments/tasks.json --output results/sol --budget-usd 40
uv run workarena-baseline summary results/sol
```

Inspect the smoke results before running the pilot. A budget stop can leave
tasks unfinished. The default provider is `deepseek`.

Use a new output directory for each batch. Keep the checked-in manifests fixed
for later model comparisons. The pilot has 50 task/seed pairs: 25 L1 tasks and
25 L2 tasks, sampled across categories. It is a pilot, not a full WorkArena
benchmark or a leaderboard result.

## Method and results

V1 completed 13 of 50 tasks. Of its 37 failures, 30 were invalid actions under
the fixed JSON interface. These results use the original L1 runs and the full
L2 subset repeated after the shared-instance fix. The earlier L2 runs are
excluded. These results measure the model, provider, and harness together;
they do not isolate model capability.

V2 clarifies the common action prompt with explicit JSON argument types, limits,
and complete examples. The parser, failure rules, model settings, task pairs,
and run limits are unchanged. Invalid action formats count as task failures for
both models. There is no format repair or retry. Keep V1 and V2 results separate,
and use the same V2 prompt for both provider baselines.

The agent receives text observations from the browser accessibility tree and
keeps the full text history. It does not send screenshots. It selects one
allowed browser action per model call. WorkArena grades success;
the model cannot mark its own task as successful. Action and time limits stop
long runs. Task failures and infrastructure errors have separate statuses.

Defaults are 30 model turns, at most 30 browser actions, 8,192 output tokens per
call, and 600 seconds per task. The task timeout is a soft limit, checked between
calls. It cannot cut off stalled setup or a browser or model call already in
progress. The model request has a separate 120-second timeout.

The model IDs are `deepseek-flash` and `gpt-6.1-sol`. DeepSeek Flash names V4.1
Flash as of 2026-10-05. Both use high reasoning effort, JSON output, and the
same text history and action prompt. Sol uses Standard processing. Providers
can change aliases. Raw responses preserve model IDs and fingerprints when
supplied; they do not guarantee a fixed model snapshot.

Each task saves a `trace.jsonl` file and a `run.json` result. The trace includes
observations, model responses, actions, usage, and errors. The batch also saves
its settings. API cost is an estimate, not an invoice. DeepSeek uses peak rates
and reserves $0.786432 before each call. Sol uses Standard rates and reserves
$6.53 to cover its input and output limits at the highest Standard rates.
These reserves cover the model limits, not only the requested output cap. If
a call might have been billed but its usage is unknown, its reserve stays
charged to the local budget. Sol uses the full long-context rate above 272,000
input tokens. If cache-write counts are absent, it prices all cache misses as
writes. Cache-write prices replace ordinary input prices; they are not added.
The runner can therefore stop with unused credit in the provider account.
The budget applies to one batch; it does not include other processes or earlier
batches. See [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing/)
and [OpenAI pricing](https://developers.openai.com/api/docs/pricing).

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

## Offline routing analysis

These commands use saved results only. They do not call model APIs or run a
browser. Keep their outputs under the ignored `results/` directory.

The frozen rule `goal-length-v1` selects DeepSeek for goals with at most 100
whitespace-separated words and Sol for longer goals. Blank or non-string goals
are rejected. The threshold is an arbitrary baseline, not a difficulty model.
It was fixed before replay scoring, after aggregate baseline results were known.
Do not tune it on these outcomes.

First save choices without scoring them. The canonical input is the first saved
DeepSeek goal for each task/seed pair. The plan records choices, goal hashes,
word counts, source hashes, and the rule hash; it does not store raw goals.

```sh
uv run python -m analysis.routing plan --paired-summary results/sol-deepseek-paired-summary.json --output results/routing-plan
uv run python -m analysis.routing score --paired-summary results/sol-deepseek-paired-summary.json --plan results/routing-plan/plan.json --output results/routing-score
```

Review the saved plan before the score step. Scoring checks its checksum and
source files. It writes `metrics.json`, `frontier.json`, and `frontier.csv` to a
new directory. The metrics compare the fixed rule with always choosing either
model. The frontier finds the cheapest observed choice for each success target;
it knows both outcomes in advance and pays for one run on every pair, including
pairs where both models failed.

This is exploratory replay, not a live router test or a blind study. Paired runs
can have different generated goal text. Results therefore include a separate
exact-goal subset; that subset is descriptive and is not an
unbiased sample. Even equal goals do not ensure equal live service state. Costs
reuse observed API estimates, exclude routing overhead, smoke, and interrupted
attempts, and do not predict a new cache schedule. The hindsight frontier is a
theoretical bound on these saved runs, not a deployable router.

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
