# CS2860 — Cooperative MARL with Ambiguous Teammate Dropout

Flat MAPPO with RIAL-style discrete communication on **RWARE** (and **MultiGrid**),
plus a controlled-dropout / delayed-heartbeat mechanism that turns "stale teammate
status" and "teammate has died" into a deliberately ambiguous distinction.

> **Central question:** does a learned communication channel let the team
> distinguish between **true teammate dropout** and a **merely stale teammate
> heartbeat**, particularly in the regime where both delay and dropout are
> present?

This branch (`feature/rware-ambiguity-mechanism`) is the **fully integrated**
version: wrapper-level dropout/heartbeat mechanism, all four study methods,
the experiment matrix runner, the analysis/plot pipeline, and the demo
rollout (with optional pyglet window).

> **Looking for the research overview?** Read [`OVERVIEW.md`](OVERVIEW.md).
> It explains the scientific question, the mechanism, the four
> methods/regimes, the pooled v3 data we already have, and the
> production-matrix plan for the final paper. This README is the
> CLI / commands reference; `OVERVIEW.md` is the research-and-paper
> companion. For per-experiment provenance and "read these numbers
> carefully" caveats see [`matrix_results/README.md`](matrix_results/README.md).

---

## Table of contents

1. [Setup](#setup)
2. [Quick start](#quick-start)
3. [Conceptual overview](#conceptual-overview)
   * [The unified env](#the-unified-env)
   * [The dropout mechanism](#the-dropout-mechanism)
   * [The delayed-heartbeat mechanism](#the-delayed-heartbeat-mechanism)
   * [The communication channel](#the-communication-channel)
   * [No-oracle invariants](#no-oracle-invariants)
4. [The four methods](#the-four-methods)
5. [The four regimes](#the-four-regimes)
6. [Training a single condition](#training-a-single-condition)
7. [Running the heuristic baseline](#running-the-heuristic-baseline)
8. [Running the experiment matrix](#running-the-experiment-matrix)
9. [Aggregation and plots](#aggregation-and-plots)
10. [Demo rollout (with optional GUI)](#demo-rollout-with-optional-gui)
11. [TensorBoard](#tensorboard)
12. [Repo layout](#repo-layout)
13. [Current status / known issues](#current-status--known-issues)
14. [Documents in `Planning/`](#documents-in-planning)

---

## Setup

```bash
cd /path/to/cs2860
uv sync
```

All commands below use `uv run`; equivalently `source .venv/bin/activate` and
drop the prefix. Python is pinned in `.python-version`; runtime deps live in
`pyproject.toml` (`uv.lock` is committed).

Key runtime deps: `torch`, `numpy`, `gymnasium`, `rware`, `pettingzoo`,
`matplotlib`, `tensorboard`, `pyglet` (for the optional render).

---

## Quick start

```bash
# 1. Smoke-test that the wrapper, MAPPO, eval, and logger all work end-to-end.
uv run python -m policies.train \
    --env rware-tiny-2ag-v2 --rollout 64 --updates 2 \
    --eval-every 1 --eval-episodes 1 --seed 0

# 2. See the dropout + heartbeat mechanism in a single deterministic rollout.
uv run python -m policies.demo_rware_dropout \
    --env rware-tiny-2ag-v2 --max-steps 30 \
    --dropout --dropout-agent 1 --dropout-time 10 \
    --heartbeat --heartbeat-period 2 --heartbeat-delay 1 \
    --seed 0

# 3. Inspect the experiment matrix that *would* be launched.
uv run python -m policies.experiments.run_rware_matrix --dry-run
```

---

## Conceptual overview

### The unified env

`policies/wrappers/unified.py` defines `UnifiedMARLEnv`, a thin state machine
that wraps any `BaseAdapter` (currently `RwareAdapter`, `MultigridAdapter`)
into a single fixed-shape, numpy-friendly contract:

| key | shape / type | meaning |
|---|---|---|
| `obs` | `float32 (n_agents, obs_dim)` | per-agent observation; zeroed for dead agents; `obs_dim` includes optional heartbeat-freshness features |
| `available_actions` | `uint8 (n_agents, n_env_actions)` | 1 = legal, 0 = masked |
| `alive` | `bool (n_agents,)` | true internal alive mask |
| `reward` | `float32 (n_agents,)` | 0 for agents dead before the step |
| `messages` | `float32 (n_agents, n_msg_tokens)` | one-hot message each agent emitted this step (see [no-oracle invariants](#no-oracle-invariants)) |
| `done` | `bool` | episode-level done |
| `info` | `dict` | always contains the `debug_*` keys below |

Stable debug keys (zeros / sentinels when the mechanism is off):

* `info["debug_true_alive"]`     — `bool (n_agents,)`
* `info["debug_dropout_fired"]`  — `bool`
* `info["debug_failed_agent"]`   — `int` (`-1` if no dropout has fired)
* `info["debug_heartbeat_age"]`  — `int64 (n_agents, n_agents)`

These keys are for analysis / heuristic / demo only; the actor never reads
them.

### The dropout mechanism

`DropoutConfig` (in `policies/wrappers/unified.py`) supports two modes:

* **Fixed:** `agent` and `time` are both set → deterministic per-episode.
* **Window:** `window_start` / `window_end` are both set → a random agent at a
  random step inside the window, drawn with a reset-seed-derived RNG so it is
  reproducible per seed.

When the dropout fires:

1. The targeted agent is marked dead **before** its action is applied that
   step, so it never produces a real env action.
2. Its row in `actions` is replaced with a NOOP at the wrapper.
3. `_alive[target] = False` permanently — the agent cannot be resurrected
   even if the underlying RWARE adapter has no per-agent termination.
4. Its reward is zeroed from this step onward.
5. Its observation is zeroed (`zero_obs_on_death=True`).

CLI flags exposed by every entrypoint that takes a wrapper:

```
--dropout                         # enable dropout
--dropout-agent INT               # fixed mode: which agent dies
--dropout-time INT                # fixed mode: at which step
--dropout-window-start INT        # window mode
--dropout-window-end INT          # window mode (exclusive)
```

### The delayed-heartbeat mechanism

`HeartbeatConfig` + `HeartbeatTracker` (in `policies/wrappers/heartbeat.py`)
implement a per-agent freshness signal:

* Each alive agent emits a heartbeat every `period` steps.
* A heartbeat takes `delay` steps to reach all teammates (in-flight messages
  are buffered).
* Per-step the wrapper appends an `(n_agents, n_agents)` freshness matrix to
  every agent's observation: `freshness[i, j]` = how many steps ago agent
  `i` last heard from agent `j`, clipped to a max sentinel value.
* On dropout: the **failed agent's last in-flight heartbeat still arrives on
  schedule**, after which there is silence. This is the entire point of the
  ambiguity: from the receiver's view, "no recent heartbeat" can mean either
  "teammate is gone" or "teammate's status simply hasn't published yet."

CLI flags:

```
--heartbeat                       # enable
--heartbeat-period INT            # default 1
--heartbeat-delay INT             # default 0
```

### The communication channel

The actor outputs **two heads**: an environment action and a discrete
**message token** in `[0, n_msg_tokens)`. At each step, every agent's
emitted token is one-hot-encoded into `(n_agents, n_msg_tokens)` and
exposed back to all agents on the next step as part of the actor input
(this is the RIAL-style channel).

`--no-comm` is the ablation: it forces `n_msg_tokens=1`, so the channel
exists structurally but carries zero bits of information.

### No-oracle invariants

Two invariants make sure the actor does **not** receive a free death signal
(any of which would defeat the central scientific question):

1. **Obs / heartbeat path:** the actor only sees the heartbeat freshness
   matrix; it never receives `info["debug_true_alive"]` or any other
   ground-truth death indicator.
2. **Message channel:** dead agents would naively emit an all-zero one-hot
   row, which is trivially distinguishable from any live one-hot. To prevent
   this oracle, dead agents now **echo their last emitted one-hot**, and the
   `_last_messages` buffer is initialised to a token-0 one-hot at reset so
   that an agent which dies before ever emitting still looks like a normal
   sender of token 0. Verified directly by probing
   `env.step(...)["messages"]` after a forced dropout.

If you want to see the leak yourself before the fix: revert
`policies/wrappers/unified.py` step() and re-run the demo — dead rows go
to all-zeros, which is a perfect dropout oracle.

---

## The four methods

| name | comm tokens | heartbeat in obs | learning | source |
|---|---|---|---|---|
| `heuristic`                  | none (1 dummy)  | yes             | no  | `policies/baselines/rware_heuristic.py` |
| `mappo-no-comm`              | 1 (dummy)       | no              | yes | `policies/train.py --no-comm`           |
| `mappo-heartbeat-only`       | 1 (dummy)       | yes             | yes | `policies/train.py --no-comm --heartbeat` |
| `mappo-heartbeat-plus-comm`  | `--n-msg-tokens` (default 8) | yes | yes | `policies/train.py --heartbeat`         |

The heuristic is a non-learning, paper-inspired hierarchical-style policy:
each agent moves greedily toward its assigned shelf request; teammates whose
heartbeat age exceeds a fixed `--stale-threshold` are treated as having
abandoned their job, and the responsibility is reallocated. When
`--heartbeat` is **not** passed, the heuristic disables its staleness gate
entirely (the wrapper still emits a clipped sentinel age that would
otherwise look "always stale"); this keeps the heuristic functional in both
heartbeat-on and heartbeat-off regimes.

---

## The four regimes

| regime          | dropout | non-zero heartbeat delay |
|-----------------|---------|--------------------------|
| `baseline`      | no      | no                       |
| `delay-only`    | no      | yes                      |
| `dropout-only`  | yes     | no                       |
| `delay-dropout` | yes     | yes (ambiguous case)     |

The matrix runner schedules non-zero delays (`D in {2, 5}` by default) only
in the regimes where delay is meaningful, and prunes degenerate cells (e.g.
`mappo-no-comm` has no heartbeat in obs and no usable comm, so its delay
sweep collapses to `D = 0`).

---

## Training a single condition

```bash
uv run python -m policies.train \
    --env rware-tiny-4ag-v2 \
    --rollout 512 --updates 1000 \
    --eval-every 25 --eval-episodes 3 \
    --seed 0 \
    --heartbeat --heartbeat-period 2 --heartbeat-delay 5 \
    --dropout --dropout-window-start 50 --dropout-window-end 200 \
    --shape-rewards --pickup-bonus 0.5 \
    --log-dir runs --run-name pol-delay-dropout-s0
```

Reward shaping (`--shape-rewards`, RWARE only) adds `+pickup-bonus` to an
agent's reward the step its `carrying_shelf` flips from `None` to a
*currently-requested* shelf. Without shaping, RWARE's stock reward is +1
per delivery and 0 otherwise; on `rware-tiny-4ag-v2` that is too sparse
for vanilla MAPPO to crack inside a CPU-friendly budget. Shaping turns
a 512k-step verification run from `eval_ret_mean = 0.0` (no shaping) into
`eval_ret_mean ≈ 80, max ≈ 166, 30/40 evals non-zero` (with shaping).

Common flags (full list via `--help`):

```
--env ENV                         # registered env id
--n-agents INT                    # MultiGrid only; rware bakes it in
--n-msg-tokens INT                # comm tokens (default 8)
--no-comm                         # ablation: force n_msg_tokens=1
--rollout INT                     # steps per PPO rollout
--updates INT                     # number of PPO updates
--eval-every INT --eval-episodes INT
--seed INT
--device {cpu,cuda}
--hidden INT --depth INT
--lr-actor / --lr-critic
--update-epochs / --minibatches / --clip-range / --entropy-coef
--gamma / --gae-lambda
--save PATH                       # save final checkpoint
--log-dir DIR --run-name NAME --no-log
# mechanism flags:
--dropout / --dropout-agent / --dropout-time
--dropout-window-start / --dropout-window-end
--heartbeat / --heartbeat-period / --heartbeat-delay
# reward shaping (RWARE only):
--shape-rewards / --pickup-bonus / --step-penalty
```

Each run (unless `--no-log`) writes:

* `<log-dir>/<run-name>/metrics.csv` — per-update scalar log
* `<log-dir>/<run-name>/config.json` — full resolved config + cli args
* `<log-dir>/<run-name>/events.out.tfevents.*` — TensorBoard

Eval is **deterministic / greedy** (argmax) over `--eval-episodes` episodes
at `--eval-every` updates; `train/ep_return_mean` is the on-policy stochastic
return averaged over the rollout's completed episodes.

---

## Running the heuristic baseline

```bash
uv run python -m policies.baselines.rware_heuristic \
    --env rware-tiny-4ag-v2 --episodes 10 \
    --max-steps 500 --stale-threshold 4 \
    --heartbeat --heartbeat-period 2 --heartbeat-delay 0 \
    --dropout --dropout-agent 0 --dropout-time 100 \
    --seed 0 \
    --log-dir runs --run-name heur-delay-dropout-s0
```

Writes the same `metrics.csv` schema as MAPPO so the aggregator/plotter can
treat it uniformly.

---

## Running the experiment matrix

```bash
# 0. Preview which (method, regime, delay, seed) cells will run.
uv run python -m policies.experiments.run_rware_matrix --dry-run

# 1. Real run on the main env. Tune --updates / --rollout / --seeds.
uv run python -m policies.experiments.run_rware_matrix \
    --env rware-tiny-4ag-v2 \
    --updates 200 --rollout 256 \
    --seeds 0 1 2 \
    --log-dir runs/exp_matrix
```

Useful narrowing flags: `--methods`, `--regimes`, `--delays`, `--limit`.

The runner shells out to `policies.train` (and `policies.baselines.rware_heuristic`
for the heuristic method) per cell, then writes everything under
`<log-dir>/<run-name>/` with run names like
`mappo-heartbeat-plus-comm__delay-dropout__d5__s2`.

### Parallel dispatch (`--max-parallel`, `--threads-per-cell`)

By default cells run **one at a time**. Pass `--max-parallel N` to run up to
N cells concurrently as separate subprocesses. Each child has its BLAS /
OpenMP / Accelerate thread pool capped via `--threads-per-cell K`
(default 1) so parallel children don't oversubscribe the cores -- without
this cap, every Python child spawns a thread pool sized to the physical
core count and N children fight each other.

Recommended sweet spot on an 8-core Apple Silicon laptop:

```bash
uv run python -m policies.experiments.run_rware_matrix \
    --methods mappo-heartbeat-only mappo-heartbeat-plus-comm \
    --regimes delay-only delay-dropout \
    --delays 30 --seeds 0 1 2 \
    --updates 1000 --rollout 512 \
    --shape-rewards \
    --dropout-mode window --dropout-window-start 200 --dropout-window-end 350 \
    --log-dir runs/exp_pilot_v4 \
    --max-parallel 3 --threads-per-cell 1 \
    --production-eval
```

What you'll see while it runs:

```
[start] mappo-heartbeat-only__delay-only__d30__s0  pid=23912  in_flight=1/3  queued=11  done=0
[start] mappo-heartbeat-only__delay-only__d30__s1  pid=23913  in_flight=2/3  queued=10  done=0
[start] mappo-heartbeat-only__delay-only__d30__s2  pid=23914  in_flight=3/3  queued=9   done=0
[done ] mappo-heartbeat-only__delay-only__d30__s0  exit=0 (ok )  cell_elapsed= 612.4s  wall= 612.4s  ...
[start] mappo-heartbeat-only__delay-dropout__d30__s0  pid=24205  in_flight=3/3  queued=8   done=1
...
```

Per-cell stdout still lands in `<log-dir>/<run-name>/stdout.log` (drop
`--no-capture` if you want clean per-cell logs in parallel mode).

### Production-grade eval (`--production-eval`)

The defaults `--eval-every 10 --eval-episodes 3` are intentionally
smoke-test-y. For a research-grade run that should produce publishable
curves, pass `--production-eval` to override these to:

* `--eval-every 25` (one greedy eval every 25 PPO updates)
* `--eval-episodes 30` (30 episodes per eval, ~10x less noisy than 3)

Equivalent to setting both flags explicitly. We saw in the v3 pilot
that `eval-episodes 3` was the dominant source of cell-to-cell noise
(see `matrix_results/README.md`); 30 episodes brings the eval SEM down
to roughly the seed-to-seed SEM, which is the right ratio.

### Ctrl+C handling

Ctrl+C in a parallel run is handled cleanly: the matrix runner sends
`SIGTERM` to all in-flight children, waits 10s for them to exit, then
escalates to `SIGKILL`. A second Ctrl+C escalates to `SIGKILL`
immediately. No queued cells are launched after the first signal.

---

## Aggregation and plots

### Aggregator (CSV summaries for full-matrix runs)

```bash
# Build per_run.csv and summary.csv from the matrix output.
uv run python -m policies.analysis.aggregate \
    --log-dir runs/exp_matrix --last-k 5

# Render the five summary figures used in the writeup.
uv run python -m policies.analysis.plot_results \
    --in-dir runs/exp_matrix --out-dir runs/exp_matrix/figures
```

Summary plots emitted by `plot_results.py`:

1. **Team return by method × regime** — overall headline.
2. **Throughput / completion** (uses `train/ep_length_mean`).
3. **Post-dropout recovery** — return after the dropout step.
4. **Ambiguous-regime comparison** — `delay-dropout` vs others.
5. **Communication comparison** — `mappo-heartbeat-only` vs
   `mappo-heartbeat-plus-comm` across regimes.

### Pilot-sized analysis (per-update curves, per-seed dots, stat tests)

The summary plotter is built for the *full* 24+-cell matrix. Pilot-sized
sweeps (4 cells × 3 seeds) need richer visualisation to separate signal
from noise.  All examples below point at the committed pooled v3 pilot
under `matrix_results/exp_pilot_v3/` (n=6 seeds, D=30, see
`matrix_results/README.md` for provenance):

```bash
# Single-pilot dashboard: per-update train + eval curves, per-seed dots
# on the final-eval bars, and a Welch t-test annotation between
# heartbeat-only and heartbeat+comm in each regime. Headline:
# "comm benefit (delay-only)", "comm benefit (delay-dropout)" and the
# interaction (drop − only).
uv run python -m policies.analysis.pilot_dashboard \
    --log-dir matrix_results/exp_pilot_v3 \
    --out matrix_results/exp_pilot_v3/dashboard.png \
    --title "v3 pilot pooled (D=30, n=6: PK s10-s12 + SAM s0-s2)"

# Cross-pilot comparison: how the comm benefit shifts as a function
# of the heartbeat delay D. Plots one panel per regime (eval return
# vs D, lines = method) plus a comm-benefit-vs-D curve with SEM bars
# and per-point Welch p-values. This is the visual answer to
# "does comm matter more when D is large?".
uv run python -m policies.analysis.compare_pilots \
    --log-dirs runs/exp_pilot_v2 runs/exp_pilot_v3 \
    --out runs/figures/compare_v2_v3.png \
    --title "v2 (D=5) vs v3 (D=30) | window dropout, n=3 seeds"

# Empirical heartbeat-age dynamics: runs random rollouts under each
# delay setting and histograms the alive-vs-dead heartbeat age that a
# receiver actually sees. The "ambiguity window" (overlap between the
# two distributions) is the empirical reason comm helps at large D and
# does nothing at small D.
uv run python -m policies.analysis.heartbeat_dynamics \
    --env rware-tiny-4ag-v2 \
    --delays 5 30 \
    --episodes 30 --max-steps 500 \
    --dropout-window-start 200 --dropout-window-end 350 \
    --out runs/figures/heartbeat_dynamics.png

# Live-watch mode: re-render the dashboard PNG every N seconds while
# training is still in progress.  metrics.csv is written incrementally
# by the trainer, so each refresh shows real partial curves.  macOS
# Preview will reload the file in-place when it changes -- pair this
# with `open <pilot>/dashboard.png` in Preview before you start.
# (use runs/<pilot> for live local training; matrix_results/<pilot> for
#  the committed snapshots.)
uv run python -m policies.analysis.pilot_dashboard \
    --log-dir runs/exp_pilot_v3 --watch 30
```

### Verifying the dashboard numbers

Plotters can lie -- a buggy aggregation step or a miscalibrated
statistical test can produce confidently-wrong figures.  To keep the
plots trustworthy we ship `policies.analysis.verify_dashboard`, which
re-derives every drawn number from independent sources and asserts
agreement:

```bash
uv run python -m policies.analysis.verify_dashboard
```

It runs three checks:

1. **Per-cell finals**: re-reads each `metrics.csv` from scratch and
   confirms the per-seed values the dashboard exposes match a manual
   recompute exactly (rel_tol = 1e-9).
2. **Welch t-test**: re-runs every (regime, delay) comparison through
   `scipy.stats.ttest_ind(equal_var=False)` and asserts the p-values
   our plotter draws agree to within 0.03 absolute *and* land on the
   same significance label (`ns`/`*`/`**`/`***`).
3. **Heartbeat dynamics**: collects 20 random-policy rollouts at each
   delay D and asserts `median(alive_age) == D` and
   `median(dead_age) == max_age_clip`, exactly as the
   `HeartbeatTracker` docstring predicts.

This script caught a real bug (a Cornish-Fisher tail approximation in
the in-house t-test was reporting `p = 0.022 *` for a result whose
exact p was `0.146 ns`); the dashboards now use scipy directly.

Why these three coexist with the summary plotter:

* `aggregate.py` + `plot_results.py` are designed for the final
  publication-style matrix and produce *summary* bar charts only.
* `pilot_dashboard.py` shows the *per-update curves* and *per-seed
  spread* you need to interpret a pilot. With n=3 seeds, the bar
  charts hide whether a +10 gap is signal or one outlier seed.
* `compare_pilots.py` lets two or more pilots share an x-axis (`D`),
  which is the only way to see the *interaction* between delay and
  comm benefit.
* `heartbeat_dynamics.py` is mechanism-level: it does not depend on
  any trained policy and explains *why* the comm benefit moves with D
  by showing the alive-vs-dead age overlap directly.

Metric definitions (full list in `Planning/ExperimentPlan.md`):

* `final_eval_return` — last `eval/ep_return_mean` written to `metrics.csv`.
  For the heuristic, this is the per-episode team return.
* `train_return_lastK` — mean of `train/ep_return_mean` over the last `K=5`
  rows. Smoothes single-episode noise.
* `train_return_max` — best `train/ep_return_mean` ever observed
  (sanity-check: did the policy ever learn anything?).
* `ep_length_mean` — last `train/ep_length_mean`. RWARE has a fixed step
  budget, so shorter mean ep_length under the same return is throughput.
* `deliveries_mean` — heuristic-only column.

---

## Demo rollout (with optional GUI)

```bash
# Heuristic policy, deterministic rollout, dropout fires at t=50 (4-agent env).
uv run python -m policies.demo_rware_dropout \
    --env rware-tiny-4ag-v2 \
    --max-steps 200 \
    --dropout --dropout-agent 0 --dropout-time 50 \
    --heartbeat --heartbeat-period 2 --heartbeat-delay 5 \
    --seed 0

# Same, but with a trained MAPPO checkpoint:
uv run python -m policies.demo_rware_dropout \
    --env rware-tiny-4ag-v2 \
    --checkpoint runs/<run-name>/final.pt \
    --dropout --dropout-agent 0 --dropout-time 50 \
    --heartbeat --heartbeat-period 2 --heartbeat-delay 5 \
    --seed 0

# Live RWARE pyglet window (locally on macOS only; sandboxed shells block this).
uv run python -m policies.demo_rware_dropout \
    --env rware-tiny-4ag-v2 \
    --dropout --dropout-agent 0 --dropout-time 50 \
    --render --render-pause 0.15
```

The text output prints per-step `alive`, `true_alive`, `heartbeat_age`,
`env_act`, `msg_tok`, `step_R`, `cum_R`, with `<-- DROPOUT` marking the
exact step the dropout fires. The 2-D heartbeat-age matrix is reduced to a
per-sender minimum age for legibility.

---

## TensorBoard

```bash
uv run tensorboard --logdir runs
```

Open the URL it prints (typically <http://127.0.0.1:6006>).

---

## Repo layout

| Path | Role |
|---|---|
| `policies/train.py` | CLI entry: MAPPO training |
| `policies/mappo/` | MAPPO trainer, networks, runner, rollout buffer |
| `policies/wrappers/unified.py` | `UnifiedMARLEnv` + `DropoutConfig` |
| `policies/wrappers/heartbeat.py` | `HeartbeatConfig` + `HeartbeatTracker` |
| `policies/wrappers/rware_adapter.py` | RWARE → unified adapter |
| `policies/wrappers/multigrid_adapter.py` | MultiGrid → unified adapter |
| `policies/baselines/rware_heuristic.py` | Non-learning heuristic + CLI |
| `policies/experiments/run_rware_matrix.py` | Methods × regimes × delays × seeds runner |
| `policies/analysis/aggregate.py` | Build per_run.csv + summary.csv |
| `policies/analysis/plot_results.py` | Five-figure summary plot pipeline |
| `policies/analysis/pilot_dashboard.py` | Per-pilot dashboard (curves + per-seed dots + scipy Welch t-tests, with `--watch` for live re-rendering) |
| `policies/analysis/compare_pilots.py` | Cross-pilot comparison (comm benefit vs delay D) |
| `policies/analysis/heartbeat_dynamics.py` | Empirical alive/dead heartbeat-age distributions |
| `policies/analysis/verify_dashboard.py` | Cross-checks dashboard outputs against the aggregator, scipy, and the analytical heartbeat prediction |
| `policies/demo_rware_dropout.py` | Single-rollout demo (text + optional pyglet) |
| `policies/logger.py` | CSV + TensorBoard scalar logger |
| `policies/summarize_runs.py` | Tail-and-summarise existing runs |
| `envs/sample_envs.py` | Helpers for env discovery |
| `runs/` | Run outputs (gitignored) |
| `logs/` | Free-form text logs (gitignored) |
| `Planning/` | Course-facing planning docs |

---

## Current status / known issues

* **Pipeline is end-to-end functional.** All four methods launch, dropout
  and heartbeat fire as configured, the matrix runner produces compatible
  per-run output, the aggregator builds CSVs, the plot script renders all
  five figures, and the demo runs (with optional pyglet GUI in non-sandboxed
  shells).
* **Message-channel oracle leak — fixed.** Earlier, dead agents emitted an
  all-zero one-hot, which made dropout trivially detectable from `messages`
  alone and defeated the central ambiguity story. Dead agents now echo
  their last emitted token; the buffer is initialised to a token-0 one-hot
  so an agent that dies before ever emitting still looks like a normal
  token-0 sender. See [no-oracle invariants](#no-oracle-invariants).
* **Sparse-reward proof-of-learning.** Vanilla MAPPO on raw
  `rware-tiny-4ag-v2` in 200k steps gave 13.5% of training rollouts with
  at least one delivery and `eval_ret_mean = 0.0` across all evals. With
  `--shape-rewards --pickup-bonus 0.5`, the same env at 512k steps with
  `mappo-heartbeat-plus-comm` reaches `train_ret_mean ≈ 277` (last 100
  updates), `eval_ret_mean ≈ 83` (last 10 evals), `eval_max = 166`,
  `30/40 evals non-zero`, in **~10.6 minutes** of wall-time on CPU.
  Conclusion: shaping is required for the matrix to produce non-trivial
  results on the 4-agent env in CPU-friendly budgets; the matrix runner
  should be invoked with `--shape-rewards` (or the equivalent flag wired
  through to per-cell trainings).
* **Heuristic is a floor, not an upper bound.** The heuristic does
  one-step greedy heading, no real path planning, and accepts RWARE's
  move-cancellation cost on illegal moves. It typically returns 0
  deliveries — which is the expected "non-learning floor" reading.

See `Planning/ExperimentPlan.md` for the full study design and the writeup
claim the figures are designed to make falsifiable.

---

## Documents in `Planning/`

* `Planning/ExperimentPlan.md` — full study design, methods, regimes,
  delays, seeds, metric definitions, expected key claims.
* `Planning/Tasks.md` — running task tracker.
* `Planning/Important.md` — repo conventions and runtime deps notes.

---

## Branch context

This is `feature/rware-ambiguity-mechanism` — the integrated branch combining
the original baseline + comm pipeline, the wrapper-level dropout/heartbeat
mechanism, the heuristic baseline, the experiment matrix runner, the
analysis/plot pipeline, and the demo. `main` does not yet have these
components; merging is the user's call.
