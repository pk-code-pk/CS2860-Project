# Experiment Plan -- Ambiguous Teammate Disappearance in RWARE

## Final project statement

> Can learned communication help a cooperative team distinguish between
> **true teammate dropout** and merely **stale teammate status signals**?

Concretely: a single agent is removed mid-episode (true dropout). The
remaining agents see each teammate's status only via a periodic
**heartbeat** that can be artificially delayed by `D` steps. When `D > 0`,
"no heartbeat for the last D steps" is *ambiguous*: the teammate may
genuinely have dropped, or they may simply not have published yet. The
hypothesis is that a learned communication channel (RIAL-style discrete
tokens emitted by each agent and consumed by every other agent next
step) lets the team disambiguate these cases better than either heartbeat
alone or no inter-agent signal at all, particularly in the regime where
both delay and dropout are present.

## Environments

* **Main:**  `rware-tiny-4ag-v2` (4 cooperating warehouse agents).
* **Debug / fast iteration:**  `rware-tiny-2ag-easy-v2`.

We deliberately stay on RWARE because it has the right shape for this
question -- multiple agents, a shared global objective (deliveries), no
competitive component, and an obvious failure mode when one agent stops
moving (others must pick up the abandoned task).

## Methods compared

| name                              | comm tokens | heartbeat in obs | learning |
|-----------------------------------|-------------|------------------|----------|
| `heuristic`                       | (none)      | yes              | no       |
| `mappo-no-comm`                   | no          | no               | yes      |
| `mappo-heartbeat-only`            | no          | yes              | yes      |
| `mappo-heartbeat-plus-comm`       | yes         | yes              | yes      |

* The heuristic is a non-learning, paper-inspired hierarchical-style
  baseline (`policies/baselines/rware_heuristic.py`) that allocates
  requested shelves to teammates by greedy nearest-Manhattan and
  reallocates a teammate's task when its `info["debug_heartbeat_age"]`
  exceeds a fixed `stale_threshold`.
* `mappo-*` are all the same `policies/train.py` MAPPO + RIAL pipeline,
  toggled via `--no-comm` / `--heartbeat`.

## Regimes compared

| regime          | dropout | non-zero heartbeat delay |
|-----------------|---------|--------------------------|
| `baseline`      | no      | no                       |
| `delay-only`    | no      | yes                      |
| `dropout-only`  | yes     | no                       |
| `delay-dropout` | yes     | yes                      |

Dropout config defaults: `--dropout-agent 0 --dropout-time 100`. The
agent is removed at step 100 and stays removed for the rest of the
episode. (The wrapper-side mechanism branch will also support a
`--dropout-window-start/--dropout-window-end` interval; that interface
is forwarded by the matrix runner but not exercised in the default
sweep.)

## Heartbeat delays

The default sweep uses `D = 0, 2, 5`. `D = 0` collapses onto the
no-delay regimes, so the matrix runner only schedules `D in {2, 5}`
inside `delay-only` and `delay-dropout`.

## Seeds

Three seeds: `0, 1, 2`. The matrix runner accepts `--seeds` to extend.

## Default matrix size

With four methods, four regimes, two non-zero delays, and three seeds,
and after the runner drops degenerate cells (mappo-no-comm has no
heartbeat in obs and no comm, so its delay sweep is just delay=0):

* heuristic                    : 1 + 2 + 1 + 2 = 6 (delay,regime) cells x 3 seeds = 18 runs
* mappo-no-comm                : 1 + 0 + 1 + 0 = 2 (delay,regime) cells x 3 seeds =  6 runs
* mappo-heartbeat-only         : 1 + 2 + 1 + 2 = 6 (delay,regime) cells x 3 seeds = 18 runs
* mappo-heartbeat-plus-comm    : 1 + 2 + 1 + 2 = 6 (delay,regime) cells x 3 seeds = 18 runs

Total: 60 runs. Reduce via `--methods`, `--regimes`, `--delays`,
`--seeds`, `--limit` for smoke runs.

## Metric definitions

* **`final_eval_return`** -- last value of `eval/ep_return_mean` written
  to `metrics.csv` (= MAPPO greedy-eval team return at the final
  checkpoint). Heuristic runs reuse the same column with the per-episode
  team return.
* **`train_return_lastK`** -- mean of `train/ep_return_mean` over the
  last `K=5` rows. Smoothes out noisy single-episode rollouts.
* **`train_return_max`** -- best `train/ep_return_mean` over the run
  (sanity check: did the policy ever learn anything?).
* **`ep_length_mean`** -- last value of `train/ep_length_mean`. RWARE
  uses a fixed step budget so a *shorter* mean ep_length under the
  same return is a throughput signal.
* **`deliveries_mean`** -- last value of `heuristic/deliveries`
  (heuristic runs only).

The aggregator reduces the per-seed values to mean/std grouped by
(method, regime, delay).

## Expected key claim (the thing we want the plots to show)

> Learned communication helps **most** in the ambiguous regime
> (`delay-dropout` with non-zero `D`), where heartbeat alone cannot
> distinguish "teammate is gone" from "teammate's status is stale".
> Concretely we expect:

1. `baseline`: all four methods perform similarly (no dropout, no
   delay -> no information advantage from comm).
2. `delay-only`: small advantage for `mappo-heartbeat-plus-comm` over
   `mappo-heartbeat-only` (comm helps fill in for stale heartbeats).
3. `dropout-only`: heartbeat-having methods recover; `mappo-no-comm`
   degrades; comm helps moderately.
4. `delay-dropout` (the **ambiguous** case): the gap between
   `mappo-heartbeat-plus-comm` and the rest is largest. Heartbeat-only
   conflates dropout with stale signal and reallocates poorly; the
   heuristic does the same with a hard threshold; learned comm lets
   the surviving agents announce "I'm still here" through the channel
   even when their heartbeat hasn't published yet.

The figures emitted by `policies.analysis.plot_results` are designed
to make this story falsifiable: if the gap in `delay-dropout` is *not*
larger than in `delay-only` or `dropout-only`, the claim fails.

## Running the study

```bash
# 0. Preview what would be launched (no env starts).
uv run python -m policies.experiments.run_rware_matrix --dry-run

# 1. Real run on the main env, 3 seeds, default methods/regimes/delays.
#    Adjust --updates / --rollout for budget.
uv run python -m policies.experiments.run_rware_matrix \
    --env rware-tiny-4ag-v2 \
    --updates 200 --rollout 256

# 2. Aggregate -> per_run.csv + summary.csv.
uv run python -m policies.analysis.aggregate \
    --log-dir runs/exp_matrix --last-k 5

# 3. Final figures.
uv run python -m policies.analysis.plot_results \
    --in-dir runs/exp_matrix --out-dir runs/exp_matrix/figures

# 4. Demo rollout for the presentation.
uv run python -m policies.demo_rware_dropout \
    --env rware-tiny-4ag-v2 \
    --dropout --dropout-agent 0 --dropout-time 50 \
    --heartbeat --heartbeat-delay 5 \
    --max-steps 200
```

## Limitations honestly noted

* The heuristic baseline does not do real path planning; it only does
  one-step greedy heading and accepts the env's move-cancellation cost
  on illegal moves. It is meant as a non-learning *floor*, not an
  upper bound.
* The mechanism branch is now merged: `policies.wrappers.unified` exposes
  `DropoutConfig`, `HeartbeatConfig`, and the `info["debug_*"]` keys.
  Every CLI in this branch (`policies.train`, `policies.baselines.rware_heuristic`,
  `policies.demo_rware_dropout`, `policies.experiments.run_rware_matrix`)
  constructs those config objects from flat CLI flags before calling the
  wrapper factory.
* When `--heartbeat` is *not* passed, the heuristic disables its staleness
  gate entirely (the wrapper still emits a clipped sentinel age that would
  otherwise look "always stale"). This keeps the heuristic functional in
  both heartbeat-on and heartbeat-off regimes.
* RWARE has no per-agent termination natively, so a "dropped" agent in
  the mechanism branch is implemented at the wrapper layer (force-NOOP
  + `alive=False`). Reward zeroing for a dropped agent is already
  handled correctly by `UnifiedMARLEnv`.
