# Paper-faithful Hierarchical MARL baseline for RWARE

This module adds a **Cooperative-HRL / COM-Cooperative-HRL** baseline
in the spirit of

> Ghavamzadeh, Mahadevan, Makar.
> "Hierarchical Multi-Agent Reinforcement Learning."

It is an alternative algorithm path, **not** a variant of the existing
flat MAPPO pipeline.  The MAPPO code under `policies/mappo/` is
untouched.

## What you get

```
policies/hierarchical/
  subtasks.py     # cooperative-subtask enum + state abstraction + termination
  low_level.py    # hand-coded primitive-action executor per subtask
  learner.py      # tabular Cooperative SMDP Q-learner (+ --no-cooperative ablation)
  controller.py   # per-team hierarchical controller + shared assignments table
  runner.py       # UnifiedMARLEnv-compatible episode driver
  train.py        # CLI entry point
```

## Paper-to-implementation mapping

| Paper concept | Implementation |
|---|---|
| Two-level task graph (root -> cooperative subtasks -> primitive actions) | `HighSubtask` enum for the cooperative level + `LowLevelExecutor` for primitives |
| Cooperative subtasks | `HANDLE_SLOT_k` (one per request-queue slot, up to `MAX_SLOTS=8`) + `IDLE` + `RECOVER` |
| Initiation / termination predicates | `available_cooperative_subtasks` and `subtask_terminated` |
| SMDP Q-learning at the cooperative level | `CooperativeSMDPQLearner.smdp_update` using `G = Σ γⁱ rᵢ + γᵏ maxQ(s',·)` |
| Cooperative conditioning of Q-values on teammates' chosen subtasks | Q-key includes `teammate_assignments` tuple (see `AbstractState`) |
| Homogeneous cooperative agents | single shared Q-table across all agents |
| COM-Cooperative-HRL (communication at cooperative boundaries) | `--comm` flag; when on, the controller broadcasts each new subtask into a shared assignments table read by all teammates at *their* boundaries |
| Reactive / stale teammate inference (no-comm condition) | without `--comm`, the shared table is only refreshed by an agent when that specific agent itself replans, so every other teammate sees a stale snapshot |
| Cooperative recovery when a teammate is unavailable | `RECOVER` cooperative subtask gated by heartbeat staleness (`--stale-threshold`) |

## Approximations (honest list)

1. **Only a two-level hierarchy.** The paper allows deeper task graphs;
   RWARE does not require that depth, and the paper's own experiments
   run with a shallow tree. Two levels suffice to exercise every
   conceptually interesting component of the algorithm (cooperative
   option selection, SMDP updates, comm-boundary).
2. **Hand-coded low-level executor.** The paper learns Q-values at the
   primitive level too. On RWARE's flattened observation (dim ≈ 70 per
   agent) a tabular primitive learner would be intractable, and a
   function-approximator primitive learner would re-introduce the
   exact MAPPO-style apparatus we are trying to stay away from on this
   branch. We hand-code primitive motion and delegate it to the same
   greedy step-toward used by the reactive heuristic baseline, so the
   *only* thing that differentiates HRL-paper from heuristic in this
   repo is the cooperative *learning* layer.
3. **Tabular Q over a compact abstract state.** We discretize each
   agent's state to `(carrying_code, closest_free_slot, teammate_assignments)`.
   This is smaller than the paper's toy warehouse state space, but
   captures what matters for coordination.
4. **Shared Q across agents.** The paper assumes homogeneous
   cooperative agents; we exploit that explicitly. The ablation
   `--no-cooperative` removes teammate conditioning to give an
   independent-hierarchical baseline, which is a known classic ablation
   of the paper.
5. **Heartbeat-age signal feeds `RECOVER`.** In the paper, cooperative
   recovery is handled by how the Q-function is queried after a
   teammate departure. RWARE has no native "teammate dead" signal in
   obs (see the existing dropout mechanism), so we approximate the
   recovery predicate using the heartbeat-freshness features that the
   unified wrapper already publishes.  When `--heartbeat` is off, ages
   are zero and `RECOVER` never becomes available -- but the rest of
   the HRL stack still runs normally.
6. **Option step-budget cap (`--max-option-steps`).** A hard cap to
   protect us when the env's collision handling causes an option to
   stall indefinitely. The paper has no explicit cap but also no
   cancelable moves.

## Not implemented / explicitly omitted

* **Deep task graphs / deeper hierarchy** (e.g. separate "pickup" and
  "deliver" sub-options under each `HANDLE_SLOT_k`). Would need more
  environment introspection and a larger Q key; the paper's own
  experiments report most cooperative gains at the top level.
* **Semi-Markov PPO at the high level.** The paper is firmly tabular
  SMDP Q-learning; we follow that, and keep PPO confined to the MAPPO
  baseline branch.
* **Primitive-level learning.** See approximation #2.

## CLI cheat sheet

Smoke test (debug env, fast):

```
uv run python -m policies.hierarchical.train \
    --env rware-tiny-2ag-easy-v2 \
    --episodes 30 --max-steps 200 --eval-every 10 --eval-episodes 2 \
    --run-name hrl-smoke-2ag
```

Main env, no-comm variant:

```
uv run python -m policies.hierarchical.train \
    --env rware-tiny-4ag-v2 \
    --episodes 300 --max-steps 500 --eval-every 25 --eval-episodes 5 \
    --run-name hrl-main-4ag
```

Main env, **COM-Cooperative-HRL**:

```
uv run python -m policies.hierarchical.train \
    --env rware-tiny-4ag-v2 \
    --episodes 300 --max-steps 500 --eval-every 25 --eval-episodes 5 \
    --comm --run-name hrl-comm-main-4ag
```

Ablation (no teammate conditioning):

```
uv run python -m policies.hierarchical.train \
    --env rware-tiny-4ag-v2 --episodes 300 --no-cooperative \
    --run-name hrl-indep-main-4ag
```

With the existing ambiguity mechanisms (dropout + heartbeat) for
mechanism-focused comparisons:

```
uv run python -m policies.hierarchical.train \
    --env rware-tiny-4ag-v2 --episodes 300 \
    --heartbeat --heartbeat-delay 3 \
    --dropout --dropout-window-start 60 --dropout-window-end 120 \
    --comm --run-name hrl-comm-main-4ag-mech
```

## How this differs from flat MAPPO

Flat MAPPO chooses one of 5 primitive RWARE actions per agent every
step, on a 70-dim observation vector, using a shared actor-critic.
Coordination emerges implicitly through the team-shared value function
and (optionally) an added communication channel.

The HRL-paper baseline chooses a **cooperative subtask** at boundary
time (every ~20 -- `max_option_steps` primitive steps), using a
tabular Q-table keyed on a *coordinated* state (teammate subtasks
included), and delegates primitive motion to a hand-coded executor.
Coordination is explicit and interpretable (you can read off who is
going to which shelf at any moment). The `--comm` flag controls
whether teammates see up-to-date or stale snapshots of each other's
cooperative commitments -- exactly the toggle the paper calls
"COM-Cooperative HRL" vs. plain "Cooperative HRL".

## Logged metrics

Every episode the run writes one row to `metrics.csv` with:

* `train/ep_return`, `train/ep_length`, `train/ep_deliveries`
* `train/epsilon` (the shared learner's current exploration rate)
* `hrl/q_table_size` (number of distinct abstract-state keys seen)
* `eval/ep_return_mean` etc. on eval episodes

This schema is a strict subset of what MAPPO and heuristic runs log,
so the existing `policies/analysis/` aggregators work without changes.
