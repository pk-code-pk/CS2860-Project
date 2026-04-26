# Randomized Targeted Request-Intent Dropout at t=50

This run checks whether the randomized targeted-dropout result is only an
early-failure artifact. It keeps the same randomized request-intent target rule
as the `t=25` robustness run, but moves dropout from episode step `25` to step
`50`.

## Experiment Setup

- Environment: `rware-medium-2ag-easy-v2`
- Seeds: `0-7` (`n = 8`)
- Methods: `mappo-no-comm`, `mappo-comm`, `mappo-intent-aux`
- Regime: `dropout-only`
- Updates: `1000`
- Rollout length: `512`
- Evaluation: every `25` updates, `30` eval episodes
- Reward shaping: enabled with pickup bonus `0.5`
- Heartbeat: disabled
- Message echo: enabled (`disable_message_echo = false`)
- Dropout time: episode step `50`
- Dropout strategy: `request-intent-random`

## Randomized Dropout Mechanism

At dropout time, the wrapper samples the failed agent from the highest non-empty
request-relevance tier:

1. Live agents carrying currently requested shelves.
2. Otherwise, live agents assigned to request slots by the intent-label
   heuristic.
3. Otherwise, live agents tied for closest distance to a requested shelf.
4. If no requests exist, all live agents.

Sampling uses the reset-seeded dropout RNG, so the randomized target is
reproducible for a fixed training seed. Dead agents echo their last live one-hot
message, so the message channel is not an all-zero death oracle.

## Final Results

Primary metric: mean of the last 5 evaluation returns per seed.

| Method | Mean | SD |
| --- | ---: | ---: |
| `mappo-no-comm` | 1.20 | 1.43 |
| `mappo-comm` | 2.36 | 1.90 |
| `mappo-intent-aux` | 8.34 | 2.32 |

Matched-seed comparisons:

| Comparison | Mean Diff | Paired t-test p | Wilcoxon p |
| --- | ---: | ---: | ---: |
| `mappo-comm - mappo-no-comm` | +1.17 | 0.1406 | 0.1484 |
| `mappo-intent-aux - mappo-no-comm` | +7.14 | 0.00045 | 0.0078 |
| `mappo-intent-aux - mappo-comm` | +5.97 | 0.0046 | 0.0156 |

Final-window message grounding accuracy averaged about `0.62`.

## Paper Interpretation

This is the strongest robustness result so far. It shows the intent-grounded
communication benefit is not limited to early dropout at `t=25`: under later
randomized targeted dropout at `t=50`, intent-grounded communication
significantly beats both no communication and unconstrained learned
communication.

Plain learned communication remains positive on average but is not significant,
which sharpens the paper's main claim: reliable dropout recovery requires
task-grounded intent messages, not just an unconstrained discrete communication
channel.

## Files

- `summary.json`: aggregate statistics, per-run values, and matched-seed tests.
- `aggregate_summary.csv`: method-level means and standard deviations.
- `per_run_summary.csv`: per-seed metrics used for the aggregate.
- `../intent_grounded_v1_targeted_random_t50/rware-medium-2ag-easy-v2/`: raw run
  logs, configs, manifest, metrics, and TensorBoard event files.
