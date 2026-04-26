# Randomized Targeted Request-Intent Dropout Results

This run is the robustness check for deterministic targeted dropout. Instead of
always dropping the single top-ranked request-relevant agent, the wrapper samples
from the highest non-empty request-relevance tier at dropout time.

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
- Dropout time: episode step `25`
- Dropout strategy: `request-intent-random`

## Randomized Dropout Mechanism

At the dropout step, the wrapper checks task-relevance tiers in order:

1. Live agents carrying currently requested shelves.
2. Otherwise, live agents assigned to request slots by the intent-label
   heuristic.
3. Otherwise, live agents tied for closest distance to a requested shelf.
4. If no requests exist, all live agents.

The failed agent is sampled uniformly from the highest non-empty tier using the
wrapper's reset-seeded dropout RNG. This keeps failures task-relevant while
avoiding the criticism that the experiment always kills the single most
important agent. For a fixed training seed, the sampled dropout trajectory is
reproducible.

Dead agents still echo their last live one-hot message, so communication is not
a perfect death oracle.

## Final Results

Primary metric: mean of the last 5 evaluation returns per seed.

| Method | Mean | SD |
| --- | ---: | ---: |
| `mappo-no-comm` | 0.43 | 0.44 |
| `mappo-comm` | 1.72 | 1.95 |
| `mappo-intent-aux` | 5.27 | 2.90 |

Matched-seed comparisons:

| Comparison | Mean Diff | Paired t-test p | Wilcoxon p |
| --- | ---: | ---: | ---: |
| `mappo-comm - mappo-no-comm` | +1.29 | 0.1221 | 0.1094 |
| `mappo-intent-aux - mappo-no-comm` | +4.84 | 0.0021 | 0.0078 |
| `mappo-intent-aux - mappo-comm` | +3.55 | 0.0597 | 0.0781 |

Intent-grounded communication remains significant against no communication under
randomized targeted failures. Plain learned communication remains positive on
average but is no longer significant at `n = 8`, which strengthens the paper's
mechanistic framing: task-grounded messages are more reliable than unconstrained
discrete communication when failures are sampled from request-relevant agents.

Final-window message grounding accuracy averaged about `0.58`.

## Paper Interpretation

This robustness check supports the stronger claim:

> When teammate failures are sampled from task-relevant agents, intent-grounded
> communication significantly improves recovery over no communication.

Together with deterministic targeted dropout, this shows the positive result is
not only an artifact of always removing the single highest-priority agent. The
effect is strongest for grounded intent communication, while unconstrained
learned communication is more fragile under randomized targeted failures.

## Files

- `summary.json`: aggregate statistics, per-run values, and matched-seed tests.
- `aggregate_summary.csv`: method-level means and standard deviations.
- `per_run_summary.csv`: per-seed metrics used for the aggregate.
- `../intent_grounded_v1_targeted_random/rware-medium-2ag-easy-v2/`: raw run
  logs, configs, manifest, metrics, and TensorBoard event files.
