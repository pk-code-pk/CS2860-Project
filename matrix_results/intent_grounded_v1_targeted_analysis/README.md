# Targeted Request-Intent Dropout Results

This result bundle is the clean positive communication result for the paper.
It tests whether learned communication helps when teammate failure removes an
agent that was doing request-relevant work.

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
- Dropout strategy: `request-intent`

## Dropout Mechanism

Unlike fixed or random dropout, targeted request-intent dropout selects the
failed agent at runtime from the current RWARE state. At the dropout step, the
wrapper drops the live agent with the highest request relevance:

1. An agent carrying a currently requested shelf.
2. Otherwise, an agent assigned to a request slot by the intent-label heuristic.
3. Otherwise, the live agent closest to any requested shelf.
4. Fallback: the first live agent.

The failure is permanent. The failed agent acts with `noop`, receives zero
reward, has zeroed observations, and cannot be resurrected by the underlying
environment.

Messages are not a death oracle in this experiment. Dead agents echo their last
live one-hot message, so the receiver can use stale communicated intent but
cannot identify death from an all-zero message row.

## Final Results

Primary metric: mean of the last 5 evaluation returns per seed.

| Method | Mean | SD |
| --- | ---: | ---: |
| `mappo-no-comm` | 0.39 | 0.53 |
| `mappo-comm` | 3.22 | 2.41 |
| `mappo-intent-aux` | 7.30 | 4.93 |

Pairwise comparisons use matched seeds against `mappo-no-comm`.

| Comparison | Mean Diff | Paired t-test p | Wilcoxon p |
| --- | ---: | ---: | ---: |
| `mappo-comm - mappo-no-comm` | +2.82 | 0.0073 | 0.0156 |
| `mappo-intent-aux - mappo-no-comm` | +6.91 | 0.0066 | 0.0078 |

`mappo-intent-aux` also learned the auxiliary message grounding task, with
final-window grounding accuracy averaging about `0.57`.

## Paper Interpretation

This supports the paper claim that learned communication can significantly
improve robustness under teammate dropout when the failure removes
task-relevant intent. It also supports a stronger mechanism claim: intent
grounding improves recovery beyond unconstrained learned communication in this
targeted-failure setting.

The scope should be stated carefully. This does not claim that communication
always helps under arbitrary random dropout. Earlier fixed/random dropout runs
were weaker, which is useful context: communication helps most when failure
creates abandoned-task ambiguity that the surviving agent cannot infer from its
local observation alone.

## Files

- `PAPER_ANALYSIS.md`: paper-ready interpretation with figures, CIs, effect
  sizes, paired tests, per-seed outcomes, learning dynamics, and limitations.
- `summary.json`: aggregate statistics, per-run values, and statistical tests.
- `aggregate_summary.csv`: method-level means and standard deviations.
- `per_run_summary.csv`: per-seed metrics used for the aggregate.
- `paper_stats_aggregate.csv`: enriched aggregate stats with SEM and 95% CIs.
- `paper_stats_comparisons.csv`: matched-seed comparisons, effect sizes, and
  paired/nonparametric tests.
- `paper_per_seed_table.csv`: seed-level table for paper appendix/debugging.
- `paper_analysis_summary.json`: machine-readable version of the paper analysis.
- `figures/`: PNG figures for the paper or presentation.
- `../intent_grounded_v1_targeted/rware-medium-2ag-easy-v2/`: raw run logs,
  configs, manifests, metrics, and TensorBoard event files.

Recommended figures:

1. `figures/targeted_last5_eval_bar.png` for the main result.
2. `figures/targeted_paired_differences.png` for matched-seed gains.
3. `figures/targeted_eval_learning_curves.png` for training dynamics.
4. `figures/targeted_message_grounding_accuracy.png` for the intent auxiliary
   diagnostic.
