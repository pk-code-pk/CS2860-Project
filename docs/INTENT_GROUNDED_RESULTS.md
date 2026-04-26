# Intent-Grounded Communication Results

This note summarizes the current strongest result set on
`rware-medium-2ag-easy-v2`.

## Experiment Suites

- `matrix_results/intent_grounded_v1_main/`: 8 seeds, 1000 updates, baseline
  and dropout at `t=100`.
- `matrix_results/intent_grounded_v1_stress/`: 8 seeds, 1000 updates, dropout
  stress at `t=50`.
- `matrix_results/intent_grounded_v1_noecho/`: 8 seeds, 1000 updates, dropout
  at `t=100` with dead-agent message echo disabled.
- `matrix_results/intent_grounded_v1_analysis/`: aggregate CSV/JSON summaries.

All three overnight suites completed without failed cells.

## Main Finding

The safest claim is:

> Intent-grounding significantly improves learned communication under harsher
> agent dropout, but current evidence does not prove that communication
> universally beats no communication.

Under early dropout (`t=50`), intent-grounded communication strongly outperforms
free-form learned communication:

| Method | Eval Last-5 Mean |
| --- | ---: |
| Free learned comm | 8.13 |
| Intent-grounded comm | 17.87 |

Paired over seeds 0-7, intent-grounded comm beats free comm by `+9.74`
eval return with `p = 0.0009`.

The comparison against no communication is positive but not statistically
significant:

| Method | Eval Last-5 Mean |
| --- | ---: |
| No communication | 10.43 |
| Intent-grounded comm | 17.87 |

Paired over seeds 0-7, intent-grounded comm beats no communication by `+7.43`
eval return with `p = 0.129`.

## Normal Dropout

At normal dropout timing (`t=100`), intent-grounded comm trends above free comm
but is not conventionally significant on evaluation:

| Method | Dropout Eval Last-5 Mean |
| --- | ---: |
| No communication | 20.00 |
| Free learned comm | 11.69 |
| Intent-grounded comm | 18.38 |

Intent-grounded comm vs free comm at `t=100`: `p = 0.064` on last-5 eval.

## Message Grounding

The auxiliary grounding objective learns non-random task semantics:

- `t=100` dropout: message grounding accuracy `0.572`.
- `t=50` dropout: message grounding accuracy `0.588`.
- no-echo dropout: message grounding accuracy `0.534`.

## No-Echo Ablation

Disabling dead-agent message echo improved intent-grounded comm:

| Method | Echo Eval Last-5 | No-Echo Eval Last-5 |
| --- | ---: | ---: |
| Free learned comm | 11.69 | 11.18 |
| Intent-grounded comm | 18.38 | 25.47 |

This suggests stale echoed intent is not the driver of the result. The more
defensible interpretation is that the auxiliary grounding objective stabilizes
the learned communication representation.

## Current Paper Framing

The current paper should not claim that communication always beats no
communication. It should claim:

> Free-form learned communication is unreliable under agent dropout. Grounding
> messages in task intent produces interpretable messages and significantly
> improves robustness over ungrounded learned communication under severe
> dropout.

The remaining open question is whether a harsher or more targeted dropout
condition can also establish a statistically significant advantage over no
communication. The next planned test is dropout at `t=25`.
