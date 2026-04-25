# overnight_v1 pooled results

This directory pools:

- PK seeds 0-3 from `matrix_results/overnight_v1_pk/`
- Sam seeds 4-7 from `matrix_results/overnight_v1/`

Total raw runs: 136.

## Files

- `per_run_summary.csv`: one row per run with last-50 train return, last-10 eval return, entropy, and HRL Q-table size.
- `aggregate_summary.csv`: grouped means/stddevs by phase/env/method/regime/delay.
- `per_seed_interactions.csv`: per-seed MAPPO communication-rescue interactions on `rware-medium-4ag-v2`.
- `pooled_summary.json`: machine-readable key contrasts.

## Key headline numbers

- Small headroom: dropout effect = +11.77 eval.
- Medium headroom: dropout effect = -1.98 eval.
- Medium MAPPO comm interaction = +2.25 eval.
- Medium normal comm dropout eval = 23.51.
- Medium oracle comm dropout eval = 27.84.
- Medium heartbeat-only dropout eval = 30.31.

## Interpretation

At n=8, small RWARE remains confounded by crowding (dropout improves eval), while medium RWARE reduces the confound but has only a small dropout penalty. MAPPO + RIAL-style communication does not reliably rescue dropout; the eval interaction is small and nonsignificant. Estimated message entropy remains high, and oracle death detection does not outperform heartbeat-only.
