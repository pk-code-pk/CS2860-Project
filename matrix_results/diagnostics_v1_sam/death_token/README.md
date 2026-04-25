## Grounded Communication / Death-Token Diagnostic (Task 2)

### Goal
Separate:
- "communication channel is useless"
- from "learned messages are not grounded"

### Setup
- Env: `rware-tiny-2ag-easy-v2`
- Seeds: `0 1 2 3`
- Regimes: `delay-only`, `delay-dropout`
- Delay: `30`
- Heartbeat max-age clip: `128`

### Implementation detail
- Reused existing oracle mode via `--disable-message-echo`.
- In this mode, dead agents emit all-zero messages (dropout oracle signal through comm channel).

### Commands
- Normal comparison:
  - `uv run python -m policies.experiments.run_rware_matrix --env rware-tiny-2ag-easy-v2 --methods mappo-heartbeat-only mappo-heartbeat-plus-comm --regimes delay-only delay-dropout --delays 30 --seeds 0 1 2 3 --updates 300 --rollout 512 --shape-rewards --heartbeat-max-age-clip 128 --dropout-window-start 200 --dropout-window-end 350 --log-dir matrix_results/diagnostics_v1_sam/death_token/tiny_normal --max-parallel 6 --threads-per-cell 1 --production-eval --stop-on-error`
- Oracle/death-token:
  - `uv run python -m policies.experiments.run_rware_matrix --env rware-tiny-2ag-easy-v2 --methods mappo-heartbeat-plus-comm --regimes delay-only delay-dropout --delays 30 --seeds 0 1 2 3 --updates 300 --rollout 512 --shape-rewards --heartbeat-max-age-clip 128 --dropout-window-start 200 --dropout-window-end 350 --disable-message-echo --log-dir matrix_results/diagnostics_v1_sam/death_token/tiny_oracle --max-parallel 6 --threads-per-cell 1 --production-eval --stop-on-error`

### Aggregate results

| condition | delay-only eval | delay-dropout eval |
|---|---:|---:|
| heartbeat-only | 20.53 | 12.63 |
| heartbeat + learned comm | 18.16 | 8.33 |
| heartbeat + oracle/death-token | 17.32 | 10.88 |

Key deltas (`delay-dropout`):
- Oracle minus learned comm: `+2.55`
- Oracle minus heartbeat-only: `-1.75`

### Answers
1. Does a grounded death signal improve performance?
   - Yes vs learned comm (`+2.55` under delay+dropout).
2. Is learned comm worse than fixed/oracle comm?
   - Yes on delay+dropout.
3. Does death information alone explain failure?
   - No. Oracle improves over learned comm but still does not beat heartbeat-only.

### Conclusion
- Grounding helps relative to learned discrete comm, but the main bottleneck is not only death detection.
- Likely failure mode: communication-learning interference and/or weak adaptation policy under dropout.
