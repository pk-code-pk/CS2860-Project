## 2-Agent Communication + Headroom Diagnostic (Task 1)

### Goal
Test whether communication helps dropout robustness more clearly in less crowded 2-agent RWARE settings.

### Envs checked
- Worked: `rware-tiny-2ag-easy-v2`, `rware-small-2ag-v2`, `rware-medium-2ag-v2`
- Discovery command:
  - `uv run python -c "import gymnasium as gym; import rware; print('\n'.join(sorted([k for k in gym.registry.keys() if 'rware-' in k and '2ag' in k])))"`

### Seeds
- `0 1 2 3` (minimum requested)

### Launchers used
- Headroom scan:
  - `uv run python -m policies.experiments.run_rware_matrix --env <ENV> --methods mappo-heartbeat-only --regimes delay-only delay-dropout --delays 30 --seeds 0 1 2 3 --updates 300 --rollout 512 --shape-rewards --heartbeat-max-age-clip 128 --dropout-window-start 200 --dropout-window-end 350 --log-dir matrix_results/diagnostics_v1_sam/comm_2ag/headroom/<ENV> --max-parallel 6 --threads-per-cell 1 --production-eval --stop-on-error`
- Main tiny matrix:
  - `uv run python -m policies.experiments.run_rware_matrix --env rware-tiny-2ag-easy-v2 --methods mappo-no-comm mappo-heartbeat-only mappo-heartbeat-plus-comm --regimes baseline dropout-only delay-only delay-dropout --delays 30 --seeds 0 1 2 3 --updates 400 --rollout 512 --shape-rewards --heartbeat-max-age-clip 128 --dropout-window-start 200 --dropout-window-end 350 --log-dir matrix_results/diagnostics_v1_sam/comm_2ag/main_tiny --max-parallel 6 --threads-per-cell 1 --production-eval --stop-on-error`

### Headroom results (heartbeat-only, delay=30)

| env | delay-only eval | delay-dropout eval | dropout - delay-only |
|---|---:|---:|---:|
| rware-tiny-2ag-easy-v2 | 20.53 | 12.63 | -7.90 |
| rware-small-2ag-v2 | 1.77 | 1.54 | -0.23 |
| rware-medium-2ag-v2 | 0.52 | 0.09 | -0.43 |

Interpretation: `rware-tiny-2ag-easy-v2` is the only tested 2-agent env with clear dropout headroom at this budget.

### Main method/regime table (tiny 2-agent)

| method | baseline | dropout-only | delay-only | delay-dropout |
|---|---:|---:|---:|---:|
| mappo-no-comm | 17.31 | 8.26 | n/a | n/a |
| mappo-heartbeat-only | 24.30 | 11.82 | 22.61 | 20.27 |
| mappo-heartbeat-plus-comm | 18.95 | 10.22 | 20.24 | 12.22 |

Derived contrasts:
- `delay-dropout` comm benefit over heartbeat-only: `-8.06`
- `delay-only` comm benefit over heartbeat-only: `-2.37`
- Interaction `(comm benefit under delay-dropout) - (comm benefit under delay-only)`: `-5.69`

### Answers to Task 1 questions
1. Does dropout hurt in 2-agent RWARE?
   - Yes on tiny (`-7.90` under delay+dropout vs delay-only); weakly on small/medium at this budget.
2. Does learned communication improve delay+dropout vs heartbeat-only?
   - No on tiny (`12.22` vs `20.27`).
3. Does communication help more under delay+dropout than delay-only?
   - No; interaction is negative (`-5.69`).
4. Does oracle/fixed death-token help more than learned?
   - Yes (see `../death_token/README.md`), but still below heartbeat-only.
5. If comm helps in 2-agent but not 4-agent, can we argue crowding masked the effect?
   - Not supported by this run set: even in 2-agent tiny, learned comm did not help.

### Short conclusion
- Dropout headroom exists in 2-agent tiny.
- Learned comm still does not rescue dropout there.
- This points more toward message grounding / policy interference than pure 4-agent crowding.
