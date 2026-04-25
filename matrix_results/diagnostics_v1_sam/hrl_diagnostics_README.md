## HRL Salvage Diagnostic (Task 3)

### What was implemented
- Added `--oracle-high-level` to `policies.hierarchical.train`.
- Added controller support (`ControllerConfig.oracle_high_level`) in `policies.hierarchical.controller`.
- Oracle mode keeps the existing `LowLevelExecutor` unchanged and uses a fixed high-level selector:
  - Carrying requested shelf -> corresponding `HANDLE_SLOT_k`
  - Carrying non-requested shelf -> `RECOVER` if legal, else `IDLE`
  - Not carrying -> nearest unclaimed legal requested slot
  - Fallback -> `RECOVER` or `IDLE`

### Envs run
- `rware-tiny-2ag-easy-v2` only (Step 1 diagnostic)

### Seeds / regimes
- Seeds: `0 1 2 3`
- Regimes: `delay-only`, `delay-dropout`
- Delay: `30`

### Launcher
- Batched parallel launcher over:
  - `hrl`
  - `hrl-comm`
  - `hrl-oracle-high` (`--oracle-high-level`)
- Each run:
  - `--episodes 400 --max-steps 500 --eval-every 50 --eval-episodes 10`
  - heartbeat enabled, and dropout window enabled in `delay-dropout`.

### Aggregate table (eval mean over final eval points)

| method | delay-only | delay-dropout |
|---|---:|---:|
| hrl | 0.10 | 0.03 |
| hrl-comm | 0.08 | 0.12 |
| hrl-oracle-high | 0.15 | 0.15 |

### Bottleneck interpretation
- Oracle-high-level improves over learned HRL variants, but all values remain extremely low.
- This suggests the high-level learner is part of the problem, but low-level executability is still a dominant bottleneck.
- Hierarchy is not yet viable in this setup without stronger low-level primitives and likely better state/action abstraction.
