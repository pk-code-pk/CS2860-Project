## Diagnostics v1 (Sam)

This directory contains targeted diagnostics for communication headroom, message grounding, and HRL failure mode isolation.

### 1) What we tested
- Task 1: 2-agent communication + headroom diagnostic
  - `comm_2ag/README.md`
- Task 2: grounded communication / death-token diagnostic
  - `death_token/README.md`
- Task 3: HRL oracle-high-level salvage diagnostic
  - `hrl_diagnostics_README.md`

### 2) Envs / methods / seeds
- Envs tested:
  - `rware-tiny-2ag-easy-v2`
  - `rware-small-2ag-v2`
  - `rware-medium-2ag-v2`
- Seeds:
  - `0 1 2 3`
- Main methods:
  - `mappo-no-comm`
  - `mappo-heartbeat-only`
  - `mappo-heartbeat-plus-comm`
  - oracle/death-token via `--disable-message-echo`
  - HRL: `hrl`, `hrl-comm`, `hrl-oracle-high`

### 3) Main aggregate table

| question slice | key result |
|---|---:|
| 2-agent tiny headroom (`delay-dropout - delay-only`, heartbeat-only) | `-7.90` |
| 2-agent small headroom | `-0.23` |
| 2-agent medium headroom | `-0.43` |
| tiny learned-comm benefit vs heartbeat-only (`delay-dropout`) | `-8.06` |
| tiny learned-comm benefit vs heartbeat-only (`delay-only`) | `-2.37` |
| tiny interaction (dropout comm benefit minus delay-only comm benefit) | `-5.69` |
| tiny oracle-minus-learned (`delay-dropout`) | `+2.55` |
| tiny oracle-minus-heartbeat-only (`delay-dropout`) | `-1.75` |
| HRL oracle-high vs HRL (`delay-dropout`) | `0.15 vs 0.03` |

### 4) Answers to required questions
- Does dropout hurt more in 2-agent?
  - Yes in tiny (clear negative effect), but not strongly in small/medium at this pilot budget.
- Does learned comm help in any easier/headroom setting?
  - No in tested 2-agent tiny headroom setting.
- Does oracle/fixed death-token help?
  - Yes relative to learned comm, but still not above heartbeat-only.
- Is HRL limited by low-level control or high-level abstraction?
  - Both. Oracle high-level helps somewhat, but all absolute returns remain near zero, indicating low-level executability is still a major bottleneck.

### 5) Best paper claim supported after diagnostics
Communication channels can carry useful dropout information when explicitly grounded (oracle/death-token > learned comm), but policy-gradient learned discrete messages do not ground reliably here; overall dropout robustness in RWARE is still dominated by policy/environment dynamics and action-level execution limits.

### Machine-readable summary
- `summary.json`
