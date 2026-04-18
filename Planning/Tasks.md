# Tasks

## Done

### Phase 1 – Environment wrapper + shared infrastructure
- `policies/wrappers/unified.py`: `UnifiedMARLEnv`, `BaseAdapter` protocol,
  `EnvSpec` dataclass, `make_unified_env` factory.
- `policies/wrappers/rware_adapter.py`: wraps `rware-*` FLATTENED envs,
  forces `msg_bits=0`, exposes uniform-ones availability mask.
- `policies/wrappers/multigrid_adapter.py`: wraps `MultiGrid-*` envs,
  flattens `(image, direction)`, drops `mission`, produces per-agent alive
  bits when agents terminate independently.
- Both adapters return a stacked tuple of `(obs, available_actions,
  alive_mask)` with a shared discrete action space split into env actions
  and a separate message-token output.
- Wrapper handles alive masking end-to-end: zeros dead agents' obs,
  suppresses their messages, substitutes NOOPs for their env actions, and
  zeros their rewards starting from the step they enter a dead state.

### Phase 2 – Flat MAPPO + communication baseline
- `policies/mappo/networks.py`: parameter-shared `CommActor` with two
  softmax heads (env + message), parameter-shared `CentralCritic` over
  joint obs+alive+messages, action-availability masking, agent-id one-hot.
- `policies/mappo/buffer.py`: `RolloutBuffer` with alive-aware GAE for a
  shared central value function.
- `policies/mappo/mappo.py`: `MAPPOTrainer` with clipped PPO surrogate,
  value clipping, entropy bonus, grad clipping, minibatch SGD across
  multiple epochs.
- `policies/mappo/runner.py`: `Runner.collect` for rollouts and
  `Runner.evaluate` for greedy policy rollouts.
- `policies/train.py`: CLI driver that works on both `rware-*` and
  `MultiGrid-*` envs.
- Smoke tests: both envs train without errors; sensible loss/entropy/kl
  values.

## Next up

### Phase 3 – Longer training + logging
- Wire TensorBoard / CSV logging (episode return, entropy, KL, value loss,
  wall-clock SPS) into `train.py`.
- Run full-length experiments on `rware-tiny-2ag-v2`, `rware-small-4ag-v2`
  and on `MultiGrid-Empty-Random-8x8-v0` with 2+ agents to get a baseline
  learning curve for the flat MAPPO + comms model.
- Add a no-communication ablation (`n_msg_tokens=1`) to verify the
  communication channel actually helps.

### Phase 4 – Hierarchical extension (future)
- Introduce an option / manager layer that consumes the same communication
  channel but outputs a higher-level macro action, then compare against
  this flat baseline.

### Known small TODOs
- `RwareAdapter.step` returns `alive = all-True` even when `done` fires;
  this is intentional (rware has no per-agent death), but double-check the
  GAE bootstrap once we do multi-env vectorized rollouts.
- `available_actions` is all-ones in both envs; if we ever add partial
  action masking (e.g. multigrid's "pickup" when no object is in front),
  plumb it through the adapter's `_available_actions`.
- `Runner` is single-env; for real training throughput we should add a
  vectorized runner or multiple envs.
