# Important

Quick reference for how this repo is wired. Keep this short.

## Layout

```
policies/
  wrappers/
    unified.py            UnifiedMARLEnv + BaseAdapter protocol + EnvSpec
    rware_adapter.py      RwareAdapter (tuple API, FLATTENED obs, msg_bits=0)
    multigrid_adapter.py  MultigridAdapter (dict API, image+dir flattened)
  mappo/
    networks.py           CommActor (two-head: env + msg), CentralCritic
    buffer.py             RolloutBuffer + GAE with alive masking
    mappo.py              MAPPOTrainer (PPO update, grad clip, value clip)
    runner.py             Runner.collect / Runner.evaluate
  train.py                CLI entry: `python -m policies.train --env ...`
envs/
  sample_envs.py          Pre-existing random-policy demo (untouched)
```

## Runtime deps

- `gymnasium`, `multigrid`, `rware` for the envs.
- `torch` for the MAPPO networks and optimizers.
- `tensorboard` for `RunLogger`'s TB event files (CSV works without it).

## Unified interface (the contract)

Reset returns a dict:
- `obs` float32 (N, obs_dim), zeroed for dead agents
- `available_actions` uint8 (N, n_env_actions)
- `alive` bool (N,)
- `messages` float32 (N, n_msg_tokens); one-hot of last-step token, zeros on reset
- `info` dict

Step input: int64 (N, 2) array, columns = [env_action, msg_token]. Dead
agents' rows are overwritten with `(noop_action, 0)` inside `UnifiedMARLEnv`.

Step returns the same dict plus `reward` (float32, N) and `done` (bool).
Per-agent reward is zeroed for agents already dead at the *start* of the
step. Once an agent is dead, it stays dead for the rest of the episode.

## Communication channel

RIAL-style: actor has two independent softmax heads – `env_head`
(n_env_actions) and `msg_head` (n_msg_tokens). Sampling is independent;
log-probs are added together so PPO sees a joint policy. Messages are
emitted at step t and consumed at step t+1 (the wrapper buffers
`last_messages`).

Dead agents emit no message (their row in `messages[t+1]` is all zeros).
RWARE's own `msg_bits` is forced to 0 so its built-in comms never mixes
with ours.

## Alive masking

Carried consistently through:
- obs zeroing in `UnifiedMARLEnv._apply_death_mask_to_obs`
- env-action substitution (NOOP) and msg-token suppression (0) in `step`
- reward zeroing for agents already dead at t
- actor log-prob and entropy multiplied by alive in `evaluate_actions`
- GAE in `RolloutBuffer.compute_advantages` aggregates only alive agents
  into the team reward driving the shared central critic
- PPO ratio is computed on team-level joint log-probs (sum over alive
  agents only – dead agents contribute 0 via the zeroed log-probs)

## Parameter sharing

Single `CommActor` and single `CentralCritic` for all agents. Each
per-agent actor input includes an agent-id one-hot so the shared net can
still specialize per slot.

## Central critic input

Concatenation of all agents' obs + alive bits + the full message matrix.
Outputs one scalar per joint state (shared value baseline, classic MAPPO).

## Env gotchas

- `rware-*` is registered with FLATTENED obs and `msg_bits=0`; shape is
  `(flatdim,)` per agent, no per-agent termination (only global `done`).
- `MultiGrid-*` obs is `{image (V,V,3), direction, mission}`; we flatten
  image (scaled to [0,1]) + one-hot direction and drop the mission string.
  Supports per-agent termination when `success_termination_mode='all'`.
- Multigrid truncation is global (`step_count >= max_steps`) – we report it
  as `done=True` for the whole episode.
