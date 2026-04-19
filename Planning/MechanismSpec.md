# Ambiguity mechanism spec

Environment-side mechanism that creates the "ambiguous teammate
disappearance" condition the project studies: can learned communication
help a cooperative team tell **true teammate dropout** apart from
**merely stale teammate status signals**?

All of the behaviour below lives in `policies/wrappers/`. The PPO / actor
/ critic / buffer / runner are untouched aside from removing a single
leaked feature (see §4).

## 1. Controlled permanent dropout

Lives in `UnifiedMARLEnv` with `DropoutConfig`.

### Flags

- `--dropout` — master switch (default off)
- `--dropout-agent INT` — fixed-mode agent index
- `--dropout-time INT` — fixed-mode step at which dropout fires
- `--dropout-window-start INT` / `--dropout-window-end INT` —
  window-mode inclusive start / exclusive end for a uniform-random
  dropout step

### Behaviour

- Disabled → wrapper is bit-identical to the pre-mechanism baseline
  (save for the new `info["debug_*"]` keys, which are harmless defaults).
- Enabled → **exactly one** agent fails permanently at step
  `dropout_time` (or at a random step in the window). From that step
  onward:
  - its env action is forced to NOOP (already the existing behaviour
    for dead agents inside `UnifiedMARLEnv.step`);
  - it emits no new learned message tokens;
  - it emits no new heartbeats;
  - its true alive bit is sticky-False;
  - its tensor slot stays in every returned array (shape is stable).

### Config precedence

- Fixed mode (`agent` + `time`) takes precedence when fully specified.
- Window mode requires both endpoints (inclusive start, exclusive end,
  `end > start`).
- Specifying both fixed *and* window raises `ValueError` at config
  construction — we reject ambiguous configs instead of silently picking
  one.
- Window mode is seeded by the `reset(seed=...)` call so runs are
  reproducible per seed.

## 2. Delayed heartbeat system

Lives in `policies/wrappers/heartbeat.py` (`HeartbeatTracker`,
`HeartbeatConfig`). Completely separate from learned communication.

### Flags

- `--heartbeat` — master switch (default off)
- `--heartbeat-period INT` — a heartbeat is emitted every `period`
  steps (default 1)
- `--heartbeat-delay INT` — heartbeats arrive `delay` steps after they
  were produced (default 0)

### Behaviour

- Every alive agent emits a heartbeat each period-matching step.
- A heartbeat produced at step `t` is delivered at step `t + delay` to
  every other agent. Self-freshness is always 0.
- After dropout the failed agent stops producing new heartbeats, but
  **heartbeats that were already in flight when it failed are still
  delivered on schedule** (this is the core ambiguity: the recipient
  briefly sees fresh status from an agent that is already gone).
- After the last in-flight heartbeat from a failed sender is delivered,
  that recipient's freshness for that sender grows monotonically.

### Exposed ages

`info["debug_heartbeat_age"]` is an `int64 (n_agents, n_agents)` array
where `[r, s]` is how many env steps ago recipient `r` last received a
heartbeat from sender `s`. Ages are clipped at `max_age_clip` (default
32) so obs features stay bounded.

## 3. Actor-visible observation contract

Goal: the actor should *not* get oracle teammate death. It should have
to infer it from freshness + learned messages.

Concrete implementation:

- The wrapper appends per-agent freshness features to each agent's obs
  vector (one float per sender, normalised to `[0, 1]`: 1 = just heard,
  0 = stalest). Width = `n_agents` when the heartbeat is enabled, 0
  otherwise. `EnvSpec.obs_dim` reflects the effective width.
- The MAPPO actor's input construction (`build_actor_input` in
  `policies/mappo/networks.py`) no longer concatenates the alive mask.
  The alive mask is still passed through, but only used by
  `sample_actions` / `evaluate_actions` for masking log-probs and by
  the critic (MAPPO's centralized training lets the critic keep oracle
  alive).
- Net effect: actor sees freshness, **not** oracle death. Masking,
  NOOP substitution, reward zeroing, GAE-alive-weighting, all still use
  the true alive mask internally — nothing in the training math changes.

## 4. Stable debug/info contract

Every `reset()` and `step()` adds these to `info`, even when the
mechanism is disabled:

| key | type | meaning |
|---|---|---|
| `debug_true_alive` | `bool (N,)` | ground-truth alive mask |
| `debug_dropout_fired` | `bool` | whether the scheduled dropout has fired in this episode |
| `debug_failed_agent` | `int` | agent index (-1 if none has failed) |
| `debug_heartbeat_age` | `int64 (N, N)` | per-(recipient, sender) age, clipped |

These keys are consumed by the baseline / analysis branch.

## 5. Sanity check

`uv run python -m policies.train --env rware-tiny-2ag-easy-v2 \
    --updates 2 --rollout 64 --no-log --seed 0 \
    --dropout --dropout-agent 1 --dropout-time 50 \
    --heartbeat --heartbeat-period 1 --heartbeat-delay 3`

Expected: training runs, losses finite, obs_dim is base+2, and a
per-step probe shows agent 1's true alive flips at step 50, stays False,
and the `debug_heartbeat_age[0, 1]` stays at 3 for a few steps (echo
effect) and then grows monotonically.

## 6. Assumptions / follow-ups for the RL side

- The PPO trainer keeps using the true alive mask for ratio + entropy
  masking. This is inside the wrapper's contract; no change required.
- The critic still sees true alive (centralized training, standard
  MAPPO). If we ever want a symmetric "critic blind to oracle death"
  condition, we'd swap `build_critic_input` similarly.
- Heartbeat features are appended *after* obs zeroing on death. Dead
  agents' own obs rows are zeroed, but their freshness row is still the
  recipient's perceived ages — which is correct.
- `n_msg_tokens=1` (the `--no-comm` ablation) still works; the message
  tensor is just degenerate.
