# Intent-Grounded Communication Diagnostic

This branch adds a positive-hypothesis communication test that does **not** use
heartbeat as the main signal.

## Hypothesis

Learned communication helps under agent dropout when the message channel carries
task intent/assignment information that heartbeat cannot provide.

The previous free-form message head was ungrounded: token ids had no fixed
meaning and were trained only through sparse PPO reward. The new
`mappo-intent-aux` method keeps messages learned, but adds an auxiliary
cross-entropy loss that grounds the outgoing token in the sender's RWARE task
intent.

## Intent Labels

For RWARE runs, `debug_message_intent_labels` are generated from the current
warehouse state:

- `0`: available / idle / no assigned request
- `1`: carrying a requested shelf toward delivery
- `2`: carrying a non-requested shelf
- `3+k`: assigned to request-queue slot `k`, clipped to the available token set

Because the wrapper already makes dropped agents echo their last live message,
an intent-grounded sender that drops while assigned to request slot `k` leaves a
meaningful stale signal: "this abandoned work was slot `k`." That is different
from heartbeat, which only gives liveness/freshness.

## Methods

The matrix runner now supports:

- `mappo-no-comm`: no communication channel.
- `mappo-comm`: free learned communication, no heartbeat.
- `mappo-intent-aux`: learned communication with RWARE intent grounding, no
  heartbeat.

## Recommended Run

Start with the smallest positive-hypothesis test:

```bash
uv run python scripts/run_intent_grounded_v1.py \
  --envs rware-tiny-2ag-easy-v2 \
  --seeds 0 1 2 3 \
  --max-parallel 4
```

If valid 2-agent small/medium RWARE ids exist locally, add them:

```bash
uv run python scripts/run_intent_grounded_v1.py \
  --envs rware-tiny-2ag-easy-v2 rware-small-2ag-v2 rware-medium-2ag-v2 \
  --seeds 0 1 2 3 4 5 6 7 \
  --max-parallel 6
```

Outputs go under `matrix_results/intent_grounded_v1/`.

## Interpretation

The clean positive result is:

`mappo-intent-aux` loses less return under `dropout-only` than both
`mappo-no-comm` and `mappo-comm`.

That supports the claim:

> Free-form learned messages did not reliably ground, but intent-grounded
> learned communication can improve robustness by transmitting actionable task
> assignment state rather than only liveness.
