# How to Reproduce Our Results

**Authors:** Praneel Khiantani, Sam Chen  
**Course:** CS 2860, Harvard University, Spring 2026

---

## Setup (one-time)

```bash
# 1. Clone the repo and cd into it
cd CS2860

# 2. Install dependencies (requires uv — https://docs.astral.sh/uv/)
uv sync
```

All commands below use `uv run`. Alternatively: `source .venv/bin/activate` and drop the `uv run` prefix.

**Requirements:** Python 3.10+, CPU only (no GPU needed). Runs on macOS or Linux.

---

## Quick Smoke Test (~2 min)

Verify everything works end-to-end before running full experiments:

```bash
uv run python -m policies.train \
    --env rware-tiny-2ag-v2 --rollout 64 --updates 5 \
    --eval-every 2 --eval-episodes 2 --seed 0
```

You should see training logs with `train/ep_return_mean` and `eval/ep_return_mean`.

---

## Reproducing Paper Results

### Result 1: Comm-MAPPO on tiny-4ag (Table II in paper)

This runs 4 cells x 3 seeds. ~70 min on an 8-core laptop at 3-way parallel.

```bash
uv run python -m policies.experiments.run_rware_matrix \
    --env rware-tiny-4ag-v2 \
    --methods mappo-heartbeat-only mappo-heartbeat-plus-comm \
    --regimes delay-only delay-dropout \
    --delays 30 --seeds 0 1 2 \
    --updates 1000 --rollout 512 \
    --shape-rewards \
    --dropout-window-start 200 --dropout-window-end 350 \
    --log-dir runs/reproduce_tiny \
    --max-parallel 3 --threads-per-cell 1 \
    --production-eval
```

View results:

```bash
uv run python -m policies.analysis.pilot_dashboard \
    --log-dir runs/reproduce_tiny \
    --out runs/reproduce_tiny/dashboard.png
```

### Result 2: GComm-MAPPO with intent grounding (Table V in paper)

This runs 3 methods x 8 seeds. ~3-4 hours total at 3-way parallel.

```bash
# No-communication baseline
for SEED in 0 1 2 3 4 5 6 7; do
  uv run python -m policies.train \
      --env rware-medium-2ag-easy-v2 \
      --no-comm \
      --rollout 512 --updates 1000 \
      --eval-every 25 --eval-episodes 30 \
      --shape-rewards --pickup-bonus 0.5 \
      --dropout --dropout-time 25 --dropout-target-strategy request-intent \
      --seed $SEED \
      --log-dir runs/reproduce_gcomm \
      --run-name mappo-no-comm__dropout-only__d0__s${SEED}
done

# Ungrounded RIAL communication
for SEED in 0 1 2 3 4 5 6 7; do
  uv run python -m policies.train \
      --env rware-medium-2ag-easy-v2 \
      --rollout 512 --updates 1000 \
      --eval-every 25 --eval-episodes 30 \
      --shape-rewards --pickup-bonus 0.5 \
      --dropout --dropout-time 25 --dropout-target-strategy request-intent \
      --seed $SEED \
      --log-dir runs/reproduce_gcomm \
      --run-name mappo-comm__dropout-only__d0__s${SEED}
done

# Intent-grounded (GComm-MAPPO)
for SEED in 0 1 2 3 4 5 6 7; do
  uv run python -m policies.train \
      --env rware-medium-2ag-easy-v2 \
      --message-grounding rware-intent \
      --rollout 512 --updates 1000 \
      --eval-every 25 --eval-episodes 30 \
      --shape-rewards --pickup-bonus 0.5 \
      --dropout --dropout-time 25 --dropout-target-strategy request-intent \
      --seed $SEED \
      --log-dir runs/reproduce_gcomm \
      --run-name mappo-intent-aux__dropout-only__d0__s${SEED}
done
```

View results:

```bash
uv run python -m policies.analysis.aggregate \
    --log-dir runs/reproduce_gcomm --last-k 5
```

### Result 3: Dropout headroom analysis (Table I in paper)

Test dropout impact across environment sizes:

```bash
# For each env, run heartbeat-only under delay-only and delay-dropout
for ENV in rware-tiny-4ag-v2 rware-small-4ag-v2 rware-medium-4ag-v2 rware-tiny-2ag-easy-v2; do
  uv run python -m policies.experiments.run_rware_matrix \
      --env $ENV \
      --methods mappo-heartbeat-only \
      --regimes delay-only delay-dropout \
      --delays 30 --seeds 0 1 2 3 \
      --updates 1000 --rollout 512 \
      --shape-rewards \
      --dropout-window-start 200 --dropout-window-end 350 \
      --log-dir runs/reproduce_headroom_${ENV} \
      --max-parallel 2 --threads-per-cell 1 \
      --production-eval
  done
```

---

## Viewing Pre-Computed Results

All committed results are in `matrix_results/`. To browse without re-running:

```bash
# Comm-MAPPO dashboard (tiny-4ag, n=6 pooled)
open matrix_results/exp_pilot_v4_pooled/dashboard.png

# Entropy analysis (small-4ag)
open matrix_results/smoke_small_v1_pooled/diagnostics/per_seed_dynamics.png

# GComm-MAPPO bar chart
open matrix_results/intent_grounded_v1_targeted_analysis/figures/targeted_last5_eval_bar.png

# GComm-MAPPO per-seed lines
open matrix_results/intent_grounded_v1_targeted_analysis/figures/targeted_paired_seed_lines.png

# Aggregate CSV data
cat matrix_results/intent_grounded_v1_targeted_analysis/aggregate_summary.csv
cat matrix_results/overnight_v1_pooled/aggregate_summary.csv
```

---

## TensorBoard

To view training curves interactively:

```bash
uv run tensorboard --logdir runs
# Then open http://127.0.0.1:6006
```

---

## Key Files

| File | Purpose |
|------|---------|
| `policies/train.py` | Main training entry point |
| `policies/mappo/` | MAPPO algorithm (actor, critic, PPO update) |
| `policies/wrappers/unified.py` | Unified env wrapper with dropout |
| `policies/wrappers/heartbeat.py` | Heartbeat delay tracker |
| `policies/hierarchical/` | HRL baseline (cooperative SMDP Q-learning) |
| `policies/experiments/run_rware_matrix.py` | Experiment matrix runner |
| `policies/analysis/` | Aggregation, dashboards, plotting |
| `matrix_results/` | All committed experiment results |
| `final_paper/` | LaTeX paper (zip to upload) |
