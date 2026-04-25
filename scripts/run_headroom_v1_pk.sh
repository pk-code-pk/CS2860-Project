#!/usr/bin/env bash
# Launches PK's slice of the headroom_v1 + oracle_ceiling_v1 pilots.
#
#   Phase 1A: Delay scan           (12 cells, mappo-heartbeat-only, ~2h   @ --max-parallel 4)
#   Phase 1B: No-mech baselines    ( 4 cells, mappo-no-comm,        ~40m @ --max-parallel 4)
#   Phase 1C: Oracle ceiling       ( 2 cells, mappo-heartbeat-plus-comm + --disable-message-echo, ~40m @ --max-parallel 2)
#
# Total: 18 cells on PK's lab (seeds 0-1). Sam runs the same three invocations
# with --seeds 2 3 on his lab; we pool afterwards for n=4.
#
# On failure of an invocation, the later ones are skipped so we don't waste
# compute on a broken world.

set -euo pipefail

cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/headroom_v1_pk_${TS}.log"

echo "[launcher] writing combined log to ${LOG}"
echo "[launcher] $(date)  Phase 1A (delay scan, 12 cells)" | tee -a "$LOG"

uv run python -m policies.experiments.run_rware_matrix \
  --env rware-small-4ag-v2 \
  --methods mappo-heartbeat-only \
  --regimes delay-only delay-dropout \
  --delays 30 60 100 \
  --seeds 0 1 \
  --updates 1000 --rollout 512 \
  --shape-rewards \
  --dropout-window-start 200 --dropout-window-end 350 \
  --log-dir runs/headroom_v1 \
  --max-parallel 4 --threads-per-cell 2 \
  --production-eval 2>&1 | tee -a "$LOG"

echo "[launcher] $(date)  Phase 1A done. Starting Phase 1B (no-mech baselines, 4 cells)" | tee -a "$LOG"

uv run python -m policies.experiments.run_rware_matrix \
  --env rware-small-4ag-v2 \
  --methods mappo-no-comm \
  --regimes baseline dropout-only \
  --delays 0 \
  --seeds 0 1 \
  --updates 1000 --rollout 512 \
  --shape-rewards \
  --dropout-window-start 200 --dropout-window-end 350 \
  --log-dir runs/headroom_v1 \
  --max-parallel 4 --threads-per-cell 2 \
  --production-eval 2>&1 | tee -a "$LOG"

echo "[launcher] $(date)  Phase 1B done. Starting Phase 1C (oracle ceiling, 2 cells)" | tee -a "$LOG"

uv run python -m policies.experiments.run_rware_matrix \
  --env rware-small-4ag-v2 \
  --methods mappo-heartbeat-plus-comm \
  --regimes delay-dropout \
  --delays 30 \
  --seeds 0 1 \
  --disable-message-echo \
  --updates 1000 --rollout 512 \
  --shape-rewards \
  --dropout-window-start 200 --dropout-window-end 350 \
  --log-dir runs/oracle_ceiling_v1 \
  --max-parallel 2 --threads-per-cell 2 \
  --production-eval 2>&1 | tee -a "$LOG"

echo "[launcher] $(date)  ALL PHASES COMPLETE." | tee -a "$LOG"
