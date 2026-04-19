# CS2860 — multi-agent RL (MAPPO + communication)

Flat MAPPO with RIAL-style messaging on **RWARE** and **MultiGrid** envs, unified via `policies/wrappers`.

## Setup

```bash
cd CLASSES/CS2860   # or your clone path
uv sync
```

Use `uv run …` so commands use the project venv, or `source .venv/bin/activate` and run `python` as usual.

## Train

```bash
uv run python -m policies.train --env rware-tiny-2ag-v2 --updates 50
uv run python -m policies.train --env MultiGrid-Empty-8x8-v0 --n-agents 2 --updates 100 --rollout 512
```

Common flags: `--no-comm` (ablation), `--run-name myrun`, `--log-dir runs`, `--no-log`, `--device cuda`. Full list:

```bash
uv run python -m policies.train --help
```

Each run (unless `--no-log`) writes under `runs/<run_name>/` (default name is time-based): `metrics.csv`, `config.json`, and TensorBoard event files.

## Inspect metrics

```bash
uv run python -m policies.summarize_runs --log-dir runs --tail 3
```

## Experiment matrix (RWARE dropout / heartbeat / comm study)

The branch `feature/rware-baselines-experiments` adds a non-RL heuristic baseline plus an experiment runner, aggregator, plotter, and demo. See `Planning/ExperimentPlan.md` for the full study design.

```bash
# 0. Preview what the matrix would launch (no env starts).
uv run python -m policies.experiments.run_rware_matrix --dry-run

# 1. Real run on the main env. Tune --updates / --rollout for budget.
uv run python -m policies.experiments.run_rware_matrix \
    --env rware-tiny-4ag-v2 --updates 200 --rollout 256

# 2. Aggregate -> per_run.csv + summary.csv.
uv run python -m policies.analysis.aggregate --log-dir runs/exp_matrix --last-k 5

# 3. Final figures (5 PNGs).
uv run python -m policies.analysis.plot_results \
    --in-dir runs/exp_matrix --out-dir runs/exp_matrix/figures

# 4. Demo rollout for the presentation (heuristic by default; pass
#    --checkpoint path/to/ckpt.pt for a trained MAPPO policy).
uv run python -m policies.demo_rware_dropout \
    --env rware-tiny-4ag-v2 \
    --dropout --dropout-agent 0 --dropout-time 50 \
    --heartbeat --heartbeat-delay 5 --max-steps 200

# Heuristic baseline standalone:
uv run python -m policies.baselines.rware_heuristic \
    --env rware-tiny-4ag-v2 --episodes 10 --stale-threshold 4
```

The dropout / heartbeat CLI flags (`--dropout`, `--heartbeat`, `--heartbeat-delay`, ...) are forwarded to the wrapper as `DropoutConfig` / `HeartbeatConfig` objects (see `policies/wrappers/unified.py`). Use `--dry-run` on `policies.experiments.run_rware_matrix` to inspect the planned commands before launching a real sweep.

## TensorBoard

```bash
uv run tensorboard --logdir runs
```

Then open the URL it prints (usually http://127.0.0.1:6006).

## Repo layout (short)

| Path | Role |
|------|------|
| `policies/train.py` | CLI entry: training |
| `policies/mappo/` | MAPPO trainer, networks, buffer |
| `policies/wrappers/` | Adapters → single `UnifiedMARLEnv` API |
| `policies/baselines/` | Non-learning heuristic baseline + CLI |
| `policies/experiments/` | Matrix runner over methods × regimes × delays × seeds |
| `policies/analysis/` | Aggregator + plot script for matrix outputs |
| `policies/demo_rware_dropout.py` | Single-rollout demo for presentations |
| `envs/` | Extra env helpers / samples |
| `runs/` | Run outputs (CSV, TB); listed in `.gitignore` |
| `logs/` | Text logs; listed in `.gitignore` |
| `Planning/` | Notes and task tracking for the course |
