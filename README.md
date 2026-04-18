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
| `envs/` | Extra env helpers / samples |
| `runs/` | Run outputs (CSV, TB); listed in `.gitignore` |
| `logs/` | Text logs; listed in `.gitignore` |
| `Planning/` | Notes and task tracking for the course |
