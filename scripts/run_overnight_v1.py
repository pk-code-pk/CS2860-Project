#!/usr/bin/env python3
"""
Auto-selecting overnight experiment launcher.

This script is intentionally conservative: it first screens `small` and
`medium` for dropout headroom at d=30, picks the env with the larger eval
dropout penalty, then runs the main MAPPO/oracle/HRL/heuristic suite there.

It is designed to be launched once before sleeping:

    uv run python scripts/run_overnight_v1.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "overnight_v1"
LOG_ROOT = ROOT / "logs"


def _run(cmd: list[str], log) -> None:
    """Run a command, streaming stdout/stderr into the combined launcher log."""
    line = "$ " + " ".join(cmd)
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    p = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    assert p.stdout is not None
    for out in p.stdout:
        print(out, end="", flush=True)
        log.write(out)
        log.flush()
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"command failed with exit {rc}: {' '.join(cmd)}")


def _last_values(path: Path, key: str, n: int) -> list[float]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    vals: list[float] = []
    for row in rows:
        v = row.get(key, "")
        if not v:
            continue
        x = float(v)
        if not math.isnan(x):
            vals.append(x)
    return vals[-n:]


def _run_eval_mean(path: Path) -> float:
    vals = _last_values(path, "eval/ep_return_mean", 10)
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _group_eval_mean(log_dir: Path, method: str, regime: str, delay: int, seeds: list[int]) -> float:
    vals: list[float] = []
    for seed in seeds:
        metrics = (
            log_dir
            / f"{method}__{regime}__d{delay}__s{seed}"
            / "metrics.csv"
        )
        vals.append(_run_eval_mean(metrics))
    return float(sum(vals) / len(vals))


def _select_env(headroom_root: Path, envs: list[str], seeds: list[int]) -> tuple[str, dict]:
    summary: dict[str, dict[str, float]] = {}
    for env in envs:
        log_dir = headroom_root / env
        delay_only = _group_eval_mean(
            log_dir, "mappo-heartbeat-only", "delay-only", 30, seeds
        )
        delay_dropout = _group_eval_mean(
            log_dir, "mappo-heartbeat-only", "delay-dropout", 30, seeds
        )
        penalty = delay_dropout - delay_only
        summary[env] = {
            "delay_only_eval": delay_only,
            "delay_dropout_eval": delay_dropout,
            "dropout_minus_delay_only": penalty,
        }

    # Most negative penalty = dropout hurts most. But keep the choice stable:
    # medium must beat small by at least 5 eval points to displace small.
    selected = envs[0]
    small_penalty = summary[envs[0]]["dropout_minus_delay_only"]
    for env in envs[1:]:
        if summary[env]["dropout_minus_delay_only"] <= small_penalty - 5.0:
            selected = env
            small_penalty = summary[env]["dropout_minus_delay_only"]
    return selected, summary


def _matrix_cmd(
    *,
    env: str,
    methods: list[str],
    regimes: list[str],
    delays: list[int],
    seeds: list[int],
    log_dir: Path,
    updates: int,
    max_parallel: int,
    threads_per_cell: int,
    extra: list[str] | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "policies.experiments.run_rware_matrix",
        "--env",
        env,
        "--methods",
        *methods,
        "--regimes",
        *regimes,
        "--delays",
        *(str(d) for d in delays),
        "--seeds",
        *(str(s) for s in seeds),
        "--updates",
        str(updates),
        "--rollout",
        "512",
        "--shape-rewards",
        "--heartbeat-max-age-clip",
        "128",
        "--dropout-window-start",
        "200",
        "--dropout-window-end",
        "350",
        "--log-dir",
        str(log_dir),
        "--max-parallel",
        str(max_parallel),
        "--threads-per-cell",
        str(threads_per_cell),
        "--production-eval",
    ]
    if extra:
        cmd.extend(extra)
    return cmd


def _run_hrl_grid(env: str, seeds: list[int], log_dir: Path, log, episodes: int, max_parallel: int) -> None:
    """Run HRL cells with simple subprocess batching."""
    cells: list[list[str]] = []
    for comm_flag, method_name in [(False, "hrl"), (True, "hrl-comm")]:
        for regime in ["delay-only", "delay-dropout"]:
            for seed in seeds:
                run_name = f"{method_name}__{regime}__d30__s{seed}"
                cmd = [
                    sys.executable,
                    "-m",
                    "policies.hierarchical.train",
                    "--env",
                    env,
                    "--episodes",
                    str(episodes),
                    "--max-steps",
                    "500",
                    "--eval-every",
                    "50",
                    "--eval-episodes",
                    "10",
                    "--seed",
                    str(seed),
                    "--heartbeat",
                    "--heartbeat-delay",
                    "30",
                    "--heartbeat-max-age-clip",
                    "128",
                    "--epsilon-start",
                    "0.5",
                    "--epsilon-end",
                    "0.05",
                    "--epsilon-decay-episodes",
                    str(max(1, episodes // 2)),
                    "--log-dir",
                    str(log_dir),
                    "--run-name",
                    run_name,
                ]
                if regime == "delay-dropout":
                    cmd.extend([
                        "--dropout",
                        "--dropout-window-start",
                        "200",
                        "--dropout-window-end",
                        "350",
                    ])
                if comm_flag:
                    cmd.append("--comm")
                cells.append(cmd)

    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "env": env,
        "seeds": seeds,
        "episodes": episodes,
        "runs": [{"cmd": cmd, "run_name": cmd[cmd.index("--run-name") + 1]} for cmd in cells],
    }
    (log_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[hrl] launching {len(cells)} cells -> {log_dir}", flush=True)
    log.write(f"[hrl] launching {len(cells)} cells -> {log_dir}\n")
    inflight: list[tuple[list[str], subprocess.Popen, object]] = []
    pending = list(cells)
    env_vars = os.environ.copy()
    env_vars.update({
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "PYTHONUNBUFFERED": "1",
    })
    while pending or inflight:
        while pending and len(inflight) < max_parallel:
            cmd = pending.pop(0)
            run_name = cmd[cmd.index("--run-name") + 1]
            out_dir = log_dir / run_name
            out_dir.mkdir(parents=True, exist_ok=True)
            fh = (out_dir / "stdout.log").open("w")
            fh.write("$ " + " ".join(cmd) + "\n\n")
            fh.flush()
            p = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
                env=env_vars,
            )
            print(f"[hrl start] {run_name} pid={p.pid}", flush=True)
            log.write(f"[hrl start] {run_name} pid={p.pid}\n")
            log.flush()
            inflight.append((cmd, p, fh))
        time.sleep(5)
        still: list[tuple[list[str], subprocess.Popen, object]] = []
        for cmd, p, fh in inflight:
            rc = p.poll()
            if rc is None:
                still.append((cmd, p, fh))
                continue
            fh.close()
            run_name = cmd[cmd.index("--run-name") + 1]
            print(f"[hrl done ] {run_name} exit={rc}", flush=True)
            log.write(f"[hrl done ] {run_name} exit={rc}\n")
            log.flush()
            if rc != 0:
                raise RuntimeError(f"HRL cell failed: {run_name} exit={rc}")
        inflight = still


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", nargs="+", default=["rware-small-4ag-v2", "rware-medium-4ag-v2"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--headroom-updates", type=int, default=1000)
    p.add_argument("--main-updates", type=int, default=1000)
    p.add_argument("--hrl-episodes", type=int, default=600)
    p.add_argument("--max-parallel", type=int, default=4)
    p.add_argument("--threads-per-cell", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"overnight_v1_{time.strftime('%Y%m%d_%H%M%S')}.log"
    with log_path.open("w") as log:
        print(f"[overnight] log={log_path}", flush=True)
        log.write(f"[overnight] log={log_path}\n")

        headroom_root = RUN_ROOT / "headroom"
        for env in args.envs:
            _run(
                _matrix_cmd(
                    env=env,
                    methods=["mappo-heartbeat-only"],
                    regimes=["delay-only", "delay-dropout"],
                    delays=[30],
                    seeds=args.seeds,
                    log_dir=headroom_root / env,
                    updates=args.headroom_updates,
                    max_parallel=args.max_parallel,
                    threads_per_cell=args.threads_per_cell,
                ),
                log,
            )

        selected_env, summary = _select_env(headroom_root, args.envs, args.seeds)
        selected_payload = {
            "selected_env": selected_env,
            "selection_rule": "most negative eval dropout penalty; non-small must beat small by at least 5",
            "headroom": summary,
        }
        (RUN_ROOT / "selected_env.json").write_text(json.dumps(selected_payload, indent=2))
        print(f"[overnight] selected_env={selected_env} summary={summary}", flush=True)
        log.write(f"[overnight] selected_env={selected_env} summary={summary}\n")
        log.flush()

        _run(
            _matrix_cmd(
                env=selected_env,
                methods=["mappo-no-comm"],
                regimes=["baseline", "dropout-only"],
                delays=[0],
                seeds=args.seeds,
                log_dir=RUN_ROOT / "main_mappo_no_mech",
                updates=args.main_updates,
                max_parallel=args.max_parallel,
                threads_per_cell=args.threads_per_cell,
            ),
            log,
        )
        _run(
            _matrix_cmd(
                env=selected_env,
                methods=["mappo-heartbeat-plus-comm"],
                regimes=["delay-only", "delay-dropout"],
                delays=[30],
                seeds=args.seeds,
                log_dir=RUN_ROOT / "main_mappo_comm",
                updates=args.main_updates,
                max_parallel=args.max_parallel,
                threads_per_cell=args.threads_per_cell,
            ),
            log,
        )
        _run(
            _matrix_cmd(
                env=selected_env,
                methods=["mappo-heartbeat-plus-comm"],
                regimes=["delay-dropout"],
                delays=[30],
                seeds=args.seeds,
                log_dir=RUN_ROOT / "oracle_ceiling",
                updates=args.main_updates,
                max_parallel=max(1, min(2, args.max_parallel)),
                threads_per_cell=args.threads_per_cell,
                extra=["--disable-message-echo"],
            ),
            log,
        )
        _run_hrl_grid(
            selected_env,
            args.seeds,
            RUN_ROOT / "hrl",
            log,
            episodes=args.hrl_episodes,
            max_parallel=args.max_parallel,
        )
        _run(
            _matrix_cmd(
                env=selected_env,
                methods=["heuristic"],
                regimes=["baseline", "dropout-only", "delay-only", "delay-dropout"],
                delays=[30],
                seeds=args.seeds,
                log_dir=RUN_ROOT / "heuristic",
                updates=1,
                max_parallel=args.max_parallel,
                threads_per_cell=args.threads_per_cell,
            ),
            log,
        )
        print("[overnight] ALL PHASES COMPLETE", flush=True)
        log.write("[overnight] ALL PHASES COMPLETE\n")


if __name__ == "__main__":
    main()
