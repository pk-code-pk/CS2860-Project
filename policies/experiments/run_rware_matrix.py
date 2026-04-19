"""
Matrix runner for the RWARE dropout / heartbeat / comm experiments.

Sweeps the cross-product of:

    methods x regimes x heartbeat-delays x seeds

and dispatches each combination as a subprocess invocation of either
``policies.train`` (for the MAPPO variants) or
``policies.baselines.rware_heuristic`` (for the heuristic baseline).

Run names follow a fixed schema:

    {method}__{regime}__d{delay}__s{seed}

so the analysis/aggregation script can recover the (method, regime, delay,
seed) tuple just by parsing the directory name. All runs land under one
log root (default: ``runs/exp_matrix/``).

Mechanism CLI flags forwarded to ``policies.train`` /
``policies.baselines.rware_heuristic`` (both consume them via the
``DropoutConfig`` / ``HeartbeatConfig`` constructors in
``policies.wrappers``):

    --heartbeat / --heartbeat-period N / --heartbeat-delay D
    --dropout / --dropout-agent A / --dropout-time T
    --dropout-window-start S / --dropout-window-end E

Use ``--dry-run`` to preview the planned subprocess commands without
launching them.

Usage examples
--------------

Preview the full default matrix (no env launches):

    uv run python -m policies.experiments.run_rware_matrix --dry-run

Real run on the main env, 3 seeds, default methods/regimes/delays:

    uv run python -m policies.experiments.run_rware_matrix \\
        --env rware-tiny-4ag-v2 --updates 200 --rollout 256

Restrict to a subset:

    uv run python -m policies.experiments.run_rware_matrix \\
        --methods heuristic mappo-heartbeat-plus-comm \\
        --regimes baseline dropout-only \\
        --delays 0 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Method / regime catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodSpec:
    """Static description of a method (turns into CLI flags)."""

    name: str
    kind: str  # "mappo" or "heuristic"
    heartbeat: bool  # expose heartbeat signal in obs (wrapper-side)
    comm: bool  # learn message tokens (mappo-only flag)


METHODS: dict[str, MethodSpec] = {
    "heuristic": MethodSpec(
        name="heuristic", kind="heuristic", heartbeat=True, comm=False
    ),
    "mappo-no-comm": MethodSpec(
        name="mappo-no-comm", kind="mappo", heartbeat=False, comm=False
    ),
    "mappo-heartbeat-only": MethodSpec(
        name="mappo-heartbeat-only", kind="mappo", heartbeat=True, comm=False
    ),
    "mappo-heartbeat-plus-comm": MethodSpec(
        name="mappo-heartbeat-plus-comm", kind="mappo", heartbeat=True, comm=True
    ),
}


@dataclass(frozen=True)
class RegimeSpec:
    name: str
    has_dropout: bool
    has_delay: bool  # True iff non-zero delays make sense for this regime


REGIMES: dict[str, RegimeSpec] = {
    "baseline":      RegimeSpec("baseline",      has_dropout=False, has_delay=False),
    "delay-only":    RegimeSpec("delay-only",    has_dropout=False, has_delay=True),
    "dropout-only":  RegimeSpec("dropout-only",  has_dropout=True,  has_delay=False),
    "delay-dropout": RegimeSpec("delay-dropout", has_dropout=True,  has_delay=True),
}


@dataclass
class RunPlan:
    """One concrete (method, regime, delay, seed) entry in the matrix."""

    method: MethodSpec
    regime: RegimeSpec
    delay: int
    seed: int
    run_name: str
    cmd: list[str]
    env_kwargs_used: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CLI command construction
# ---------------------------------------------------------------------------


def _build_train_cmd(
    *,
    python: str,
    env: str,
    method: MethodSpec,
    regime: RegimeSpec,
    delay: int,
    seed: int,
    log_dir: str,
    run_name: str,
    updates: int,
    rollout: int,
    n_msg_tokens: int,
    eval_every: int,
    eval_episodes: int,
    save_dir: str | None,
    dropout_agent: int,
    dropout_time: int,
    heartbeat_period: int,
    shape_rewards: bool,
    pickup_bonus: float,
    step_penalty: float,
) -> tuple[list[str], dict[str, Any]]:
    cmd: list[str] = [
        python,
        "-m",
        "policies.train",
        "--env",
        env,
        "--seed",
        str(seed),
        "--updates",
        str(updates),
        "--rollout",
        str(rollout),
        "--eval-every",
        str(eval_every),
        "--eval-episodes",
        str(eval_episodes),
        "--log-dir",
        log_dir,
        "--run-name",
        run_name,
        "--n-msg-tokens",
        str(n_msg_tokens),
    ]
    if not method.comm:
        cmd.append("--no-comm")

    used: dict[str, Any] = {}
    if method.heartbeat:
        cmd.append("--heartbeat")
        cmd.extend(["--heartbeat-period", str(heartbeat_period)])
        used["heartbeat"] = True
        used["heartbeat_period"] = heartbeat_period
        if regime.has_delay and delay > 0:
            cmd.extend(["--heartbeat-delay", str(delay)])
            used["heartbeat_delay"] = delay

    if regime.has_dropout:
        cmd.append("--dropout")
        cmd.extend(["--dropout-agent", str(dropout_agent)])
        cmd.extend(["--dropout-time", str(dropout_time)])
        used["dropout"] = True
        used["dropout_agent"] = dropout_agent
        used["dropout_time"] = dropout_time

    if shape_rewards:
        cmd.append("--shape-rewards")
        cmd.extend(["--pickup-bonus", str(pickup_bonus)])
        if step_penalty != 0.0:
            cmd.extend(["--step-penalty", str(step_penalty)])
        used["shape_rewards"] = True
        used["pickup_bonus"] = pickup_bonus
        used["step_penalty"] = step_penalty

    if save_dir is not None:
        save_path = str(Path(save_dir) / f"{run_name}.pt")
        cmd.extend(["--save", save_path])

    return cmd, used


def _build_heuristic_cmd(
    *,
    python: str,
    env: str,
    method: MethodSpec,
    regime: RegimeSpec,
    delay: int,
    seed: int,
    log_dir: str,
    run_name: str,
    episodes: int,
    max_steps: int,
    stale_threshold: int,
    dropout_agent: int,
    dropout_time: int,
    heartbeat_period: int,
) -> tuple[list[str], dict[str, Any]]:
    cmd: list[str] = [
        python,
        "-m",
        "policies.baselines.rware_heuristic",
        "--env",
        env,
        "--seed",
        str(seed),
        "--episodes",
        str(episodes),
        "--max-steps",
        str(max_steps),
        "--stale-threshold",
        str(stale_threshold),
        "--log-dir",
        log_dir,
        "--run-name",
        run_name,
    ]
    used: dict[str, Any] = {}
    # Heuristic always wants the heartbeat signal so it can detect
    # presumed-dropped teammates.
    if method.heartbeat:
        cmd.append("--heartbeat")
        cmd.extend(["--heartbeat-period", str(heartbeat_period)])
        used["heartbeat"] = True
        used["heartbeat_period"] = heartbeat_period
        if regime.has_delay and delay > 0:
            cmd.extend(["--heartbeat-delay", str(delay)])
            used["heartbeat_delay"] = delay
    if regime.has_dropout:
        cmd.append("--dropout")
        cmd.extend(["--dropout-agent", str(dropout_agent)])
        cmd.extend(["--dropout-time", str(dropout_time)])
        used["dropout"] = True
        used["dropout_agent"] = dropout_agent
        used["dropout_time"] = dropout_time
    return cmd, used


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------


def _delays_for(regime: RegimeSpec, delays: list[int]) -> list[int]:
    """
    Return the delay values to sweep for a given regime.

    * Regimes without a delay axis (baseline, dropout-only) always use delay=0.
    * Regimes with a delay axis (delay-only, delay-dropout) use the user-
      supplied positive delays. We drop zeros there because delay=0 collapses
      onto the corresponding no-delay regime.
    """
    if not regime.has_delay:
        return [0]
    pos = sorted({int(d) for d in delays if int(d) > 0})
    return pos or [max(int(d) for d in delays) if delays else 1]


def _is_meaningful(method: MethodSpec, regime: RegimeSpec) -> bool:
    """
    Skip cells that are degenerate by construction.

    * mappo-no-comm has no heartbeat in obs, so any *delay* sweep is a
      duplicate of the same regime at delay=0. We still keep dropout-only
      and baseline for it (they are the "blind" controls).
    """
    if method.heartbeat:
        return True
    # method.heartbeat == False
    return not regime.has_delay


def build_plans(
    *,
    methods: list[MethodSpec],
    regimes: list[RegimeSpec],
    delays: list[int],
    seeds: list[int],
    args: argparse.Namespace,
) -> list[RunPlan]:
    plans: list[RunPlan] = []
    python = sys.executable
    for method in methods:
        for regime in regimes:
            if not _is_meaningful(method, regime):
                continue
            for delay in _delays_for(regime, delays):
                for seed in seeds:
                    run_name = (
                        f"{method.name}__{regime.name}__d{delay}__s{seed}"
                    )
                    if method.kind == "mappo":
                        cmd, used = _build_train_cmd(
                            python=python,
                            env=args.env,
                            method=method,
                            regime=regime,
                            delay=delay,
                            seed=seed,
                            log_dir=args.log_dir,
                            run_name=run_name,
                            updates=args.updates,
                            rollout=args.rollout,
                            n_msg_tokens=args.n_msg_tokens,
                            eval_every=args.eval_every,
                            eval_episodes=args.eval_episodes,
                            save_dir=args.save_dir,
                            dropout_agent=args.dropout_agent,
                            dropout_time=args.dropout_time,
                            heartbeat_period=args.heartbeat_period,
                            shape_rewards=args.shape_rewards,
                            pickup_bonus=args.pickup_bonus,
                            step_penalty=args.step_penalty,
                        )
                    elif method.kind == "heuristic":
                        cmd, used = _build_heuristic_cmd(
                            python=python,
                            env=args.env,
                            method=method,
                            regime=regime,
                            delay=delay,
                            seed=seed,
                            log_dir=args.log_dir,
                            run_name=run_name,
                            episodes=args.heuristic_episodes,
                            max_steps=args.heuristic_max_steps,
                            stale_threshold=args.stale_threshold,
                            dropout_agent=args.dropout_agent,
                            dropout_time=args.dropout_time,
                            heartbeat_period=args.heartbeat_period,
                        )
                    else:
                        raise ValueError(f"unknown method.kind {method.kind!r}")

                    plans.append(
                        RunPlan(
                            method=method,
                            regime=regime,
                            delay=delay,
                            seed=seed,
                            run_name=run_name,
                            cmd=cmd,
                            env_kwargs_used=used,
                        )
                    )
    return plans


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _write_manifest(log_dir: Path, plans: list[RunPlan], args: argparse.Namespace) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = log_dir / "manifest.json"
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args": {k: v for k, v in vars(args).items()},
        "runs": [
            {
                "run_name": p.run_name,
                "method": p.method.name,
                "regime": p.regime.name,
                "delay": p.delay,
                "seed": p.seed,
                "kind": p.method.kind,
                "cmd": p.cmd,
                "env_kwargs": p.env_kwargs_used,
            }
            for p in plans
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def _run_one(plan: RunPlan, log_dir: Path, *, capture: bool) -> int:
    """Run a single plan. Returns the subprocess exit code."""
    out_path = log_dir / plan.run_name
    out_path.mkdir(parents=True, exist_ok=True)
    stdout_path = out_path / "stdout.log"

    print(f"\n[run] {plan.run_name}")
    print(f"      cmd: {' '.join(plan.cmd)}")

    if capture:
        with stdout_path.open("w") as f:
            f.write("$ " + " ".join(plan.cmd) + "\n\n")
            f.flush()
            proc = subprocess.run(
                plan.cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )
    else:
        proc = subprocess.run(plan.cmd, check=False)

    rc = int(proc.returncode)
    print(f"      exit={rc}")
    return rc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the RWARE dropout/heartbeat/comm experiment matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--env", default="rware-tiny-4ag-v2",
                   help="RWARE env id (use rware-tiny-2ag-easy-v2 for debugging).")
    p.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS.keys()),
        choices=list(METHODS.keys()),
    )
    p.add_argument(
        "--regimes",
        nargs="+",
        default=list(REGIMES.keys()),
        choices=list(REGIMES.keys()),
    )
    p.add_argument("--delays", nargs="+", type=int, default=[0, 2, 5],
                   help="Heartbeat delays to sweep in delay-bearing regimes.")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])

    # MAPPO knobs
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--rollout", type=int, default=256)
    p.add_argument("--n-msg-tokens", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--save-dir", default=None,
                   help="If set, MAPPO checkpoints are saved here.")

    # Heuristic knobs
    p.add_argument("--heuristic-episodes", type=int, default=20)
    p.add_argument("--heuristic-max-steps", type=int, default=500)
    p.add_argument("--stale-threshold", type=int, default=4)

    # Heartbeat / dropout knobs (forwarded to wrapper)
    p.add_argument("--heartbeat-period", type=int, default=1)
    p.add_argument("--dropout-agent", type=int, default=0)
    p.add_argument("--dropout-time", type=int, default=100)

    # Reward shaping (forwarded to MAPPO trainings only; heuristic doesn't learn).
    p.add_argument("--shape-rewards", action="store_true",
                   help="Forward --shape-rewards to every MAPPO cell. Strongly "
                        "recommended on rware-tiny-4ag-v2 — the unshaped reward "
                        "is too sparse for vanilla MAPPO inside a CPU budget.")
    p.add_argument("--pickup-bonus", type=float, default=0.5,
                   help="Per-agent pickup-bonus value passed to MAPPO when "
                        "--shape-rewards is on.")
    p.add_argument("--step-penalty", type=float, default=0.0,
                   help="Per-step penalty passed to MAPPO when --shape-rewards "
                        "is on (0 disables).")

    # Output / dispatch
    p.add_argument("--log-dir", default="runs/exp_matrix")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plans and exit; do not launch subprocesses.")
    p.add_argument("--no-capture", action="store_true",
                   help="Stream subprocess output to stdout instead of capturing to file.")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Abort the matrix on the first non-zero subprocess exit.")
    p.add_argument("--limit", type=int, default=None,
                   help="Run only the first N planned entries (smoke test).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    methods = [METHODS[m] for m in args.methods]
    regimes = [REGIMES[r] for r in args.regimes]

    plans = build_plans(
        methods=methods,
        regimes=regimes,
        delays=args.delays,
        seeds=args.seeds,
        args=args,
    )
    if args.limit is not None:
        plans = plans[: int(args.limit)]

    log_dir = Path(args.log_dir)
    manifest = _write_manifest(log_dir, plans, args)

    print(f"[matrix] env={args.env}")
    print(f"[matrix] {len(plans)} planned run(s) -> {log_dir}")
    print(f"[matrix] manifest: {manifest}")
    by_method: dict[str, int] = {}
    for p in plans:
        by_method[p.method.name] = by_method.get(p.method.name, 0) + 1
    for k, v in sorted(by_method.items()):
        print(f"[matrix]   {k:30s}  {v:3d} runs")

    if args.dry_run:
        print("\n[matrix] --dry-run set; not launching anything.")
        for plan in plans:
            print(f"  {plan.run_name}")
            print(f"    {' '.join(plan.cmd)}")
        return

    successes = 0
    failures: list[tuple[str, int]] = []
    t0 = time.time()
    for plan in plans:
        rc = _run_one(plan, log_dir, capture=not args.no_capture)
        if rc == 0:
            successes += 1
        else:
            failures.append((plan.run_name, rc))
            if args.stop_on_error:
                print(f"[matrix] stopping early after {plan.run_name} (exit={rc})")
                break

    elapsed = time.time() - t0
    print(
        f"\n[matrix] done in {elapsed:.1f}s "
        f"({successes} ok, {len(failures)} failed)"
    )
    for name, rc in failures:
        print(f"  FAILED  {name}  exit={rc}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
