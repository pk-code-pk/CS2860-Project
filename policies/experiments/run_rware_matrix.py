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

Parallel dispatch
-----------------

By default the matrix runs cells **sequentially** (``--max-parallel 1``),
which is exactly the original single-shot behaviour. Pass
``--max-parallel N`` to run up to N cells concurrently as separate
subprocesses. Each child's BLAS / OpenMP / Accelerate thread pool is
capped to ``--threads-per-cell`` (default 1) so N parallel children do
NOT oversubscribe the physical cores -- this is the standard fix for
"embarrassingly parallel subprocess" workloads.

Recommended sweet spot on an 8-core Apple Silicon laptop:

    --max-parallel 3  --threads-per-cell 1

Ctrl+C is handled cleanly: the runner sends SIGTERM to all in-flight
children, gives them ``grace_seconds`` (10s) to exit, then SIGKILLs any
survivors. A second Ctrl+C escalates immediately to SIGKILL.

Production-grade eval
---------------------

The default ``--eval-every 10 --eval-episodes 3`` is intentionally
smoke-test-y. For a real research-grade run that should produce
publishable curves, pass ``--production-eval`` to override these to
``eval-every=25 eval-episodes=30``. Equivalent to setting both flags
explicitly.

Usage examples
--------------

Preview the full default matrix (no env launches):

    uv run python -m policies.experiments.run_rware_matrix --dry-run

Real run on the main env, 3 seeds, default methods/regimes/delays,
3-way parallel, production eval:

    uv run python -m policies.experiments.run_rware_matrix \\
        --env rware-tiny-4ag-v2 --updates 1000 --rollout 512 \\
        --shape-rewards \\
        --max-parallel 3 --threads-per-cell 1 \\
        --production-eval

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
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any


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
    dropout_window_start: int | None,
    dropout_window_end: int | None,
    heartbeat_period: int,
    heartbeat_max_age_clip: int,
    shape_rewards: bool,
    pickup_bonus: float,
    step_penalty: float,
    disable_message_echo: bool,
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
        cmd.extend(["--heartbeat-max-age-clip", str(heartbeat_max_age_clip)])
        used["heartbeat"] = True
        used["heartbeat_period"] = heartbeat_period
        used["heartbeat_max_age_clip"] = heartbeat_max_age_clip
        if regime.has_delay and delay > 0:
            cmd.extend(["--heartbeat-delay", str(delay)])
            used["heartbeat_delay"] = delay

    if regime.has_dropout:
        cmd.append("--dropout")
        if dropout_window_start is not None and dropout_window_end is not None:
            # Window mode: random agent + random time inside [start, end), drawn
            # per-episode by the wrapper using the reset seed. Mutually
            # exclusive with fixed (--dropout-agent / --dropout-time) mode in
            # DropoutConfig.__post_init__, so do NOT also send those.
            cmd.extend(["--dropout-window-start", str(dropout_window_start)])
            cmd.extend(["--dropout-window-end", str(dropout_window_end)])
            used["dropout"] = True
            used["dropout_mode"] = "window"
            used["dropout_window_start"] = dropout_window_start
            used["dropout_window_end"] = dropout_window_end
        else:
            cmd.extend(["--dropout-agent", str(dropout_agent)])
            cmd.extend(["--dropout-time", str(dropout_time)])
            used["dropout"] = True
            used["dropout_mode"] = "fixed"
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

    # Oracle-ceiling experiment: only meaningful for methods with a real
    # comm channel (comm=True), since it changes the behavior of the
    # message vector. Forwarded unconditionally when requested; for
    # no-comm cells it's a no-op (dead rows are zero either way).
    if disable_message_echo:
        cmd.append("--disable-message-echo")
        used["disable_message_echo"] = True

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
    dropout_window_start: int | None,
    dropout_window_end: int | None,
    heartbeat_period: int,
    heartbeat_max_age_clip: int,
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
        cmd.extend(["--heartbeat-max-age-clip", str(heartbeat_max_age_clip)])
        used["heartbeat"] = True
        used["heartbeat_period"] = heartbeat_period
        used["heartbeat_max_age_clip"] = heartbeat_max_age_clip
        if regime.has_delay and delay > 0:
            cmd.extend(["--heartbeat-delay", str(delay)])
            used["heartbeat_delay"] = delay
    if regime.has_dropout:
        cmd.append("--dropout")
        if dropout_window_start is not None and dropout_window_end is not None:
            cmd.extend(["--dropout-window-start", str(dropout_window_start)])
            cmd.extend(["--dropout-window-end", str(dropout_window_end)])
            used["dropout"] = True
            used["dropout_mode"] = "window"
            used["dropout_window_start"] = dropout_window_start
            used["dropout_window_end"] = dropout_window_end
        else:
            cmd.extend(["--dropout-agent", str(dropout_agent)])
            cmd.extend(["--dropout-time", str(dropout_time)])
            used["dropout"] = True
            used["dropout_mode"] = "fixed"
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
                            dropout_window_start=args.dropout_window_start,
                            dropout_window_end=args.dropout_window_end,
                            heartbeat_period=args.heartbeat_period,
                            heartbeat_max_age_clip=args.heartbeat_max_age_clip,
                            shape_rewards=args.shape_rewards,
                            pickup_bonus=args.pickup_bonus,
                            step_penalty=args.step_penalty,
                            disable_message_echo=args.disable_message_echo,
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
                            dropout_window_start=args.dropout_window_start,
                            dropout_window_end=args.dropout_window_end,
                            heartbeat_period=args.heartbeat_period,
                            heartbeat_max_age_clip=args.heartbeat_max_age_clip,
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


def _build_child_env(threads_per_cell: int) -> dict[str, str]:
    """
    Construct a subprocess env that caps numerical-library thread pools.

    On macOS / Apple Silicon (and on most multi-core Linux boxes), every
    Python child that imports numpy / torch will spin up an OpenMP /
    Accelerate / MKL pool sized to physical cores. When we run several cells
    in parallel that produces severe oversubscription
    (e.g. 4 children x ~6 BLAS threads = 24 threads on an 8-core machine),
    which actually slows total wall-clock down. Capping the per-child pool
    to 1 thread is the standard fix for "embarrassingly parallel
    subprocess" workloads -- we'd rather have N independent single-threaded
    cells than N children fighting for the same cores.

    The returned dict is a *copy* of os.environ with the relevant variables
    overwritten. Pass it as the ``env=`` argument to ``subprocess.Popen``.
    """
    env = os.environ.copy()
    n = str(int(threads_per_cell))
    env["OMP_NUM_THREADS"] = n
    env["MKL_NUM_THREADS"] = n
    env["OPENBLAS_NUM_THREADS"] = n
    env["VECLIB_MAXIMUM_THREADS"] = n  # macOS Accelerate
    env["NUMEXPR_NUM_THREADS"] = n
    # PyTorch respects OMP_NUM_THREADS but also has its own knob:
    env.setdefault("PYTORCH_NUM_THREADS", n)
    return env


@dataclass
class _InFlight:
    """Bookkeeping for a currently-running child subprocess."""

    plan: RunPlan
    popen: subprocess.Popen
    log_handle: IO[str] | None  # open stdout file, or None for --no-capture
    started_at: float


class ParallelRunner:
    """
    Dispatch a list of RunPlans across at most ``max_parallel`` concurrent
    subprocesses.

    Why a custom class instead of ``concurrent.futures.ThreadPoolExecutor``?

    * We need direct access to each child's ``Popen`` object so that a
      Ctrl+C in the parent process can cleanly tear down all children
      (SIGTERM, then SIGKILL after a grace period).
    * ``futures`` would give us a Future per cell, but ``subprocess.run``
      blocks the worker thread for the entire training run -- we'd have no
      handle on the child to terminate it.

    Sequential mode (``max_parallel == 1``) goes through exactly the same
    code path; this keeps behaviour identical to the original
    sequential-only runner so we don't regress when the user doesn't opt
    into parallelism.
    """

    def __init__(
        self,
        *,
        max_parallel: int,
        threads_per_cell: int,
        capture: bool,
        stop_on_error: bool,
        grace_seconds: float = 10.0,
    ) -> None:
        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        self.max_parallel = int(max_parallel)
        self.threads_per_cell = int(threads_per_cell)
        self.capture = bool(capture)
        self.stop_on_error = bool(stop_on_error)
        self.grace_seconds = float(grace_seconds)

        self._in_flight: list[_InFlight] = []
        self._results: list[tuple[str, int, float]] = []  # (run_name, rc, elapsed)
        self._stop = False
        self._sigint_count = 0
        self._sigterm_sent_at: float | None = None  # wall-clock when we SIGTERM'd
        self._t0 = 0.0
        self._queued_count = 0

    # ----- signal handling ---------------------------------------------
    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    def _on_signal(self, signum: int, frame: Any) -> None:  # noqa: ARG002
        self._sigint_count += 1
        self._stop = True
        if self._sigint_count == 1:
            self._sigterm_sent_at = time.time()
            print(
                f"\n[matrix] caught signal {signum}; terminating "
                f"{len(self._in_flight)} in-flight child(ren) "
                f"(SIGTERM, {self.grace_seconds:.0f}s grace, then SIGKILL)..."
            )
            for fl in list(self._in_flight):
                try:
                    fl.popen.terminate()
                except ProcessLookupError:
                    pass
        else:
            print("\n[matrix] second signal -- killing children NOW")
            for fl in list(self._in_flight):
                try:
                    fl.popen.kill()
                except ProcessLookupError:
                    pass

    # ----- launch / reap -----------------------------------------------
    def _launch(self, plan: RunPlan, log_dir: Path) -> None:
        out_path = log_dir / plan.run_name
        out_path.mkdir(parents=True, exist_ok=True)
        stdout_path = out_path / "stdout.log"
        env = _build_child_env(self.threads_per_cell)

        log_handle: IO[str] | None
        if self.capture:
            log_handle = stdout_path.open("w")
            log_handle.write("$ " + " ".join(plan.cmd) + "\n\n")
            log_handle.flush()
            popen = subprocess.Popen(
                plan.cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
        else:
            log_handle = None
            popen = subprocess.Popen(plan.cmd, env=env)

        fl = _InFlight(plan=plan, popen=popen, log_handle=log_handle, started_at=time.time())
        self._in_flight.append(fl)
        print(
            f"[start] {plan.run_name}  pid={popen.pid}  "
            f"in_flight={len(self._in_flight)}/{self.max_parallel}  "
            f"queued={self._queued}  done={len(self._results)}"
        )

    def _try_reap_one(self, *, blocking: bool) -> bool:
        """
        Try to reap one finished child. Returns True if one was reaped.

        If ``blocking=True`` and there is at least one in-flight child, this
        will spin-poll until something finishes (or all are killed by a
        signal handler). The poll interval is 0.25s -- high enough that the
        overhead is negligible vs. multi-minute training runs but low
        enough that progress lines feel responsive.
        """
        if not self._in_flight:
            return False
        while True:
            for i, fl in enumerate(self._in_flight):
                rc = fl.popen.poll()
                if rc is not None:
                    self._finish(i, int(rc))
                    return True
            if not blocking:
                return False
            # If a signal is asking us to stop and the children won't die
            # within grace_seconds, escalate to SIGKILL.
            if self._stop:
                self._maybe_force_kill()
            time.sleep(0.25)

    def _finish(self, idx: int, rc: int) -> None:
        fl = self._in_flight.pop(idx)
        elapsed = time.time() - fl.started_at
        if fl.log_handle is not None:
            try:
                fl.log_handle.close()
            except Exception:
                pass
        self._results.append((fl.plan.run_name, rc, elapsed))
        wall = time.time() - self._t0
        status = "ok " if rc == 0 else "FAIL"
        print(
            f"[done ] {fl.plan.run_name}  exit={rc} ({status})  "
            f"cell_elapsed={elapsed:6.1f}s  wall={wall:6.1f}s  "
            f"in_flight={len(self._in_flight)}/{self.max_parallel}  "
            f"queued={self._queued}  done={len(self._results)}"
        )

    def _maybe_force_kill(self) -> None:
        """If we already SIGTERM'd and grace period has elapsed, SIGKILL."""
        if self._sigterm_sent_at is None:
            return
        elapsed_since_sigterm = time.time() - self._sigterm_sent_at
        if elapsed_since_sigterm > self.grace_seconds or self._sigint_count >= 2:
            for fl in list(self._in_flight):
                try:
                    fl.popen.kill()
                except ProcessLookupError:
                    pass

    # ----- top-level loop ----------------------------------------------
    @property
    def _queued(self) -> int:
        return self._queued_count

    def run(self, plans: list[RunPlan], log_dir: Path) -> tuple[int, list[tuple[str, int]]]:
        """
        Run all plans. Returns (n_successes, [(run_name, rc) for each failure]).
        """
        self._install_signal_handlers()
        self._t0 = time.time()
        self._queued_count = len(plans)

        for plan in plans:
            if self._stop:
                break
            # Wait for a slot.
            while len(self._in_flight) >= self.max_parallel:
                self._try_reap_one(blocking=True)
                if self.stop_on_error and self._results and self._results[-1][1] != 0:
                    print(
                        f"[matrix] --stop-on-error set; aborting after "
                        f"{self._results[-1][0]} (exit={self._results[-1][1]})"
                    )
                    self._stop = True
                    break
            if self._stop:
                break
            self._queued_count -= 1
            self._launch(plan, log_dir)

        # Drain whatever is left.
        while self._in_flight:
            self._try_reap_one(blocking=True)

        successes = sum(1 for _, rc, _ in self._results if rc == 0)
        failures = [(name, rc) for name, rc, _ in self._results if rc != 0]
        return successes, failures


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
    p.add_argument("--heartbeat-max-age-clip", type=int, default=32,
                   help="Clip heartbeat-age features at this value. Set this "
                        "above the largest delay in delay scans (e.g. 128 for "
                        "d=60/100) so high-delay cells remain distinguishable.")
    # Fixed-mode dropout (used iff window flags are NOT both supplied).
    p.add_argument("--dropout-agent", type=int, default=0)
    p.add_argument("--dropout-time", type=int, default=100)
    # Window-mode dropout: random agent + random time in [start, end), drawn
    # per-episode by the wrapper using the reset seed. Mutually exclusive
    # with fixed mode in DropoutConfig (enforced wrapper-side); when both
    # window flags are set, the matrix runner forwards window flags only.
    p.add_argument("--dropout-window-start", type=int, default=None,
                   help="Lower bound (inclusive) of the per-episode dropout "
                        "step window. Must be set together with --dropout-"
                        "window-end to enable window mode.")
    p.add_argument("--dropout-window-end", type=int, default=None,
                   help="Upper bound (exclusive) of the per-episode dropout "
                        "step window. Must be set together with --dropout-"
                        "window-start to enable window mode.")

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

    # Oracle-ceiling experiment (see docs/PROGRESS_REPORT.md §6, and the
    # `--disable-message-echo` flag in policies/train.py).
    p.add_argument("--disable-message-echo", action="store_true",
                   help="Forward --disable-message-echo to every MAPPO cell. "
                        "Dead agents emit all-zero messages instead of "
                        "echoing their last one-hot, creating a perfect "
                        "dropout oracle through the message channel. Used "
                        "for the oracle-ceiling experiment that bounds how "
                        "much any comm method could possibly rescue dropout. "
                        "NOT a production flag.")

    # Output / dispatch
    p.add_argument("--log-dir", default="runs/exp_matrix")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plans and exit; do not launch subprocesses.")
    p.add_argument("--no-capture", action="store_true",
                   help="Stream subprocess output to stdout instead of capturing to file. "
                        "With --max-parallel > 1 this produces interleaved output; "
                        "prefer the default (capture per cell) when running in parallel.")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Abort the matrix on the first non-zero subprocess exit.")
    p.add_argument("--limit", type=int, default=None,
                   help="Run only the first N planned entries (smoke test).")
    p.add_argument("--max-parallel", type=int, default=1,
                   help="Max number of cells to run concurrently as subprocesses. "
                        "Default 1 (sequential, byte-equivalent to the original "
                        "single-shot behaviour). On an 8-core Apple Silicon laptop, "
                        "3-4 is the typical sweet spot when --threads-per-cell=1.")
    p.add_argument("--threads-per-cell", type=int, default=1,
                   help="OMP / MKL / Accelerate thread cap injected into each "
                        "child subprocess. Keep at 1 when running with "
                        "--max-parallel > 1 to avoid BLAS oversubscription on "
                        "multi-core CPUs. Increase only if you're running a "
                        "single cell (max-parallel=1) on a machine where you "
                        "want BLAS to use all cores for that cell.")
    p.add_argument("--production-eval", action="store_true",
                   help="Convenience flag: overrides --eval-every and "
                        "--eval-episodes with production-grade values "
                        "(eval-every=25, eval-episodes=30) so eval noise stops "
                        "dominating the per-cell signal. Equivalent to passing "
                        "--eval-every 25 --eval-episodes 30 explicitly.")
    return p.parse_args()


def main() -> None:
    # Force line-buffered stdout so [matrix] / [start] / [done] lines flush
    # immediately when stdout is redirected to a file (e.g. backgrounded
    # via nohup / `&`). Without this, Python block-buffers stdout in
    # non-tty mode and the runner appears silent for minutes at a time
    # even though it's working fine.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    args = _parse_args()
    if args.production_eval:
        args.eval_every = 25
        args.eval_episodes = 30
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
    print(
        f"[matrix] dispatch: max_parallel={args.max_parallel}  "
        f"threads_per_cell={args.threads_per_cell}  "
        f"capture={'on' if not args.no_capture else 'off'}"
    )
    if args.production_eval:
        print(
            f"[matrix] --production-eval: eval_every={args.eval_every}  "
            f"eval_episodes={args.eval_episodes}"
        )
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

    if args.no_capture and args.max_parallel > 1:
        print(
            "[matrix] WARNING: --no-capture + --max-parallel > 1 produces "
            "interleaved subprocess output. Per-cell logs will NOT be "
            "written to runs/<dir>/<cell>/stdout.log. Consider dropping "
            "--no-capture if you want clean per-cell logs."
        )

    runner = ParallelRunner(
        max_parallel=args.max_parallel,
        threads_per_cell=args.threads_per_cell,
        capture=not args.no_capture,
        stop_on_error=args.stop_on_error,
    )
    t0 = time.time()
    successes, failures = runner.run(plans, log_dir)
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
