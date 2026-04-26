"""Run the intent-grounded communication diagnostic matrix.

This is intentionally small: it tests whether learned communication helps
dropout recovery when messages are grounded in task intent rather than liveness
heartbeat features.

Example:

    uv run python scripts/run_intent_grounded_v1.py --seeds 0 1 2 3 --max-parallel 4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch the no-heartbeat intent-grounded comm diagnostic."
    )
    p.add_argument(
        "--envs",
        nargs="+",
        default=["rware-tiny-2ag-easy-v2"],
        help="RWARE env ids to test. Add valid 2-agent small/medium ids if available.",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--rollout", type=int, default=512)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--message-grounding-coef", type=float, default=0.2)
    p.add_argument("--dropout-agent", type=int, default=0)
    p.add_argument("--dropout-time", type=int, default=100)
    p.add_argument("--max-parallel", type=int, default=2)
    p.add_argument("--threads-per-cell", type=int, default=1)
    p.add_argument("--log-root", default="matrix_results/intent_grounded_v1")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(args.log_root)
    root.mkdir(parents=True, exist_ok=True)

    for env in args.envs:
        log_dir = root / env
        cmd = [
            sys.executable,
            "-m",
            "policies.experiments.run_rware_matrix",
            "--env",
            env,
            "--methods",
            "mappo-no-comm",
            "mappo-comm",
            "mappo-intent-aux",
            "--regimes",
            "baseline",
            "dropout-only",
            "--seeds",
            *[str(s) for s in args.seeds],
            "--updates",
            str(args.updates),
            "--rollout",
            str(args.rollout),
            "--eval-every",
            str(args.eval_every),
            "--eval-episodes",
            str(args.eval_episodes),
            "--dropout-agent",
            str(args.dropout_agent),
            "--dropout-time",
            str(args.dropout_time),
            "--message-grounding-coef",
            str(args.message_grounding_coef),
            "--shape-rewards",
            "--heartbeat-max-age-clip",
            "128",
            "--max-parallel",
            str(args.max_parallel),
            "--threads-per-cell",
            str(args.threads_per_cell),
            "--log-dir",
            str(log_dir),
        ]
        if args.dry_run:
            cmd.append("--dry-run")

        print("[intent-grounded]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
