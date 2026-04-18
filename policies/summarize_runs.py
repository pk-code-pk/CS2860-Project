"""
Tiny CLI to print the last-K metrics of every run under a log directory.

Usage:
    uv run python -m policies.summarize_runs --log-dir runs --tail 3
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _fmt(x: str) -> str:
    try:
        return f"{float(x):.3f}"
    except (ValueError, TypeError):
        return str(x)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="runs")
    p.add_argument("--tail", type=int, default=3)
    p.add_argument(
        "--keys",
        default="step,train/ep_return_mean,train/ep_length_mean,"
                "eval/ep_return_mean,loss/entropy,loss/approx_kl",
    )
    args = p.parse_args()

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"no such dir: {log_dir}")
        return

    for run_dir in sorted(log_dir.iterdir()):
        csv_path = run_dir / "metrics.csv"
        if not csv_path.exists():
            continue
        rows = _read_csv(csv_path)
        if not rows:
            continue
        print(f"\n== {run_dir.name} == ({len(rows)} rows)")
        header_cols = [k for k in keys if k in rows[0]]
        print("  " + "  ".join(f"{c:>22}" for c in header_cols))
        for row in rows[-args.tail :]:
            print("  " + "  ".join(f"{_fmt(row.get(c, '')):>22}" for c in header_cols))


if __name__ == "__main__":
    main()
