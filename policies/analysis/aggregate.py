"""
Aggregate the output of ``policies.experiments.run_rware_matrix`` into
analysis-friendly CSVs.

Walks ``--log-dir`` (default ``runs/exp_matrix``) and treats every
sub-directory containing a ``metrics.csv`` as one run. The run-name
schema is::

    {method}__{regime}__d{delay}__s{seed}

so the (method, regime, delay, seed) tuple is recovered by parsing the
directory name. ``config.json`` is also loaded to pull additional
metadata (env id, dropout time, heartbeat period, ...).

Two CSVs are produced under ``--out-dir`` (default = log-dir):

* ``per_run.csv``       -- one row per run with the final/last-K metrics.
* ``summary.csv``       -- mean +/- std across seeds, grouped by
                           (method, regime, delay).

Metrics extracted from ``metrics.csv``:

* ``final_eval_return``    -- last non-empty value in ``eval/ep_return_mean``
                              (if any), otherwise ``train/ep_return_mean``.
* ``train_return_lastK``   -- mean of the last ``--last-k`` train rows.
* ``train_return_max``     -- max ``train/ep_return_mean`` across the run.
* ``ep_length_mean``       -- last value of ``train/ep_length_mean``.
* ``deliveries_mean``      -- last value of ``heuristic/deliveries`` if present.
* ``rows``                 -- number of rows in ``metrics.csv``.

Usage::

    uv run python -m policies.analysis.aggregate \\
        --log-dir runs/exp_matrix --out-dir runs/exp_matrix --last-k 5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Run-name parser
# --------------------------------------------------------------------------

# {method}__{regime}__d{delay}__s{seed}
# method may itself contain hyphens, e.g. "mappo-heartbeat-plus-comm".
_RUN_NAME_RE = re.compile(
    r"^(?P<method>.+?)__(?P<regime>[a-z0-9-]+)__d(?P<delay>-?\d+)__s(?P<seed>-?\d+)$"
)


@dataclass(frozen=True)
class RunKey:
    method: str
    regime: str
    delay: int
    seed: int


def parse_run_name(name: str) -> RunKey | None:
    m = _RUN_NAME_RE.match(name)
    if not m:
        return None
    return RunKey(
        method=m["method"],
        regime=m["regime"],
        delay=int(m["delay"]),
        seed=int(m["seed"]),
    )


# --------------------------------------------------------------------------
# CSV / config helpers
# --------------------------------------------------------------------------


def _safe_float(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _last_non_empty(rows: list[dict[str, str]], key: str) -> float | None:
    for row in reversed(rows):
        v = _safe_float(row.get(key))
        if v is not None:
            return v
    return None


def _first_not_none(*values: float | None) -> float | None:
    """Return the first value that is not ``None`` (treats 0.0 as valid)."""
    for v in values:
        if v is not None:
            return v
    return None


def _mean_last_k(rows: list[dict[str, str]], key: str, k: int) -> float | None:
    if not rows:
        return None
    vals: list[float] = []
    for row in reversed(rows):
        v = _safe_float(row.get(key))
        if v is not None:
            vals.append(v)
        if len(vals) >= k:
            break
    if not vals:
        return None
    return sum(vals) / len(vals)


def _max_value(rows: list[dict[str, str]], key: str) -> float | None:
    best: float | None = None
    for row in rows:
        v = _safe_float(row.get(key))
        if v is None:
            continue
        if best is None or v > best:
            best = v
    return best


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


# --------------------------------------------------------------------------
# Per-run extraction
# --------------------------------------------------------------------------


@dataclass
class PerRunRow:
    run_name: str
    method: str
    regime: str
    delay: int
    seed: int
    env: str
    final_eval_return: float | None
    train_return_lastK: float | None
    train_return_max: float | None
    ep_length_mean: float | None
    deliveries_mean: float | None
    rows: int
    heartbeat: bool | None
    dropout: bool | None
    dropout_agent: int | None
    dropout_time: int | None
    heartbeat_delay: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "method": self.method,
            "regime": self.regime,
            "delay": self.delay,
            "seed": self.seed,
            "env": self.env,
            "final_eval_return": _fmt(self.final_eval_return),
            "train_return_lastK": _fmt(self.train_return_lastK),
            "train_return_max": _fmt(self.train_return_max),
            "ep_length_mean": _fmt(self.ep_length_mean),
            "deliveries_mean": _fmt(self.deliveries_mean),
            "rows": self.rows,
            "heartbeat": self.heartbeat,
            "dropout": self.dropout,
            "dropout_agent": self.dropout_agent,
            "dropout_time": self.dropout_time,
            "heartbeat_delay": self.heartbeat_delay,
        }


def _fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.6f}"


def extract_run(run_dir: Path, *, last_k: int) -> PerRunRow | None:
    key = parse_run_name(run_dir.name)
    if key is None:
        return None
    rows = _read_csv(run_dir / "metrics.csv")
    if not rows:
        return None
    cfg = _read_config(run_dir / "config.json")

    return PerRunRow(
        run_name=run_dir.name,
        method=key.method,
        regime=key.regime,
        delay=key.delay,
        seed=key.seed,
        env=str(cfg.get("env", "")),
        # Note: use an explicit None-check fallback rather than `or`, because
        # a real eval return of 0.0 is falsy in Python and would otherwise be
        # silently overridden by the train return.
        final_eval_return=_first_not_none(
            _last_non_empty(rows, "eval/ep_return_mean"),
            _last_non_empty(rows, "train/ep_return_mean"),
        ),
        train_return_lastK=_mean_last_k(rows, "train/ep_return_mean", last_k),
        train_return_max=_max_value(rows, "train/ep_return_mean"),
        ep_length_mean=_last_non_empty(rows, "train/ep_length_mean"),
        deliveries_mean=_last_non_empty(rows, "heuristic/deliveries"),
        rows=len(rows),
        heartbeat=_to_bool(cfg.get("heartbeat")),
        dropout=_to_bool(cfg.get("dropout")),
        dropout_agent=_to_int(cfg.get("dropout_agent")),
        dropout_time=_to_int(cfg.get("dropout_time")),
        heartbeat_delay=_to_int(cfg.get("heartbeat_delay")),
    )


def _to_bool(x: Any) -> bool | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in ("true", "1", "yes")
    return bool(x)


def _to_int(x: Any) -> int | None:
    if x is None or x == "":
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Summary across seeds
# --------------------------------------------------------------------------


def _mean_std(values: list[float]) -> tuple[float, float, int]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, 1
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var), n


def summarise(per_run: list[PerRunRow]) -> list[dict[str, Any]]:
    """
    Group by (method, regime, delay) and reduce the final_eval_return /
    train_return_lastK columns into mean +/- std with a seed count.
    """
    groups: dict[tuple[str, str, int], list[PerRunRow]] = {}
    for row in per_run:
        key = (row.method, row.regime, row.delay)
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (method, regime, delay), rows in sorted(groups.items()):
        final = [r.final_eval_return for r in rows if r.final_eval_return is not None]
        train = [r.train_return_lastK for r in rows if r.train_return_lastK is not None]
        ep_len = [r.ep_length_mean for r in rows if r.ep_length_mean is not None]
        env = next((r.env for r in rows if r.env), "")

        f_mean, f_std, f_n = _mean_std(final)
        t_mean, t_std, _ = _mean_std(train)
        l_mean, l_std, _ = _mean_std(ep_len)

        out.append(
            {
                "method": method,
                "regime": regime,
                "delay": delay,
                "env": env,
                "n_seeds": f_n,
                "seeds": ",".join(str(r.seed) for r in sorted(rows, key=lambda r: r.seed)),
                "final_return_mean": _fmt_num(f_mean),
                "final_return_std": _fmt_num(f_std),
                "train_return_lastK_mean": _fmt_num(t_mean),
                "train_return_lastK_std": _fmt_num(t_std),
                "ep_length_mean": _fmt_num(l_mean),
                "ep_length_std": _fmt_num(l_std),
            }
        )
    return out


def _fmt_num(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.6f}"


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate matrix-runner outputs into per_run.csv + summary.csv"
    )
    p.add_argument("--log-dir", default="runs/exp_matrix",
                   help="Directory containing per-run subdirs")
    p.add_argument("--out-dir", default=None,
                   help="Where to write per_run.csv + summary.csv (defaults to --log-dir)")
    p.add_argument("--last-k", type=int, default=5,
                   help="Average over the last K train rows for train_return_lastK")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"log dir does not exist: {log_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else log_dir

    per_run: list[PerRunRow] = []
    skipped: list[str] = []
    for run_dir in sorted(p for p in log_dir.iterdir() if p.is_dir()):
        row = extract_run(run_dir, last_k=int(args.last_k))
        if row is None:
            skipped.append(run_dir.name)
            continue
        per_run.append(row)

    print(f"[aggregate] {len(per_run)} runs ingested from {log_dir}")
    if skipped:
        print(f"[aggregate] skipped {len(skipped)} dir(s) (no metrics.csv or bad name):")
        for s in skipped[:5]:
            print(f"             - {s}")
        if len(skipped) > 5:
            print(f"             ... and {len(skipped) - 5} more")

    if not per_run:
        print("[aggregate] no usable runs; nothing to write")
        return

    per_run_path = out_dir / "per_run.csv"
    fieldnames_per_run = list(per_run[0].as_dict().keys())
    write_csv(per_run_path, [r.as_dict() for r in per_run], fieldnames_per_run)
    print(f"[aggregate] wrote {per_run_path}  ({len(per_run)} rows)")

    summary_rows = summarise(per_run)
    summary_path = out_dir / "summary.csv"
    fieldnames_summary = list(summary_rows[0].keys()) if summary_rows else []
    write_csv(summary_path, summary_rows, fieldnames_summary)
    print(f"[aggregate] wrote {summary_path}  ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
