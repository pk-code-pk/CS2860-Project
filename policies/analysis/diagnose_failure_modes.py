"""
Diagnose likely communication failure modes from matrix-runner outputs.

This script reads run folders named:

    {method}__{regime}__d{delay}__s{seed}

and computes a compact report aimed at answering:

1) Is there enough dropout headroom for communication to rescue?
2) Does communication help more under delay-dropout than delay-only?
3) Are comm messages likely ungrounded (high message entropy proxy)?
4) Is training materially less stable when communication is enabled?

The entropy diagnosis uses the same decomposition style used in project notes:

    total_entropy(hb+comm) ~= action_entropy + message_entropy

where action_entropy is estimated from matching hb-only cells.
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


RUN_RE = re.compile(
    r"^(?P<method>.+?)__(?P<regime>[a-z0-9-]+)__d(?P<delay>-?\d+)__s(?P<seed>-?\d+)$"
)


@dataclass(frozen=True)
class RunKey:
    method: str
    regime: str
    delay: int
    seed: int


@dataclass
class RunStats:
    key: RunKey
    env: str
    n_msg_tokens: int
    train_last_k_mean: float | None
    train_last_k_std: float | None
    eval_last_k_mean: float | None
    entropy_last_k_mean: float | None
    collapse_frac: float | None
    cv_last_k: float | None


def _safe_float(v: Any) -> float | None:
    if v in (None, "", "nan"):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _last_finite(values: list[float], k: int) -> list[float]:
    finite = [v for v in values if math.isfinite(v)]
    return finite[-k:] if finite else []


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    m = _mean(values)
    if m is None:
        return None
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def _parse_run_dir(run_dir: Path) -> RunStats | None:
    m = RUN_RE.match(run_dir.name)
    if not m:
        return None
    rows = _read_csv(run_dir / "metrics.csv")
    if not rows:
        return None
    cfg = _read_json(run_dir / "config.json")

    train_vals = [_safe_float(r.get("train/ep_return_mean")) for r in rows]
    eval_vals = [_safe_float(r.get("eval/ep_return_mean")) for r in rows]
    ent_vals = [_safe_float(r.get("loss/entropy")) for r in rows]
    train_finite = [v for v in train_vals if v is not None]

    train_last = _last_finite([v for v in train_finite if v is not None], 200)
    eval_last = _last_finite([v for v in eval_vals if v is not None], 10)
    ent_last = _last_finite([v for v in ent_vals if v is not None], 100)

    train_mean = _mean(train_last)
    train_std = _std(train_last)
    collapse_thresh = 0.15 * train_mean if (train_mean is not None and train_mean > 0) else None
    collapse_frac = None
    if collapse_thresh is not None and train_last:
        collapse_frac = sum(1 for x in train_last if x <= collapse_thresh) / len(train_last)

    cv_last = None
    if train_mean is not None and train_std is not None and abs(train_mean) > 1e-9:
        cv_last = train_std / abs(train_mean)

    return RunStats(
        key=RunKey(
            method=m["method"],
            regime=m["regime"],
            delay=int(m["delay"]),
            seed=int(m["seed"]),
        ),
        env=str(cfg.get("env", "")),
        n_msg_tokens=_safe_int(
            cfg.get("n_msg_tokens_effective", cfg.get("n_msg_tokens", 1)),
            default=1,
        ),
        train_last_k_mean=train_mean,
        train_last_k_std=train_std,
        eval_last_k_mean=_mean(eval_last),
        entropy_last_k_mean=_mean(ent_last),
        collapse_frac=collapse_frac,
        cv_last_k=cv_last,
    )


def _group_mean(runs: list[RunStats], metric: str) -> float | None:
    vals: list[float] = []
    for r in runs:
        v = getattr(r, metric)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            vals.append(float(v))
    return _mean(vals)


def _subset(
    runs: list[RunStats], *, method: str | None = None, regime: str | None = None, delay: int | None = None
) -> list[RunStats]:
    out: list[RunStats] = []
    for r in runs:
        if method is not None and r.key.method != method:
            continue
        if regime is not None and r.key.regime != regime:
            continue
        if delay is not None and r.key.delay != delay:
            continue
        out.append(r)
    return out


def _fmt(v: float | None, nd: int = 2) -> str:
    if v is None or not math.isfinite(v):
        return "n/a"
    return f"{v:.{nd}f}"


def _diagnose(log_dir: Path, *, delay: int | None) -> dict[str, Any]:
    runs = [
        x
        for d in sorted(p for p in log_dir.iterdir() if p.is_dir())
        if (x := _parse_run_dir(d)) is not None
    ]
    if not runs:
        raise SystemExit(f"no parseable run dirs with metrics.csv under {log_dir}")

    delays = sorted({r.key.delay for r in runs})
    chosen_delay = delay if delay is not None else (max(delays) if delays else 0)
    runs = _subset(runs, delay=chosen_delay)

    hb_only = "mappo-heartbeat-only"
    hb_comm = "mappo-heartbeat-plus-comm"

    hb_only_do = _subset(runs, method=hb_only, regime="delay-only")
    hb_only_dd = _subset(runs, method=hb_only, regime="delay-dropout")
    hb_comm_do = _subset(runs, method=hb_comm, regime="delay-only")
    hb_comm_dd = _subset(runs, method=hb_comm, regime="delay-dropout")

    # Eval-level interaction and headroom.
    hb_dropout_penalty = None
    if hb_only_do and hb_only_dd:
        hb_dropout_penalty = _group_mean(hb_only_dd, "eval_last_k_mean") - _group_mean(
            hb_only_do, "eval_last_k_mean"
        )

    comm_benefit_do = None
    comm_benefit_dd = None
    if hb_only_do and hb_comm_do:
        comm_benefit_do = _group_mean(hb_comm_do, "eval_last_k_mean") - _group_mean(
            hb_only_do, "eval_last_k_mean"
        )
    if hb_only_dd and hb_comm_dd:
        comm_benefit_dd = _group_mean(hb_comm_dd, "eval_last_k_mean") - _group_mean(
            hb_only_dd, "eval_last_k_mean"
        )

    interaction = None
    if comm_benefit_do is not None and comm_benefit_dd is not None:
        interaction = comm_benefit_dd - comm_benefit_do

    # Entropy decomposition proxy.
    action_entropy_do = _group_mean(hb_only_do, "entropy_last_k_mean")
    action_entropy_dd = _group_mean(hb_only_dd, "entropy_last_k_mean")
    total_comm_entropy_do = _group_mean(hb_comm_do, "entropy_last_k_mean")
    total_comm_entropy_dd = _group_mean(hb_comm_dd, "entropy_last_k_mean")

    msg_entropy_do = None
    msg_entropy_dd = None
    if action_entropy_do is not None and total_comm_entropy_do is not None:
        msg_entropy_do = max(total_comm_entropy_do - action_entropy_do, 0.0)
    if action_entropy_dd is not None and total_comm_entropy_dd is not None:
        msg_entropy_dd = max(total_comm_entropy_dd - action_entropy_dd, 0.0)

    # Infer token count from comm runs (fall back to 8 for reporting scale).
    comm_tokens = next((r.n_msg_tokens for r in runs if r.key.method == hb_comm), 8)
    msg_max = math.log(max(comm_tokens, 2))
    msg_entropy_ratio_do = (msg_entropy_do / msg_max) if (msg_entropy_do is not None and msg_max > 0) else None
    msg_entropy_ratio_dd = (msg_entropy_dd / msg_max) if (msg_entropy_dd is not None and msg_max > 0) else None

    # Stability comparison.
    hb_cv = _group_mean(hb_only_dd + hb_only_do, "cv_last_k")
    comm_cv = _group_mean(hb_comm_dd + hb_comm_do, "cv_last_k")
    hb_collapse = _group_mean(hb_only_dd + hb_only_do, "collapse_frac")
    comm_collapse = _group_mean(hb_comm_dd + hb_comm_do, "collapse_frac")

    likely_issues: list[dict[str, Any]] = []

    if hb_dropout_penalty is None or abs(hb_dropout_penalty) < 3.0:
        likely_issues.append(
            {
                "issue": "weak_dropout_headroom",
                "severity": "high",
                "evidence": f"hb-only dropout minus delay-only eval = {_fmt(hb_dropout_penalty)}",
                "interpretation": (
                    "Dropout barely changes baseline eval, so maximum possible communication rescue is small."
                ),
            }
        )
    if msg_entropy_ratio_dd is not None and msg_entropy_ratio_dd >= 0.75:
        likely_issues.append(
            {
                "issue": "ungrounded_message_channel",
                "severity": "high",
                "evidence": (
                    f"estimated message entropy ratio under delay-dropout = {_fmt(msg_entropy_ratio_dd, 3)} "
                    f"(1.0 means near-uniform random)"
                ),
                "interpretation": (
                    "Comm symbols appear weakly grounded; the sender likely emits high-entropy tokens."
                ),
            }
        )
    if comm_cv is not None and hb_cv is not None and comm_cv > hb_cv * 1.2:
        likely_issues.append(
            {
                "issue": "comm_training_instability",
                "severity": "medium",
                "evidence": f"train CV last-200 updates: hb-only={_fmt(hb_cv,3)} vs hb+comm={_fmt(comm_cv,3)}",
                "interpretation": "Adding comm increases policy oscillation and can mask any small benefit.",
            }
        )
    if interaction is not None and interaction <= 0:
        likely_issues.append(
            {
                "issue": "missing_dropout_specific_comm_gain",
                "severity": "high",
                "evidence": (
                    f"interaction (comm benefit dd - comm benefit do) = {_fmt(interaction)} "
                    f"[do={_fmt(comm_benefit_do)}, dd={_fmt(comm_benefit_dd)}]"
                ),
                "interpretation": (
                    "Communication is not providing the hypothesized dropout-specific rescue effect."
                ),
            }
        )
    if comm_collapse is not None and hb_collapse is not None and comm_collapse > hb_collapse + 0.1:
        likely_issues.append(
            {
                "issue": "comm_collapse_tendency",
                "severity": "medium",
                "evidence": (
                    f"low-return-state fraction in last-200 updates: hb-only={_fmt(hb_collapse,3)} "
                    f"vs hb+comm={_fmt(comm_collapse,3)}"
                ),
                "interpretation": (
                    "Comm runs spend more training time in collapse-like troughs."
                ),
            }
        )

    return {
        "log_dir": str(log_dir),
        "delay": chosen_delay,
        "n_runs": len(runs),
        "methods_detected": sorted({r.key.method for r in runs}),
        "regimes_detected": sorted({r.key.regime for r in runs}),
        "core_metrics": {
            "hb_dropout_minus_delay_eval": hb_dropout_penalty,
            "comm_benefit_delay_only_eval": comm_benefit_do,
            "comm_benefit_delay_dropout_eval": comm_benefit_dd,
            "interaction_eval": interaction,
            "action_entropy_delay_only": action_entropy_do,
            "action_entropy_delay_dropout": action_entropy_dd,
            "total_comm_entropy_delay_only": total_comm_entropy_do,
            "total_comm_entropy_delay_dropout": total_comm_entropy_dd,
            "estimated_msg_entropy_delay_only": msg_entropy_do,
            "estimated_msg_entropy_delay_dropout": msg_entropy_dd,
            "estimated_msg_entropy_ratio_delay_only": msg_entropy_ratio_do,
            "estimated_msg_entropy_ratio_delay_dropout": msg_entropy_ratio_dd,
            "hb_only_train_cv": hb_cv,
            "hb_plus_comm_train_cv": comm_cv,
            "hb_only_collapse_frac": hb_collapse,
            "hb_plus_comm_collapse_frac": comm_collapse,
            "n_msg_tokens_effective": comm_tokens,
        },
        "likely_issues": likely_issues,
    }


def _render_report_text(result: dict[str, Any]) -> str:
    m = result["core_metrics"]
    lines = [
        f"Diagnosis report for: {result['log_dir']}",
        f"delay analyzed: d={result['delay']}  |  runs={result['n_runs']}",
        "",
        "Core contrasts (eval):",
        f"- hb-only dropout minus delay-only: {_fmt(m['hb_dropout_minus_delay_eval'])}",
        f"- comm benefit in delay-only: {_fmt(m['comm_benefit_delay_only_eval'])}",
        f"- comm benefit in delay-dropout: {_fmt(m['comm_benefit_delay_dropout_eval'])}",
        f"- interaction (dd - do): {_fmt(m['interaction_eval'])}",
        "",
        "Grounding proxy (entropy decomposition):",
        (
            f"- estimated msg entropy ratio: delay-only={_fmt(m['estimated_msg_entropy_ratio_delay_only'],3)}, "
            f"delay-dropout={_fmt(m['estimated_msg_entropy_ratio_delay_dropout'],3)} "
            "(closer to 1.0 => near-uniform messages)"
        ),
        "",
        "Stability:",
        f"- train CV (last-200 updates): hb-only={_fmt(m['hb_only_train_cv'],3)}, hb+comm={_fmt(m['hb_plus_comm_train_cv'],3)}",
        f"- collapse-like fraction: hb-only={_fmt(m['hb_only_collapse_frac'],3)}, hb+comm={_fmt(m['hb_plus_comm_collapse_frac'],3)}",
        "",
        "Likely issues (ranked):",
    ]
    issues = result.get("likely_issues", [])
    if not issues:
        lines.append("- No strong issue signatures triggered by current thresholds.")
    else:
        for idx, issue in enumerate(issues, start=1):
            lines.append(
                f"{idx}. [{issue['severity']}] {issue['issue']}: {issue['evidence']} | {issue['interpretation']}"
            )
    lines.append("")
    lines.append("Note: this is diagnostic evidence, not strict causal proof.")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose likely communication failure modes from matrix results."
    )
    p.add_argument(
        "--log-dir",
        required=True,
        help="Matrix result directory (contains run subdirs with metrics.csv/config.json).",
    )
    p.add_argument(
        "--delay",
        type=int,
        default=None,
        help="Specific delay bucket to analyze (default: largest delay in log-dir).",
    )
    p.add_argument(
        "--out-json",
        default=None,
        help="Optional path for machine-readable JSON output.",
    )
    p.add_argument(
        "--out-report",
        default=None,
        help="Optional path for a human-readable text report.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"log dir does not exist: {log_dir}")

    result = _diagnose(log_dir, delay=args.delay)
    report = _render_report_text(result)
    print(report)

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2))
        print(f"\n[diagnose] wrote JSON: {out_json}")

    if args.out_report:
        out_report = Path(args.out_report)
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(report + "\n")
        print(f"[diagnose] wrote report: {out_report}")


if __name__ == "__main__":
    main()
