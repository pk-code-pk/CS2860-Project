"""
Final-figure plotter for the RWARE dropout/heartbeat/comm experiments.

Reads ``per_run.csv`` and ``summary.csv`` produced by
``policies.analysis.aggregate`` and emits the five mandatory figures:

  1. ``team_return_vs_delay.png``         -- team return vs heartbeat delay
                                              for every method.
  2. ``ep_length_vs_delay.png``           -- episode length (proxy for
                                              throughput / completion) vs delay.
  3. ``post_dropout_methods.png``         -- bar chart of final return per
                                              method in dropout-only.
  4. ``ambiguous_regime_methods.png``     -- bar chart of final return per
                                              method in delay-dropout (the
                                              "ambiguous" regime).
  5. ``comm_ablation.png``                -- mappo-no-comm vs
                                              mappo-heartbeat-only vs
                                              mappo-heartbeat-plus-comm,
                                              one bar group per regime.

If the aggregated CSV is missing a particular cell (e.g. no runs for a
method/regime pair) the plotter gracefully skips it and notes the gap in
the figure's title.

Usage::

    uv run python -m policies.analysis.plot_results \\
        --in-dir runs/exp_matrix --out-dir runs/exp_matrix/figures
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe; we only ever save to file.
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


@dataclass
class SummaryRow:
    method: str
    regime: str
    delay: int
    n_seeds: int
    final_return_mean: float
    final_return_std: float
    ep_length_mean: float
    ep_length_std: float


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v


def _load_summary(path: Path) -> list[SummaryRow]:
    if not path.exists():
        raise SystemExit(
            f"summary CSV not found: {path}\n"
            f"Run `python -m policies.analysis.aggregate --log-dir <log-dir>` first."
        )
    rows: list[SummaryRow] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                SummaryRow(
                    method=r["method"],
                    regime=r["regime"],
                    delay=int(r["delay"]),
                    n_seeds=int(r.get("n_seeds") or 0),
                    final_return_mean=_safe_float(r.get("final_return_mean")),
                    final_return_std=_safe_float(r.get("final_return_std"), 0.0),
                    ep_length_mean=_safe_float(r.get("ep_length_mean")),
                    ep_length_std=_safe_float(r.get("ep_length_std"), 0.0),
                )
            )
    return rows


# --------------------------------------------------------------------------
# Method styling
# --------------------------------------------------------------------------


_METHOD_ORDER = [
    "heuristic",
    "mappo-no-comm",
    "mappo-heartbeat-only",
    "mappo-heartbeat-plus-comm",
]
_METHOD_COLORS = {
    "heuristic":                  "#7f7f7f",
    "mappo-no-comm":              "#1f77b4",
    "mappo-heartbeat-only":       "#ff7f0e",
    "mappo-heartbeat-plus-comm":  "#2ca02c",
}
_METHOD_MARKERS = {
    "heuristic":                  "s",
    "mappo-no-comm":              "o",
    "mappo-heartbeat-only":       "^",
    "mappo-heartbeat-plus-comm":  "D",
}
_REGIME_ORDER = ["baseline", "delay-only", "dropout-only", "delay-dropout"]


def _method_style(method: str, idx: int) -> dict[str, Any]:
    return {
        "color": _METHOD_COLORS.get(method, f"C{idx}"),
        "marker": _METHOD_MARKERS.get(method, "o"),
    }


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------


def _ensure_outdir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {path}")


def _filter(rows: list[SummaryRow], **filters: Any) -> list[SummaryRow]:
    out = []
    for r in rows:
        ok = True
        for k, v in filters.items():
            if getattr(r, k) != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def _by_method_then_x(
    rows: list[SummaryRow],
    x_attr: str,
    y_attr: str,
    yerr_attr: str | None = None,
) -> dict[str, list[tuple[float, float, float]]]:
    """Return {method: [(x, y, yerr)]} sorted by x."""
    out: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for r in rows:
        x = getattr(r, x_attr)
        y = getattr(r, y_attr)
        if np.isnan(y):
            continue
        yerr = getattr(r, yerr_attr) if yerr_attr else 0.0
        if np.isnan(yerr):
            yerr = 0.0
        out[r.method].append((float(x), float(y), float(yerr)))
    for m in out:
        out[m].sort(key=lambda t: t[0])
    return out


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def plot_team_return_vs_delay(
    rows: list[SummaryRow], out_dir: Path, env_label: str
) -> None:
    """
    Team return vs heartbeat delay.

    We only plot the regimes whose delay axis is meaningful
    (delay-only and delay-dropout), one panel each, with all methods
    overlaid in each panel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    panels = [("delay-only", axes[0]), ("delay-dropout", axes[1])]
    for regime, ax in panels:
        sub = _filter(rows, regime=regime)
        series = _by_method_then_x(sub, "delay", "final_return_mean", "final_return_std")
        if not series:
            ax.set_title(f"{regime}\n(no data)")
        else:
            ax.set_title(f"regime: {regime}")
        for i, method in enumerate(_METHOD_ORDER):
            pts = series.get(method)
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            errs = [p[2] for p in pts]
            ax.errorbar(
                xs,
                ys,
                yerr=errs,
                label=method,
                capsize=3,
                **_method_style(method, i),
            )
        ax.set_xlabel("heartbeat delay (steps)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("final eval team return (mean over seeds)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4,
                   bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Team return vs heartbeat delay  ({env_label})")
    fig.subplots_adjust(bottom=0.2)
    _save(fig, out_dir / "team_return_vs_delay.png")


def plot_ep_length_vs_delay(
    rows: list[SummaryRow], out_dir: Path, env_label: str
) -> None:
    """
    Episode length vs delay. In RWARE we use a fixed step budget (env-side
    truncation) so a *shorter* mean ep_length under the same number of
    deliveries means higher throughput; we surface both panels for clarity.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sub = [r for r in rows if r.regime in ("delay-only", "delay-dropout")]
    series = _by_method_then_x(sub, "delay", "ep_length_mean", "ep_length_std")
    for i, method in enumerate(_METHOD_ORDER):
        pts = series.get(method)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        errs = [p[2] for p in pts]
        ax.errorbar(
            xs, ys, yerr=errs, label=method, capsize=3,
            **_method_style(method, i),
        )
    ax.set_xlabel("heartbeat delay (steps)")
    ax.set_ylabel("mean episode length (steps)")
    ax.set_title(
        f"Episode length vs heartbeat delay  ({env_label})\n"
        "delay-bearing regimes only; lower ~ faster completion"
    )
    ax.grid(True, alpha=0.3)
    if ax.has_data():
        ax.legend(loc="best", ncol=2)
    _save(fig, out_dir / "ep_length_vs_delay.png")


def _bar_chart_by_method(
    rows: list[SummaryRow],
    out_path: Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    methods = [m for m in _METHOD_ORDER if any(r.method == m for r in rows)]
    if not methods:
        print(f"[plot] skip (no data): {out_path.name}")
        return
    means = []
    stds = []
    counts = []
    for m in methods:
        sub = [r for r in rows if r.method == m]
        # If a method appears at multiple delays in the same regime (it should
        # not, but be safe), pool by simple average over delays first.
        per_delay = defaultdict(list)
        for r in sub:
            per_delay[r.delay].append(r.final_return_mean)
        flat = [v for vals in per_delay.values() for v in vals if not np.isnan(v)]
        if not flat:
            means.append(np.nan)
            stds.append(0.0)
            counts.append(0)
        else:
            means.append(float(np.mean(flat)))
            stds.append(float(np.std(flat)))
            counts.append(len(flat))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(methods))
    bars = ax.bar(
        xs,
        means,
        yerr=stds,
        capsize=4,
        color=[_METHOD_COLORS.get(m, f"C{i}") for i, m in enumerate(methods)],
    )
    for x, n, mean in zip(xs, counts, means):
        if not np.isnan(mean):
            ax.text(x, mean, f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_path)


def plot_post_dropout(rows: list[SummaryRow], out_dir: Path, env_label: str) -> None:
    sub = _filter(rows, regime="dropout-only")
    _bar_chart_by_method(
        sub,
        out_dir / "post_dropout_methods.png",
        title=f"Post-dropout team return  ({env_label}, regime=dropout-only)",
        ylabel="final eval return (mean over seeds)",
    )


def plot_ambiguous_regime(
    rows: list[SummaryRow], out_dir: Path, env_label: str
) -> None:
    sub = _filter(rows, regime="delay-dropout")
    _bar_chart_by_method(
        sub,
        out_dir / "ambiguous_regime_methods.png",
        title=(
            f"Ambiguous regime (delay + dropout)  ({env_label})\n"
            "stale signals + true dropout coexist; comm should help most here"
        ),
        ylabel="final eval return (pooled across delays, mean over seeds)",
    )


def plot_comm_ablation(rows: list[SummaryRow], out_dir: Path, env_label: str) -> None:
    """
    Grouped bar chart: rows = regimes, groups = the three mappo variants.
    """
    methods = [
        "mappo-no-comm",
        "mappo-heartbeat-only",
        "mappo-heartbeat-plus-comm",
    ]
    regimes = [r for r in _REGIME_ORDER if any(row.regime == r for row in rows)]
    if not regimes:
        print("[plot] skip comm_ablation (no data)")
        return

    means = np.full((len(regimes), len(methods)), np.nan)
    stds = np.zeros_like(means)
    for i, regime in enumerate(regimes):
        for j, method in enumerate(methods):
            sub = [r for r in rows if r.regime == regime and r.method == method
                   and not np.isnan(r.final_return_mean)]
            if not sub:
                continue
            vals = [r.final_return_mean for r in sub]
            errs = [r.final_return_std for r in sub if not np.isnan(r.final_return_std)]
            means[i, j] = float(np.mean(vals))
            stds[i, j] = float(np.mean(errs)) if errs else 0.0

    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.25
    xs = np.arange(len(regimes))
    for j, method in enumerate(methods):
        offsets = xs + (j - 1) * width
        ax.bar(
            offsets,
            means[:, j],
            width=width,
            yerr=stds[:, j],
            capsize=3,
            label=method,
            color=_METHOD_COLORS.get(method, f"C{j}"),
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(regimes, rotation=10, ha="right")
    ax.set_ylabel("final eval return (mean over delays + seeds)")
    ax.set_title(
        f"Communication ablation across regimes  ({env_label})\n"
        "no-comm vs heartbeat-only vs heartbeat+comm"
    )
    ax.grid(True, axis="y", alpha=0.3)
    if ax.has_data():
        ax.legend(loc="best")
    _save(fig, out_dir / "comm_ablation.png")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot the RWARE matrix-runner outputs.")
    p.add_argument("--in-dir", default="runs/exp_matrix",
                   help="Directory containing summary.csv + per_run.csv.")
    p.add_argument("--out-dir", default=None,
                   help="Where to write the PNGs (defaults to <in-dir>/figures).")
    p.add_argument("--env-label", default=None,
                   help="Override the env label printed in figure titles.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (in_dir / "figures")
    _ensure_outdir(out_dir)

    summary_path = in_dir / "summary.csv"
    rows = _load_summary(summary_path)
    print(f"[plot] loaded {len(rows)} summary rows from {summary_path}")
    if not rows:
        print("[plot] no rows; nothing to plot")
        return

    env_label = args.env_label or _infer_env_label(in_dir, rows)

    plot_team_return_vs_delay(rows, out_dir, env_label)
    plot_ep_length_vs_delay(rows, out_dir, env_label)
    plot_post_dropout(rows, out_dir, env_label)
    plot_ambiguous_regime(rows, out_dir, env_label)
    plot_comm_ablation(rows, out_dir, env_label)
    print(f"[plot] all figures under {out_dir}")


def _infer_env_label(in_dir: Path, rows: list[SummaryRow]) -> str:
    """Try ``per_run.csv`` first; fall back to a generic label."""
    per_run = in_dir / "per_run.csv"
    if per_run.exists():
        with per_run.open() as f:
            for r in csv.DictReader(f):
                env = r.get("env") or ""
                if env:
                    return env
    return "rware"


if __name__ == "__main__":
    main()
