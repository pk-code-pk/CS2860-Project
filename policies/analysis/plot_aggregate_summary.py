"""
Bar charts from a pooled ``aggregate_summary.csv`` (e.g. intent-grounded
overnight analysis: columns ``suite``, ``method``, ``regime``, metrics).

For each ``suite`` value, one subplot: grouped bars over ``method`` and
``regime`` (side-by-side within each method group). Missing cells are
omitted (no bar). Error bars use the matching ``*_sd`` column when
``--metric`` ends in ``_mean``.

Usage::

    uv run python -m policies.analysis.plot_aggregate_summary \\
        --csv matrix_results/intent_grounded_v1_analysis/aggregate_summary.csv \\
        --out matrix_results/intent_grounded_v1_analysis/figures/pooled_by_suite.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Consistent colors across all panels (clarifies baseline vs dropout).
_REGIME_COLOR: dict[str, str] = {
    "baseline": "#2E5F8C",
    "dropout-only": "#C45C3A",
}
# Fallback colors for unexpected regime names
_FALLBACK = ["#6A994E", "#BC4749", "#6C757D", "#9B59B6", "#E2A100"]


def _color_for_regime(rg: str, j: int) -> str:
    return _REGIME_COLOR.get(rg, _FALLBACK[j % len(_FALLBACK)])


def _error_col_for(metric: str) -> str | None:
    if metric.endswith("_mean"):
        return metric[: -len("_mean")] + "sd"
    return None


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _float(row: dict[str, Any], key: str) -> float | None:
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _method_tick(mu: str, short: bool) -> str:
    if not short:
        return mu
    if mu == "mappo-no-comm":
        return "no-comm"
    if mu == "mappo-comm":
        return "comm"
    if mu == "mappo-intent-aux":
        return "intent-aux"
    return mu


def _metric_label(metric: str) -> str:
    return metric.replace("_", " ")


def _plot(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    method_order: list[str],
    regime_order: list[str],
    suites: list[str] | None,
    suptitle: str | None,
    short_method_labels: bool,
    figsize_row: tuple[float, float],
    layout: str,
    sharey: bool,
) -> plt.Figure:
    err_key = _error_col_for(metric)

    by_su: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        s = row.get("suite", "")
        if not s:
            continue
        by_su.setdefault(str(s), []).append(row)

    suite_keys = sorted(by_su.keys())
    if suites is not None:
        suite_keys = [s for s in suites if s in by_su] or suite_keys

    n = max(1, len(suite_keys))
    w, h_row = figsize_row
    h_total = h_row * (1.15 if n > 1 and layout == "rows" else 1.0)

    if layout == "cols":
        ncols, nrows = n, 1
    else:
        nrows, ncols = n, 1

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(w * ncols, h_row * nrows) if layout == "cols" else (w, h_total),
        squeeze=False,
        sharey=sharey,
        constrained_layout=False,
    )
    ax_list: list[plt.Axes] = [axes.flat[i] for i in range(n)]

    # All regime names in the CSV, stable order (for legend + colors)
    all_reg: list[str] = []
    _seen: set[str] = set()
    for r in regime_order:
        if r in _seen:
            continue
        if any(
            str(row.get("regime")) == r
            for sk in suite_keys
            for row in by_su[sk]
        ):
            all_reg.append(r)
            _seen.add(r)
    for sk in suite_keys:
        for row in by_su[sk]:
            r = str(row.get("regime", ""))
            if r and r not in _seen:
                all_reg.append(r)
                _seen.add(r)

    used_regimes_in_plot: set[str] = set()

    for ax, sk in zip(ax_list, suite_keys):
        sub = by_su[sk]
        idx: dict[tuple[str, str], dict[str, Any]] = {}
        for row in sub:
            m, rg = str(row.get("method", "")), str(row.get("regime", ""))
            if m and rg:
                idx[(m, rg)] = row

        reg_avail = {str(r.get("regime", "")) for r in sub if r.get("regime")}
        reg_list = [r for r in regime_order if r in reg_avail] + sorted(
            reg_avail - set(regime_order)
        )

        methods_use: list[str] = []
        for m in method_order:
            if any((m, rg) in idx for rg in reg_list):
                methods_use.append(m)
        for m in method_order:
            if m not in methods_use and any((m, str(rg)) in idx for rg in reg_list):
                methods_use.append(m)

        if not methods_use or not reg_list:
            ax.set_title(f"{sk}  (no data for chosen filters)")
            ax.set_axis_off()
            continue

        x0 = np.arange(len(methods_use), dtype=float)
        nbr = len(reg_list)
        width = min(0.78 / max(1, nbr), 0.32)
        offset = (np.arange(nbr, dtype=float) - (nbr - 1) / 2.0) * width

        for j, rg in enumerate(reg_list):
            try:
                pal_idx = all_reg.index(rg)
            except ValueError:
                pal_idx = j
            color = _color_for_regime(rg, pal_idx)
            used_regimes_in_plot.add(rg)

            ys: list[float] = []
            yerr: list[float] = []
            for mu in methods_use:
                row = idx.get((mu, rg))
                if row is None:
                    ys.append(0.0)
                    yerr.append(0.0)
                else:
                    yv = _float(row, metric)
                    ys.append(0.0 if yv is None else yv)
                    if err_key and row is not None:
                        ev = _float(row, err_key)
                        yerr.append(0.0 if ev is None or ev < 0 else ev)
                    else:
                        yerr.append(0.0)

            pos = x0 + offset[j]
            mask = [(mu, rg) in idx for mu in methods_use]
            if not any(mask):
                continue
            ax.bar(
                pos[mask],
                [ys[i] for i, m in enumerate(mask) if m],
                width,
                yerr=[yerr[i] for i, m in enumerate(mask) if m] if err_key else None,
                ecolor="#333",
                capsize=2.5,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                label=rg,
            )

        tick = [_method_tick(mu, short_method_labels) for mu in methods_use]
        ax.set_xticks(x0)
        ax.set_xticklabels(
            tick,
            rotation=0,
            ha="center",
            fontsize=10,
        )
        ax.margins(x=0.08)
        st = sk
        if len(reg_list) == 1:
            st = f"{sk}  (only: {reg_list[0]})"
        ax.set_title(st, fontsize=11, pad=6, fontweight="semibold")
        ax.grid(axis="y", alpha=0.32, zorder=0, linestyle="--", linewidth=0.8)
        ax.set_axisbelow(True)

    for ax in ax_list[len(suite_keys) :]:
        ax.set_axis_off()
        ax.set_visible(False)

    ytxt = _metric_label(metric) if not err_key else f"{_metric_label(metric)} (mean ± s.d.)"
    if hasattr(fig, "supylabel"):
        fig.supylabel(
            ytxt,
            x=0.04,
            fontsize=10.5,
            fontweight="medium",
        )
    else:
        ax_list[0].set_ylabel(ytxt, fontsize=10.5, fontweight="medium")

    # One legend for the whole figure (no overlap with first panel)
    if used_regimes_in_plot:
        handles = [
            mpatches.Patch(
                color=_color_for_regime(
                    r, all_reg.index(r) if r in all_reg else 0
                ),
                label=r,
            )
            for r in all_reg
            if r in used_regimes_in_plot
        ]
        if handles:
            fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=min(3, len(handles)),
                frameon=True,
                fancybox=True,
                fontsize=9,
                title="regime",
                title_fontsize=9,
            )

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="semibold", y=0.985)

    fig.tight_layout(rect=(0, 0.1, 1, 0.95 if suptitle else 0.98))
    return fig


def _parse_list(s: str | None) -> list[str] | None:
    if not s or not s.strip():
        return None
    return [p.strip() for p in s.split(",") if p.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Bar plots from aggregate_summary.csv")
    p.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to aggregate_summary.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: <csv_dir>/aggregate_summary_bars.png)",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="eval_last5_mean",
        help="Numeric column to plot (must exist in the CSV).",
    )
    p.add_argument(
        "--method-order",
        type=str,
        default="mappo-no-comm,mappo-comm,mappo-intent-aux",
        help="Comma-separated method order on the x axis.",
    )
    p.add_argument(
        "--regime-order",
        type=str,
        default="baseline,dropout-only",
        help="Comma-separated regime order (legend / bar sub-groups).",
    )
    p.add_argument(
        "--suites",
        type=str,
        default=None,
        help="If set, comma-separated suite filter and order (e.g. main,stress_t50).",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure suptitle.",
    )
    p.add_argument(
        "--short-labels",
        action="store_true",
        help="Shorter x tick labels for the three MAPPO ablations.",
    )
    p.add_argument(
        "--figsize",
        type=str,
        default="10,3.2",
        help="Base width, height in inches of one panel (height is per row in stacked layout).",
    )
    p.add_argument(
        "--layout",
        choices=("rows", "cols"),
        default="rows",
        help="rows: one subplot per suite, stacked. cols: side-by-side in one row.",
    )
    p.add_argument(
        "--sharey",
        action="store_true",
        help="Share y across subplots (same scale for all suites; can hide small-difference details).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI.",
    )
    args = p.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    w, h = (float(x.strip()) for x in args.figsize.split(",")[:2])

    rows = _load_rows(args.csv)
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")
    if args.metric not in rows[0]:
        raise SystemExit(
            f"column {args.metric!r} not in CSV; first columns: {list(rows[0].keys())[:12]}..."
        )

    out = args.out
    if out is None:
        out = args.csv.parent / "aggregate_summary_bars.png"

    fig = _plot(
        rows,
        metric=args.metric,
        method_order=_parse_list(args.method_order) or [],
        regime_order=_parse_list(args.regime_order) or [],
        suites=_parse_list(args.suites),
        suptitle=args.title,
        short_method_labels=args.short_labels,
        figsize_row=(w, h),
        layout=args.layout,
        sharey=args.sharey,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"[plot_aggregate_summary] wrote {out}")


if __name__ == "__main__":
    main()
