"""
Cross-pilot comparison plotter.

Reads multiple pilot directories produced by
``policies.experiments.run_rware_matrix`` and produces a single PNG that
visualises how the headline numbers (per-cell final eval, dropout cost,
comm benefit) shift as a function of the heartbeat delay ``D`` across
pilots.

This is the visual equivalent of the table::

    D=5  (v2):  hb-only delay-only=82.8  delay-dropout=105.5   comm-effect=...
    D=30 (v3):  hb-only delay-only=99.2  delay-dropout= 84.4   comm-effect=...

Each pilot directory is treated as one "delay setting" -- its delay is
inferred from the cell directory names (``__d{delay}__`` token). If a
pilot contains multiple delays, it is split into one column per delay.

Why this exists
---------------

The single-pilot dashboard tells you what happened *within* one D
setting; this plot tells you whether the predicted *interaction* between
``D`` and "comm helps under dropout" actually shows up in the data.
The headline scientific claim of the project ("comm helps when stale
vs gone is genuinely ambiguous, i.e. when D is large enough") can only
be supported by a curve, not a single point.

Usage
-----

::

    uv run python -m policies.analysis.compare_pilots \\
        --log-dirs runs/exp_pilot_v2 runs/exp_pilot_v3 \\
        --out runs/figures/compare_pilots.png
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from policies.analysis.pilot_dashboard import (
    CellKey,
    _METHOD_COLOURS,
    _METHOD_SHORT,
    _final_eval_per_seed,
    _welch_t_p,
    load_pilot,
)


# ---------------------------------------------------------------------------
# Aggregation across pilots
# ---------------------------------------------------------------------------


@dataclass
class CellStats:
    """Per-(method, regime, delay) cell summary stats across seeds."""
    method: str
    regime: str
    delay: int
    seeds: list[float]   # final-eval values, one per seed
    mean: float
    std: float
    n: int


def _summarise_pilot(
    cells: dict[CellKey, dict],
    *,
    last_k: int,
) -> dict[tuple[str, str, int], CellStats]:
    out: dict[tuple[str, str, int], CellStats] = {}
    methods = sorted({k.method for k in cells})
    regimes = sorted({k.regime for k in cells})
    delays = sorted({k.delay for k in cells})
    for m in methods:
        for r in regimes:
            for d in delays:
                vals = _final_eval_per_seed(cells, m, r, d, last_k)
                if not vals:
                    continue
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                out[(m, r, d)] = CellStats(
                    method=m, regime=r, delay=d, seeds=vals,
                    mean=mean, std=std, n=len(vals),
                )
    return out


def _gather_all(
    log_dirs: list[Path], *, last_k: int
) -> dict[tuple[str, str, int], CellStats]:
    """
    Merge per-pilot stats into a single map keyed by (method, regime, delay).
    Later pilots overwrite earlier ones for duplicate cells -- caller is
    responsible for not passing inconsistent overlapping pilots.
    """
    merged: dict[tuple[str, str, int], CellStats] = {}
    for d in log_dirs:
        cells = load_pilot(d)
        if not cells:
            print(f"[compare] WARN: no recognisable cells under {d}; skipping")
            continue
        merged.update(_summarise_pilot(cells, last_k=last_k))
    return merged


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot_per_regime(
    ax,
    stats: dict[tuple[str, str, int], CellStats],
    regime: str,
    delays: list[int],
    methods: list[str],
):
    """
    For one regime: x-axis = delay, lines = method (mean +/- std error
    bars), per-seed dots scattered on each (delay, method) grid point.
    """
    rng = np.random.default_rng(0)
    for m_idx, method in enumerate(methods):
        xs: list[int] = []
        ms: list[float] = []
        ss: list[float] = []
        all_seeds: list[tuple[int, float]] = []
        for d in delays:
            cs = stats.get((method, regime, d))
            if cs is None:
                continue
            xs.append(d)
            ms.append(cs.mean)
            ss.append(cs.std)
            all_seeds.extend((d, v) for v in cs.seeds)
        if not xs:
            continue
        ax.errorbar(
            xs, ms, yerr=ss,
            color=_METHOD_COLOURS.get(method, "k"),
            linewidth=1.6, marker="o", markersize=6,
            capsize=4,
            label=_METHOD_SHORT.get(method, method),
        )
        # Per-seed dots, jittered horizontally per method to disambiguate.
        x_jit = (m_idx - (len(methods) - 1) / 2) * 0.6
        for d, v in all_seeds:
            jitter = rng.uniform(-0.25, 0.25)
            ax.scatter(
                d + x_jit + jitter, v,
                color=_METHOD_COLOURS.get(method, "k"),
                alpha=0.5, s=18, edgecolor="white", linewidth=0.5,
            )
    ax.set_title(f"{regime}", fontsize=10)
    ax.set_xlabel("heartbeat delay D", fontsize=8)
    ax.set_ylabel("eval return (mean of last K)", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(fontsize=7, loc="best")


def _plot_comm_benefit_curve(
    ax,
    stats: dict[tuple[str, str, int], CellStats],
    delays: list[int],
):
    """
    Right panel: comm benefit (mean +/- propagated stdev) vs D, with one
    line per regime. Dotted horizontal line at zero. Welch t-test p-value
    annotated above each point.
    """
    regime_styles = {
        "delay-only":    ("#1f77b4", "o", "delay-only"),
        "delay-dropout": ("#d62728", "s", "delay-dropout"),
    }
    for regime, (colour, marker, label) in regime_styles.items():
        xs: list[int] = []
        gs: list[float] = []
        es: list[float] = []
        ps: list[float] = []
        for d in delays:
            a = stats.get(("mappo-heartbeat-only", regime, d))
            b = stats.get(("mappo-heartbeat-plus-comm", regime, d))
            if a is None or b is None:
                continue
            gap = b.mean - a.mean
            # Propagate stdev assuming independent groups, dividing by
            # sqrt(n) gives the SEM-of-difference; we plot SEM rather
            # than raw stdev so the eye reads "is this distinguishable
            # from zero?" correctly.
            sem_a = a.std / math.sqrt(max(a.n, 1))
            sem_b = b.std / math.sqrt(max(b.n, 1))
            sem_gap = math.sqrt(sem_a ** 2 + sem_b ** 2)
            _, p = _welch_t_p(a.seeds, b.seeds)
            xs.append(d); gs.append(gap); es.append(sem_gap); ps.append(p)
        if not xs:
            continue
        ax.errorbar(
            xs, gs, yerr=es,
            color=colour, marker=marker, markersize=7,
            linewidth=1.6, capsize=4, label=label,
        )
        # Annotate p-value next to each point; alternate vertical offset
        # by regime so the two curves don't write on top of each other.
        offset_sign = +1 if regime == "delay-only" else -1
        for x, g, e, p in zip(xs, gs, es, ps):
            if not math.isfinite(p):
                continue
            stars = (
                "***" if p < 0.001 else
                "**"  if p < 0.01  else
                "*"   if p < 0.05  else
                "ns"
            )
            ax.annotate(
                f"p={p:.2f} {stars}",
                xy=(x, g),
                xytext=(8, 12 * offset_sign),
                textcoords="offset points",
                ha="left",
                fontsize=7,
                color=colour,
            )

    ax.axhline(0, color="k", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_title(
        "comm benefit (hb+comm − hb-only) vs D    "
        "[positive ⇒ comm helps, negative ⇒ comm hurts]",
        fontsize=10,
    )
    ax.set_xlabel("heartbeat delay D", fontsize=8)
    ax.set_ylabel("Δ eval return (SEM bars across seeds)", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(fontsize=8, loc="best")


def make_compare_figure(
    stats: dict[tuple[str, str, int], CellStats],
    out_path: Path,
    title: str,
):
    methods = sorted({k[0] for k in stats}, key=lambda m: list(_METHOD_COLOURS).index(m)
                     if m in _METHOD_COLOURS else 99)
    regimes = sorted({k[1] for k in stats})
    delays = sorted({k[2] for k in stats})

    # Layout: top row has one subplot per regime; bottom row has a single
    # full-width comm-benefit subplot. We use GridSpec so the bottom row
    # actually spans the full figure width regardless of n_regimes.
    n_regimes = max(len(regimes), 1)
    fig = plt.figure(figsize=(4.8 * n_regimes, 8.0))
    gs = fig.add_gridspec(2, n_regimes, height_ratios=[1.0, 1.1])
    top_axes = [fig.add_subplot(gs[0, j]) for j in range(n_regimes)]
    bottom_ax = fig.add_subplot(gs[1, :])

    for j, regime in enumerate(regimes):
        _plot_per_regime(top_axes[j], stats, regime, delays, methods)
    _plot_comm_benefit_curve(bottom_ax, stats, delays)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-pilot comparison: per-cell finals as a function "
                    "of heartbeat delay D, plus a comm-benefit-vs-D curve."
    )
    p.add_argument("--log-dirs", nargs="+", required=True,
                   help="One or more pilot dirs (e.g. runs/exp_pilot_v2 "
                        "runs/exp_pilot_v3).")
    p.add_argument("--out", required=True,
                   help="Output PNG path.")
    p.add_argument("--title", default="comm benefit vs heartbeat delay D",
                   help="Figure title.")
    p.add_argument("--last-k", type=int, default=10,
                   help="How many trailing eval entries to average per seed.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    log_dirs = [Path(p) for p in args.log_dirs]
    for d in log_dirs:
        if not d.exists():
            raise SystemExit(f"log dir not found: {d}")
    stats = _gather_all(log_dirs, last_k=int(args.last_k))
    if not stats:
        raise SystemExit("no usable cells found in any pilot dir")
    make_compare_figure(stats, Path(args.out), args.title)


if __name__ == "__main__":
    main()
