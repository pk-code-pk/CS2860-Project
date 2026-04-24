"""
Single-pilot dashboard for the RWARE dropout/heartbeat/comm experiments.

Reads the per-cell ``metrics.csv`` files emitted by
``policies.experiments.run_rware_matrix`` and produces a multi-panel
PNG that lets a human actually understand a pilot at a glance:

  * Top row    -- per-update *train* return curves, one subplot per regime,
                  one line per (method, seed) lightly drawn, with a thick
                  per-method mean +/- std band on top.
  * Middle row -- per-update *eval* return curves with the same layout.
  * Bottom row -- per-cell final-eval distribution: per-seed scatter dots
                  on top of a per-method bar (mean) with std error-bars.
                  Welch's t-test p-value annotated on each adjacent bar
                  pair (heartbeat-only vs heartbeat+comm) within the same
                  regime.

The headline gap (``comm-advantage delay-dropout - comm-advantage delay-only``)
is printed as a sub-title on the figure so you do not have to do mental
arithmetic.

Why this exists
---------------

The aggregator-driven plotter in ``policies.analysis.plot_results`` is
designed for the *full* matrix and produces summary bar charts only.
For pilot-sized sweeps (4 cells x N seeds) we need to *see* the noise
to interpret the result, which means: per-seed traces, training
trajectories, and explicit confidence quantification. A single 8x9-inch
figure beats reading three CSV tables in your head.

Usage
-----

::

    uv run python -m policies.analysis.pilot_dashboard \\
        --log-dir runs/exp_pilot_v3 \\
        --out runs/exp_pilot_v3/dashboard.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Same schema as policies.analysis.aggregate.parse_run_name, copied to
# avoid a cross-module import dependency on the aggregator's PerRunRow
# (we want raw curves, not just last-K reductions).
_RUN_NAME_RE = re.compile(
    r"^(?P<method>.+?)__(?P<regime>[a-z0-9-]+)__d(?P<delay>-?\d+)__s(?P<seed>-?\d+)$"
)


@dataclass(frozen=True)
class CellKey:
    method: str
    regime: str
    delay: int
    seed: int


def _safe_float(x) -> float | None:
    if x in (None, "", "nan"):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _read_curves(path: Path) -> dict[str, np.ndarray]:
    """Return arrays of ``step, train_return, eval_return`` (NaN where missing)."""
    if not path.exists():
        return {"step": np.array([]), "train": np.array([]), "eval": np.array([])}
    steps: list[float] = []
    trains: list[float] = []
    evals: list[float] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            s = _safe_float(row.get("step"))
            t = _safe_float(row.get("train/ep_return_mean"))
            e = _safe_float(row.get("eval/ep_return_mean"))
            steps.append(s if s is not None else float("nan"))
            trains.append(t if t is not None else float("nan"))
            evals.append(e if e is not None else float("nan"))
    return {
        "step": np.asarray(steps),
        "train": np.asarray(trains),
        "eval": np.asarray(evals),
    }


def load_pilot(log_dir: Path) -> dict[CellKey, dict[str, np.ndarray]]:
    out: dict[CellKey, dict[str, np.ndarray]] = {}
    for d in sorted(p for p in log_dir.iterdir() if p.is_dir()):
        m = _RUN_NAME_RE.match(d.name)
        if not m:
            continue
        key = CellKey(
            method=m["method"],
            regime=m["regime"],
            delay=int(m["delay"]),
            seed=int(m["seed"]),
        )
        out[key] = _read_curves(d / "metrics.csv")
    return out


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _last_mean(arr: np.ndarray, k: int) -> float:
    """Mean of the last ``k`` non-NaN values; NaN if none."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite[-k:]))


def _welch_t_p(a: list[float], b: list[float]) -> tuple[float, float]:
    """
    Two-sample Welch's t-test (unequal variances). Returns ``(t, p_two_sided)``.

    Delegates to ``scipy.stats.ttest_ind(equal_var=False)`` for an exact
    Student-t survival function. We previously rolled a normal-approximation
    of the t tail to avoid the scipy dep, but at the n=3-per-group sample
    sizes that pilot runs use, the approximation under-stated p-values by
    ~5x in the worst case (e.g. it labelled a 0.146 result as 0.022) and
    flipped the visible '*' / 'ns' label. ``policies.analysis.verify_dashboard``
    catches any future regression of this kind by comparing every drawn
    p-value to scipy directly.
    """
    a = [float(x) for x in a if math.isfinite(float(x))]
    b = [float(x) for x in b if math.isfinite(float(x))]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    if np.var(a, ddof=1) == 0 and np.var(b, ddof=1) == 0:
        return float("nan"), float("nan")
    from scipy import stats
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


# Method order + colours: keep heartbeat-only and heartbeat-plus-comm
# adjacent in every panel so the gap is visually obvious.
_METHOD_ORDER = (
    "mappo-no-comm",
    "mappo-heartbeat-only",
    "mappo-heartbeat-plus-comm",
    "heuristic",
)
_METHOD_COLOURS = {
    "mappo-no-comm": "#888888",
    "mappo-heartbeat-only": "#1f77b4",
    "mappo-heartbeat-plus-comm": "#d62728",
    "heuristic": "#2ca02c",
}
_METHOD_SHORT = {
    "mappo-no-comm": "no-comm",
    "mappo-heartbeat-only": "hb-only",
    "mappo-heartbeat-plus-comm": "hb+comm",
    "heuristic": "heur",
}


def _present_methods(cells: dict[CellKey, dict]) -> list[str]:
    seen = {k.method for k in cells}
    return [m for m in _METHOD_ORDER if m in seen]


def _present_regimes(cells: dict[CellKey, dict]) -> list[str]:
    order = ["baseline", "delay-only", "dropout-only", "delay-dropout"]
    seen = {k.regime for k in cells}
    return [r for r in order if r in seen]


def _stack_curves(
    cells: dict[CellKey, dict],
    method: str,
    regime: str,
    delay: int,
    column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect per-seed curves for one cell into a 2-D array (n_seeds, n_steps),
    NaN-padding to the max length. Also returns the matching step axis
    (taken from the longest seed).
    """
    seed_arrs: list[tuple[np.ndarray, np.ndarray]] = []
    for k, curves in cells.items():
        if k.method != method or k.regime != regime or k.delay != delay:
            continue
        seed_arrs.append((curves["step"], curves[column]))
    if not seed_arrs:
        return np.empty((0, 0)), np.empty((0,))
    L = max(len(c) for _, c in seed_arrs)
    step = max((s for s, _ in seed_arrs), key=len)
    M = np.full((len(seed_arrs), L), np.nan, dtype=float)
    for i, (_, c) in enumerate(seed_arrs):
        M[i, : len(c)] = c
    return M, step


def _plot_curves_panel(
    ax,
    cells: dict[CellKey, dict],
    regime: str,
    delay: int,
    column: str,
    *,
    smooth: int,
    title: str,
    ylabel: str,
):
    methods = _present_methods(cells)
    plotted = False
    for method in methods:
        M, step = _stack_curves(cells, method, regime, delay, column)
        if M.size == 0 or step.size == 0:
            continue
        x = step[: M.shape[1]]
        # Light per-seed traces.
        for row in M:
            sm = _moving_average(row, smooth)
            ax.plot(
                x[: len(sm)], sm,
                color=_METHOD_COLOURS.get(method, "k"),
                alpha=0.18, linewidth=0.7,
            )
        # Mean +/- std band over seeds, ignoring NaN columns (column has 0
        # finite values gives NaN which matplotlib hides cleanly).
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(M, axis=0)
            std = np.nanstd(M, axis=0)
        sm_mean = _moving_average(mean, smooth)
        sm_std = _moving_average(std, smooth)
        L = len(sm_mean)
        ax.plot(
            x[:L], sm_mean,
            color=_METHOD_COLOURS.get(method, "k"),
            linewidth=1.8,
            label=_METHOD_SHORT.get(method, method),
        )
        ax.fill_between(
            x[:L], sm_mean - sm_std, sm_mean + sm_std,
            color=_METHOD_COLOURS.get(method, "k"),
            alpha=0.12,
        )
        plotted = True
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("env steps", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.25, linestyle=":")
    if plotted:
        ax.legend(fontsize=7, loc="best")


def _moving_average(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or arr.size == 0:
        return arr
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    # Fallback: replace NaNs with previous finite value for the EMA;
    # then re-mask leading NaNs to keep the curve start clean.
    filled = arr.copy()
    last = float("nan")
    for i in range(len(filled)):
        if math.isnan(filled[i]):
            filled[i] = last
        else:
            last = filled[i]
    pad = np.concatenate([[filled[0]] * (k - 1), filled])
    out = np.convolve(pad, np.ones(k) / k, mode="valid")
    # Restore NaNs at the very start where there were no finite values yet.
    if not finite[0]:
        first_finite = next((i for i, v in enumerate(finite) if v), len(arr))
        out[:first_finite] = np.nan
    return out


# ---------------------------------------------------------------------------
# Bottom-row summary panel
# ---------------------------------------------------------------------------


def _final_eval_per_seed(
    cells: dict[CellKey, dict], method: str, regime: str, delay: int, last_k: int
) -> list[float]:
    out: list[float] = []
    for k, curves in cells.items():
        if k.method != method or k.regime != regime or k.delay != delay:
            continue
        v = _last_mean(curves["eval"], last_k)
        if math.isfinite(v):
            out.append(v)
    return out


def _plot_summary_panel(
    ax,
    cells: dict[CellKey, dict],
    regime: str,
    delay: int,
    *,
    last_k: int,
    title: str,
):
    methods = _present_methods(cells)
    if not methods:
        ax.set_title(f"{title} (no data)", fontsize=10)
        return

    xs = np.arange(len(methods), dtype=float)
    means: list[float] = []
    stds: list[float] = []
    per_seed: list[list[float]] = []
    for m in methods:
        vals = _final_eval_per_seed(cells, m, regime, delay, last_k)
        per_seed.append(vals)
        if vals:
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0))
        else:
            means.append(float("nan"))
            stds.append(0.0)

    bars = ax.bar(
        xs, means, yerr=stds,
        capsize=4,
        color=[_METHOD_COLOURS.get(m, "k") for m in methods],
        alpha=0.55,
    )
    # Per-seed dots in a tight x-jitter so you can count the seeds.
    rng = np.random.default_rng(0)
    for i, vals in enumerate(per_seed):
        if not vals:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            xs[i] + jitter, vals,
            color="black", s=18, zorder=3,
            edgecolor="white", linewidth=0.5,
        )

    # Welch's t-test annotation between heartbeat-only vs heartbeat-plus-comm
    # if both are present in this cell.
    try:
        i_hb = methods.index("mappo-heartbeat-only")
        i_co = methods.index("mappo-heartbeat-plus-comm")
        t, p = _welch_t_p(per_seed[i_hb], per_seed[i_co])
        gap = means[i_co] - means[i_hb]
        if math.isfinite(p):
            stars = (
                "***" if p < 0.001 else
                "**"  if p < 0.01  else
                "*"   if p < 0.05  else
                "ns"
            )
            top = max(
                m + s for m, s in zip(means, stds) if math.isfinite(m)
            )
            y = top * 1.10 if top > 0 else 1.0
            ax.plot(
                [xs[i_hb], xs[i_co]], [y, y],
                color="k", linewidth=0.8,
            )
            ax.text(
                (xs[i_hb] + xs[i_co]) / 2, y * 1.02,
                f"$\\Delta$={gap:+.1f}  p={p:.2f} {stars}",
                ha="center", fontsize=7,
            )
    except ValueError:
        pass

    ax.set_xticks(xs)
    ax.set_xticklabels([_METHOD_SHORT.get(m, m) for m in methods], fontsize=8)
    ax.set_ylabel("eval return (mean of last K)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")


# ---------------------------------------------------------------------------
# Top-line summary text
# ---------------------------------------------------------------------------


def _format_headline(cells: dict[CellKey, dict], delay: int, last_k: int) -> str:
    """
    Build a single-line summary of the comm interaction effect:
    ``(comm benefit in delay-dropout) - (comm benefit in delay-only)``.
    """
    def gap(regime: str) -> tuple[float, float, int, int]:
        a = _final_eval_per_seed(
            cells, "mappo-heartbeat-only", regime, delay, last_k
        )
        b = _final_eval_per_seed(
            cells, "mappo-heartbeat-plus-comm", regime, delay, last_k
        )
        if not a or not b:
            return float("nan"), float("nan"), len(a), len(b)
        ma = float(np.mean(a))
        mb = float(np.mean(b))
        return mb - ma, float(np.std(a + b, ddof=1) if len(a + b) > 1 else 0.0), len(a), len(b)

    g_only, _, na_o, nb_o = gap("delay-only")
    g_drop, _, na_d, nb_d = gap("delay-dropout")
    # Use compact two-line layout so the headline never overflows the figure.
    line1_bits = []
    if math.isfinite(g_only):
        line1_bits.append(f"Δcomm (delay-only)={g_only:+.1f} (n={na_o})")
    if math.isfinite(g_drop):
        line1_bits.append(f"Δcomm (delay-dropout)={g_drop:+.1f} (n={na_d})")
    line1 = "    ".join(line1_bits)
    line2 = (
        f"interaction (drop − only) = {(g_drop - g_only):+.1f}"
        if math.isfinite(g_only) and math.isfinite(g_drop)
        else ""
    )
    return f"{line1}\n{line2}".strip()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def make_dashboard(
    cells: dict[CellKey, dict],
    *,
    out_path: Path,
    title: str,
    last_k: int,
    smooth: int,
):
    regimes = _present_regimes(cells)
    if not regimes:
        raise SystemExit("no recognisable regimes found; nothing to plot")
    # Use the most common delay across cells for the curve panels; if you
    # have multiple delays in one log-dir, run the plotter once per delay.
    delays = sorted({k.delay for k in cells})
    if len(delays) > 1:
        # Pick the largest non-zero delay as headline, but also note the
        # ambiguity in the suptitle.
        delay = max(delays)
        title += f" [multiple delays present: {delays}; using d={delay}]"
    else:
        delay = delays[0]

    n_regimes = len(regimes)
    fig, axes = plt.subplots(
        nrows=3,
        ncols=n_regimes,
        figsize=(4.0 * n_regimes, 9.0),
        squeeze=False,
    )

    for j, regime in enumerate(regimes):
        _plot_curves_panel(
            axes[0][j],
            cells,
            regime,
            delay,
            "train",
            smooth=smooth,
            title=f"train return  |  {regime}  d={delay}",
            ylabel="train return (shaped)" if j == 0 else "",
        )
        _plot_curves_panel(
            axes[1][j],
            cells,
            regime,
            delay,
            "eval",
            smooth=smooth,
            title=f"eval return  |  {regime}  d={delay}",
            ylabel="eval return (unshaped)" if j == 0 else "",
        )
        _plot_summary_panel(
            axes[2][j],
            cells,
            regime,
            delay,
            last_k=last_k,
            title=f"final eval (last {last_k})  |  {regime}",
        )

    suptitle = title
    headline = _format_headline(cells, delay, last_k)
    if headline:
        suptitle += "\n" + headline
    # Lower the suptitle font slightly so the (multi-line) headline always
    # fits within the figure width, then reserve vertical space for it.
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[dashboard] wrote {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-pilot dashboard: training curves, eval curves, "
                    "per-cell finals with per-seed dots and a Welch t-test "
                    "annotation."
    )
    p.add_argument("--log-dir", required=True,
                   help="One pilot directory (e.g. runs/exp_pilot_v3).")
    p.add_argument("--out", default=None,
                   help="Output PNG path (default: <log-dir>/dashboard.png).")
    p.add_argument("--title", default=None,
                   help="Figure title; default = log-dir basename.")
    p.add_argument("--last-k", type=int, default=10,
                   help="How many of the trailing eval entries to average for "
                        "the bottom-row 'final eval' panel.")
    p.add_argument("--smooth", type=int, default=10,
                   help="Moving-average window for the curve panels (in "
                        "metrics-csv rows, i.e. PPO updates).")
    p.add_argument("--watch", type=float, default=None, metavar="SECONDS",
                   help="Live-watch mode: re-render the PNG every SECONDS "
                        "seconds until interrupted. Useful while training is "
                        "still in progress -- the trainer writes metrics.csv "
                        "incrementally so partial curves are real. The PNG "
                        "is overwritten in-place; macOS Preview will reload "
                        "automatically when the file changes.")
    return p.parse_args()


def _render_once(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"log dir not found: {log_dir}")
    cells = load_pilot(log_dir)
    if not cells:
        # In watch mode an empty dir is normal early in training; just warn.
        print(f"[dashboard] no pilot cells under {log_dir} yet; skipping render")
        return
    out = Path(args.out) if args.out else log_dir / "dashboard.png"
    title = args.title or f"pilot: {log_dir.name}"
    make_dashboard(cells, out_path=out, title=title,
                   last_k=int(args.last_k), smooth=int(args.smooth))


def main() -> None:
    import time
    args = _parse_args()
    if args.watch is None:
        _render_once(args)
        return
    interval = max(float(args.watch), 1.0)
    print(f"[dashboard] watch mode: re-rendering every {interval:.1f}s "
          f"(Ctrl+C to exit)")
    try:
        while True:
            t0 = time.time()
            _render_once(args)
            elapsed = time.time() - t0
            sleep_for = max(interval - elapsed, 0.0)
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\n[dashboard] watch mode stopped")


if __name__ == "__main__":
    main()
