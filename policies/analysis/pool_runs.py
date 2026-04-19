"""
Pool one or more matrix-runner output directories into a single combined
directory, so cross-lab pooled analysis (e.g. PK + Sam) is a single
``cp -r`` followed by a single dashboard invocation.

Usage
-----

Suppose PK ran::

    runs/exp_pilot_v4/                              # PK's slice
        manifest.json
        mappo-heartbeat-only__delay-only__d30__s0/
        mappo-heartbeat-only__delay-only__d30__s1/
        mappo-heartbeat-only__delay-only__d30__s2/
        ...

and Sam ran the *identical* command but with ``--seeds 3 4 5`` on his
laptop, ending up with::

    /path/to/sam/runs/exp_pilot_v4/
        manifest.json
        mappo-heartbeat-only__delay-only__d30__s3/
        mappo-heartbeat-only__delay-only__d30__s4/
        mappo-heartbeat-only__delay-only__d30__s5/
        ...

Then::

    uv run python -m policies.analysis.pool_runs \\
        --srcs runs/exp_pilot_v4 /path/to/sam/runs/exp_pilot_v4 \\
        --out runs/exp_pilot_v4_pooled

produces a single pooled directory with all 24 per-cell subdirs and a
``pooled_manifest.json`` recording where each cell came from.

Why this exists
---------------

The matrix runner already names per-cell directories
``{method}__{regime}__d{delay}__s{seed}/`` so two labs running disjoint
seed sets cannot collide on a directory level. ``pool_runs`` just
codifies the convention with hard checks:

* refuses to clobber an existing destination cell unless ``--overwrite``
* errors out if two source directories produce the *same* run_name
  (i.e. you accidentally ran with overlapping seeds)
* refuses to pool runs that disagree on critical config (env id,
  methods, regimes, delays, updates, rollout, eval-episodes,
  shape-rewards, dropout-window-{start,end}). This catches the
  "we had different code versions" / "different hyperparams" mistake
  before it pollutes the pooled dashboard.
* writes a ``pooled_manifest.json`` so downstream readers can answer
  "which cell came from which lab?" without having to git-blame.

If you really want to pool runs that disagree on config, pass
``--allow-config-mismatch`` and accept the consequences (a warning is
printed).

Pairs cleanly with the existing analysis tools::

    uv run python -m policies.analysis.pilot_dashboard \\
        --log-dir runs/exp_pilot_v4_pooled \\
        --out runs/exp_pilot_v4_pooled/dashboard.png \\
        --title "v4 pooled (PK seeds 0-2 + Sam seeds 3-5, n=6)"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Manifest fields whose values must be identical across all source
# directories (because they materially affect the per-cell numbers).
# Anything *not* on this list is allowed to differ between labs --
# notably max_parallel, threads_per_cell, log_dir, capture, save_dir
# and (importantly) seeds.
_CRITICAL_CONFIG_KEYS = (
    "env",
    "methods",
    "regimes",
    "delays",
    "updates",
    "rollout",
    "n_msg_tokens",
    "eval_every",
    "eval_episodes",
    "heartbeat_period",
    "dropout_agent",
    "dropout_time",
    "dropout_window_start",
    "dropout_window_end",
    "shape_rewards",
    "pickup_bonus",
    "step_penalty",
    "stale_threshold",
    "heuristic_episodes",
    "heuristic_max_steps",
)


def _read_manifest(src: Path) -> dict[str, Any]:
    p = src / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(
            f"No manifest.json in {src}. Did you point at a matrix-runner "
            f"output directory? (one level up from the per-cell dirs)"
        )
    return json.loads(p.read_text())


def _list_run_dirs(src: Path) -> list[Path]:
    """Per-cell directories are anything containing a metrics.csv file."""
    out: list[Path] = []
    for child in sorted(src.iterdir()):
        if child.is_dir() and (child / "metrics.csv").exists():
            out.append(child)
    return out


def _check_config_consistency(
    manifests: list[tuple[Path, dict[str, Any]]],
    *,
    allow_mismatch: bool,
) -> list[str]:
    """Return a list of warning strings for any keys that disagree."""
    warnings: list[str] = []
    if len(manifests) < 2:
        return warnings
    base_path, base = manifests[0]
    base_args = base.get("args", {})
    for src_path, m in manifests[1:]:
        m_args = m.get("args", {})
        for key in _CRITICAL_CONFIG_KEYS:
            a = base_args.get(key)
            b = m_args.get(key)
            if a != b:
                msg = (
                    f"config mismatch on '{key}': "
                    f"{base_path.name}={a!r}  vs  {src_path.name}={b!r}"
                )
                warnings.append(msg)
    if warnings and not allow_mismatch:
        for w in warnings:
            print(f"[pool_runs] ERROR: {w}", file=sys.stderr)
        print(
            "[pool_runs] Refusing to pool. Pass --allow-config-mismatch to "
            "override (you almost never want to).",
            file=sys.stderr,
        )
        sys.exit(2)
    elif warnings:
        for w in warnings:
            print(f"[pool_runs] WARN: {w}")
    return warnings


def _copy_cell(src_cell: Path, dst_cell: Path, *, overwrite: bool) -> None:
    if dst_cell.exists():
        if not overwrite:
            raise FileExistsError(
                f"destination cell already exists: {dst_cell}. "
                f"Pass --overwrite to replace it."
            )
        shutil.rmtree(dst_cell)
    shutil.copytree(src_cell, dst_cell)


def pool(
    srcs: list[Path],
    out: Path,
    *,
    overwrite: bool = False,
    allow_config_mismatch: bool = False,
) -> dict[str, Any]:
    """
    Pool ``srcs`` into ``out``. Returns the pooled-manifest dict.
    """
    if not srcs:
        raise ValueError("Need at least one source directory.")
    out.mkdir(parents=True, exist_ok=True)

    # 1. Read all manifests up front so we can enforce consistency BEFORE
    #    we copy anything. (Half-pooled directories are confusing.)
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for src in srcs:
        if not src.exists():
            raise FileNotFoundError(f"source not found: {src}")
        manifests.append((src, _read_manifest(src)))

    warnings = _check_config_consistency(
        manifests, allow_mismatch=allow_config_mismatch
    )

    # 2. Build the cell -> source map and detect collisions on run_name.
    cell_to_source: dict[str, Path] = {}
    for src, _ in manifests:
        for cell_dir in _list_run_dirs(src):
            run_name = cell_dir.name
            if run_name in cell_to_source:
                raise FileExistsError(
                    f"Two source directories both contain run_name "
                    f"{run_name!r}: {cell_to_source[run_name]} and {src}. "
                    f"Did you run with overlapping --seeds?"
                )
            cell_to_source[run_name] = cell_dir

    # 3. Copy cells.
    n_copied = 0
    for run_name, src_cell in sorted(cell_to_source.items()):
        dst_cell = out / run_name
        _copy_cell(src_cell, dst_cell, overwrite=overwrite)
        n_copied += 1

    # 4. Write pooled manifest summarising the merge.
    pooled = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "out": str(out),
        "n_cells": n_copied,
        "n_sources": len(srcs),
        "config_warnings": warnings,
        "sources": [
            {
                "src": str(src),
                "manifest_args": m.get("args", {}),
                "n_cells_from_src": sum(
                    1 for v in cell_to_source.values()
                    if v.parent == src
                ),
                "cells_from_src": sorted(
                    v.name for v in cell_to_source.values()
                    if v.parent == src
                ),
            }
            for src, m in manifests
        ],
    }
    out_manifest = out / "pooled_manifest.json"
    out_manifest.write_text(json.dumps(pooled, indent=2))
    return pooled


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Pool one or more matrix-runner output directories into a "
            "single combined directory for cross-lab analysis."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--srcs",
        nargs="+",
        required=True,
        help="Source matrix-runner output directories. Each must contain "
             "a manifest.json and one subdirectory per cell.",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Destination directory for the pooled output. Will be created "
             "if it does not exist.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination cells of the same name. "
             "Off by default so accidental re-pools don't silently shadow "
             "an existing pool.",
    )
    p.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help="Pool even if source manifests disagree on critical config "
             "keys (env, methods, regimes, delays, updates, rollout, "
             "eval-episodes, shape-rewards, dropout-window-*). You almost "
             "never want this; it's an escape hatch for the rare case "
             "where you intentionally pool runs from slightly different "
             "configs.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    srcs = [Path(s) for s in args.srcs]
    out = Path(args.out)
    pooled = pool(
        srcs,
        out,
        overwrite=args.overwrite,
        allow_config_mismatch=args.allow_config_mismatch,
    )
    print(f"[pool_runs] pooled {pooled['n_cells']} cell(s) from "
          f"{pooled['n_sources']} source(s) -> {out}")
    print(f"[pool_runs] manifest: {out / 'pooled_manifest.json'}")
    for src_info in pooled["sources"]:
        print(
            f"[pool_runs]   {src_info['src']}  -> "
            f"{src_info['n_cells_from_src']} cell(s)"
        )
    if pooled["config_warnings"]:
        print(
            f"[pool_runs] NOTE: {len(pooled['config_warnings'])} config "
            f"warning(s) were tolerated via --allow-config-mismatch. "
            f"See pooled_manifest.json -> config_warnings."
        )


if __name__ == "__main__":
    main()
