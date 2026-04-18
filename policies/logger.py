"""
Light-weight training logger.

Writes two things per call to ``log_scalars``:
  * One row to ``{run_dir}/metrics.csv`` (header is inferred on first write).
  * One TensorBoard scalar event per key, under ``{run_dir}/tb/``.

Also dumps ``{run_dir}/config.json`` with the full ``argparse.Namespace`` /
dict passed in at construction, so runs are self-documenting.

This is intentionally tiny – we do not need anything fancier than scalar
curves for Phase 3.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping


class RunLogger:
    def __init__(
        self,
        log_dir: str | os.PathLike,
        run_name: str | None = None,
        config: Mapping[str, Any] | None = None,
        use_tensorboard: bool = True,
    ):
        self.run_name = run_name or time.strftime("run-%Y%m%d-%H%M%S")
        self.run_dir = Path(log_dir) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_file = None
        self._csv_writer = None
        self._fieldnames: list[str] | None = None

        self._tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))
            except Exception as exc:  # pragma: no cover - pure convenience
                print(f"[logger] tensorboard disabled: {exc}")
                self._tb = None

        if config is not None:
            self._dump_config(dict(config))

    # ------------------------------------------------------------------

    def _dump_config(self, config: dict[str, Any]) -> None:
        safe: dict[str, Any] = {}
        for k, v in config.items():
            try:
                json.dumps(v)
                safe[k] = v
            except TypeError:
                safe[k] = repr(v)
        (self.run_dir / "config.json").write_text(json.dumps(safe, indent=2))

    def _open_csv(self, fieldnames: list[str]) -> None:
        self._csv_file = self._csv_path.open("w", newline="")
        self._fieldnames = fieldnames
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self._csv_writer.writeheader()

    # ------------------------------------------------------------------

    def log_scalars(self, step: int, metrics: Mapping[str, float]) -> None:
        row = {"step": int(step), **{k: float(v) for k, v in metrics.items()}}
        if self._csv_writer is None:
            self._open_csv(list(row.keys()))
        else:
            # Add any new keys as empty columns (rare – keep schema stable).
            assert self._fieldnames is not None
            missing = [k for k in row.keys() if k not in self._fieldnames]
            if missing:
                # CSV schema is append-only; just ignore new keys silently for
                # simplicity. TB still sees everything.
                row = {k: v for k, v in row.items() if k in self._fieldnames}
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        if self._tb is not None:
            for k, v in metrics.items():
                self._tb.add_scalar(k, float(v), int(step))

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
            self._tb = None

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
