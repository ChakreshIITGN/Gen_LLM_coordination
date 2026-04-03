"""
Experiment logger. Writes JSONL trajectories + JSON config/metrics.
One log event per simulation step — full trace for later analysis.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import ExperimentConfig


class ExperimentLogger:
    """Creates a timestamped run directory and writes artifacts into it."""

    def __init__(self, config: ExperimentConfig):
        tag = config.logging.experiment_tag
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(config.logging.output_dir) / tag / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._traj_path = self.run_dir / "trajectory.jsonl"
        self._traj_file = self._traj_path.open("a", encoding="utf-8")

        # save config as first artifact
        config.to_json(self.run_dir / "config.json")

    def log_step(self, event: dict[str, Any]) -> None:
        """Append one step record to trajectory.jsonl."""
        self._traj_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._traj_file.flush()

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        """Write final metrics.json."""
        path = self.run_dir / "metrics.json"
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def close(self) -> None:
        self._traj_file.close()
