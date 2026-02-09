"""Logging utilities for experiments."""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars and arrays to native Python types."""

    def default(self, o):
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        if isinstance(o, (np.floating, np.float64, np.float32)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


def save_config(config: Dict[str, Any], output_dir: Path):
    """Save experiment configuration."""
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, cls=NumpyJSONEncoder)


def save_metrics(metrics: Dict[str, Any], output_dir: Path):
    """Save experiment metrics."""
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyJSONEncoder)


def save_trajectory(trajectory: list[Dict[str, Any]], output_dir: Path):
    """Save episode trajectory as JSONL."""
    trajectory_path = output_dir / "trajectory.jsonl"
    with open(trajectory_path, "w") as f:
        for record in trajectory:
            f.write(json.dumps(record, cls=NumpyJSONEncoder) + "\n")


def create_output_dir(base_dir: str, model_provider: str, model_name: str, config: Dict[str, Any]) -> Path:
    """
    Create output directory with model-specific naming.

    Format: {base_dir}/{model_provider}_{model_name}_{timestamp}/
    """
    # Sanitize model name for filesystem
    safe_model_name = model_name.replace("/", "_").replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{model_provider}_{safe_model_name}_{timestamp}"

    output_dir = Path(base_dir) / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir
