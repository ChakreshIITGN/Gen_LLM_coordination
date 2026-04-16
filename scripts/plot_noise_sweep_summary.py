#!/usr/bin/env python3
"""Regenerate noise-sweep figures from a saved ``noise_sweep_summary*.csv`` (no experiment rerun)."""
from __future__ import annotations

from pathlib import Path

import click
import pandas as pd

from taxisim.experiments.noise_sweep import plot_noise_sweep_summary_figures


@click.command()
@click.argument("csv_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for PNG outputs (default: same directory as the CSV).",
)
@click.option(
    "--suffix",
    type=str,
    default="",
    help="Filename suffix fragment matching the sweep, e.g. _model-HuggingFaceTB-SmolLM2-1.7B-Instruct",
)
@click.option(
    "--timestamp",
    type=str,
    default=None,
    help="UTC timestamp string for output filenames (default: new timestamp).",
)
def main(csv_path: Path, output_dir: Path | None, suffix: str, timestamp: str | None) -> None:
    df = pd.read_csv(csv_path)
    out = output_dir or csv_path.parent
    plot_noise_sweep_summary_figures(df, out, filename_suffix=suffix, ts=timestamp)


if __name__ == "__main__":
    main()
