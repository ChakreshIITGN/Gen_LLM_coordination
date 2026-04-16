#!/usr/bin/env python3
"""
Augment an existing noise_sweep_summary*.csv with LLM action fractions from
per-noise JSONL exports written by ``ExperimentRunner.save_results`` (basename
``noise_sweep_llm_n<noise>_<suffix>_<timestamp>.jsonl`` — **no** ``_episodes`` in
the name). Files named ``*_episodes.jsonl`` are a different format (one record
per episode with optional ``steps``); they are skipped unless you pass
``--include-episodes-jsonl``. If the run used ``log_every_step=False``, those
episode files have empty ``steps`` and **cannot** be used to recover action
fractions — use the ``save_results`` JSONL or rely on ``run_noise_sweep``’s
in-memory summary (which already includes action columns).

**Use the same experiment's summary CSV as the JSONLs you merge.** The summary
rows (success_rate, mean_steps, …) are not recomputed from JSONL; only action
fractions are. Mixing a TinyLlama summary with SmolLM2 JSONLs (or vice versa)
produces impossible plots (e.g. ~56% success with 100% ``stay`` in far-start
1D tasks).

``llm_parse_failure_rate`` cannot be reconstructed from these JSONLs (reasonings
are not stored), so it is left as NaN unless already present in the summary CSV.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import click
import pandas as pd

from taxisim.experiments.noise_sweep import plot_noise_sweep_summary_figures

_LLM_JSONL_RE = re.compile(r"^noise_sweep_llm_n(\d+(?:\.\d+)?)")


def _parse_noise_from_filename(path: Path) -> float | None:
    m = _LLM_JSONL_RE.match(path.name)
    if not m:
        return None
    return float(m.group(1))


def _aggregate_action_fractions_from_jsonl(path: Path) -> dict[str, float]:
    c: Counter[str] = Counter()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        acts = row.get("actions")
        if acts:
            if isinstance(acts, str):
                c.update([acts])
            else:
                c.update(acts)
        else:
            # ``*_episodes.jsonl`` lines: per-step actions under ``steps`` when log_every_step=True
            for step in row.get("steps") or []:
                a = step.get("action")
                if a:
                    c.update([str(a).lower()])
    total = sum(c.values())
    if total <= 0:
        return {
            "action_frac_left": float("nan"),
            "action_frac_right": float("nan"),
            "action_frac_stay": float("nan"),
        }
    return {
        "action_frac_left": float(c.get("left", 0)) / total,
        "action_frac_right": float(c.get("right", 0)) / total,
        "action_frac_stay": float(c.get("stay", 0)) / total,
    }


def _discover_llm_jsonls(
    results_dir: Path,
    include: str | None,
    exclude: str | None,
    *,
    include_episodes_jsonl: bool,
) -> list[Path]:
    candidates: list[Path] = []
    for p in sorted(results_dir.glob("noise_sweep_llm_n*.jsonl")):
        if "_episodes" in p.name and not include_episodes_jsonl:
            continue
        if include and include not in p.name:
            continue
        if exclude and exclude in p.name:
            continue
        if _parse_noise_from_filename(p) is None:
            continue
        candidates.append(p)
    # One file per noise level; prefer save_results (no _episodes) over episode logs
    by_n: dict[float, Path] = {}
    for p in sorted(candidates, key=lambda x: ("_episodes" in x.name, str(x))):
        n = _parse_noise_from_filename(p)
        assert n is not None
        if n not in by_n:
            by_n[n] = p
    return [by_n[k] for k in sorted(by_n.keys())]


@click.command()
@click.option(
    "--summary-csv",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Existing noise_sweep_summary*.csv (with or without action columns).",
)
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to scan for noise_sweep_llm_n*.jsonl (default: parent of summary CSV).",
)
@click.option(
    "--include-substring",
    type=str,
    default=None,
    help="Only use JSONL files whose basename contains this (e.g. TinyLlama).",
)
@click.option(
    "--exclude-substring",
    type=str,
    default=None,
    help="Skip JSONL files whose basename contains this (e.g. model- to keep bare Qwen-style n0.jsonl).",
)
@click.option(
    "--output-csv",
    type=click.Path(path_type=Path),
    default=None,
    help="Write merged CSV here (default: results_dir / noise_sweep_summary_augmented_<timestamp>.csv).",
)
@click.option(
    "--plot/--no-plot",
    default=True,
    help="After writing CSV, regenerate figures via plot_noise_sweep_summary_figures.",
)
@click.option(
    "--plot-suffix",
    type=str,
    default="",
    help="Filename suffix for PNG outputs (e.g. _model-TinyLlama-...); default: inferred from JSONL names.",
)
@click.option(
    "--include-episodes-jsonl/--no-include-episodes-jsonl",
    default=False,
    help="Also use *_episodes.jsonl (episode logs). Requires per-step data in ``steps`` or top-level ``actions``; "
    "empty ``steps`` (log_every_step=False) yields NaN action fractions.",
)
def main(
    summary_csv: Path,
    results_dir: Path | None,
    include_substring: str | None,
    exclude_substring: str | None,
    output_csv: Path | None,
    plot: bool,
    plot_suffix: str,
    include_episodes_jsonl: bool,
) -> None:
    rd = results_dir or summary_csv.parent
    rd = Path(rd)

    files = _discover_llm_jsonls(
        rd,
        include_substring,
        exclude_substring,
        include_episodes_jsonl=include_episodes_jsonl,
    )
    if not files:
        n_ep = len(list(rd.glob("noise_sweep_llm_n*_episodes.jsonl")))
        n_plain = len(
            [
                p
                for p in rd.glob("noise_sweep_llm_n*.jsonl")
                if "_episodes" not in p.name
            ]
        )
        hint = (
            f" Found {n_plain} save_results-style file(s) and {n_ep} *_episodes.jsonl. "
            "By default only save_results exports are used (no *_episodes in the name). "
            "Pass --include-episodes-jsonl to include episode logs, or ensure "
            "`noise_sweep_llm_n*_model-*_<timestamp>.jsonl` exists from ExperimentRunner.save_results."
        )
        raise click.ClickException(
            f"No matching noise_sweep_llm_n*.jsonl under {rd}. "
            "Adjust --include-substring / --exclude-substring."
            + (hint if n_ep or n_plain else "")
        )

    by_noise: dict[float, dict[str, float]] = {}
    for p in files:
        n = _parse_noise_from_filename(p)
        assert n is not None
        by_noise[n] = _aggregate_action_fractions_from_jsonl(p)

    df = pd.read_csv(summary_csv)
    for col in ("action_frac_left", "action_frac_right", "action_frac_stay", "llm_parse_failure_rate"):
        if col not in df.columns:
            df[col] = float("nan")

    llm_mask = df["agent_name"] == "llm"
    updated = 0
    for n, fracs in sorted(by_noise.items(), key=lambda x: x[0]):
        noise_key = float(n)
        row_mask = llm_mask & (df["noise_sigma"].astype(float) == noise_key)
        if not row_mask.any():
            click.echo(f"warning: no llm row for noise_sigma={noise_key} in summary; skipping {n}", err=True)
            continue
        for k, v in fracs.items():
            df.loc[row_mask, k] = v
        updated += int(row_mask.sum())

    if updated == 0:
        raise click.ClickException("No LLM rows were updated; check noise_sigma alignment with JSONL filenames.")

    if df.loc[llm_mask, "action_frac_stay"].isna().all():
        click.echo(
            "warning: all LLM action_frac_* are NaN — JSONL had no top-level ``actions`` and no per-step "
            "``steps`` entries (typical when log_every_step=False). Use ExperimentRunner.save_results "
            "JSONL exports, enable per-step logging for *_episodes.jsonl, or use run_noise_sweep output CSV.",
            err=True,
        )

    # Catch mixed provenance: stay-only policies cannot achieve high success when starts are far from goal.
    bad = df.loc[
        llm_mask
        & df["action_frac_stay"].notna()
        & (df["action_frac_stay"] >= 0.95)
        & (df["success_rate"] > 0.15)
    ]
    if len(bad) > 0:
        click.echo(
            "warning: LLM rows have action_frac_stay >= 0.95 but success_rate > 0.15 — "
            "the base summary likely belongs to a different model run than these JSONLs. "
            "Regenerate figures from a single run's noise_sweep_summary*.csv or pass "
            "--summary-csv from that same run.",
            err=True,
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = output_csv or (rd / f"noise_sweep_summary_augmented_{ts}.csv")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    click.echo(f"Wrote {out_path}")

    if plot:
        suffix = plot_suffix
        if not suffix and files:
            m = re.match(r"^noise_sweep_llm_n\d+(?:\.\d+)?(.*)$", files[0].stem)
            if m and m.group(1):
                suffix = m.group(1) if m.group(1).startswith("_") else f"_{m.group(1)}"
        paths = plot_noise_sweep_summary_figures(df, out_path.parent, filename_suffix=suffix, ts=ts)
        for p in paths:
            click.echo(f"  plot: {p}")


if __name__ == "__main__":
    main()
