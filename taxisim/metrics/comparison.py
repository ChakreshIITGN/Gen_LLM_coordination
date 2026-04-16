from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compare_agents(
    agent_results: dict[str, list[float]],
    baseline_name: str = "random",
) -> pd.DataFrame:
    """
    For each agent, compute: mean, std, median,
    Mann-Whitney U vs baseline (p-value), effect size (Cohen's d).
    Returns a DataFrame with agents as rows.
    """
    rows: list[dict] = []
    base = np.array(agent_results.get(baseline_name, []), dtype=float)
    for name, vals in agent_results.items():
        v = np.array(vals, dtype=float)
        mean = float(np.mean(v)) if v.size else float("nan")
        std = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
        med = float(np.median(v)) if v.size else float("nan")
        p_val = float("nan")
        d_cohen = float("nan")
        if name != baseline_name and base.size and v.size:
            try:
                _, p_val = stats.mannwhitneyu(v, base, alternative="two-sided")
            except ValueError:
                p_val = 1.0
            pooled = np.sqrt((np.var(v, ddof=1) + np.var(base, ddof=1)) / 2) if v.size > 1 and base.size > 1 else np.nan
            if pooled and pooled > 0:
                d_cohen = float((np.mean(v) - np.mean(base)) / pooled)
        rows.append(
            {
                "agent": name,
                "mean": mean,
                "std": std,
                "median": med,
                "mannwhitney_p_vs_baseline": p_val,
                "cohens_d_vs_baseline": d_cohen,
            }
        )
    return pd.DataFrame(rows)


def summary_table(
    agent_results: dict[str, dict],
) -> pd.DataFrame:
    """
    Multi-metric summary table. Agents as rows, metrics as columns.
    Each cell: "mean ± std".
    """
    agents = list(agent_results.keys())
    metric_names: set[str] = set()
    for m in agent_results.values():
        metric_names.update(m.keys())
    cols = sorted(metric_names)
    data: dict[str, list[str]] = {c: [] for c in cols}
    index: list[str] = []
    for ag in agents:
        index.append(ag)
        for c in cols:
            series = agent_results[ag].get(c, [])
            arr = np.array(series, dtype=float)
            if arr.size == 0:
                data[c].append("n/a")
            else:
                m = float(np.mean(arr))
                s = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
                data[c].append(f"{m:.4f} ± {s:.4f}")
    return pd.DataFrame(data, index=index)
