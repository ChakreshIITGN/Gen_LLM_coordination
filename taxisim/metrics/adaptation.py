from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats

from taxisim.metrics.navigation import steps_to_goal


def fold_change_score(
    trajectories_low_bg: list[list[dict]],
    trajectories_high_bg: list[list[dict]],
    metric_fn: Callable[[list[dict]], float | int | None] | None = None,
    goal_threshold: float = 1.0,
) -> dict:
    """
    Computes whether agent performance is similar under low vs high background.
    """
    if metric_fn is None:

        def _stg(tr: list[dict]) -> float | None:
            v = steps_to_goal(tr, goal_threshold=goal_threshold)
            return float(v) if v is not None else None

        metric_fn = _stg  # type: ignore[assignment]

    def _collect(series: list[list[dict]]) -> list[float]:
        out: list[float] = []
        for tr in series:
            v = metric_fn(tr)  # type: ignore[misc]
            if v is None:
                out.append(float("nan"))
            else:
                out.append(float(v))
        arr = np.array(out, dtype=float)
        return arr[~np.isnan(arr)].tolist()

    low = np.array(_collect(trajectories_low_bg), dtype=float)
    high = np.array(_collect(trajectories_high_bg), dtype=float)
    low_mean = float(np.nanmean(low)) if low.size else float("nan")
    high_mean = float(np.nanmean(high)) if high.size else float("nan")
    low_std = float(np.nanstd(low, ddof=1)) if low.size > 1 else 0.0
    high_std = float(np.nanstd(high, ddof=1)) if high.size > 1 else 0.0

    ratio = float(high_mean / low_mean) if low_mean not in (0.0, np.nan) else float("nan")

    p_value = 1.0
    if low.size and high.size:
        try:
            _, p_value = stats.mannwhitneyu(low, high, alternative="two-sided")
        except ValueError:
            p_value = 1.0

    is_fc = bool(abs(ratio - 1.0) < 0.3) if not np.isnan(ratio) else False

    return {
        "low_bg_mean": low_mean,
        "high_bg_mean": high_mean,
        "high_bg_std": high_std,
        "low_bg_std": low_std,
        "ratio": ratio,
        "p_value": float(p_value),
        "is_fold_change_invariant": is_fc,
    }


def adaptation_index(trajectory: list[dict]) -> float:
    """
    Measures whether agent adapts to background: correlation between
    background concentration level and directional accuracy.
    If correlation is near 0: agent is background-independent (E. coli-like).
    If correlation is negative: high background impairs performance.
    Returns correlation coefficient.
    """
    xs: list[float] = []
    ys: list[float] = []
    prev_d: float | None = None
    for step in trajectory:
        bg = step.get("background_level")
        if bg is None:
            continue
        d = float(step.get("distance_to_source", 0.0))
        a = step.get("action_taken", "stay")
        if prev_d is not None and a != "stay" and a != "reset":
            closer = 1.0 if d < prev_d else 0.0
            xs.append(float(bg))
            ys.append(closer)
        prev_d = d
    if len(xs) < 2:
        return 0.0
    r, _ = stats.pearsonr(xs, ys)
    return float(r)
