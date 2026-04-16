from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from taxisim.agents.base import Agent
from taxisim.environments.base import Environment, StepResult
from taxisim.environments.linear_1d import Linear1DEnvironment
from taxisim.experiments.runner import ExperimentConfig, ExperimentRunner
from taxisim.metrics.adaptation import fold_change_score
from taxisim.metrics.navigation import steps_to_goal


class BackgroundInfoWrapper(Environment):
    """Wraps an environment and adds ``background_level`` to each ``info`` dict."""

    def __init__(self, inner: Environment, background_level: float):
        self._inner = inner
        self.background_level = float(background_level)
        self.noise_sigma = inner.noise_sigma
        self.max_steps = inner.max_steps
        self.seed = inner.seed
        self.rng = inner.rng
        self.step_count = inner.step_count

    @property
    def action_space(self) -> list[str]:
        return self._inner.action_space

    def concentration(self, position: Any) -> float:
        return self._inner.concentration(position)

    def reset(self) -> StepResult:
        r = self._inner.reset()
        self.step_count = self._inner.step_count
        info = dict(r.info)
        info["background_level"] = self.background_level
        return StepResult(
            observation=r.observation,
            true_concentration=r.true_concentration,
            reward=r.reward,
            done=r.done,
            info=info,
        )

    def step(self, action: str) -> StepResult:
        r = self._inner.step(action)
        self.step_count = self._inner.step_count
        info = dict(r.info)
        info["background_level"] = self.background_level
        return StepResult(
            observation=r.observation,
            true_concentration=r.true_concentration,
            reward=r.reward,
            done=r.done,
            info=info,
        )


def _start_position_for_background_1d(
    bg: float,
    source: float,
    decay_length: float,
    length: float,
) -> float:
    """Pick a position in [0, length] with concentration ~= bg (two-branch exponential)."""
    bg = float(np.clip(bg, 1e-6, 99.999))
    d = -decay_length * math.log(bg / 100.0)
    c1 = source - d
    c2 = source + d
    candidates = [c for c in (c1, c2) if 0.0 <= c <= length]
    if not candidates:
        return float(np.clip(source, 0.0, length))
    return float(candidates[0])


def _normalized_filename_suffix(value: str) -> str:
    if not value:
        return ""
    return value if value.startswith("_") else f"_{value}"


def run_fold_change_experiment(
    agents: dict[str, Agent],
    background_levels: list[float] = [5.0, 20.0, 50.0, 80.0],
    noise_sigma: float = 5.0,
    n_episodes: int = 50,
    seed: int = 42,
    results_dir: str = "results",
    filename_suffix: str = "",
    source_position: float = 50.0,
    decay_length: float = 10.0,
    length: int = 100,
    max_steps: int = 200,
) -> pd.DataFrame:
    """
    Tests whether agents are background-concentration-invariant.
    Shifts start position so initial concentration matches background_level.
    """
    sns.set_theme(style="whitegrid")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = _normalized_filename_suffix(filename_suffix)
    rows: list[dict] = []
    trajectories_by_agent_bg: dict[str, dict[float, list[list[dict]]]] = {
        name: {bg: [] for bg in background_levels} for name in agents
    }

    for bg in background_levels:
        start = _start_position_for_background_1d(bg, source_position, decay_length, float(length))
        for agent_name, agent in agents.items():
            trajs: list[list[dict]] = []
            for ep in range(n_episodes):
                inner = Linear1DEnvironment(
                    length=length,
                    source_position=source_position,
                    decay_length=decay_length,
                    noise_sigma=noise_sigma,
                    max_steps=max_steps,
                    seed=seed,
                    start_position=start,
                )
                env = BackgroundInfoWrapper(inner, bg)
                exp = ExperimentConfig(
                    experiment_name=f"fold_change_{agent_name}_bg{bg}{suffix}",
                    n_episodes=1,
                    max_steps=max_steps,
                    seed=seed,
                    results_dir=results_dir,
                    log_every_step=True,
                )
                runner = ExperimentRunner(exp)
                summary = runner.run_episode(env, agent, ep)
                trajs.append(summary["trajectory"])
            trajectories_by_agent_bg[agent_name][bg] = trajs
            succ = float(
                np.mean([steps_to_goal(t, goal_threshold=1.0) is not None for t in trajs])
            )
            stgs = [steps_to_goal(t, goal_threshold=1.0) for t in trajs]
            vals = [float(s) for s in stgs if s is not None]
            mean_st = float(np.mean(vals)) if vals else float("nan")
            rows.append(
                {
                    "agent_name": agent_name,
                    "background_level": bg,
                    "success_rate": succ,
                    "mean_steps": mean_st,
                }
            )

    out = pd.DataFrame(rows)

    # Per-agent fold-change score: lowest vs highest background lists
    fc_rows: list[dict] = []
    low_bg = min(background_levels)
    high_bg = max(background_levels)
    for name in agents:
        low_tr = trajectories_by_agent_bg[name][low_bg]
        high_tr = trajectories_by_agent_bg[name][high_bg]
        fc = fold_change_score(low_tr, high_tr)
        fc_rows.append(
            {
                "agent_name": name,
                "fold_change_score": fc["ratio"],
                "is_fold_change_invariant": fc["is_fold_change_invariant"],
                "p_value": fc["p_value"],
            }
        )
    fc_df = pd.DataFrame(fc_rows)
    out = out.merge(fc_df, on="agent_name", how="left")

    heat = out.pivot(index="agent_name", columns="background_level", values="success_rate")
    fig, ax = plt.subplots(figsize=(8, max(3, len(agents) * 0.4)))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set_title("Fold-change: success rate heatmap")
    fig.tight_layout()
    fig.savefig(Path(results_dir) / f"fold_change_heatmap{suffix}_{ts}.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    pal = sns.color_palette("colorblind", n_colors=len(agents))
    names = fc_df["agent_name"].tolist()
    ratios = fc_df["fold_change_score"].tolist()
    ax2.bar(names, ratios, color=[pal[i % len(pal)] for i in range(len(names))])
    ax2.axhline(1.0, color="gray", ls="--")
    ax2.set_ylabel("fold_change_score (high_bg / low_bg)")
    ax2.set_title("Fold-change score by agent")
    fig2.tight_layout()
    fig2.savefig(Path(results_dir) / f"fold_change_ratio{suffix}_{ts}.png", dpi=150)
    plt.close(fig2)

    out.to_csv(Path(results_dir) / f"fold_change_summary{suffix}_{ts}.csv", index=False)
    return out
