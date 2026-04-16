from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from taxisim.agents.base import Agent, AgentConfig
from taxisim.environments.linear_1d import Linear1DEnvironment
from taxisim.experiments.runner import ExperimentConfig, ExperimentRunner


def _normalized_filename_suffix(value: str) -> str:
    if not value:
        return ""
    return value if value.startswith("_") else f"_{value}"


def run_memory_ablation(
    agent_class: Type[Agent],
    agent_kwargs: dict[str, Any],
    history_lengths: list[int] = [1, 2, 5, 10, 20],
    env_kwargs: dict | None = None,
    n_episodes: int = 50,
    seed: int = 42,
    results_dir: str = "results",
    filename_suffix: str = "",
) -> pd.DataFrame:
    """
    For each history_length, run n_episodes and collect metrics.
    Returns DataFrame with columns: history_length, success_rate,
    mean_steps, mean_path_efficiency, mean_fraction_closer, std_steps.
    Also produces and saves plots.
    """
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("colorblind")
    env_kwargs = env_kwargs or {}
    rows: list[dict] = []
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = _normalized_filename_suffix(filename_suffix)

    for hl in history_lengths:
        # Merge so ``seed`` is not duplicated when callers include it in ``env_kwargs``.
        env = Linear1DEnvironment(**{**env_kwargs, "seed": seed})
        kw = {k: v for k, v in agent_kwargs.items() if k != "config"}
        cfg = AgentConfig(
            name=kw.get("name", agent_class.__name__),
            history_length=hl,
            seed=seed,
        )
        kw.pop("name", None)
        agent = agent_class(config=cfg, action_space=env.action_space, **kw)
        exp = ExperimentConfig(
            experiment_name=f"memory_ablation_hl{hl}{suffix}",
            n_episodes=n_episodes,
            max_steps=env.max_steps,
            seed=seed,
            results_dir=results_dir,
            log_every_step=False,
        )
        runner = ExperimentRunner(exp)
        df_ep = runner.run_experiment(env, agent, reset_agent_per_episode=True)
        rows.append(
            {
                "history_length": hl,
                "success_rate": float(df_ep["success"].mean()),
                "mean_steps": float(df_ep["steps"].mean()),
                "mean_path_efficiency": float(df_ep["path_efficiency"].mean()),
                "mean_fraction_closer": float(df_ep["fraction_closer"].mean()),
                "std_steps": float(df_ep["steps"].std(ddof=1)) if len(df_ep) > 1 else 0.0,
            }
        )

    out = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(out["history_length"], out["success_rate"], marker="o", color=palette[0])
    ax.set_xlabel("history_length")
    ax.set_ylabel("success_rate")
    ax.set_title("Memory ablation: success rate vs history length")
    fig.tight_layout()
    fig.savefig(
        Path(results_dir) / f"memory_ablation_success{suffix}_{ts}.png",
        dpi=150,
    )
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(out["history_length"], out["mean_path_efficiency"], marker="o", color=palette[1])
    ax2.set_xlabel("history_length")
    ax2.set_ylabel("mean path_efficiency")
    ax2.set_title("Memory ablation: path efficiency vs history length")
    fig2.tight_layout()
    fig2.savefig(
        Path(results_dir) / f"memory_ablation_path_efficiency{suffix}_{ts}.png",
        dpi=150,
    )
    plt.close(fig2)

    out.to_csv(Path(results_dir) / f"memory_ablation_summary{suffix}_{ts}.csv", index=False)
    return out
