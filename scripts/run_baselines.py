#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.table import Table

from taxisim.agents.base import AgentConfig
from taxisim.agents.bayesian_agent import BayesianAgent
from taxisim.agents.hillclimb_agent import HillClimbAgent
from taxisim.agents.random_agent import RandomAgent
from taxisim.agents.temporal_diff_agent import TemporalDiffAgent
from taxisim.environments.gradient_2d import Gradient2DEnvironment
from taxisim.environments.linear_1d import Linear1DEnvironment
from taxisim.experiments.runner import ExperimentConfig, ExperimentRunner, wilson_score_interval


@click.command()
@click.option("--env", type=click.Choice(["1d", "2d"]), default="1d")
@click.option("--n-episodes", type=int, default=50)
@click.option("--max-steps", type=int, default=200)
@click.option("--noise", type=float, default=0.0)
@click.option("--seed", type=int, default=42)
@click.option("--output-dir", type=str, default="results")
@click.option("--history-length", type=int, default=5)
def main(
    env: str,
    n_episodes: int,
    max_steps: int,
    noise: float,
    seed: int,
    output_dir: str,
    history_length: int,
) -> None:
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("colorblind")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if env == "1d":
        base_env = Linear1DEnvironment(
            noise_sigma=noise,
            max_steps=max_steps,
            seed=seed,
        )
    else:
        base_env = Gradient2DEnvironment(
            noise_sigma=noise,
            max_steps=max_steps,
            seed=seed,
        )

    action_space = base_env.action_space

    def make_cfg(name: str) -> AgentConfig:
        return AgentConfig(name=name, history_length=history_length, seed=seed)

    agents = {
        "random": RandomAgent(make_cfg("random"), action_space),
        "hillclimb": HillClimbAgent(make_cfg("hillclimb"), action_space),
        "temporal_diff": TemporalDiffAgent(make_cfg("temporal_diff"), action_space),
    }

    if env == "1d":
        agents["bayesian"] = BayesianAgent(
            make_cfg("bayesian"),
            action_space,
            env_length=int(base_env.length),
            decay_length=base_env.decay_length,
            noise_sigma=max(noise, 1e-6),
        )
    else:
        agents["bayesian"] = BayesianAgent(
            make_cfg("bayesian"),
            action_space,
            env_length=int(base_env.width),
            decay_length=base_env.decay_length,
            noise_sigma=max(noise, 1e-6),
            width=float(base_env.width),
            height=float(base_env.height),
        )

    records: list[pd.DataFrame] = []
    for name, agent in agents.items():
        if env == "1d":
            e = Linear1DEnvironment(
                noise_sigma=noise,
                max_steps=max_steps,
                seed=seed,
            )
        else:
            e = Gradient2DEnvironment(
                noise_sigma=noise,
                max_steps=max_steps,
                seed=seed,
            )
        exp = ExperimentConfig(
            experiment_name=f"baselines_{name}_{env}",
            n_episodes=n_episodes,
            max_steps=max_steps,
            seed=seed,
            results_dir=output_dir,
            log_every_step=False,
        )
        runner = ExperimentRunner(exp)
        df = runner.run_experiment(e, agent, reset_agent_per_episode=True)
        df["agent"] = name
        records.append(df)

    all_df = pd.concat(records, ignore_index=True)
    csv_path = Path(output_dir) / f"baselines_{env}_{ts}.csv"
    all_df.to_csv(csv_path, index=False)

    console = Console()
    table = Table(title=f"Baseline summary ({env})")
    table.add_column("Agent")
    table.add_column("success_rate")
    table.add_column("95% Wilson CI")
    table.add_column("mean_steps ± std")
    table.add_column("mean_path_efficiency")
    for name in agents:
        sub = all_df[all_df["agent"] == name]
        succ = int(sub["success"].sum())
        n = len(sub)
        lo, hi = wilson_score_interval(succ, n)
        table.add_row(
            name,
            f"{sub['success'].mean():.3f}",
            f"[{lo:.3f}, {hi:.3f}]",
            f"{sub['steps'].mean():.2f} ± {sub['steps'].std():.2f}",
            f"{sub['path_efficiency'].mean():.3f}",
        )
    console.print(table)

    # Four plots: bar success, box steps, bar path efficiency, bar mean reward
    fig, ax = plt.subplots(figsize=(8, 5))
    order = list(agents.keys())
    means = [all_df[all_df["agent"] == n]["success"].mean() for n in order]
    ax.bar(order, means, color=[palette[i % len(palette)] for i in range(len(order))])
    ax.set_ylabel("success_rate")
    ax.set_title("Baselines: success rate")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / f"baselines_bar_success_{env}_{ts}.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=all_df, x="agent", y="steps", order=order, palette="colorblind", ax=ax2)
    ax2.set_title("Baselines: steps distribution")
    fig2.tight_layout()
    fig2.savefig(Path(output_dir) / f"baselines_box_steps_{env}_{ts}.png", dpi=150)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    pe = [all_df[all_df["agent"] == n]["path_efficiency"].mean() for n in order]
    ax3.bar(order, pe, color=[palette[i % len(palette)] for i in range(len(order))])
    ax3.set_ylabel("mean path_efficiency")
    ax3.set_title("Baselines: path efficiency")
    fig3.tight_layout()
    fig3.savefig(Path(output_dir) / f"baselines_bar_path_efficiency_{env}_{ts}.png", dpi=150)
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=all_df, x="agent", y="total_reward", order=order, palette="colorblind", ax=ax4)
    ax4.set_title("Baselines: total reward distribution")
    fig4.tight_layout()
    fig4.savefig(Path(output_dir) / f"baselines_box_reward_{env}_{ts}.png", dpi=150)
    plt.close(fig4)


if __name__ == "__main__":
    main()
