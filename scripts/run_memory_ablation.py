#!/usr/bin/env python3
from __future__ import annotations

import re

import click

from taxisim.agents.hillclimb_agent import HillClimbAgent
from taxisim.agents.llm_agent import LLMAgent
from taxisim.agents.temporal_diff_agent import TemporalDiffAgent
from taxisim.environments.linear_1d import Linear1DEnvironment
from taxisim.experiments.memory_ablation import run_memory_ablation


def _model_tag(model_name: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name.strip())
    tag = tag.strip("-_.")
    return tag or "model"


@click.command()
@click.option("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
@click.option("--backend", type=str, default="huggingface")
@click.option("--prompt-style", type=str, default="abstract")
@click.option("--n-episodes", type=int, default=50)
@click.option("--noise", type=float, default=0.0)
@click.option("--seed", type=int, default=42)
@click.option("--output-dir", type=str, default="results")
def main(
    model: str,
    backend: str,
    prompt_style: str,
    n_episodes: int,
    noise: float,
    seed: int,
    output_dir: str,
) -> None:
    model_tag = _model_tag(model)
    env = Linear1DEnvironment(noise_sigma=noise, seed=seed)
    env_kwargs = {"noise_sigma": noise, "seed": seed}

    run_memory_ablation(
        LLMAgent,
        {
            "name": "llm",
            "model_name": model,
            "backend": backend,
            "prompt_style": prompt_style,
            "temperature": 0.0,
            "env_params": {"length": int(env.length)},
        },
        env_kwargs=env_kwargs,
        n_episodes=n_episodes,
        seed=seed,
        results_dir=output_dir,
        filename_suffix=f"model-{model_tag}",
    )

    run_memory_ablation(
        HillClimbAgent,
        {"name": "hillclimb"},
        env_kwargs=env_kwargs,
        n_episodes=n_episodes,
        seed=seed,
        results_dir=output_dir,
        filename_suffix=f"model-{model_tag}",
    )

    run_memory_ablation(
        TemporalDiffAgent,
        {"name": "temporal_diff"},
        env_kwargs=env_kwargs,
        n_episodes=n_episodes,
        seed=seed,
        results_dir=output_dir,
        filename_suffix=f"model-{model_tag}",
    )


if __name__ == "__main__":
    main()
