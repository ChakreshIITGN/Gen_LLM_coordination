#!/usr/bin/env python3
from __future__ import annotations

import re

import click

from taxisim.agents.base import AgentConfig
from taxisim.agents.bayesian_agent import BayesianAgent
from taxisim.agents.hillclimb_agent import HillClimbAgent
from taxisim.agents.llm_agent import LLMAgent, resolve_llm_device
from taxisim.agents.random_agent import RandomAgent
from taxisim.agents.temporal_diff_agent import TemporalDiffAgent
from taxisim.environments.linear_1d import Linear1DEnvironment
from taxisim.experiments.fold_change import run_fold_change_experiment


def _model_tag(model_name: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name.strip())
    tag = tag.strip("-_.")
    return tag or "model"


@click.command()
@click.option("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
@click.option("--backend", type=str, default="huggingface")
@click.option("--n-episodes", type=int, default=50)
@click.option("--noise", type=float, default=5.0)
@click.option("--seed", type=int, default=42)
@click.option("--output-dir", type=str, default="results")
@click.option(
    "--device",
    type=str,
    default=None,
    help="HuggingFace model device: auto, cuda, cpu, cuda:0. "
    "Default: cuda when PyTorch sees a GPU, else auto.",
)
def main(
    model: str,
    backend: str,
    n_episodes: int,
    noise: float,
    seed: int,
    output_dir: str,
    device: str | None,
) -> None:
    llm_device = resolve_llm_device(device)
    model_tag = _model_tag(model)
    env = Linear1DEnvironment(noise_sigma=noise, seed=seed)
    cfg = AgentConfig

    agents = {
        "random": RandomAgent(cfg("random", history_length=5, seed=seed), env.action_space),
        "hillclimb": HillClimbAgent(cfg("hillclimb", history_length=5, seed=seed), env.action_space),
        "temporal_diff": TemporalDiffAgent(
            cfg("temporal_diff", history_length=5, seed=seed),
            env.action_space,
        ),
        "bayesian": BayesianAgent(
            cfg("bayesian", history_length=5, seed=seed),
            env.action_space,
            env_length=int(env.length),
            decay_length=env.decay_length,
            noise_sigma=noise,
        ),
        "llm": LLMAgent(
            cfg("llm", history_length=5, seed=seed),
            env.action_space,
            model_name=model,
            backend=backend,
            device=llm_device,
            temperature=0.0,
            env_params={"length": int(env.length)},
        ),
    }

    run_fold_change_experiment(
        agents,
        noise_sigma=noise,
        n_episodes=n_episodes,
        seed=seed,
        results_dir=output_dir,
        filename_suffix=f"model-{model_tag}",
        source_position=env.source_position,
        decay_length=env.decay_length,
        length=int(env.length),
        max_steps=env.max_steps,
    )


if __name__ == "__main__":
    main()
