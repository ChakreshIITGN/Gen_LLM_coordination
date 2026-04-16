#!/usr/bin/env python3
from __future__ import annotations

import re

import click

from taxisim.agents.base import AgentConfig
from taxisim.agents.bayesian_agent import BayesianAgent
from taxisim.agents.hillclimb_agent import HillClimbAgent
from taxisim.agents.llm_agent import LLMAgent
from taxisim.agents.random_agent import RandomAgent
from taxisim.agents.temporal_diff_agent import TemporalDiffAgent
from taxisim.environments.linear_1d import Linear1DEnvironment
from taxisim.experiments.noise_sweep import run_noise_sweep


def _model_tag(model_name: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name.strip())
    tag = tag.strip("-_.")
    return tag or "model"


@click.command()
@click.option("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
@click.option("--backend", type=str, default="huggingface")
@click.option("--n-episodes", type=int, default=50)
@click.option("--noise", type=float, default=0.0, help="Ignored; sweep uses its own levels.")
@click.option("--seed", type=int, default=42)
@click.option("--output-dir", type=str, default="results")
@click.option(
    "--noise-level",
    "noise_levels",
    type=float,
    multiple=True,
    help="If set, only these observation-noise values (repeatable). Default: full sweep.",
)
def main(
    model: str,
    backend: str,
    n_episodes: int,
    noise: float,
    seed: int,
    output_dir: str,
    noise_levels: tuple[float, ...],
) -> None:
    _ = noise
    model_tag = _model_tag(model)
    env = Linear1DEnvironment(seed=seed)
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
            noise_sigma=1.0,
        ),
        "llm": LLMAgent(
            cfg("llm", history_length=5, seed=seed),
            env.action_space,
            model_name=model,
            backend=backend,
            temperature=0.0,
            env_params={"length": int(env.length)},
        ),
    }

    run_noise_sweep(
        agents,
        noise_levels=list(noise_levels) if noise_levels else None,
        env_kwargs={"seed": seed},
        n_episodes=n_episodes,
        seed=seed,
        results_dir=output_dir,
        filename_suffix=f"model-{model_tag}",
    )


if __name__ == "__main__":
    main()
