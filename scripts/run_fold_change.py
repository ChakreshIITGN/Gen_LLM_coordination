#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import cast

import click
import torch

from taxisim.agents.base import AgentConfig
from taxisim.agents.bayesian_agent import BayesianAgent
from taxisim.agents.hillclimb_agent import HillClimbAgent
from taxisim.agents.llm_agent import LLMAgent, resolve_llm_device
from taxisim.agents.random_agent import RandomAgent
from taxisim.agents.temporal_diff_agent import TemporalDiffAgent
from taxisim.environments.linear_1d import Linear1DEnvironment
from taxisim.experiments.fold_change import FoldChangeStartMode, run_fold_change_experiment


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
@click.option(
    "--start-mode",
    type=click.Choice(["auto", "left", "right", "fixed"], case_sensitive=False),
    default="auto",
    help="How to set start position (independent of background offset). "
    "auto: legacy (prefer source-fixed_distance if in bounds, else source+fixed_distance). "
    "left/right: clip(source ± fixed_distance). fixed: use --start-position.",
)
@click.option(
    "--fixed-distance",
    type=float,
    default=25.0,
    show_default=True,
    help="Distance from source for auto/left/right start modes.",
)
@click.option(
    "--start-position",
    type=float,
    default=None,
    help="Absolute start coordinate; required when --start-mode=fixed.",
)
@click.option(
    "--llm-only",
    is_flag=True,
    default=False,
    help="Run only the LLM agent (skip CPU baselines). Use this to keep the GPU busy; "
    "otherwise the default order runs four numpy agents before the LLM at each background level.",
)
def main(
    model: str,
    backend: str,
    n_episodes: int,
    noise: float,
    seed: int,
    output_dir: str,
    device: str | None,
    start_mode: str,
    fixed_distance: float,
    start_position: float | None,
    llm_only: bool,
) -> None:
    if start_mode == "fixed" and start_position is None:
        raise click.UsageError("--start-position is required when --start-mode=fixed")
    llm_device = resolve_llm_device(device)
    model_tag = _model_tag(model)
    env = Linear1DEnvironment(noise_sigma=noise, seed=seed)
    cfg = AgentConfig

    click.echo(
        f"PyTorch {torch.__version__}, torch.cuda.is_available()={torch.cuda.is_available()}",
        err=True,
    )
    if backend == "huggingface" and not torch.cuda.is_available() and device is None:
        click.echo(
            "Note: CUDA is not visible to PyTorch; install a CUDA-enabled wheel from "
            "https://pytorch.org/get-started/locally/ (pip default is often CPU-only). "
            "The driver CUDA version from nvidia-smi does not install PyTorch GPU support by itself.",
            err=True,
        )
    click.echo(f"LLM device: {llm_device}", err=True)

    if not llm_only:
        click.echo(
            "Note: full fold-change runs four CPU-only agents before the LLM at each background "
            "level; the GPU stays idle during those runs. Use --llm-only for LLM-only runs.",
            err=True,
        )

    llm_agent = LLMAgent(
        cfg("llm", history_length=5, seed=seed),
        env.action_space,
        model_name=model,
        backend=backend,
        device=llm_device,
        temperature=0.0,
        env_params={"length": int(env.length)},
    )
    if llm_only:
        agents = {"llm": llm_agent}
    else:
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
            "llm": llm_agent,
        }

    if backend == "huggingface":
        llm_agent.preload_model()

    click.echo(
        f"Model ready. Running fold-change experiment ({len(agents)} agents, "
        f"{n_episodes} episodes per cell) → {output_dir}",
        err=True,
    )
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
        fixed_start_distance=fixed_distance,
        start_mode=cast(FoldChangeStartMode, start_mode.lower()),
        fixed_start_position=start_position,
    )


if __name__ == "__main__":
    main()
