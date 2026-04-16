from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm

from taxisim.agents.base import Agent
from taxisim.agents.hillclimb_agent import HillClimbAgent
from taxisim.agents.random_agent import RandomAgent
from taxisim.agents.temporal_diff_agent import TemporalDiffAgent
from taxisim.environments.base import Environment
from taxisim.metrics.navigation import (
    final_distance,
    fraction_closer,
    mean_reward,
    path_efficiency,
    steps_to_goal,
)


def wilson_score_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for binomial proportion."""
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1.0 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denom
    margin = (
        z
        * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
        / denom
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass
class ExperimentConfig:
    experiment_name: str
    n_episodes: int = 50
    max_steps: int = 200
    seed: int = 42
    results_dir: str = "results"
    log_every_step: bool = True  # if False, only log episode summary


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._make_results_dir()

    def _make_results_dir(self) -> None:
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)

    def _goal_threshold(self, env: Environment) -> float:
        from taxisim.environments.gradient_2d import Gradient2DEnvironment
        from taxisim.environments.linear_1d import Linear1DEnvironment

        if isinstance(env, Linear1DEnvironment):
            return 1.0
        if isinstance(env, Gradient2DEnvironment):
            return 1.5
        return 1.0

    def _apply_episode_seed(self, env: Environment, agent: Agent, episode_idx: int) -> None:
        ep_seed = int(self.config.seed + episode_idx)
        env.rng = np.random.default_rng(ep_seed)
        env.seed = ep_seed
        if isinstance(agent, RandomAgent):
            agent._rng = np.random.default_rng(ep_seed + 11)
        if isinstance(agent, HillClimbAgent):
            agent._rng = np.random.default_rng(ep_seed + 13)
        if isinstance(agent, TemporalDiffAgent):
            agent._rng = np.random.default_rng(ep_seed + 17)

    def run_episode(
        self,
        env: Environment,
        agent: Agent,
        episode_idx: int,
    ) -> dict:
        """
        Run a single episode. Returns episode summary dict.
        """
        self._apply_episode_seed(env, agent, episode_idx)
        agent.reset()
        init = env.reset()
        obs = init.observation
        pos = init.info["position"]
        done = init.done
        trajectory: list[dict] = [dict(init.info)]
        rewards: list[float] = []
        actions: list[str] = []
        observations: list[float] = [obs]
        reasonings: list[str] = []

        step_meta: list[dict] = []
        ts = datetime.now(timezone.utc).isoformat()

        while not done:
            step = agent.act(obs, pos)
            actions.append(step.action)
            reasonings.append(step.reasoning)
            out = env.step(step.action)
            rewards.append(out.reward)
            trajectory.append(dict(out.info))
            if self.config.log_every_step:
                step_meta.append(
                    {
                        "action": step.action,
                        "reasoning": step.reasoning,
                        "reward": out.reward,
                        "info": dict(out.info),
                    }
                )
            obs = out.observation
            pos = out.info["position"]
            observations.append(obs)
            done = out.done

        gt = self._goal_threshold(env)
        stg = steps_to_goal(trajectory, goal_threshold=gt)
        success = stg is not None
        ep_summary = {
            "episode": episode_idx,
            "agent_name": agent.config.name,
            "env_name": env.__class__.__name__,
            "success": success,
            "steps": int(stg if stg is not None else len(actions)),
            "final_distance": final_distance(trajectory),
            "path_efficiency": path_efficiency(trajectory, goal_threshold=gt),
            "fraction_closer": fraction_closer(trajectory),
            "total_reward": float(sum(rewards)),
            "mean_reward": mean_reward(trajectory, rewards),
        }

        meta: dict[str, Any] = {
            "experiment": self.config.experiment_name,
            "agent": agent.config.name,
            "model_name": getattr(agent, "model_name", ""),
            "prompt_style": getattr(agent, "prompt_style", ""),
            "temperature": getattr(agent, "temperature", None),
            "history_length": agent.config.history_length,
            "noise_sigma": getattr(env, "noise_sigma", None),
            "seed": self.config.seed,
            "episode": episode_idx,
            "timestamp": ts,
        }

        record = {
            "meta": meta,
            "episode_summary": ep_summary,
            "steps": step_meta if self.config.log_every_step else [],
        }

        log_path = Path(self.config.results_dir) / f"{self.config.experiment_name}_episodes.jsonl"
        with jsonlines.open(log_path, mode="a") as w:
            w.write(record)

        return {
            **ep_summary,
            "trajectory": trajectory if self.config.log_every_step else [],
            "actions": actions,
            "observations": observations,
            "reasonings": reasonings,
            "_raw_log": record,
        }

    def run_experiment(
        self,
        env: Environment,
        agent: Agent,
        reset_agent_per_episode: bool = True,
    ) -> pd.DataFrame:
        """
        Run n_episodes episodes. Returns DataFrame with one row per episode.
        Also saves full results to JSON Lines file in results_dir.
        """
        rows: list[dict] = []
        for ep in tqdm(
            range(self.config.n_episodes),
            desc=self.config.experiment_name,
        ):
            if not reset_agent_per_episode and ep > 0:
                pass
            summary = self.run_episode(env, agent, ep)
            row = {k: v for k, v in summary.items() if k not in ("trajectory", "reasonings", "_raw_log")}
            rows.append(row)

        df = pd.DataFrame(rows)
        successes = int(df["success"].sum()) if "success" in df else 0
        n = len(df)
        lo, hi = wilson_score_interval(successes, n)
        df.attrs["wilson_success_rate_95"] = (lo, hi)
        self.save_results(df, self.config.experiment_name)
        return df

    def save_results(self, results: pd.DataFrame, name: str) -> str:
        """
        Save to results/{name}_{timestamp}.csv and .jsonl
        Returns path to CSV.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base = Path(self.config.results_dir) / f"{name}_{ts}"
        csv_path = base.with_suffix(".csv")
        results.to_csv(csv_path, index=False)
        jl_path = base.with_suffix(".jsonl")
        with jsonlines.open(jl_path, mode="w") as w:
            for _, row in results.iterrows():
                w.write(row.to_dict())
        return str(csv_path)
