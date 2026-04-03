"""
Experiment runner. Orchestrates: world setup -> policy -> step loop -> logging -> metrics.
Single-agent, synchronous. Phase 2 adds ParallelExperimentRunner.
"""
from __future__ import annotations

from .core.config import ExperimentConfig
from .core.logging import ExperimentLogger
from .core.metrics import compute_ecoli_metrics
from .core.policies import build_policy
from .core.world import EcoliWorld


class ExperimentRunner:
    """Run a single-agent experiment end-to-end from config."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self) -> dict:
        """Execute the experiment. Returns metrics dict."""
        cfg = self.config

        if cfg.experiment_type != "ecoli":
            raise NotImplementedError(f"Only 'ecoli' supported for now, got '{cfg.experiment_type}'")

        # setup
        world = EcoliWorld(cfg.world)
        policy = build_policy(
            cfg.policy, cfg.llm, seed=cfg.seed,
            reward_fn=world.reward_at,
        )
        logger = ExperimentLogger(cfg)

        # initial state
        start = cfg.agent.start_position
        position = start if isinstance(start, int) else start[0]
        last_reward: float | None = None
        trajectory: list[dict] = []

        print(f"Running: {cfg.experiment_name} | policy={cfg.policy.type} | "
              f"steps={cfg.episode_length} | model={cfg.llm.model}")
        print(f"Artifacts: {logger.run_dir}")

        # step loop
        for step in range(cfg.episode_length):
            obs = world.observe(step, position, last_reward)
            action, raw = policy.select_action(obs)
            new_pos, reward = world.step(position, action)

            # log event
            event = {
                "step": step,
                "position_before": position,
                "position_after": new_pos,
                "action_type": action.type,
                "reason": action.reason,
                "reward": round(reward, 6),
                "raw_llm_output": raw,
            }
            logger.log_step(event)
            trajectory.append({
                "step": step,
                "position": new_pos,
                "reward": reward,
                "action_type": action.type,
                "reason": action.reason,
            })

            # advance state
            position = new_pos
            last_reward = reward

            # progress indicator every 10%
            if (step + 1) % max(1, cfg.episode_length // 10) == 0:
                print(f"  step {step + 1}/{cfg.episode_length} | pos={position} | reward={reward:.4f}")

        # compute and save metrics
        metrics = compute_ecoli_metrics(trajectory)
        logger.save_metrics(metrics)
        logger.close()

        print(f"Done. Total reward: {metrics['total_reward']} | "
              f"Coverage: {metrics['coverage_unique_positions']} positions")
        return metrics
