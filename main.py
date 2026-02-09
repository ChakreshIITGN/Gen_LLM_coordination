"""Main experiment runner for Stage 1: LLM as inference-only controller."""

import argparse
import json
from pathlib import Path
import numpy as np

from src.llm_abm.core.schemas import ExperimentConfig, EpisodeConfig, ModelProvider, Action
from src.llm_abm.core.world import ChemotaxisPOMDP
from src.llm_abm.core.policies import LLMChemotaxisPolicy
from src.llm_abm.core.metrics import EpisodeMetrics, ExperimentMetrics
from src.llm_abm.core.logging import create_output_dir, save_config, save_metrics, save_trajectory
from src.llm_abm.utils.seed import set_seed


def run_episode(
    env: ChemotaxisPOMDP, policy: LLMChemotaxisPolicy, episode_id: int, rng: np.random.Generator
) -> EpisodeMetrics:
    """Run a single episode."""
    metrics = EpisodeMetrics(episode_id=episode_id)
    trajectory = []

    # Reset environment with random source location
    source_location = rng.integers(0, 21)
    obs = env.reset(source_location=source_location)

    # Get initial hidden state for logging
    hidden_state = env.get_hidden_state()

    step = 0
    while True:
        # Select action using policy
        action, raw_llm_output = policy.select_action(obs)

        # Execute action
        next_obs, reward, done, info = env.step(action)

        # Update metrics
        if action == Action.RUN:
            metrics.num_runs += 1
        elif action == Action.TUMBLE:
            metrics.num_tumbles += 1

        # Log trajectory
        trajectory_record = {
            "step": step,
            "observation": {
                "concentrations": obs.concentrations,
                "last_action": obs.last_action.value if obs.last_action else None,
            },
            "action": action.value,
            "raw_llm_output": raw_llm_output,
            "hidden_state": {
                "position": hidden_state.position,
                "direction": hidden_state.direction,
                "source_location": hidden_state.source_location,
            },
            "info": info,
            "reward": reward,
            "done": done,
        }
        trajectory.append(trajectory_record)

        # Check for success
        if reward > 0.0 and not metrics.success:
            metrics.success = True
            metrics.steps_to_hit = step + 1

        metrics.total_steps = step + 1
        metrics.final_distance_to_source = info["distance_to_source"]

        if done:
            break

        obs = next_obs
        hidden_state = env.get_hidden_state()
        step += 1

    metrics.trajectory = trajectory
    return metrics


def run_experiment(config: ExperimentConfig):
    """Run a full experiment."""
    print(f"\n{'='*60}")
    print(f"Running experiment: {config.model_provider.value} / {config.model_name}")
    print(f"Episodes: {config.num_episodes}")
    print(f"{'='*60}\n")

    # Set random seed
    rng = set_seed(config.seed)

    # Initialize policy
    print(f"Initializing LLM policy...")
    policy = LLMChemotaxisPolicy(model_provider=config.model_provider, model_name=config.model_name)
    print(f"Policy initialized.\n")

    # Create output directory
    output_dir = create_output_dir(
        base_dir=config.output_dir,
        model_provider=config.model_provider.value,
        model_name=config.model_name,
        config=config.model_dump(),
    )
    print(f"Output directory: {output_dir}\n")

    # Initialize metrics
    experiment_metrics = ExperimentMetrics(
        model_provider=config.model_provider.value, model_name=config.model_name, num_episodes=config.num_episodes
    )

    # Run episodes
    for episode_id in range(config.num_episodes):
        print(f"Episode {episode_id + 1}/{config.num_episodes}...", end=" ", flush=True)

        # Create environment for this episode
        env = ChemotaxisPOMDP(config.episode_config, rng=rng)

        # Run episode
        episode_metrics = run_episode(env, policy, episode_id, rng)
        experiment_metrics.episodes.append(episode_metrics)

        # Print episode result
        if episode_metrics.success:
            print(f"✓ Success in {episode_metrics.steps_to_hit} steps")
        else:
            print(f"✗ Failed (final distance: {episode_metrics.final_distance_to_source})")

    # Save results
    print(f"\nSaving results to {output_dir}...")

    # Save configuration
    save_config(config.model_dump(), output_dir)

    # Save metrics
    metrics_dict = experiment_metrics.to_dict()
    save_metrics(metrics_dict, output_dir)

    # Save all trajectories
    all_trajectories = []
    for episode_metrics in experiment_metrics.episodes:
        for record in episode_metrics.trajectory:
            record["episode_id"] = episode_metrics.episode_id
            all_trajectories.append(record)
    save_trajectory(all_trajectories, output_dir)

    # Print summary
    summary = experiment_metrics.compute_summary()
    print(f"\n{'='*60}")
    print("Experiment Summary")
    print(f"{'='*60}")
    print(f"Hit rate: {summary['hit_rate']:.2%}")
    if summary["mean_steps_to_hit"]:
        print(f"Mean steps to hit: {summary['mean_steps_to_hit']:.1f}")
    print(f"Mean total steps: {summary['mean_total_steps']:.1f}")
    print(f"Mean runs: {summary['mean_runs']:.1f}")
    print(f"Mean tumbles: {summary['mean_tumbles']:.1f}")
    print(f"Excess tumbles: {summary['excess_tumbles']:.1f}")
    print(f"{'='*60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Stage 1: LLM as inference-only controller for E. coli chemotaxis")

    # Model configuration
    parser.add_argument(
        "--model-provider",
        type=str,
        choices=["ollama", "huggingface"],
        required=True,
        help="LLM provider: 'ollama' or 'huggingface'",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model identifier (e.g., 'mistral' for Ollama or 'microsoft/phi-2' for HuggingFace)",
    )

    # Experiment configuration
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to run (default: 10)")
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory for results (default: results)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: None for random)")

    # Episode configuration
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps per episode (default: 100)")
    parser.add_argument(
        "--noise-std", type=float, default=0.1, help="Observation noise standard deviation (default: 0.1)"
    )
    parser.add_argument(
        "--observation-history", type=int, default=5, help="Number of past observations to include (default: 5)"
    )
    parser.add_argument(
        "--slip-probability", type=float, default=0.0, help="Probability of direction slip on RUN (default: 0.0)"
    )
    parser.add_argument(
        "--success-distance", type=int, default=1, help="Success if within this distance of source (default: 1)"
    )
    parser.add_argument(
        "--decay-length", type=float, default=3.0, help="Concentration decay length scale (default: 3.0)"
    )

    args = parser.parse_args()

    # Build configuration
    episode_config = EpisodeConfig(
        max_steps=args.max_steps,
        noise_std=args.noise_std,
        observation_history=args.observation_history,
        slip_probability=args.slip_probability,
        success_distance=args.success_distance,
        decay_length=args.decay_length,
    )

    experiment_config = ExperimentConfig(
        model_provider=ModelProvider(args.model_provider),
        model_name=args.model_name,
        num_episodes=args.num_episodes,
        episode_config=episode_config,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # Run experiment
    run_experiment(experiment_config)


if __name__ == "__main__":
    main()
