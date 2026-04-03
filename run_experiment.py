#!/usr/bin/env python3
"""
CLI entrypoint for running experiments.
Usage: python run_experiment.py --config experiments/configs/ecoli_memory.json
"""
import argparse
import json

from src.llm_abm.core.config import ExperimentConfig
from src.llm_abm.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run an LLM-ABM experiment")
    parser.add_argument(
        "--config", required=True,
        help="Path to experiment config JSON file",
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    runner = ExperimentRunner(config)
    metrics = runner.run()

    print("\n── Metrics ──")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
