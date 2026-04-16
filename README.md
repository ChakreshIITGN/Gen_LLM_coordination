# taxisim

## Overview

**taxisim** is a research library for studying whether large language models can perform gradient-following navigation analogous to *E. coli* chemotaxis. It provides modular environments (1D line and 2D grid with smooth concentration fields), classical baselines, an LLM-driven agent with multiple backends, prompts (abstract vs biological framing), metrics, and multi-episode experiment runners with structured logging.

## Installation

```bash
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and set API keys if you use OpenAI or Anthropic, and optionally `HF_TOKEN` for higher Hugging Face Hub rate limits and gated models.

```bash
cp .env.example .env
```

The **`LLMAgent`** loads a nearby `.env` automatically (via `python-dotenv`’s `find_dotenv()` the first time the model is used), so `HF_TOKEN` does not need to be exported in the shell. If you still see unauthenticated Hub warnings, confirm `.env` is in the project directory (or parent) and contains `HF_TOKEN=hf_...` with no quotes unless your token includes spaces.

## Quick Start

```python
from taxisim.agents.base import AgentConfig
from taxisim.agents.random_agent import RandomAgent
from taxisim.environments.linear_1d import Linear1DEnvironment
from taxisim.experiments.runner import ExperimentConfig, ExperimentRunner

env = Linear1DEnvironment(seed=0)
agent = RandomAgent(AgentConfig(name="random", seed=0), env.action_space)
runner = ExperimentRunner(ExperimentConfig(experiment_name="demo", n_episodes=1, seed=0))
runner.run_experiment(env, agent)
```

## Running Experiments

Baseline sweep (all classical agents, 1D or 2D):

```bash
python scripts/run_baselines.py --env 1d --n-episodes 50 --output-dir results
python scripts/run_baselines.py --env 2d --n-episodes 50 --output-dir results
```

Memory ablation (LLM plus hill-climb and temporal-difference baselines):
```Qwen/Qwen2-0.5B-Instruct```

```bash
python scripts/run_memory_ablation.py --model Qwen/Qwen2-0.5B-Instruct --backend huggingface --n-episodes 50 --output-dir results
```

Noise sweep:

```bash
python scripts/run_noise_sweep.py --model Qwen/Qwen2-0.5B-Instruct --backend huggingface --n-episodes 50 --output-dir results
```

Fold-change (background concentration invariance):

```bash
python scripts/run_fold_change.py --model Qwen/Qwen2-0.5B-Instruct --backend huggingface --n-episodes 50 --output-dir results
```

## Supported Models

The primary target is **small Hugging Face causal / chat models (on the order of ≤1B parameters)** for fast iteration. Example checkpoints people commonly use with this repo include:

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B, chat-tuned)
- `Qwen/Qwen2-0.5B-Instruct` (0.5B, instruction-tuned)
- `HuggingFaceTB/SmolLM2-1.7B-Instruct` (1.7B, instruct family)
- `microsoft/phi-1_5` (1.3B, base-style; abstract prompts often work best)

OpenAI and Anthropic backends accept whatever model id your account exposes (e.g. `gpt-4o-mini`).

## Results Structure

Each experiment writes under `results/` (or your `--output-dir`):

- **Per-experiment CSV** from `ExperimentRunner.save_results`: `results/<name>_<timestamp>.csv`
- **Episode-level JSON Lines** (when enabled): `results/<experiment_name>_episodes.jsonl` with `meta`, `episode_summary`, and optional per-step `steps`
- **Figures** from experiment modules and `run_baselines.py`: PNG files at 150 DPI with timestamps in the filename

For `run_noise_sweep.py`, `run_memory_ablation.py`, and `run_fold_change.py`, filenames now also include a model tag (derived from `--model`) so outputs remain distinguishable across model swaps.

Success rates in summaries are accompanied by **Wilson score 95% intervals** on the aggregated DataFrame (`df.attrs["wilson_success_rate_95"]` where applicable).

## Extending

**New environment:** Subclass `taxisim.environments.base.Environment`, implement `reset`, `step`, `action_space`, and `concentration`, and return `StepResult` whose `info` includes at least `position`, `distance_to_source`, and `step_num`. Register the class in `taxisim.environments.__init__` for imports and add tests under `tests/test_environments.py`.

**New agent:** Subclass `taxisim.agents.base.Agent`, implement `act`, and use `AgentStep` to return the chosen string action plus optional reasoning text (required for LLM agents). Call `update_history` after you consume the latest observation so prompts and baselines stay consistent. Export the class from `taxisim.agents.__init__` and add tests in `tests/test_agents.py`.
