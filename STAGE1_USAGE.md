# Stage 1: LLM as Inference-Only Controller

This document describes how to use Stage 1 of the E. coli chemotaxis POMDP experiment.

## Overview

Stage 1 tests whether an LLM can implement the fixed bacterial chemotaxis rule from prompt alone, without any training or learning.

## Environment

The POMDP models E. coli chemotaxis in a 1D world:
- **State (hidden)**: Position (0-20), direction (-1 or +1), source location
- **Observation**: Noisy concentration history (last k readings)
- **Actions**: RUN (continue) or TUMBLE (reorient)
- **Reward**: Sparse (1.0 if within distance ≤1 of source, 0.0 otherwise)

## Usage

### With Ollama

First, make sure Ollama is running and you have a model pulled:

```bash
ollama pull mistral
```

Then run:

```bash
python main.py \
    --model-provider ollama \
    --model-name mistral \
    --num-episodes 10 \
    --noise-std 0.1 \
    --max-steps 100
```

### With HuggingFace

Start with a **small model** for faster runs and lower memory:

```bash
# SmolLM-360M (~360M params) — smallest, good for quick tests
python main.py \
    --model-provider huggingface \
    --model-name HuggingFaceTB/SmolLM-360M-Instruct \
    --num-episodes 10 \
    --noise-std 0.1 \
    --max-steps 100
```

Other small options:
- **TinyLlama 1.1B**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Qwen2 0.5B**: `Qwen/Qwen2-0.5B-Instruct`
- **Phi-2 2.7B** (larger): `microsoft/phi-2`

## Command-Line Arguments

### Model Configuration
- `--model-provider`: `ollama` or `huggingface` (required)
- `--model-name`: Model identifier (required)
  - For Ollama: e.g., `mistral`, `llama2`, `tinyllama`
  - For HuggingFace (start small): `HuggingFaceTB/SmolLM-360M-Instruct`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `Qwen/Qwen2-0.5B-Instruct`; larger: `microsoft/phi-2`

### Experiment Configuration
- `--num-episodes`: Number of episodes to run (default: 10)
- `--output-dir`: Output directory (default: `results`)
- `--seed`: Random seed (default: None)

### Episode Configuration
- `--max-steps`: Maximum steps per episode (default: 100)
- `--noise-std`: Observation noise standard deviation (default: 0.1)
- `--observation-history`: Number of past observations k (default: 5)
- `--slip-probability`: Probability of direction slip on RUN (default: 0.0)
- `--success-distance`: Success if within this distance of source (default: 1)
- `--decay-length`: Concentration decay length scale (default: 3.0)

## Output

Results are saved in `{output_dir}/{model_provider}_{model_name}_{timestamp}/`:

- `config.json`: Experiment configuration
- `metrics.json`: Summary metrics and per-episode results
- `trajectory.jsonl`: Full trajectory log (one JSON record per step)

### Metrics

- **Hit rate**: Fraction of episodes that reached the source
- **Mean steps to hit**: Average steps to success (for successful episodes)
- **Mean total steps**: Average episode length
- **Mean runs/tumbles**: Action distribution
- **Excess tumbles**: Tumbles beyond runs (indicates inefficient behavior)

## Example Output

```
============================================================
Running experiment: ollama / mistral
Episodes: 10
============================================================

Initializing LLM policy...
Policy initialized.

Output directory: results/ollama_mistral_20240101_120000

Episode 1/10... ✓ Success in 23 steps
Episode 2/10... ✗ Failed (final distance: 3)
Episode 3/10... ✓ Success in 45 steps
...

============================================================
Experiment Summary
============================================================
Hit rate: 60.00%
Mean steps to hit: 34.0
Mean total steps: 78.5
Mean runs: 45.2
Mean tumbles: 33.3
Excess tumbles: 0.0
============================================================
```

## Prompt Format

The LLM receives a minimal prompt with:
- Recent concentration readings: `[0.123, 0.145, 0.167, ...]`
- Last action (if available): `RUN` or `TUMBLE`
- Instruction to respond with `RUN` or `TUMBLE`

The prompt is kept constant across episodes (no accumulating transcript).

## Notes

- The LLM does **not** receive position, direction, or source location
- Each episode uses a random source location
- The prompt is designed to test temporal inference (comparing concentration trends)
- Results are evaluated on unseen source locations and noise seeds
