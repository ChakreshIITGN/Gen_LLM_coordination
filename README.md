# Gen_LLM_coordination
Repo for the codes on generalising and studying the coordination between LLM agennts

### Python environment setup (using `uv`)

This project uses **`uv`** for fast, reproducible Python environments.

#### 1. Install `uv`

**macOS / Linux**
```sh
    curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then to initiate

```sh
    uv venv
    uv sync
```

For Mac/Linux
```sh
    source .venv/bin/activate
```

Powershell
```sh
    .\.venv\Scripts\Activate.ps1
```

# LLM-as-Agent ABM — Minimal Ring World Experiments

This repo is a minimal, **scientifically interpretable** starting point for using a **local LLM as an agent policy** inside an agent-based simulation (ABM).

The focus is on:
- clear world rules
- strict action interfaces (JSON → validated)
- trajectory logging
- **behavioral** evaluation (strategy inference from metrics), not self-reported reasoning

---

## What we built today

### Core idea: LLM = policy
A **policy** maps an agent’s observation to an action:

> observation → action

In this project:
- the simulator defines the world and its transition rules
- the LLM only chooses actions based on a structured observation
- the simulator validates and applies actions deterministically

No “free-form text actions”.

---

## Minimal environment: Ring line world

### World
- 1D ring of length **L = 20**, positions `0..19`
- periodic boundary (wrap-around):
  - `19 + 1 → 0`
  - `0 - 1 → 19`

### Rewards
- a set of reward locations with values (e.g., `{3: 5.0, 9: 10.0, 14: 7.0}`)
- if the agent lands on a reward position, it collects it (and the reward is removed)

### Energy + movement cost
- agent has energy `E`
- movement consumes energy
- **variable movement**:
  - action requests `steps` (e.g., move 5 steps)
  - actual executed steps are **clipped** by available energy
- reward also replenishes energy:
  - energy gain is proportional to reward value:
    - `ΔE = floor(alpha * reward_value)`
    - current default: `alpha = 0.3`
- energy is capped:
  - `E <- min(MAX_ENERGY, E + ΔE)`
  - prevents runaway “infinite energy” loops

### Episode termination
An episode ends when:
- max steps `T` is reached (e.g., `T=500`), OR
- energy reaches 0

---

## Agent / LLM assumptions (very important)

### No learning by default
The LLM **does not learn** during simulation:
- no weight updates
- no persistent memory across episodes
- each run is independent unless you explicitly feed history back in the prompt

### What we measure instead
We infer “strategy” from **observable behavior**:
- total reward
- coverage (unique positions visited)
- time to first reward
- action and direction counts conditioned on energy
- requested vs executed steps (shows infeasible planning / clipping)

This is more reproducible than relying on model explanations.

---

## Project outputs

Each run writes:
- `trajectory.jsonl` — one JSON record per step (full trace)
- `metrics.json` — summary metrics for quick comparisons
- `config.json` — resolved experiment parameters (for reproducibility)

Example step record fields include:
- `t`, `obs`, `action`, `raw_llm_output`
- `x_before`, `x_after`, `energy_before`, `energy_after`
- `reward_gained`, `reward_total_so_far`
- `steps_executed`, `energy_gained_from_reward`

---

## Requirements

### Local LLM runtime
We use **Ollama** locally. You must have it installed and running.

Pull a model (example):
```bash
ollama pull mistral


