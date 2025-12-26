Here’s a simple repo structure that’s “serious enough” for CI/CD + Docker later, but still easy for experiments.

## Repo layout

```text
llm-abm-lab/
├─ README.md
├─ pyproject.toml
├─ .gitignore
├─ .env.example
├─ LICENSE
├─ Makefile
├─ docker/
│  ├─ Dockerfile
│  └─ docker-compose.yml
├─ scripts/
│  ├─ setup.sh
│  └─ run_exp.sh
├─ src/
│  └─ llm_abm/
│     ├─ __init__.py
│     ├─ core/
│     │  ├─ schemas.py        # pydantic models (World, Obs, Action)
│     │  ├─ world.py          # state transitions (pure functions)
│     │  ├─ policies.py       # LLM policy + baselines (random, greedy)
│     │  ├─ logging.py        # JSONL logger + run metadata
│     │  └─ metrics.py        # trajectory metrics (coverage, regret, etc.)
│     ├─ integrations/
│     │  └─ ollama_client.py  # OpenAI-compatible call wrapper
│     └─ utils/
│        └─ seed.py
├─ experiments/
│  ├─ exp_0_minimal_setup/
│  │  ├─ README.md            # design principle + what it tests
│  │  ├─ config.yaml          # world + model params
│  │  ├─ run.py               # entrypoint for this experiment
│  │  └─ expected/            # optional: golden outputs for tests
│  ├─ exp_1_ring_line_rewards/
│  │  ├─ README.md
│  │  ├─ config.yaml
│  │  └─ run.py
│  └─ exp_2_partial_observability/
│     ├─ README.md
│     ├─ config.yaml
│     └─ run.py
├─ tests/
│  ├─ test_world_transitions.py
│  ├─ test_schemas_validation.py
│  └─ test_metrics.py
└─ .github/
   └─ workflows/
      └─ ci.yml
```

### Why this works

* `src/llm_abm/core` is your reusable “engine”
* `experiments/exp_*` are thin wrappers: config + run script + experiment notes
* tests cover the engine; experiments can be messy without breaking CI

---

## What goes in `exp_0_minimal_setup`

Purpose: prove the plumbing works *without LLM*.

* world step function
* random policy
* logging to `runs/<timestamp>/trajectory.jsonl`
* metrics computed at end

This makes CI stable and fast.

---

## README design (top-level)

Include:

1. **Project goal** (LLM as policy in ABM)
2. **Design principles** (black-box policy, schema contracts, logging-first, reproducibility)
3. **Quickstart**

   * python env setup
   * run exp_0
   * run exp_1 (later includes Ollama)
4. **Experiments index** (table linking to each `experiments/exp_*/README.md`)
5. **Reproducibility**

   * seeds, configs, run artifacts
6. **CI/CD**

   * tests, lint (optional), docker (optional)

---

## “Naive but working” CI/CD from day 1

Keep it minimal:

### Testing

* `pytest` only
* unit tests for:

  * world transitions (including ring wrap)
  * schema validation
  * metrics correctness

### GitHub Actions

* run `pytest` on push/PR
* no Docker build yet (add later)

### Docker (optional but ready)

* Dockerfile that runs `pip install -e .` and can execute `python experiments/exp_0_minimal_setup/run.py`
* docker-compose later for Ollama integration (so others can run local LLM)

---

## Suggested conventions (keep it clean)

* Every experiment folder has:

  * `README.md` with: hypothesis, world design, observation/action schemas, metrics
  * `config.yaml` with: seed, world params, policy params, logging path
  * `run.py` that calls shared engine
* Every run produces:

  * `runs/exp_*/<timestamp>/config_resolved.yaml`
  * `trajectory.jsonl`
  * `metrics.json`

---

If you want, next message I can write:

* a ready-to-copy **top-level README.md skeleton** (short, practical)
* a minimal **ci.yml** (pytest)
* a minimal **pyproject.toml** (dependencies: pydantic, pytest, pyyaml)
