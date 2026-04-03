"""
FastAPI backend for the experiment dashboard.
Serves experiment data AND launches experiments.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# resolve project root (dashboard/backend/main.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = PROJECT_ROOT / "experiments" / "runs"
CONFIGS_DIR = PROJECT_ROOT / "experiments" / "configs"

app = FastAPI(title="LLM-ABM Lab", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job tracker for running experiments ───────────────────

_jobs: dict[str, dict[str, Any]] = {}


# ── Read endpoints ──────────────────────────────────────────────────

@app.get("/api/experiments")
def list_experiments():
    """List all experiment runs, grouped by tag."""
    if not RUNS_DIR.exists():
        return []

    runs = []
    for tag_dir in sorted(RUNS_DIR.iterdir()):
        if not tag_dir.is_dir() or tag_dir.name.startswith("."):
            continue
        for run_dir in sorted(tag_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            config_path = run_dir / "config.json"
            metrics_path = run_dir / "metrics.json"
            run_info = {
                "id": f"{tag_dir.name}/{run_dir.name}",
                "tag": tag_dir.name,
                "timestamp": run_dir.name,
                "has_config": config_path.exists(),
                "has_metrics": metrics_path.exists(),
            }
            # attach summary fields from config if available
            if config_path.exists():
                cfg = json.loads(config_path.read_text())
                run_info["experiment_name"] = cfg.get("experiment_name", "")
                run_info["policy_type"] = cfg.get("policy", {}).get("type", "")
                run_info["model"] = cfg.get("llm", {}).get("model", "")
                run_info["episode_length"] = cfg.get("episode_length", 0)
            # attach headline metrics if available
            if metrics_path.exists():
                m = json.loads(metrics_path.read_text())
                run_info["total_reward"] = m.get("total_reward")
                run_info["coverage"] = m.get("coverage_unique_positions")
            runs.append(run_info)
    return runs


# ── Static /api/experiments/* routes MUST come before {tag}/{timestamp} catch-all ──

@app.get("/api/experiments/compare")
def compare_runs(runs: str):
    """Side-by-side metrics. Pass comma-separated run IDs: tag/timestamp,tag/timestamp"""
    run_ids = [r.strip() for r in runs.split(",") if r.strip()]
    results = []
    for run_id in run_ids:
        parts = run_id.split("/")
        if len(parts) != 2:
            continue
        tag, ts = parts
        metrics_path = RUNS_DIR / tag / ts / "metrics.json"
        config_path = RUNS_DIR / tag / ts / "config.json"
        entry = {"id": run_id}
        if metrics_path.exists():
            entry["metrics"] = json.loads(metrics_path.read_text())
        if config_path.exists():
            cfg = json.loads(config_path.read_text())
            entry["policy"] = cfg.get("policy", {}).get("type", "")
            entry["model"] = cfg.get("llm", {}).get("model", "")
        results.append(entry)
    return results


# ── Parameterized {tag}/{timestamp} routes ─────────────────────────

@app.get("/api/experiments/{tag}/{timestamp}/config")
def get_config(tag: str, timestamp: str):
    path = RUNS_DIR / tag / timestamp / "config.json"
    if not path.exists():
        raise HTTPException(404, "Config not found")
    return json.loads(path.read_text())


@app.get("/api/experiments/{tag}/{timestamp}/metrics")
def get_metrics(tag: str, timestamp: str):
    path = RUNS_DIR / tag / timestamp / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "Metrics not found")
    return json.loads(path.read_text())


@app.get("/api/experiments/{tag}/{timestamp}/trajectory")
def get_trajectory(tag: str, timestamp: str, limit: int = 500, offset: int = 0):
    """Paginated trajectory data."""
    path = RUNS_DIR / tag / timestamp / "trajectory.jsonl"
    if not path.exists():
        raise HTTPException(404, "Trajectory not found")

    lines = path.read_text().strip().splitlines()
    total = len(lines)
    selected = lines[offset : offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "data": [json.loads(line) for line in selected],
    }


@app.get("/api/experiments/{tag}/{timestamp}/trajectory/summary")
def get_trajectory_summary(tag: str, timestamp: str):
    """Pre-computed arrays for charting — avoids sending raw JSONL to frontend."""
    path = RUNS_DIR / tag / timestamp / "trajectory.jsonl"
    if not path.exists():
        raise HTTPException(404, "Trajectory not found")

    steps, positions, rewards, actions, reasons = [], [], [], [], []
    cumulative_reward = 0.0
    cum_rewards = []

    for line in path.read_text().strip().splitlines():
        row = json.loads(line)
        steps.append(row["step"])
        positions.append(row["position_after"])
        rewards.append(row["reward"])
        cumulative_reward += row["reward"]
        cum_rewards.append(round(cumulative_reward, 4))
        actions.append(row["action_type"])
        reasons.append(row.get("reason", ""))

    return {
        "steps": steps,
        "positions": positions,
        "rewards": rewards,
        "cumulative_rewards": cum_rewards,
        "actions": actions,
        "reasons": reasons,
    }


# ── Execution endpoints ─────────────────────────────────────────────

class RunRequest(BaseModel):
    config: dict  # raw ExperimentConfig JSON


class BatchRunRequest(BaseModel):
    config: dict
    num_repeats: int = 1
    seeds: list[int] | None = None


class MultiPolicyRequest(BaseModel):
    config: dict  # base config (world, agent, llm, episode_length, logging)
    policies: list[str] = ["greedy", "random", "llm_memory", "llm_no_memory"]
    num_repeats: int = 1


def _resolve_run_dir(config: dict) -> Path | None:
    """Find the latest run directory for a given experiment tag."""
    tag = config.get("logging", {}).get("experiment_tag", "")
    if not tag:
        return None
    tag_dir = RUNS_DIR / tag
    if not tag_dir.exists():
        return None
    runs = sorted(tag_dir.iterdir(), reverse=True)
    return runs[0] if runs else None


def _create_job(config: dict) -> tuple[str, Path]:
    """Create a job entry and temp config file. Returns (job_id, config_path)."""
    job_id = str(uuid.uuid4())[:8]
    config_path = PROJECT_ROOT / f".tmp_config_{job_id}.json"
    config_path.write_text(json.dumps(config, indent=2))
    _jobs[job_id] = {
        "status": "running",
        "started_at": time.time(),
        "config_path": str(config_path),
        "output": "",
        "run_id": None,
        "run_dir": None,
    }
    return job_id, config_path


@app.post("/api/experiments/run")
async def run_experiment(req: RunRequest):
    """Launch an experiment in a subprocess. Returns job_id for status polling."""
    job_id, config_path = _create_job(req.config)
    asyncio.create_task(_run_in_background(job_id, config_path, req.config))
    return {"job_id": job_id, "status": "running"}


@app.post("/api/experiments/batch-run")
async def batch_run(req: BatchRunRequest):
    """Launch N repeats of an experiment with different seeds."""
    seeds = req.seeds or list(range(req.num_repeats))
    job_ids = []
    for seed in seeds[:req.num_repeats]:
        config = json.loads(json.dumps(req.config))  # deep copy
        config["seed"] = seed
        # make tag unique per seed
        base_tag = config.get("logging", {}).get("experiment_tag", "experiment")
        config.setdefault("logging", {})["experiment_tag"] = f"{base_tag}_seed{seed}"
        job_id, config_path = _create_job(config)
        job_ids.append(job_id)
        asyncio.create_task(_run_in_background(job_id, config_path, config))
    batch_id = str(uuid.uuid4())[:8]
    _jobs[f"batch_{batch_id}"] = {
        "status": "running",
        "job_ids": job_ids,
        "started_at": time.time(),
    }
    return {"batch_id": batch_id, "job_ids": job_ids, "status": "running"}


async def _run_in_background(job_id: str, config_path: Path, config: dict):
    """Run the experiment CLI as a subprocess."""
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(PROJECT_ROOT / "run_experiment.py"),
            "--config", str(config_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode() if stdout else ""
        _jobs[job_id]["output"] = output
        _jobs[job_id]["status"] = "completed" if proc.returncode == 0 else "failed"
        # resolve run directory for live step reading
        run_dir = _resolve_run_dir(config)
        if run_dir:
            _jobs[job_id]["run_dir"] = str(run_dir)
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["output"] = str(e)
    finally:
        config_path.unlink(missing_ok=True)


@app.get("/api/experiments/jobs/active")
def active_jobs():
    """Count of currently running experiments."""
    running = [jid for jid, j in _jobs.items() if j["status"] == "running" and not jid.startswith("batch_")]
    return {"count": len(running), "job_ids": running}


@app.get("/api/experiments/job/{job_id}")
def get_job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    return _jobs[job_id]


@app.get("/api/experiments/job/{job_id}/steps")
def get_job_steps(job_id: str):
    """Read partial trajectory.jsonl for live step viewing during experiment."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = _jobs[job_id]
    # try to find run_dir from job or by resolving config
    run_dir = job.get("run_dir")
    if not run_dir:
        cfg_path = job.get("config_path")
        if cfg_path and Path(cfg_path).exists():
            cfg = json.loads(Path(cfg_path).read_text())
            resolved = _resolve_run_dir(cfg)
            if resolved:
                run_dir = str(resolved)
                job["run_dir"] = run_dir
    if not run_dir:
        return {"steps": [], "status": job["status"]}
    traj = Path(run_dir) / "trajectory.jsonl"
    if not traj.exists():
        return {"steps": [], "status": job["status"]}
    lines = traj.read_text().strip().splitlines()
    steps = [json.loads(line) for line in lines]
    return {"steps": steps, "status": job["status"]}


@app.post("/api/experiments/multi-policy-run")
async def multi_policy_run(req: MultiPolicyRequest):
    """Run all selected policies × N repeats. Returns grouped job IDs."""
    all_job_ids: dict[str, list[str]] = {}
    base_tag = req.config.get("logging", {}).get("experiment_tag", "experiment")

    for policy_type in req.policies:
        policy_jobs = []
        for seed in range(req.num_repeats):
            config = json.loads(json.dumps(req.config))
            config["seed"] = seed
            config["policy"] = {"type": policy_type}
            if policy_type in ("llm_memory", "llm_no_memory"):
                config["policy"]["memory_k"] = req.config.get("policy", {}).get("memory_k", 10)
            config.setdefault("logging", {})["experiment_tag"] = (
                f"{base_tag}_{policy_type}" if req.num_repeats == 1
                else f"{base_tag}_{policy_type}_seed{seed}"
            )
            config["experiment_name"] = f"{base_tag} ({policy_type}, seed={seed})"
            job_id, config_path = _create_job(config)
            policy_jobs.append(job_id)
            asyncio.create_task(_run_in_background(job_id, config_path, config))
        all_job_ids[policy_type] = policy_jobs

    batch_id = str(uuid.uuid4())[:8]
    _jobs[f"multi_{batch_id}"] = {
        "status": "running",
        "policy_jobs": all_job_ids,
        "started_at": time.time(),
    }
    return {"batch_id": batch_id, "policy_jobs": all_job_ids, "status": "running"}


@app.get("/api/experiments/analysis")
def get_analysis():
    """Group all runs by policy type, compute average metrics per policy."""
    if not RUNS_DIR.exists():
        return {"policies": {}}

    policy_data: dict[str, list[dict]] = {}
    for tag_dir in sorted(RUNS_DIR.iterdir()):
        if not tag_dir.is_dir() or tag_dir.name.startswith("."):
            continue
        for run_dir in sorted(tag_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            config_path = run_dir / "config.json"
            metrics_path = run_dir / "metrics.json"
            if not config_path.exists() or not metrics_path.exists():
                continue
            cfg = json.loads(config_path.read_text())
            metrics = json.loads(metrics_path.read_text())
            policy_type = cfg.get("policy", {}).get("type", "unknown")
            run_id = f"{tag_dir.name}/{run_dir.name}"
            policy_data.setdefault(policy_type, []).append({
                "run_id": run_id,
                "tag": tag_dir.name,
                "timestamp": run_dir.name,
                "metrics": metrics,
                "experiment_name": cfg.get("experiment_name", ""),
            })

    # Compute averages per policy
    result: dict[str, dict] = {}
    for policy, runs in policy_data.items():
        avg_metrics: dict[str, float] = {}
        metric_keys = set()
        for r in runs:
            for k, v in r["metrics"].items():
                if isinstance(v, (int, float)):
                    metric_keys.add(k)
        for k in metric_keys:
            vals = [r["metrics"][k] for r in runs if isinstance(r["metrics"].get(k), (int, float))]
            if vals:
                avg_metrics[k] = round(sum(vals) / len(vals), 4)
        result[policy] = {
            "count": len(runs),
            "runs": [{"run_id": r["run_id"], "tag": r["tag"], "timestamp": r["timestamp"]} for r in runs],
            "avg_metrics": avg_metrics,
        }
    return {"policies": result}


@app.post("/api/experiments/validate-config")
def validate_config(req: RunRequest):
    """Validate config JSON without running."""
    try:
        # import here to avoid circular deps at module level
        from src.llm_abm.core.config import ExperimentConfig
        ExperimentConfig.model_validate(req.config)
        return {"valid": True, "errors": None}
    except Exception as e:
        return {"valid": False, "errors": str(e)}


@app.post("/api/experiments/preview-prompt")
def preview_prompt(req: RunRequest):
    """Show the exact system prompt and sample step-0 observation the LLM will receive.
    For scientific transparency — user reviews before launching."""
    try:
        from src.llm_abm.core.config import ExperimentConfig
        from src.llm_abm.core.world import EcoliWorld
        from src.llm_abm.core.policies import ECOLI_SYSTEM_PROMPT, _build_obs_content

        cfg = ExperimentConfig.model_validate(req.config)
        world = EcoliWorld(cfg.world)

        start_pos = cfg.agent.start_position
        if isinstance(start_pos, list):
            start_pos = start_pos[0]

        initial_reward = world.reward_at(start_pos)
        obs = world.observe(step=0, position=start_pos, reward=initial_reward)
        sample_observation = _build_obs_content(obs)

        return {
            "system_prompt": ECOLI_SYSTEM_PROMPT,
            "sample_observation": sample_observation,
            "policy_type": cfg.policy.type,
            "memory_k": cfg.policy.memory_k,
            "model": cfg.llm.model,
            "temperature": cfg.llm.temperature,
            "world_length": cfg.world.length,
            "include_gradient": cfg.world.include_gradient,
            "start_position": start_pos,
            "episode_length": cfg.episode_length,
        }
    except Exception as e:
        raise HTTPException(400, str(e))


# ── Environment setup endpoints ─────────────────────────────────────

@app.get("/api/setup/status")
async def setup_status():
    """Check if Ollama is running and what models are available."""
    import httpx
    ollama_ok = False
    models = []
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                ollama_ok = True
                models = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass

    return {
        "ollama_running": ollama_ok,
        "models_available": models,
        "python_version": sys.version,
    }


@app.get("/api/configs/templates")
def list_config_templates():
    """List available example configs for the wizard."""
    if not CONFIGS_DIR.exists():
        return []
    templates = []
    for f in sorted(CONFIGS_DIR.glob("*.json")):
        cfg = json.loads(f.read_text())
        templates.append({
            "filename": f.name,
            "experiment_name": cfg.get("experiment_name", f.stem),
            "policy_type": cfg.get("policy", {}).get("type", ""),
            "config": cfg,
        })
    return templates
