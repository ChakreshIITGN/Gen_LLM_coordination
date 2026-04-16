from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from taxisim.agents.base import Agent
from taxisim.agents.llm_agent import LLMAgent
from taxisim.environments.linear_1d import Linear1DEnvironment
from taxisim.experiments.runner import ExperimentConfig, ExperimentRunner


def _normalized_filename_suffix(value: str) -> str:
    if not value:
        return ""
    return value if value.startswith("_") else f"_{value}"


def _markers_by_agent_name(names: list[str]) -> dict[str, str]:
    """Distinct markers so overlapping curves stay readable; ``llm`` uses a diamond."""
    baseline_markers = ("o", "s", "^", "v", "P", "X", "*", "h", "8", "<", ">")
    sorted_names = sorted(names)
    out: dict[str, str] = {}
    bi = 0
    for name in sorted_names:
        if name == "llm":
            out[name] = "D"
        else:
            out[name] = baseline_markers[bi % len(baseline_markers)]
            bi += 1
    return out


def _aggregate_action_fractions(df_ep: pd.DataFrame) -> dict[str, float]:
    """Fraction of each primitive action over all steps in the episode table."""
    nan = float("nan")
    out = {
        "action_frac_left": nan,
        "action_frac_right": nan,
        "action_frac_stay": nan,
    }
    if "actions" not in df_ep.columns or len(df_ep) == 0:
        return out
    c: Counter[str] = Counter()
    for acts in df_ep["actions"]:
        if acts is None or (isinstance(acts, float) and pd.isna(acts)):
            continue
        if isinstance(acts, (str, bytes)):
            c.update([str(acts)])
            continue
        try:
            c.update(acts)
        except TypeError:
            continue
    total = sum(c.values())
    if total <= 0:
        return out
    out["action_frac_left"] = float(c.get("left", 0)) / total
    out["action_frac_right"] = float(c.get("right", 0)) / total
    out["action_frac_stay"] = float(c.get("stay", 0)) / total
    return out


def plot_noise_sweep_summary_figures(
    out: pd.DataFrame,
    results_dir: str | Path,
    *,
    filename_suffix: str = "",
    ts: str | None = None,
) -> list[str]:
    """
    Write standard noise-sweep PNGs from an aggregated summary DataFrame
    (same schema as ``run_noise_sweep`` output). Returns paths of figures written.
    """
    sns.set_theme(style="whitegrid")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if ts is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = _normalized_filename_suffix(filename_suffix)
    names = sorted(out["agent_name"].unique())
    markers = _markers_by_agent_name(list(names))
    palette = sns.color_palette("colorblind", n_colors=max(3, len(names)))
    written: list[str] = []

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(names):
        sub = out[out["agent_name"] == name].sort_values("noise_sigma")
        ax.plot(
            sub["noise_sigma"],
            sub["success_rate"],
            marker=markers[name],
            markersize=7,
            label=name,
            color=palette[i % len(palette)],
        )
    ax.set_xlabel("noise_sigma")
    ax.set_ylabel("success_rate")
    ax.set_title("Noise sweep: success rate vs observation noise")
    ax.legend()
    fig.tight_layout()
    p1 = results_dir / f"noise_sweep_success{suffix}_{ts}.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    written.append(str(p1))

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(names):
        sub = out[out["agent_name"] == name].sort_values("noise_sigma")
        ax2.plot(
            sub["noise_sigma"],
            sub["mean_steps"],
            marker=markers[name],
            markersize=7,
            label=name,
            color=palette[i % len(palette)],
        )
    ax2.set_xlabel("noise_sigma")
    ax2.set_ylabel("mean_steps")
    ax2.set_title("Noise sweep: mean steps vs observation noise")
    ax2.legend()
    fig2.tight_layout()
    p2 = results_dir / f"noise_sweep_mean_steps{suffix}_{ts}.png"
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    written.append(str(p2))

    if "action_frac_right" in out.columns and out["action_frac_right"].notna().any():
        fig_r, ax_r = plt.subplots(figsize=(9, 5))
        for i, name in enumerate(names):
            sub = out[out["agent_name"] == name].sort_values("noise_sigma")
            if sub["action_frac_right"].notna().any():
                ax_r.plot(
                    sub["noise_sigma"],
                    sub["action_frac_right"],
                    marker=markers[name],
                    markersize=7,
                    label=name,
                    color=palette[i % len(palette)],
                )
        ax_r.set_xlabel("noise_sigma")
        ax_r.set_ylabel("action_frac_right (pooled over steps)")
        ax_r.set_title("Noise sweep: fraction of right actions vs observation noise")
        ax_r.set_ylim(-0.02, 1.02)
        ax_r.legend()
        fig_r.tight_layout()
        p3 = results_dir / f"noise_sweep_action_frac_right{suffix}_{ts}.png"
        fig_r.savefig(p3, dpi=150)
        plt.close(fig_r)
        written.append(str(p3))

    llm = out[out["agent_name"] == "llm"].sort_values("noise_sigma").copy()
    if len(llm) > 0 and "action_frac_left" in llm.columns and llm["action_frac_left"].notna().any():
        if "llm_parse_failure_rate" not in llm.columns:
            llm["llm_parse_failure_rate"] = float("nan")
        fig_l, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        x = llm["noise_sigma"].to_numpy()
        axes[0].plot(x, llm["action_frac_left"], marker="o", markersize=6, label="left")
        axes[0].plot(x, llm["action_frac_right"], marker="s", markersize=6, label="right")
        axes[0].plot(x, llm["action_frac_stay"], marker="^", markersize=6, label="stay")
        axes[0].set_ylabel("action fraction")
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].legend(loc="best")
        axes[0].set_title("LLM: pooled action fractions vs noise")

        axes[1].plot(
            x,
            llm["llm_parse_failure_rate"].to_numpy(dtype=float),
            marker="o",
            markersize=6,
            color=palette[min(len(palette) - 1, 4)],
        )
        axes[1].set_ylabel("parse_failure_rate")
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].set_title("LLM: parse failure rate (within noise cell)")

        axes[2].plot(
            x,
            llm["mean_path_efficiency"],
            marker="D",
            markersize=6,
            color=palette[min(len(palette) - 1, 3)],
        )
        axes[2].set_xlabel("noise_sigma")
        axes[2].set_ylabel("mean_path_efficiency")
        axes[2].set_ylim(-0.02, 1.02)
        axes[2].set_title("LLM: mean path efficiency vs noise")
        fig_l.tight_layout()
        p4 = results_dir / f"noise_sweep_llm_diagnostics{suffix}_{ts}.png"
        fig_l.savefig(p4, dpi=150)
        plt.close(fig_l)
        written.append(str(p4))

    return written


def run_noise_sweep(
    agents: dict[str, Agent],
    noise_levels: list[float] | None = None,
    env_kwargs: dict | None = None,
    n_episodes: int = 50,
    seed: int = 42,
    results_dir: str = "results",
    filename_suffix: str = "",
) -> pd.DataFrame:
    """
    For each agent × noise_level combination, run n_episodes.
    Returns DataFrame with success/steps/path metrics plus, when available,
    pooled action fractions over all steps and (for LLM) parse-failure rate
    within that noise cell.
    """
    if noise_levels is None:
        noise_levels = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]

    env_kwargs = env_kwargs or {}
    rows: list[dict] = []
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = _normalized_filename_suffix(filename_suffix)

    for noise in noise_levels:
        for agent_name, agent in agents.items():
            if hasattr(agent, "noise_sigma"):
                agent.noise_sigma = max(float(noise), 1e-9)
            ek = {**env_kwargs, "noise_sigma": noise, "seed": seed}
            env = Linear1DEnvironment(**ek)
            exp = ExperimentConfig(
                experiment_name=f"noise_sweep_{agent_name}_n{noise}{suffix}",
                n_episodes=n_episodes,
                max_steps=env.max_steps,
                seed=seed,
                results_dir=results_dir,
                log_every_step=False,
            )
            runner = ExperimentRunner(exp)
            pf0 = pa0 = None
            if isinstance(agent, LLMAgent):
                pf0, pa0 = int(agent.parse_failures), int(agent.parse_attempts)
            df_ep = runner.run_experiment(env, agent, reset_agent_per_episode=True)
            fracs = _aggregate_action_fractions(df_ep)
            parse_rate = float("nan")
            if isinstance(agent, LLMAgent) and pa0 is not None and pf0 is not None:
                d_pa = int(agent.parse_attempts) - pa0
                d_pf = int(agent.parse_failures) - pf0
                parse_rate = float(d_pf) / float(d_pa) if d_pa > 0 else float("nan")
            rows.append(
                {
                    "agent_name": agent_name,
                    "noise_sigma": noise,
                    "success_rate": float(df_ep["success"].mean()),
                    "mean_steps": float(df_ep["steps"].mean()),
                    "mean_path_efficiency": float(df_ep["path_efficiency"].mean()),
                    "std_steps": float(df_ep["steps"].std(ddof=1)) if len(df_ep) > 1 else 0.0,
                    **fracs,
                    "llm_parse_failure_rate": parse_rate,
                }
            )

    out = pd.DataFrame(rows)
    plot_noise_sweep_summary_figures(out, results_dir, filename_suffix=filename_suffix, ts=ts)
    out.to_csv(Path(results_dir) / f"noise_sweep_summary{suffix}_{ts}.csv", index=False)
    return out
