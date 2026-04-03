"""
Behavioral metrics computed from trajectories.
Strategy is inferred from these — never from self-reported reasoning.
"""
from __future__ import annotations

from collections import Counter


def compute_ecoli_metrics(trajectory: list[dict]) -> dict:
    """
    Compute behavioral metrics from an E. coli trajectory.
    Each entry in trajectory has: step, position, reward, action_type, reason.
    """
    if not trajectory:
        return {}

    positions = [e["position"] for e in trajectory]
    rewards = [e["reward"] for e in trajectory]
    actions = [e["action_type"] for e in trajectory]
    reasons = [e.get("reason", "") for e in trajectory]

    total_reward = sum(rewards)
    coverage = len(set(positions))

    # first step with non-zero reward
    first_reward_step = None
    for e in trajectory:
        if e["reward"] > 0:
            first_reward_step = e["step"]
            break

    # action distribution
    action_counts = dict(Counter(actions))

    # directional bias: net displacement over time
    displacements = []
    for i in range(1, len(positions)):
        displacements.append(positions[i] - positions[i - 1])
    net_displacement = sum(displacements)

    # said-vs-did: does "Left"/"Right" in reason match actual action?
    # simple heuristic — count agreement between stated direction words and action
    said_did_matches = 0
    said_did_total = 0
    for act, reason in zip(actions, reasons):
        reason_lower = reason.lower()
        if act in ("Left", "Right"):
            said_did_total += 1
            if act.lower() in reason_lower:
                said_did_matches += 1
    said_did_rate = said_did_matches / said_did_total if said_did_total > 0 else None

    # reward trend: mean reward in first half vs second half
    mid = len(rewards) // 2
    reward_first_half = sum(rewards[:mid]) / max(mid, 1)
    reward_second_half = sum(rewards[mid:]) / max(len(rewards) - mid, 1)

    return {
        "total_reward": round(total_reward, 4),
        "coverage_unique_positions": coverage,
        "first_reward_step": first_reward_step,
        "action_counts": action_counts,
        "net_displacement": net_displacement,
        "said_vs_did_rate": said_did_rate,
        "reward_first_half_mean": round(reward_first_half, 4),
        "reward_second_half_mean": round(reward_second_half, 4),
        "steps_run": len(trajectory),
        "final_position": positions[-1],
    }
