from __future__ import annotations

from typing import Callable


def steps_to_goal(
    trajectory: list[dict],
    goal_threshold: float = 1.0,
) -> int | None:
    """
    Steps taken to reach goal. Returns None if goal not reached.
    Counts `step()` calls (rows where action_taken != 'reset') until distance_to_source <= threshold.
    """
    steps = 0
    for step in trajectory:
        if step.get("action_taken") == "reset":
            continue
        steps += 1
        if float(step.get("distance_to_source", 1e9)) <= goal_threshold:
            return steps
    return None


def path_efficiency(
    trajectory: list[dict],
    goal_threshold: float = 1.0,
) -> float:
    """
    Ratio of optimal (straight-line) distance to actual steps taken.
    = optimal_distance / actual_steps_taken
    Clipped to [0, 1]. Returns 0 if goal not reached.
    """
    if not trajectory:
        return 0.0
    stg = steps_to_goal(trajectory, goal_threshold=goal_threshold)
    if stg is None:
        return 0.0
    d0 = None
    for step in trajectory:
        if step.get("action_taken") == "reset":
            d0 = float(step.get("distance_to_source", 0.0))
            break
    if d0 is None:
        d0 = float(trajectory[0].get("distance_to_source", 0.0))
    optimal_distance = max(d0, 1e-9)
    actual_steps = stg
    ratio = optimal_distance / float(actual_steps)
    return float(max(0.0, min(1.0, ratio)))


def final_distance(trajectory: list[dict]) -> float:
    """Distance to source at final step."""
    if not trajectory:
        return float("nan")
    return float(trajectory[-1].get("distance_to_source", float("nan")))


def fraction_closer(trajectory: list[dict]) -> float:
    """
    Fraction of non-stay steps that moved closer to source.
    (proxy for directional accuracy)
    """
    prev_d: float | None = None
    moved = 0
    closer = 0
    for step in trajectory:
        if step.get("action_taken") in (None, "reset"):
            prev_d = float(step["distance_to_source"])
            continue
        a = step.get("action_taken", "stay")
        d = float(step["distance_to_source"])
        if prev_d is not None and a != "stay":
            moved += 1
            if d < prev_d:
                closer += 1
        prev_d = d
    if moved == 0:
        return 0.0
    return closer / moved


def mean_reward(trajectory: list[dict], rewards: list[float]) -> float:
    """Mean reward per step."""
    if not rewards:
        return 0.0
    return float(sum(rewards) / len(rewards))


def success_rate(
    trajectories: list[list[dict]],
    goal_threshold: float = 1.0,
) -> float:
    """
    Across N episodes, fraction where goal was reached.
    trajectories is a list of N trajectory lists.
    """
    if not trajectories:
        return 0.0
    ok = 0
    for tr in trajectories:
        if steps_to_goal(tr, goal_threshold=goal_threshold) is not None:
            ok += 1
    return ok / len(trajectories)
