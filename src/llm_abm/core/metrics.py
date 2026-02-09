"""Metrics tracking for chemotaxis experiments."""

from typing import List, Dict, Any
from dataclasses import dataclass, field
from .schemas import Action


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""

    episode_id: int
    success: bool = False
    steps_to_hit: int = -1  # -1 if never hit
    total_steps: int = 0
    num_runs: int = 0
    num_tumbles: int = 0
    final_distance_to_source: int = -1
    trajectory: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "episode_id": self.episode_id,
            "success": self.success,
            "steps_to_hit": self.steps_to_hit,
            "total_steps": self.total_steps,
            "num_runs": self.num_runs,
            "num_tumbles": self.num_tumbles,
            "final_distance_to_source": self.final_distance_to_source,
            "trajectory_length": len(self.trajectory),
        }


@dataclass
class ExperimentMetrics:
    """Aggregated metrics across episodes."""

    model_provider: str
    model_name: str
    num_episodes: int
    episodes: List[EpisodeMetrics] = field(default_factory=list)

    def compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        if not self.episodes:
            return {}

        successes = [e.success for e in self.episodes]
        steps_to_hit = [e.steps_to_hit for e in self.episodes if e.success]
        total_steps = [e.total_steps for e in self.episodes]
        num_runs = [e.num_runs for e in self.episodes]
        num_tumbles = [e.num_tumbles for e in self.episodes]
        final_distances = [e.final_distance_to_source for e in self.episodes]

        summary = {
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "num_episodes": self.num_episodes,
            "hit_rate": sum(successes) / len(successes) if successes else 0.0,
            "median_steps_to_hit": float(sorted(steps_to_hit)[len(steps_to_hit) // 2]) if steps_to_hit else None,
            "mean_steps_to_hit": sum(steps_to_hit) / len(steps_to_hit) if steps_to_hit else None,
            "mean_total_steps": sum(total_steps) / len(total_steps) if total_steps else 0.0,
            "mean_runs": sum(num_runs) / len(num_runs) if num_runs else 0.0,
            "mean_tumbles": sum(num_tumbles) / len(num_tumbles) if num_tumbles else 0.0,
            "mean_final_distance": sum(final_distances) / len(final_distances) if final_distances else 0.0,
            "excess_tumbles": (
                sum(max(0, t - r) for t, r in zip(num_tumbles, num_runs)) / len(num_tumbles) if num_tumbles else 0.0
            ),
        }

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {"summary": self.compute_summary(), "episodes": [e.to_dict() for e in self.episodes]}
