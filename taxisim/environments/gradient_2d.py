from __future__ import annotations

import math
from typing import Any

import numpy as np

from taxisim.environments.base import Environment, StepResult


class Gradient2DEnvironment(Environment):
    """2D grid with radial exponential decay from a point source."""

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        source_position: tuple[float, float] = (10.0, 10.0),
        decay_length: float = 3.0,
        noise_sigma: float = 0.0,
        max_steps: int = 300,
        seed: int = 42,
        start_position: tuple[float, float] | None = None,
    ):
        super().__init__(noise_sigma=noise_sigma, max_steps=max_steps, seed=seed)
        self.width = float(width)
        self.height = float(height)
        self.source_position = (float(source_position[0]), float(source_position[1]))
        self.decay_length = decay_length
        self.start_position = start_position
        self._position: tuple[float, float] = (0.0, 0.0)
        self._goal_threshold = 1.5

    @property
    def action_space(self) -> list[str]:
        return ["north", "south", "east", "west", "stay"]

    def concentration(self, position: Any) -> float:
        x, y = float(position[0]), float(position[1])
        sx, sy = self.source_position
        r = math.sqrt((x - sx) ** 2 + (y - sy) ** 2)
        return 100.0 * math.exp(-r / self.decay_length)

    def _distance_to_source(self, pos: tuple[float, float]) -> float:
        sx, sy = self.source_position
        return math.sqrt((pos[0] - sx) ** 2 + (pos[1] - sy) ** 2)

    def _clip(self, pos: tuple[float, float]) -> tuple[float, float]:
        return (
            float(np.clip(pos[0], 0.0, self.width)),
            float(np.clip(pos[1], 0.0, self.height)),
        )

    def _sample_start(self) -> tuple[float, float]:
        """Uniform over positions with distance to source > 20 (or max feasible)."""
        sx, sy = self.source_position
        # Try regions outside radius 20 from source within bounds
        min_r = 20.0
        for _ in range(10000):
            x = float(self.rng.uniform(0.0, self.width))
            y = float(self.rng.uniform(0.0, self.height))
            if self._distance_to_source((x, y)) > min_r:
                return (x, y)
        # Fallback: corners
        corners = [(0.0, 0.0), (self.width, 0.0), (0.0, self.height), (self.width, self.height)]
        return max(corners, key=lambda p: self._distance_to_source(p))

    def reset(self) -> StepResult:
        self.step_count = 0
        if self.start_position is not None:
            self._position = self._clip(
                (float(self.start_position[0]), float(self.start_position[1]))
            )
        else:
            self._position = self._sample_start()

        true_c = self.concentration(self._position)
        noisy = self._add_noise(true_c)
        dist = self._distance_to_source(self._position)
        info = {
            "position": self._position,
            "distance_to_source": dist,
            "step_num": self.step_count,
            "true_concentration": true_c,
            "noisy_concentration": noisy,
            "action_taken": "reset",
        }
        return StepResult(
            observation=noisy,
            true_concentration=true_c,
            reward=0.0,
            done=False,
            info=info,
        )

    def step(self, action: str) -> StepResult:
        x, y = self._position
        old_dist = self._distance_to_source(self._position)

        if action == "north":
            y = min(self.height, y + 1.0)
        elif action == "south":
            y = max(0.0, y - 1.0)
        elif action == "east":
            x = min(self.width, x + 1.0)
        elif action == "west":
            x = max(0.0, x - 1.0)
        elif action == "stay":
            pass
        else:
            raise ValueError(f"Invalid action: {action}")

        self._position = (x, y)
        self.step_count += 1
        new_dist = self._distance_to_source(self._position)

        reward = 0.0
        if new_dist < old_dist:
            reward += 1.0
        elif new_dist > old_dist:
            reward -= 0.1

        done = False
        if new_dist <= self._goal_threshold:
            reward += 100.0
            done = True
        elif self.step_count >= self.max_steps:
            reward -= 1.0
            done = True

        true_c = self.concentration(self._position)
        noisy = self._add_noise(true_c)
        info = {
            "position": self._position,
            "distance_to_source": new_dist,
            "step_num": self.step_count,
            "true_concentration": true_c,
            "noisy_concentration": noisy,
            "action_taken": action,
        }
        return StepResult(
            observation=noisy,
            true_concentration=true_c,
            reward=reward,
            done=done,
            info=info,
        )
