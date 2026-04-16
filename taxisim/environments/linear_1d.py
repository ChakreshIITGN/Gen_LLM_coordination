from __future__ import annotations

from typing import Any

import numpy as np

from taxisim.environments.base import Environment, StepResult


class Linear1DEnvironment(Environment):
    """1D line world with exponential decay concentration toward a fixed source."""

    def __init__(
        self,
        length: int = 100,
        source_position: float = 50.0,
        decay_length: float = 10.0,
        noise_sigma: float = 0.0,
        max_steps: int = 200,
        seed: int = 42,
        start_position: float | None = None,
    ):
        super().__init__(noise_sigma=noise_sigma, max_steps=max_steps, seed=seed)
        self.length = float(length)
        self.source_position = source_position
        self.decay_length = decay_length
        self.start_position = start_position
        self._position: float = 0.0
        self._goal_threshold = 1.0

    @property
    def action_space(self) -> list[str]:
        return ["left", "right", "stay"]

    def concentration(self, position: Any) -> float:
        x = float(position)
        dist = abs(x - self.source_position)
        return 100.0 * np.exp(-dist / self.decay_length)

    def _distance_to_source(self, position: float) -> float:
        return abs(position - self.source_position)

    def _clip_position(self, x: float) -> float:
        return float(np.clip(x, 0.0, self.length))

    def _sample_start(self) -> float:
        """Uniform over positions with distance to source > 20."""
        s = self.source_position
        L = self.length
        regions: list[tuple[float, float]] = []
        if s - 20.0 > 0.0:
            regions.append((0.0, s - 20.0))
        if s + 20.0 < L:
            regions.append((s + 20.0, L))
        if not regions:
            return float(self.rng.uniform(0.0, L))
        lengths = [b - a for a, b in regions]
        total = sum(lengths)
        r = self.rng.uniform(0, total)
        acc = 0.0
        for (a, b), ln in zip(regions, lengths):
            acc += ln
            if r <= acc:
                return float(self.rng.uniform(a, b))
        a, b = regions[-1]
        return float(self.rng.uniform(a, b))

    def reset(self) -> StepResult:
        self.step_count = 0
        if self.start_position is not None:
            self._position = self._clip_position(float(self.start_position))
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
        old_pos = self._position
        old_dist = self._distance_to_source(old_pos)

        if action == "left":
            self._position = self._clip_position(self._position - 1.0)
        elif action == "right":
            self._position = self._clip_position(self._position + 1.0)
        elif action == "stay":
            pass
        else:
            raise ValueError(f"Invalid action: {action}")

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
