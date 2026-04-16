from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StepResult:
    observation: float  # noisy concentration reading at current position
    true_concentration: float  # ground truth (for logging only, never given to agent)
    reward: float
    done: bool
    info: dict  # must include: position, distance_to_source, step_num


class Environment(ABC):
    """Abstract base for all taxisim environments."""

    def __init__(self, noise_sigma: float = 0.0, max_steps: int = 200, seed: int = 42):
        self.noise_sigma = noise_sigma
        self.max_steps = max_steps
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.step_count = 0

    @abstractmethod
    def reset(self) -> StepResult:
        """Reset environment. Returns initial observation."""

    @abstractmethod
    def step(self, action: str) -> StepResult:
        """Take action, return StepResult. Action is a string from action_space."""

    @property
    @abstractmethod
    def action_space(self) -> list[str]:
        """List of valid action strings."""

    @abstractmethod
    def concentration(self, position: Any) -> float:
        """True (noiseless) concentration at given position."""

    def _add_noise(self, value: float) -> float:
        if self.noise_sigma == 0.0:
            return value
        return float(value + self.rng.normal(0, self.noise_sigma))
