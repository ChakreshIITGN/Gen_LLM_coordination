from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

from taxisim.agents.base import Agent, AgentConfig, AgentStep


class BayesianAgent(Agent):
    """
    Ideal agent: knows C(x|source) functional form, infers source via Bayesian updates,
    moves toward posterior mean of source location.

    If observations include a known additive bias ``b`` (``obs = C + b + noise``),
    set ``observation_bias=b`` so likelihoods use ``obs - b`` (equivalent to ``N(C, σ²)``).
    """

    def __init__(
        self,
        config: AgentConfig,
        action_space: list[str],
        env_length: int = 100,
        decay_length: float = 10.0,
        peak_concentration: float = 100.0,
        noise_sigma: float = 1.0,
        observation_bias: float = 0.0,
        grid_bins_1d: int = 100,
        grid_shape_2d: tuple[int, int] = (10, 10),
        width: float = 20.0,
        height: float = 20.0,
    ):
        super().__init__(config, action_space)
        self.env_length = float(env_length)
        self.decay_length = decay_length
        self.peak_concentration = peak_concentration
        self.noise_sigma = max(float(noise_sigma), 1e-9)
        # Known additive shift on observations: obs = C(x|s) + bias + noise (e.g. fold-change wrapper).
        self.observation_bias = float(observation_bias)
        self.grid_bins_1d = grid_bins_1d
        self.grid_shape_2d = grid_shape_2d
        self.width = width
        self.height = height
        self._is_2d = "north" in action_space
        self._posterior: np.ndarray
        self._grid_1d: np.ndarray | None = None
        self._grid_sx: np.ndarray | None = None
        self._grid_sy: np.ndarray | None = None
        self._reset_posterior()

    def _reset_posterior(self) -> None:
        if self._is_2d:
            gx, gy = self.grid_shape_2d
            self._posterior = np.ones((gx, gy), dtype=np.float64)
            self._posterior /= self._posterior.sum()
            sx = np.linspace(0.0, self.width, gx)
            sy = np.linspace(0.0, self.height, gy)
            self._grid_sx, self._grid_sy = np.meshgrid(sx, sy, indexing="ij")
        else:
            self._grid_1d = np.linspace(0.0, self.env_length, self.grid_bins_1d)
            self._posterior = np.ones(self.grid_bins_1d, dtype=np.float64)
            self._posterior /= self._posterior.sum()

    def reset(self) -> None:
        super().reset()
        self._reset_posterior()

    def _conc_1d(self, x: float, source: float) -> float:
        dist = abs(x - source)
        return self.peak_concentration * math.exp(-dist / self.decay_length)

    def _conc_2d(self, x: float, y: float, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
        """Vectorized concentration over candidate source locations (meshgrid arrays)."""
        r = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
        return self.peak_concentration * np.exp(-r / self.decay_length)

    def _update_1d(self, obs: float, x: float) -> None:
        assert self._grid_1d is not None
        means = np.array([self._conc_1d(x, s) for s in self._grid_1d])
        centered = float(obs) - self.observation_bias
        likelihood = stats.norm.pdf(centered, loc=means, scale=self.noise_sigma)
        self._posterior *= likelihood
        total = self._posterior.sum()
        if total > 0:
            self._posterior /= total
        else:
            self._reset_posterior()

    def _update_2d(self, obs: float, pos: tuple[float, float]) -> None:
        assert self._grid_sx is not None and self._grid_sy is not None
        x, y = pos
        means = self._conc_2d(x, y, self._grid_sx, self._grid_sy)
        centered = float(obs) - self.observation_bias
        likelihood = stats.norm.pdf(centered, loc=means, scale=self.noise_sigma)
        self._posterior *= likelihood
        total = self._posterior.sum()
        if total > 0:
            self._posterior /= total
        else:
            self._reset_posterior()

    def act(self, observation: float, position: Any) -> AgentStep:
        if self._is_2d:
            pos = (float(position[0]), float(position[1]))
            self._update_2d(observation, pos)
            assert self._grid_sx is not None and self._grid_sy is not None
            mean_sx = float(np.sum(self._posterior * self._grid_sx))
            mean_sy = float(np.sum(self._posterior * self._grid_sy))
            dx = mean_sx - pos[0]
            dy = mean_sy - pos[1]
            if abs(dx) >= abs(dy):
                action = "east" if dx > 0 else "west"
            else:
                action = "north" if dy > 0 else "south"
            if action not in self.action_space:
                action = "stay"
            entropy = float(-np.sum(self._posterior * np.log(self._posterior + 1e-300)))
            reasoning = f"posterior_mean=({mean_sx:.3f},{mean_sy:.3f}), H={entropy:.3f}"
        else:
            x = float(position)
            self._update_1d(observation, x)
            assert self._grid_1d is not None
            mean_s = float(np.sum(self._posterior * self._grid_1d))
            map_i = int(np.argmax(self._posterior))
            map_est = float(self._grid_1d[map_i])
            entropy = float(-np.sum(self._posterior * np.log(self._posterior + 1e-300)))
            if mean_s > x:
                action = "right"
            elif mean_s < x:
                action = "left"
            else:
                action = "stay"
            reasoning = f"posterior_mean={mean_s:.3f}, MAP={map_est:.3f}, H={entropy:.3f}"

        self.update_history(observation, position)
        return AgentStep(action=action, reasoning=reasoning)
