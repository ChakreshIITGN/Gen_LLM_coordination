from __future__ import annotations

import numpy as np

from taxisim.agents.base import Agent, AgentConfig, AgentStep


class TemporalDiffAgent(Agent):
    """Running mean + trend-based direction persistence (loosely methylation-like)."""

    def __init__(self, config: AgentConfig, action_space: list[str]):
        super().__init__(config, action_space)
        self._rng = np.random.default_rng(config.seed + 17)
        self._dir_1d: str = "right"
        self._dir_2d: str = "east"

    def reset(self) -> None:
        super().reset()
        self._dir_1d = "right"
        self._dir_2d = "east"

    def _running_mean(self, obs: float) -> float:
        window = list(self.history) + [obs]
        n = min(len(window), self.config.history_length)
        return float(np.mean(window[-n:]))

    def _opposite_1d(self, d: str) -> str:
        return "left" if d == "right" else "right"

    def _opposite_2d(self, d: str) -> str:
        opp = {"north": "south", "south": "north", "east": "west", "west": "east"}
        return opp.get(d, d)

    def _random_1d(self) -> str:
        return str(self._rng.choice(["left", "right"]))

    def _random_cardinal(self) -> str:
        opts = [a for a in ("north", "south", "east", "west") if a in self.action_space]
        return str(self._rng.choice(opts)) if opts else "stay"

    def act(self, observation: float, position) -> AgentStep:
        is_2d = "north" in self.action_space
        hist = list(self.history)
        mean_est = self._running_mean(observation)
        grad_est = 0.0
        if len(hist) >= 1:
            grad_est = float(observation - hist[-1])

        reasoning = f"running_mean≈{mean_est:.4f}, last_step_delta≈{grad_est:.4f}"

        if len(hist) < 2:
            action = self._dir_2d if is_2d else self._dir_1d
            if action not in self.action_space:
                action = self.action_space[0]
            self.update_history(observation, position)
            return AgentStep(action=action, reasoning=reasoning + "; warmup.")

        prev_trend = hist[-1] - hist[-2]
        increasing = observation > hist[-1]

        if is_2d:
            if increasing or prev_trend > 0:
                action = self._dir_2d
            else:
                if self._rng.random() < 0.8:
                    self._dir_2d = self._opposite_2d(self._dir_2d)
                    action = self._dir_2d
                else:
                    action = self._random_cardinal()
                    self._dir_2d = action if action != "stay" else self._dir_2d
        else:
            if increasing or prev_trend > 0:
                action = self._dir_1d
            else:
                if self._rng.random() < 0.8:
                    self._dir_1d = self._opposite_1d(self._dir_1d)
                    action = self._dir_1d
                else:
                    action = self._random_1d()
                    self._dir_1d = action

        if action not in self.action_space:
            action = "stay"
        self.update_history(observation, position)
        return AgentStep(action=action, reasoning=reasoning)
