from __future__ import annotations

import numpy as np

from taxisim.agents.base import Agent, AgentConfig, AgentStep


class HillClimbAgent(Agent):
    """Compare current observation to previous; persist or reverse direction."""

    def __init__(self, config: AgentConfig, action_space: list[str]):
        super().__init__(config, action_space)
        self._rng = np.random.default_rng(config.seed)
        self._last_action: str | None = None
        self._dir_2d: str | None = None  # one of north/south/east/west

    def reset(self) -> None:
        super().reset()
        self._last_action = None
        self._dir_2d = None

    def _default_forward_1d(self) -> str:
        return "right" if "right" in self.action_space else self.action_space[0]

    def _default_forward_2d(self) -> str:
        for a in ("east", "north", "south", "west"):
            if a in self.action_space:
                return a
        return self.action_space[0]

    def _opposite_1d(self, a: str) -> str:
        if a == "left":
            return "right"
        if a == "right":
            return "left"
        return "stay"

    def _opposite_2d(self, a: str) -> str:
        opp = {"north": "south", "south": "north", "east": "west", "west": "east"}
        return opp.get(a, a)

    def _random_cardinal_2d(self) -> str:
        opts = [a for a in ("north", "south", "east", "west") if a in self.action_space]
        return str(self._rng.choice(opts)) if opts else "stay"

    def act(self, observation: float, position) -> AgentStep:
        is_2d = "north" in self.action_space

        if not is_2d:
            return self._act_1d(observation, position)
        return self._act_2d(observation, position)

    def _act_1d(self, observation: float, position) -> AgentStep:
        reasoning_parts: list[str] = []
        if len(self.history) < 1:
            action = self._default_forward_1d()
            reasoning_parts.append("No prior observation; default forward.")
        else:
            prev = self.history[-1]
            reasoning_parts.append(f"Compare current {observation:.4f} to previous {prev:.4f}.")
            if observation > prev:
                action = self._last_action if self._last_action in ("left", "right") else self._default_forward_1d()
                reasoning_parts.append("Signal increased; persist direction.")
            elif observation < prev:
                if self._last_action in ("left", "right"):
                    action = self._opposite_1d(self._last_action)
                else:
                    action = self._default_forward_1d()
                reasoning_parts.append("Signal decreased; reverse direction.")
            else:
                action = self._rng.choice(["left", "right"])
                reasoning_parts.append("Tie; random left/right.")

        self._last_action = action
        self.update_history(observation, position)
        return AgentStep(action=action, reasoning=" ".join(reasoning_parts))

    def _act_2d(self, observation: float, position) -> AgentStep:
        reasoning_parts: list[str] = []
        if len(self.history) < 1:
            self._dir_2d = self._default_forward_2d()
            action = self._dir_2d
            reasoning_parts.append("No prior observation; default cardinal.")
        else:
            prev = self.history[-1]
            reasoning_parts.append(f"Compare current {observation:.4f} to previous {prev:.4f}.")
            if observation > prev:
                d = self._dir_2d if self._dir_2d in self.action_space else self._default_forward_2d()
                action = d
                reasoning_parts.append("Signal increased; persist direction.")
            elif observation < prev:
                self._dir_2d = self._opposite_2d(self._dir_2d or self._default_forward_2d())
                action = self._dir_2d
                reasoning_parts.append("Signal decreased; reverse direction.")
            else:
                self._dir_2d = self._random_cardinal_2d()
                action = self._dir_2d
                reasoning_parts.append("Tie; random cardinal.")

        if action not in self.action_space:
            action = "stay"
        self._dir_2d = action if action in ("north", "south", "east", "west") else self._dir_2d
        self.update_history(observation, position)
        return AgentStep(action=action, reasoning=" ".join(reasoning_parts))
