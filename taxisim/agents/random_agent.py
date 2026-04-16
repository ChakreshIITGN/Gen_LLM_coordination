import numpy as np

from taxisim.agents.base import Agent, AgentConfig, AgentStep


class RandomAgent(Agent):
    """Uniform random policy over the action space."""

    def __init__(self, config: AgentConfig, action_space: list[str]):
        super().__init__(config, action_space)
        self._rng = np.random.default_rng(config.seed)

    def act(self, observation: float, position) -> AgentStep:
        action = str(self._rng.choice(self.action_space))
        self.update_history(observation, position)
        return AgentStep(action=action, reasoning="random")
