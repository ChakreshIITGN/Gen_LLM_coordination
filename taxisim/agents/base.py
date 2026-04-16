from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    name: str
    history_length: int = 5  # how many past observations to retain
    seed: int = 42


@dataclass
class AgentStep:
    action: str
    reasoning: str = ""  # explanation (for LLM agents, the raw response)
    metadata: dict = field(default_factory=dict)


class Agent(ABC):
    def __init__(self, config: AgentConfig, action_space: list[str]):
        self.config = config
        self.action_space = action_space
        self.history: list[float] = []  # concentration history
        self.position_history: list = []  # position history

    def update_history(self, observation: float, position) -> None:
        self.history.append(observation)
        self.position_history.append(position)
        if len(self.history) > self.config.history_length:
            self.history.pop(0)
        if len(self.position_history) > self.config.history_length:
            self.position_history.pop(0)

    def reset(self) -> None:
        self.history = []
        self.position_history = []

    @abstractmethod
    def act(self, observation: float, position) -> AgentStep:
        """Given current observation and position, return action."""
