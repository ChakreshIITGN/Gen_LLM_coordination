"""Core schemas for the E. coli chemotaxis POMDP."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Action(str, Enum):
    """Action primitives for chemotaxis."""
    RUN = "RUN"
    TUMBLE = "TUMBLE"


class ModelProvider(str, Enum):
    """LLM provider options."""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class POMDPState(BaseModel):
    """Hidden state of the POMDP."""
    position: int = Field(ge=0, le=20, description="Agent position in 1D world")
    direction: int = Field(description="Direction: -1 (left) or +1 (right)")
    source_location: int = Field(ge=0, le=20, description="Source location x*")
    background_offset: float = Field(default=0.0, description="Background concentration offset")
    
    def __init__(self, **data):
        super().__init__(**data)
        assert self.direction in [-1, 1], "Direction must be -1 or +1"


class Observation(BaseModel):
    """Observation available to the agent."""
    concentrations: list[float] = Field(description="Last k noisy concentration observations")
    last_action: Optional[Action] = Field(default=None, description="Last action taken")


class EpisodeConfig(BaseModel):
    """Configuration for a single episode."""
    max_steps: int = Field(default=100, description="Horizon H")
    noise_std: float = Field(default=0.1, description="Observation noise σ")
    observation_history: int = Field(default=5, description="Number of past observations k")
    slip_probability: float = Field(default=0.0, description="Probability of direction slip on RUN")
    success_distance: int = Field(default=1, description="Success if within this distance of source")
    decay_length: float = Field(default=3.0, description="Concentration decay length scale")
    
    # Source location will be randomized per episode
    source_location: Optional[int] = Field(default=None, description="Source location (random if None)")


class ExperimentConfig(BaseModel):
    """Configuration for an experiment run."""
    model_provider: ModelProvider = Field(description="LLM provider (ollama or huggingface)")
    model_name: str = Field(description="Model identifier")
    num_episodes: int = Field(default=10, description="Number of episodes to run")
    episode_config: EpisodeConfig = Field(default_factory=EpisodeConfig)
    output_dir: str = Field(default="results", description="Output directory for results")
    seed: Optional[int] = Field(default=None, description="Random seed")
