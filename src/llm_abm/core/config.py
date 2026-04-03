"""
Experiment configuration hierarchy.
All configs serialize to/from JSON for reproducibility.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


class RewardConfig(BaseModel):
    """Reward landscape definition."""
    type: Literal["exponential_decay", "discrete_locations"] = "exponential_decay"

    # exponential_decay params
    peak_position: float = 20.0
    initial_value: float = 10.0
    decay_rate: float = 0.1

    # discrete_locations params (position -> value)
    locations: dict[int, float] = Field(default_factory=dict)


class WorldConfig(BaseModel):
    """World geometry and reward setup."""
    length: int = 50
    boundary: Literal["blocking", "periodic"] = "blocking"
    reward: RewardConfig = Field(default_factory=RewardConfig)
    include_gradient: bool = True  # expose neighbor rewards in observation

    # ring world energy system (ignored for ecoli)
    start_energy: int = 100
    move_cost: int = 1
    max_energy: int = 100
    energy_alpha: float = 0.5  # reward -> energy conversion


class AgentConfig(BaseModel):
    """Agent initialization."""
    num_agents: int = 1
    start_position: int | list[int] = 30


class LLMConfig(BaseModel):
    """LLM provider settings."""
    provider: Literal["ollama"] = "ollama"
    model: str = "mistral"
    temperature: float = 1.0
    max_tokens: int = 250
    base_url: str = "http://localhost:11434"
    max_concurrent: int = 4  # for Phase 2 parallel runner


class PolicyConfig(BaseModel):
    """Policy selection and parameters."""
    type: Literal["llm_memory", "llm_no_memory", "random", "greedy"] = "llm_memory"
    memory_k: int = 5  # last k decisions visible to LLM


class LoggingConfig(BaseModel):
    """Output directory and tagging."""
    output_dir: str = "experiments/runs"
    experiment_tag: str = "default"


class ExperimentConfig(BaseModel):
    """Top-level experiment config. Load from JSON, save as artifact."""
    experiment_name: str = "unnamed"
    experiment_type: Literal["ecoli", "ring_world"] = "ecoli"
    world: WorldConfig = Field(default_factory=WorldConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    episode_length: int = 500
    seed: int = 0

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        """Load config from a JSON file."""
        return cls.model_validate_json(Path(path).read_text())

    def to_json(self, path: str | Path) -> None:
        """Save config as a JSON artifact."""
        Path(path).write_text(
            self.model_dump_json(indent=2), encoding="utf-8"
        )
