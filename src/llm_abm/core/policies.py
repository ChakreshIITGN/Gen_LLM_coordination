"""
Policy implementations for E. coli and Ring World experiments.
Each policy instance owns its own state — no global message lists.

Policy = observation -> action. The LLM is just one possible policy.
"""
from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from typing import Any

from pydantic import ValidationError

from .config import LLMConfig, PolicyConfig
from .schemas import EcoliAction, EcoliObservation
from ..integrations.ollama_client import OllamaClient


# ── System prompt for E. coli LLM policies ─────────────────────────

ECOLI_SYSTEM_PROMPT = (
    "You are an agent on a 1D line. Your goal is to maximize total reward collected over time.\n"
    "Each step you observe your position, the reward at your current position, "
    "and (when available) the reward at adjacent positions.\n"
    "Output ONLY valid JSON.\n"
    'Schema: {"type": "Left"|"Right"|"Wait", "reason": "string"}\n'
    "Rules:\n"
    "- type must be exactly one of: Left, Right, Wait\n"
    "- reason must be one short sentence (max 200 chars)\n"
    "- Do not add extra fields or explanations"
)


# ── Base policy ─────────────────────────────────────────────────────

class BasePolicy(ABC):
    """All policies implement select_action and reset."""

    @abstractmethod
    def select_action(self, obs: EcoliObservation) -> tuple[EcoliAction, str]:
        """Returns (validated_action, raw_output_string)."""
        ...

    def reset(self) -> None:
        """Clear per-episode state. Override if stateful."""
        pass


# ── LLM with conversational memory ─────────────────────────────────

class LLMMemoryPolicy(BasePolicy):
    """
    Feeds last k (observation, action) pairs as chat history.
    This is the core test: can in-context memory enable adaptation?
    """

    def __init__(self, llm_config: LLMConfig, policy_config: PolicyConfig):
        self.client = OllamaClient(llm_config)
        self.memory_k = policy_config.memory_k
        # per-instance chat history — the critical fix over the old global list
        self._messages: list[dict] = [
            {"role": "system", "content": ECOLI_SYSTEM_PROMPT}
        ]

    def select_action(self, obs: EcoliObservation) -> tuple[EcoliAction, str]:
        # build user message with current observation
        user_msg = {
            "role": "user",
            "content": _build_obs_content(obs),
        }
        self._messages.append(user_msg)

        raw = self.client.call(self._messages)

        # parse and validate
        action = _parse_ecoli_action(raw)

        # store assistant response for memory continuity
        self._messages.append({"role": "assistant", "content": raw})

        # trim to system + last k exchanges (each exchange = user + assistant = 2 msgs)
        self._trim_history()

        return action, raw

    def _trim_history(self) -> None:
        """Keep system prompt + last memory_k exchanges."""
        max_msgs = 1 + self.memory_k * 2  # system + k*(user + assistant)
        if len(self._messages) > max_msgs:
            self._messages = [self._messages[0]] + self._messages[-self.memory_k * 2:]

    def reset(self) -> None:
        self._messages = [{"role": "system", "content": ECOLI_SYSTEM_PROMPT}]


# ── LLM without memory (stateless) ─────────────────────────────────

class LLMNoMemoryPolicy(BasePolicy):
    """
    Fresh prompt each step — no history. Stateless baseline.
    Tests: what can the LLM do with a single observation?
    """

    def __init__(self, llm_config: LLMConfig):
        self.client = OllamaClient(llm_config)

    def select_action(self, obs: EcoliObservation) -> tuple[EcoliAction, str]:
        messages = [
            {"role": "system", "content": ECOLI_SYSTEM_PROMPT},
            {"role": "user", "content": _build_obs_content(obs)},
        ]
        raw = self.client.call(messages)
        return _parse_ecoli_action(raw), raw


# ── Random baseline ─────────────────────────────────────────────────

class RandomPolicy(BasePolicy):
    """Uniform random over Left/Right/Wait. No LLM. Null hypothesis."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def select_action(self, obs: EcoliObservation) -> tuple[EcoliAction, str]:
        choice = self.rng.choice(["Left", "Right", "Wait"])
        action = EcoliAction(type=choice, reason="random baseline")
        return action, f'{{"type":"{choice}","reason":"random baseline"}}'


# ── Greedy baseline ─────────────────────────────────────────────────

class GreedyPolicy(BasePolicy):
    """
    Always moves toward higher reward gradient.
    Optimal single-step policy — tests ceiling performance.
    Needs access to the world to evaluate reward_at().
    """

    def __init__(self, reward_fn):
        """reward_fn: callable(x) -> float, e.g. world.reward_at"""
        self.reward_fn = reward_fn

    def select_action(self, obs: EcoliObservation) -> tuple[EcoliAction, str]:
        left_r = self.reward_fn(obs.position - 1)
        right_r = self.reward_fn(obs.position + 1)
        here_r = self.reward_fn(obs.position)

        if left_r >= right_r and left_r > here_r:
            choice = "Left"
        elif right_r > here_r:
            choice = "Right"
        else:
            choice = "Wait"

        action = EcoliAction(type=choice, reason=f"greedy: L={left_r:.2f} R={right_r:.2f}")
        return action, json.dumps(action.model_dump())


# ── Helpers ─────────────────────────────────────────────────────────

def _build_obs_content(obs: EcoliObservation) -> dict:
    """Build the user message content from an observation.
    Includes gradient info only when available (not None)."""
    content: dict = {
        "position": obs.position,
        "reward_at_position": obs.reward,
        "step": obs.step,
        "instruction": "Choose your next action: Left, Right, or Wait.",
    }
    if obs.reward_left is not None:
        content["reward_left"] = obs.reward_left
    if obs.reward_right is not None:
        content["reward_right"] = obs.reward_right
    return content


def _parse_ecoli_action(raw: str) -> EcoliAction:
    """Parse LLM output to EcoliAction. Falls back to Wait on failure."""
    try:
        data = json.loads(raw)
        return EcoliAction(**data)
    except (json.JSONDecodeError, ValidationError):
        return EcoliAction(type="Wait", reason="parse error — fallback to Wait")


def build_policy(
    policy_config: PolicyConfig,
    llm_config: LLMConfig,
    seed: int = 0,
    reward_fn=None,
) -> BasePolicy:
    """Factory: instantiate the right policy from config."""
    match policy_config.type:
        case "llm_memory":
            return LLMMemoryPolicy(llm_config, policy_config)
        case "llm_no_memory":
            return LLMNoMemoryPolicy(llm_config)
        case "random":
            return RandomPolicy(seed=seed)
        case "greedy":
            if reward_fn is None:
                raise ValueError("GreedyPolicy requires reward_fn (world.reward_at)")
            return GreedyPolicy(reward_fn)
        case _:
            raise ValueError(f"Unknown policy type: {policy_config.type}")
