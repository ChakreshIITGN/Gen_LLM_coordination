"""
Action and observation schemas for each experiment type.
Pydantic models enforce strict contracts between simulator and policy.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, conint, model_validator


# ── E. coli chemotaxis ──────────────────────────────────────────────

class EcoliAction(BaseModel):
    """E. coli action: single step Left/Right/Wait with a reason string."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["Left", "Right", "Wait"]
    reason: str = Field(min_length=1, max_length=200)

    @property
    def displacement(self) -> int:
        """Signed step: -1 (Left), +1 (Right), 0 (Wait)."""
        return {"Left": -1, "Right": 1, "Wait": 0}[self.type]


class EcoliObservation(BaseModel):
    """What the E. coli agent sees each step."""
    step: int
    position: int
    reward: Optional[float] = None       # reward at current position
    reward_left: Optional[float] = None  # reward at position - 1
    reward_right: Optional[float] = None # reward at position + 1


# ── Ring world ──────────────────────────────────────────────────────

class RingWorldAction(BaseModel):
    """Ring world action: Move(dir, steps) or Wait."""
    type: Literal["M", "W"]
    dir: Optional[Literal["L", "R"]] = None
    steps: Optional[conint(ge=1)] = None
    max_steps: int = Field(default=20, exclude=True)

    @model_validator(mode="after")
    def _validate_move(self):
        if self.type == "W":
            return self
        if self.dir not in ("L", "R") or self.steps is None:
            raise ValueError("M requires dir in {L,R} and steps >= 1")
        if self.steps > self.max_steps:
            raise ValueError(f"steps must be <= {self.max_steps}")
        return self

    @property
    def displacement(self) -> int:
        if self.type == "W":
            return 0
        sign = -1 if self.dir == "L" else 1
        return sign * (self.steps or 0)


class RingWorldObservation(BaseModel):
    """What the ring world agent sees each step."""
    step: int
    ring_length: int
    position: int
    energy: int
    visible_rewards: dict[int, float] = Field(default_factory=dict)
