from typing import Dict, Literal, Optional, Tuple
from pydantic import BaseModel, Field, conint, model_validator


ActionType = Literal["M", "W"]
Dir = Literal["L", "R"]

class Observation(BaseModel):
    t: int
    L: int
    x: int
    energy: int
    visible_rewards: Dict[int, float] = Field(default_factory=dict)

class Action(BaseModel):
    type: ActionType
    dir: Optional[Dir] = None
    steps: Optional[conint(ge=1)] = None  # we'll cap in validation

    # this is an internal constraint not output by the LLM
    max_steps : int = Field(default=20, exclude=True)  # for validation use only

    @model_validator(mode="after")
    def _validate_shape(self):
        if self.type == "W":
            return self  # dir/steps can be None
        # M requires dir and steps
        if self.dir not in ("L", "R") or self.steps is None:
            raise ValueError("M requires dir in {L,R} and steps >= 1")
        # enforce a hard cap so the LLM can't request huge jumps
        if self.steps > self.max_steps:
            raise ValueError(f"steps must be <= {self.max_steps}")
        return self