"""
World dynamics for each experiment type.
The world is deterministic — only the policy introduces stochasticity.
"""
from __future__ import annotations

import math

from .config import WorldConfig
from .schemas import EcoliAction, EcoliObservation, RingWorldAction, RingWorldObservation


# ── E. coli chemotaxis world ────────────────────────────────────────

class EcoliWorld:
    """
    1D line with exponential decay reward landscape.
    Blocking boundaries: agent cannot go below 0 or above L.
    No energy system — pure chemotaxis test.
    """

    def __init__(self, config: WorldConfig):
        self.L = config.length
        self.include_gradient = config.include_gradient
        rc = config.reward
        self.peak = rc.peak_position
        self.initial = rc.initial_value
        self.rate = rc.decay_rate

    def reward_at(self, x: float) -> float:
        """Exponential decay centered at peak. Zero below peak."""
        if x < self.peak:
            return 0.0
        return self.initial * math.exp(-self.rate * (x - self.peak))

    def step(self, position: int, action: EcoliAction) -> tuple[int, float]:
        """Apply action, return (new_position, reward)."""
        new_pos = position + action.displacement
        new_pos = max(0, min(self.L, new_pos))  # blocking boundaries
        return new_pos, self.reward_at(new_pos)

    def observe(self, step: int, position: int, reward: float | None) -> EcoliObservation:
        return EcoliObservation(
            step=step,
            position=position,
            reward=reward,
            reward_left=self.reward_at(position - 1) if self.include_gradient else None,
            reward_right=self.reward_at(position + 1) if self.include_gradient else None,
        )


# ── Ring world ──────────────────────────────────────────────────────

class RingWorld:
    """
    1D periodic ring with discrete reward pickups and energy system.
    Movement costs energy; rewards replenish energy.
    """

    def __init__(self, config: WorldConfig):
        self.L = config.length
        self.move_cost = config.move_cost
        self.max_energy = config.max_energy
        self.alpha = config.energy_alpha
        # mutable — rewards removed on collection
        self.rewards: dict[int, float] = dict(config.reward.locations)

    def ring_dist(self, a: int, b: int) -> int:
        """Shortest distance on ring."""
        d = abs(a - b)
        return min(d, self.L - d)

    def step(
        self, position: int, energy: int, action: RingWorldAction,
    ) -> tuple[int, int, float, int]:
        """
        Apply action under energy constraints.
        Returns (new_pos, new_energy, reward_gained, steps_executed).
        """
        if energy <= 0 or action.type == "W":
            reward, e_gain = self._collect(position)
            return position, min(self.max_energy, energy + e_gain), reward, 0

        # clip steps to what energy can afford
        affordable = min(action.steps or 0, energy // self.move_cost)
        if affordable <= 0:
            return position, energy, 0.0, 0

        dx = -1 if action.dir == "L" else 1
        new_pos = (position + dx * affordable) % self.L
        new_energy = energy - affordable * self.move_cost

        reward, e_gain = self._collect(new_pos)
        new_energy = min(self.max_energy, new_energy + e_gain)
        return new_pos, new_energy, reward, affordable

    def _collect(self, position: int) -> tuple[float, int]:
        """Collect reward at position if present. Returns (value, energy_gain)."""
        if position not in self.rewards:
            return 0.0, 0
        val = self.rewards.pop(position)
        return val, int(math.floor(self.alpha * val))

    def observe(
        self, step: int, position: int, energy: int, vis_radius: int = 1,
    ) -> RingWorldObservation:
        visible = {
            pos: val for pos, val in self.rewards.items()
            if self.ring_dist(position, pos) <= vis_radius
        }
        return RingWorldObservation(
            step=step, ring_length=self.L, position=position,
            energy=energy, visible_rewards=visible,
        )
