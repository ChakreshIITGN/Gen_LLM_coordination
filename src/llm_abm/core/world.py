"""POMDP environment for E. coli chemotaxis."""

import numpy as np
from typing import Optional
from .schemas import POMDPState, Observation, Action, EpisodeConfig


class ChemotaxisPOMDP:
    """
    1D POMDP for E. coli chemotaxis.

    State (hidden):
    - Position: x ∈ {0, ..., 20}
    - Direction: d ∈ {-1, +1}
    - Source location: x* (random per episode)
    - Background offset: b (optional)

    Observation:
    - Noisy concentration history: o_t = C(x_t) + ε_t
    - Last k observations provided

    Actions:
    - RUN: move one step in direction d (with slip prob)
    - TUMBLE: randomize direction, then optionally move
    """

    def __init__(self, config: EpisodeConfig, rng: Optional[np.random.Generator] = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()

        # Initialize state
        self.state: Optional[POMDPState] = None
        self.observation_history: list[float] = []
        self.step_count = 0
        self.last_action: Optional[Action] = None

    def reset(self, source_location: Optional[int] = None) -> Observation:
        """Reset the environment for a new episode."""
        # Determine source location
        if source_location is None:
            if self.config.source_location is not None:
                source_location = self.config.source_location
            else:
                source_location = self.rng.integers(0, 21)

        # Random starting position (not at source)
        start_pos = self.rng.integers(0, 21)
        while start_pos == source_location:
            start_pos = self.rng.integers(0, 21)

        # Random starting direction
        direction = self.rng.choice([-1, 1])

        self.state = POMDPState(
            position=start_pos,
            direction=direction,
            source_location=source_location,
            background_offset=0.0,  # Can be extended later for adaptation tests
        )

        self.observation_history = []
        self.step_count = 0
        self.last_action = None

        # Get initial observation
        obs = self._get_observation()
        return obs

    def _concentration(self, position: int) -> float:
        """
        Compute concentration at position.
        Uses exponential decay: C(x) = exp(-|x - x*| / decay_length) + background
        """
        distance = abs(position - self.state.source_location)
        concentration = np.exp(-distance / self.config.decay_length) + self.state.background_offset
        return float(concentration)

    def _get_observation(self) -> Observation:
        """Generate noisy observation."""
        true_concentration = self._concentration(self.state.position)
        noise = self.rng.normal(0, self.config.noise_std)
        noisy_concentration = true_concentration + noise

        # Add to history
        self.observation_history.append(noisy_concentration)

        # Keep only last k observations
        k = self.config.observation_history
        if len(self.observation_history) > k:
            self.observation_history = self.observation_history[-k:]

        return Observation(concentrations=self.observation_history.copy(), last_action=self.last_action)

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        Execute one step.

        Returns:
            observation: new observation
            reward: sparse reward (1.0 if reached source, 0.0 otherwise)
            done: whether episode is done
            info: additional info
        """
        self.step_count += 1
        self.last_action = action

        # Apply action
        if action == Action.RUN:
            # Check for slip
            if self.rng.random() < self.config.slip_probability:
                self.state.direction *= -1

            # Move one step in current direction
            new_position = self.state.position + self.state.direction

            # Boundary handling: wrap around or clamp
            # Using wrap-around for simplicity (0-20 ring)
            if new_position < 0:
                new_position = 20
            elif new_position > 20:
                new_position = 0

            self.state.position = new_position

        elif action == Action.TUMBLE:
            # Randomize direction (50/50)
            self.state.direction = self.rng.choice([-1, 1])
            # Optionally move after tumble (for now, we move)
            new_position = self.state.position + self.state.direction
            if new_position < 0:
                new_position = 20
            elif new_position > 20:
                new_position = 0
            self.state.position = new_position

        # Get new observation
        obs = self._get_observation()

        # Check success (within success_distance of source)
        distance_to_source = abs(self.state.position - self.state.source_location)
        reward = 1.0 if distance_to_source <= self.config.success_distance else 0.0

        # Check if done
        done = reward > 0.0 or self.step_count >= self.config.max_steps  # Success  # Timeout

        info = {
            "position": self.state.position,
            "direction": self.state.direction,
            "source_location": self.state.source_location,
            "distance_to_source": distance_to_source,
            "true_concentration": self._concentration(self.state.position),
            "step": self.step_count,
        }

        return obs, reward, done, info

    def get_hidden_state(self) -> POMDPState:
        """Get the hidden state (for debugging/evaluation only)."""
        return self.state
