"""LLM-based policy for chemotaxis."""

import re
from typing import Optional
from .schemas import Action, Observation, ModelProvider
from ..integrations.ollama_client import OllamaClient
from ..integrations.huggingface_client import HuggingFaceClient


class LLMChemotaxisPolicy:
    """
    LLM-based policy that maps observations to RUN/TUMBLE actions.

    Uses a minimal prompt with last k concentrations and last action.
    """

    def __init__(self, model_provider: ModelProvider, model_name: str, **client_kwargs):
        """
        Initialize LLM policy.

        Args:
            model_provider: "ollama" or "huggingface"
            model_name: Model identifier
            **client_kwargs: Additional kwargs for client initialization
        """
        self.model_provider = model_provider
        self.model_name = model_name

        if model_provider == ModelProvider.OLLAMA:
            self.client = OllamaClient(model_name, **client_kwargs)
        elif model_provider == ModelProvider.HUGGINGFACE:
            self.client = HuggingFaceClient(model_name, **client_kwargs)
        else:
            raise ValueError(f"Unknown model provider: {model_provider}")

    def _create_prompt(self, observation: Observation) -> str:
        """
        Create minimal prompt from observation.

        Prompt format:
        - Last k concentrations
        - Last action (if available)
        - Request RUN or TUMBLE
        """
        concentrations = observation.concentrations

        # Format concentrations
        conc_str = ", ".join([f"{c:.3f}" for c in concentrations])

        prompt = f"""You are controlling a simple organism that senses chemical concentration over time.

Recent concentration readings: [{conc_str}]
"""

        if observation.last_action:
            prompt += f"Last action: {observation.last_action.value}\n"

        prompt += """
Based on the concentration trend, choose an action:
- RUN: continue in current direction (if concentration is improving)
- TUMBLE: reorient randomly (if concentration is worsening)

Respond with only: RUN or TUMBLE
"""

        return prompt

    def _parse_action(self, llm_output: str) -> Action:
        """
        Parse action from LLM output.

        Looks for RUN or TUMBLE in the output (case-insensitive).
        """
        output_upper = llm_output.upper().strip()

        # Try exact match first
        if "RUN" in output_upper and "TUMBLE" not in output_upper:
            return Action.RUN
        elif "TUMBLE" in output_upper:
            return Action.TUMBLE

        # Try regex patterns
        run_match = re.search(r"\bRUN\b", output_upper)
        tumble_match = re.search(r"\bTUMBLE\b", output_upper)

        if run_match and (tumble_match is None or run_match.start() < tumble_match.start()):
            return Action.RUN
        elif tumble_match:
            return Action.TUMBLE

        # Default to TUMBLE if unclear
        return Action.TUMBLE

    def select_action(self, observation: Observation) -> tuple[Action, str]:
        """
        Select action based on observation.

        Returns:
            action: Selected action
            raw_output: Raw LLM output for logging
        """
        prompt = self._create_prompt(observation)
        if self.model_provider == ModelProvider.OLLAMA:
            raw_output = self.client.generate(prompt, max_tokens=10, temperature=0.3)
        else:
            raw_output = self.client.generate(prompt, max_new_tokens=10, temperature=0.3)
        action = self._parse_action(raw_output)

        return action, raw_output
