from __future__ import annotations

import logging
import re
import warnings
from typing import Any

import torch

from taxisim.agents.base import Agent, AgentConfig, AgentStep
from taxisim.prompts.builder import PromptBuilder

logger = logging.getLogger(__name__)

_dotenv_loaded = False


def _load_dotenv_once() -> None:
    """Load ``.env`` into the process so ``HF_TOKEN``, API keys, etc. are visible to libraries."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    try:
        from dotenv import find_dotenv, load_dotenv

        env_path = find_dotenv()
        load_dotenv(env_path) if env_path else load_dotenv()
    except ImportError:
        pass
    _dotenv_loaded = True


class LLMAgent(Agent):
    """
    LLM-driven chemotaxis agent (HuggingFace, OpenAI, or Anthropic).

    Uses greedy decoding by default (temperature=0). Loading large models (e.g. many billions
    of parameters) can be slow and memory-heavy; prefer ≤1B-parameter checkpoints for initial runs.
    """

    def __init__(
        self,
        config: AgentConfig,
        action_space: list[str],
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        prompt_style: str = "abstract",  # "abstract" or "biological"
        temperature: float = 0.0,  # 0 for greedy/deterministic
        max_new_tokens: int = 128,
        backend: str = "huggingface",  # "huggingface", "openai", or "anthropic"
        device: str = "auto",
        api_key: str | None = None,
        env_params: dict[str, Any] | None = None,
    ):
        super().__init__(config, action_space)
        self.model_name = model_name
        self.prompt_style = prompt_style
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.backend = backend
        self.device = device
        self.api_key = api_key
        self.env_type = "2d" if "north" in action_space else "1d"
        self.env_params = env_params or (
            {"width": 20, "height": 20} if self.env_type == "2d" else {"length": 100}
        )
        self._prompt_builder = PromptBuilder(
            style=prompt_style,
            env_type=self.env_type,
            history_length=config.history_length,
        )
        self._hf_model = None
        self._hf_tokenizer = None
        self._openai_client = None
        self._anthropic_client = None
        self.parse_failures = 0
        self.parse_attempts = 0
        self._warn_large_model()

    def _warn_large_model(self) -> None:
        name = self.model_name.lower()
        if any(x in name for x in ("7b", "13b", "70b", "65b", "34b", "30b")):
            warnings.warn(
                f"Model name '{self.model_name}' suggests a large checkpoint; "
                "taxisim targets small (≤~1B) models first.",
                UserWarning,
                stacklevel=2,
            )

    def _ensure_backend(self) -> None:
        _load_dotenv_once()
        if self.backend == "huggingface":
            if self._hf_model is None:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tok = AutoTokenizer.from_pretrained(self.model_name)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto" if self.device == "auto" else None,
                    torch_dtype=torch.float16,
                )
                if self.device != "auto":
                    model = model.to(self.device)
                model.eval()
                self._hf_tokenizer = tok
                self._hf_model = model
        elif self.backend == "openai":
            if self._openai_client is None:
                from openai import OpenAI

                self._openai_client = OpenAI(api_key=self.api_key)
        elif self.backend == "anthropic":
            if self._anthropic_client is None:
                import anthropic

                self._anthropic_client = anthropic.Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _generate_hf(self, system: str, user: str) -> str:
        assert self._hf_tokenizer is not None and self._hf_model is not None
        tok = self._hf_tokenizer
        model = self._hf_model
        if getattr(tok, "chat_template", None):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            prompt = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = system + "\n\n" + user

        inputs = tok(prompt, return_tensors="pt")
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        do_sample = self.temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        text = tok.decode(new_tokens, skip_special_tokens=True)
        return text

    def _generate_openai(self, system: str, user: str) -> str:
        assert self._openai_client is not None
        resp = self._openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""

    def _generate_anthropic(self, system: str, user: str) -> str:
        assert self._anthropic_client is not None
        msg = self._anthropic_client.messages.create(
            model=self.model_name,
            max_tokens=self.max_new_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=self.temperature,
        )
        parts = msg.content
        if not parts:
            return ""
        return "".join(getattr(b, "text", str(b)) for b in parts)

    def _parse_action(self, response_text: str) -> str:
        """
        Extract action from LLM response.
        Prefer the last valid action token (models often explain first, then answer).
        Fallback: first occurrence, then substring match. If none found, return "stay".
        """
        text = response_text.strip().lower()
        valid = set(self.action_space)

        # Last non-empty line: often "left" or "Action: right"
        for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
            line_clean = re.sub(r"^[^a-z]*", "", line)
            line_clean = re.sub(r"[^a-z]+$", "", line_clean)
            for sep in (":", "—", "-"):
                if sep in line_clean:
                    parts = line_clean.split(sep, 1)
                    line_clean = parts[-1].strip()
            last_tok = re.findall(r"[a-z]+", line_clean)
            if last_tok and last_tok[-1] in valid:
                return last_tok[-1]

        words = re.findall(r"[a-z]+", text)
        # Prefer last occurrence (explanations often precede the chosen action)
        for w in reversed(words):
            if w in valid:
                return w
        # substring: prefer longer action names first (north before northward won't match)
        for a in sorted(valid, key=len, reverse=True):
            if a in text:
                return a
        logger.warning("Could not parse action from LLM response; using stay. Response: %r", response_text)
        return "stay"

    def act(self, observation: float, position: Any) -> AgentStep:
        self._ensure_backend()
        prompts = self._prompt_builder.build(
            history=self.history,
            current_observation=observation,
            position=position,
            env_params=self.env_params,
        )
        system, user = prompts["system"], prompts["user"]

        if self.backend == "huggingface":
            raw = self._generate_hf(system, user)
        elif self.backend == "openai":
            raw = self._generate_openai(system, user)
        else:
            raw = self._generate_anthropic(system, user)

        self.parse_attempts += 1
        action = self._parse_action(raw)
        if action not in self.action_space:
            action = "stay"
        if action == "stay" and raw.strip():
            # Heuristic: count as parse failure if we fell back (already logged in _parse_action)
            words = re.findall(r"[a-z]+", raw.lower())
            if not any(w in self.action_space for w in words):
                self.parse_failures += 1
        elif action == "stay" and not raw.strip():
            self.parse_failures += 1

        self.update_history(observation, position)
        meta = {
            "parse_failure": action == "stay" and not any(
                a in raw.lower() for a in self.action_space
            ),
            "model_name": self.model_name,
            "backend": self.backend,
        }
        return AgentStep(action=action, reasoning=raw, metadata=meta)
