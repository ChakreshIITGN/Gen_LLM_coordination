"""
Ollama LLM client. Sync for Phase 1, async pool added in Phase 2.
Wraps the ollama Python SDK — handles message serialization and retries.
"""
from __future__ import annotations

import json
import time
from typing import Any

from ollama import chat, ChatResponse

from ..core.config import LLMConfig


def _serialize_content(content: Any) -> str:
    """Convert dicts/lists to JSON strings for Ollama; pass strings through."""
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


class OllamaClient:
    """Synchronous Ollama chat client with retry logic."""

    def __init__(self, config: LLMConfig):
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    def call(self, messages: list[dict], retries: int = 3) -> str:
        """
        Send messages to Ollama, return raw response text.
        Retries on failure with exponential backoff.
        """
        # serialize any dict/list content fields to JSON strings
        serialized = [
            {"role": m["role"], "content": _serialize_content(m["content"])}
            for m in messages
        ]

        for attempt in range(retries):
            try:
                resp: ChatResponse = chat(
                    model=self.model,
                    messages=serialized,
                    options={
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                )
                return resp["message"]["content"].strip()
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
