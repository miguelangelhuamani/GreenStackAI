# llm_client.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class LLMConfig:
    # Pick ONE:
    # - "openai_responses" (recommended)
    # - "openai_chat"      (older, still supported)
    provider: str = "openai_responses"

    # Example models shown in OpenAI docs include gpt-5.* (pick what your team has access to)
    model: str = "gpt-5-mini"

    temperature: float = 0.2
    max_output_tokens: int = 1400

    # Keep cost bounded
    reasoning_effort: str = "low"  # "low"|"medium"|"high" depending on model support
    retries: int = 3
    retry_backoff_s: float = 0.8

    # Optional: for local gateways / proxies
    base_url: Optional[str] = None


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        if cfg.provider.startswith("openai"):
            if OpenAI is None:
                raise RuntimeError("Missing dependency. Run: pip install openai")
            kwargs = {}
            if cfg.base_url:
                kwargs["base_url"] = cfg.base_url
            # SDK reads OPENAI_API_KEY automatically per quickstart/docs :contentReference[oaicite:1]{index=1}
            self.client = OpenAI(**kwargs)

    def generate(self, developer_instructions: str, user_input: str) -> str:
        """
        Returns plain text content.
        Keeps your agent logic "yours" (no frameworks); only wraps the raw API call.
        """
        last_err = None
        for attempt in range(self.cfg.retries):
            try:
                if self.cfg.provider == "openai_responses":
                    # Responses API usage from OpenAI docs :contentReference[oaicite:2]{index=2}
                    resp = self.client.responses.create(
                        model=self.cfg.model,
                        reasoning={"effort": self.cfg.reasoning_effort},
                        input=[
                            {"role": "developer", "content": developer_instructions},
                            {"role": "user", "content": user_input},
                        ],
                        temperature=self.cfg.temperature,
                        max_output_tokens=self.cfg.max_output_tokens,
                    )
                    return resp.output_text

                if self.cfg.provider == "openai_chat":
                    # Chat Completions example in official reference :contentReference[oaicite:3]{index=3}
                    completion = self.client.chat.completions.create(
                        model=self.cfg.model,
                        messages=[
                            {"role": "developer", "content": developer_instructions},
                            {"role": "user", "content": user_input},
                        ],
                        temperature=self.cfg.temperature,
                    )
                    return completion.choices[0].message.content or ""

                raise ValueError(f"Unknown provider: {self.cfg.provider}")

            except Exception as e:
                last_err = e
                time.sleep(self.cfg.retry_backoff_s * (2 ** attempt))

        raise RuntimeError(f"LLM call failed after retries. Last error: {last_err}")