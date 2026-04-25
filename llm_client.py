# llm_client.py
from __future__ import annotations
import os
import time
import requests
from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMConfig:
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 1500
    temperature: float = 0.2
    retries: int = 3
    retry_backoff_s: float = 1.0
    api_key: Optional[str] = None  # falls back to ANTHROPIC_API_KEY env var
    provider: Optional[str] = None # Added for compatibility with run_eval.py instantiation

class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        # Force model to Claude regardless of what was passed by run_eval.py
        self.cfg.model = "claude-sonnet-4-5-20250929"
        
        self.api_key = cfg.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "No API key found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key in LLMConfig."
            )
        self.url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

    def generate(self, developer_instructions: str, user_input: str) -> str:
        """
        Sends developer_instructions as the system prompt and user_input
        as the user message. Returns the response text.
        Compatible with the existing generate() call signature used
        throughout agent_skeleton.py and all baselines.
        """
        payload = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "system": developer_instructions,
            "messages": [
                {"role": "user", "content": user_input}
            ],
        }

        last_err = None
        for attempt in range(self.cfg.retries):
            try:
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                resp_json = response.json()
                content = resp_json.get("content", resp_json)
                if isinstance(content, list):
                    return content[0]["text"]
                elif isinstance(content, dict):
                    return content.get("text", str(content))
                else:
                    return str(content)

            except requests.exceptions.HTTPError as e:
                # 529 = Anthropic overloaded, retry
                if e.response is not None and e.response.status_code in (429, 529):
                    last_err = e
                    time.sleep(self.cfg.retry_backoff_s * (2 ** attempt))
                    continue
                raise
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.retry_backoff_s * (2 ** attempt))

        raise RuntimeError(f"Claude API call failed after {self.cfg.retries} retries. Last error: {last_err}")

if __name__ == "__main__":
    client = LLMClient(LLMConfig())
    response = client.generate(
        developer_instructions="You are a helpful assistant.",
        user_input="Reply with exactly: CLAUDE_OK"
    )
    print(f"Smoke test response: {response}")
    assert "CLAUDE_OK" in response, f"Unexpected response: {response}"
    print("✅ LLM client smoke test passed.")