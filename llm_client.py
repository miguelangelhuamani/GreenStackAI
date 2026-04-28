# llm_client.py
from __future__ import annotations
import os
import time
import requests
from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMConfig:
    model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    max_tokens: int = 2048
    temperature: float = 0.2
    retries: int = 8
    retry_backoff_s: float = 15.0
    api_key: Optional[str] = None
    provider: Optional[str] = None
    base_url: Optional[str] = None

class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        provider = (cfg.provider or os.environ.get("LLM_PROVIDER") or "tinker").lower()
        if provider in {"openai", "openai_responses", "openai_compatible", "tinker"}:
            self.provider = "openai_compatible"
            self._init_openai_compatible()
        else:
            self.provider = "anthropic"
            self._init_anthropic()

    def _init_anthropic(self) -> None:
        self.api_key = self.cfg.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("No API key found. Set ANTHROPIC_API_KEY or pass api_key in LLMConfig.")
        self.url = (self.cfg.base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/") + "/v1/messages"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

    def _init_openai_compatible(self) -> None:
        self.api_key = (
            self.cfg.api_key
            or os.environ.get("TINKER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise RuntimeError("No API key found. Set TINKER_API_KEY or OPENAI_API_KEY.")
        base_url = (
            self.cfg.base_url
            or os.environ.get("TINKER_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def generate(self, developer_instructions: str, user_input: str) -> str:
        """
        Sends developer_instructions as the system prompt and user_input
        as the user message. Returns the response text.
        Compatible with the existing generate() call signature used
        throughout agent_skeleton.py and all baselines.
        """
        if self.provider == "anthropic":
            payload = {
                "model": self.cfg.model,
                "max_tokens": self.cfg.max_tokens,
                "temperature": self.cfg.temperature,
                "system": developer_instructions,
                "messages": [{"role": "user", "content": user_input}],
            }
        else:
            payload = {
                "model": self.cfg.model,
                "max_tokens": self.cfg.max_tokens,
                "temperature": self.cfg.temperature,
                "messages": [
                    {"role": "system", "content": developer_instructions},
                    {"role": "user", "content": user_input},
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
                if not response.ok:
                    try:
                        err_body = response.json()
                    except Exception:
                        err_body = response.text
                    print(f"[Claude API ERROR {response.status_code}] {err_body}")
                    response.raise_for_status()
                resp_json = response.json()
                if self.provider == "anthropic":
                    content = resp_json.get("content", resp_json)
                    if isinstance(content, list):
                        return content[0].get("text", "")
                    if isinstance(content, dict):
                        return content.get("text", str(content))
                    return str(content)
                choices = resp_json.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
                return str(resp_json)

            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code in (429, 500, 502, 503, 504, 529):
                    last_err = e
                    retry_after = float(e.response.headers.get("retry-after", self.cfg.retry_backoff_s * (2 ** attempt)))
                    print(f"[Retryable HTTP {e.response.status_code}] Waiting {retry_after:.1f}s before retry {attempt+1}/{self.cfg.retries}...")
                    time.sleep(retry_after)
                    continue
                raise
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.retry_backoff_s * (2 ** attempt))

        raise RuntimeError(f"LLM API call failed after {self.cfg.retries} retries. Last error: {last_err}")

if __name__ == "__main__":
    client = LLMClient(LLMConfig())
    response = client.generate(
        developer_instructions="You are a helpful assistant.",
        user_input="Reply with exactly: CLAUDE_OK"
    )
    print(f"Smoke test response: {response}")
    assert "CLAUDE_OK" in response, f"Unexpected response: {response}"
    print("✅ LLM client smoke test passed.")
