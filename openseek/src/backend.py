"""Backend that talks to FlagScale's vLLM serving endpoint (OpenAI-compatible).

Also provides a deterministic StubBackend so the orchestrator runs end-to-end
offline. The real run hits http://0.0.0.0:2026/v1/completions per the official
baseline's llm_config.yaml.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class BackendConfig:
    type: str = "openai"  # openai | stub
    base_url: str = "http://0.0.0.0:2026/v1"
    model: str = "../Qwen3-4B"
    api_key: str = "EMPTY"
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    request_timeout: int = 600
    max_retries: int = 4


class StubBackend:
    """Returns canned text built from `responder(prompt)`, for tests/dev."""

    def __init__(self, cfg: BackendConfig, responder: Callable[[str], str] | None = None):
        self.cfg = cfg
        self._responder = responder or (lambda p: "<label>STUB</label>")

    def complete(self, prompt: str, n: int = 1, temperature: float | None = None) -> list[str]:
        return [self._responder(prompt) for _ in range(n)]


class OpenAIBackend:
    """OpenAI-compatible /v1/completions client -- works for FlagScale+vLLM and
    any other OpenAI-compatible server (DashScope, OpenRouter, self-hosted)."""

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("backend.type=openai requires `pip install openai`") from exc
        self._client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            timeout=cfg.request_timeout,
        )

    def complete(self, prompt: str, n: int = 1, temperature: float | None = None) -> list[str]:
        temp = self.cfg.temperature if temperature is None else temperature
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self._client.completions.create(
                    model=self.cfg.model,
                    prompt=prompt,
                    max_tokens=self.cfg.max_tokens,
                    temperature=temp,
                    top_p=self.cfg.top_p,
                    n=n,
                )
                outputs = [c.text or "" for c in resp.choices]
                while len(outputs) < n:
                    outputs.append(outputs[-1] if outputs else "")
                return outputs[:n]
            except Exception as exc:
                last_err = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenAI backend failed after retries: {last_err}")


def build_backend(cfg: BackendConfig, stub_responder: Callable[[str], str] | None = None):
    if cfg.type == "stub":
        return StubBackend(cfg, stub_responder)
    if cfg.type == "openai":
        return OpenAIBackend(cfg)
    raise ValueError(f"Unknown backend.type: {cfg.type!r}")
