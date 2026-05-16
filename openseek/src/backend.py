"""Backend that talks to FlagScale's vLLM serving endpoint (OpenAI-compatible).

Also provides a deterministic StubBackend so the orchestrator runs end-to-end
offline. The real run hits http://0.0.0.0:2026/v1/completions per the official
baseline's llm_config.yaml -- but the same interface works against any
OpenAI-compatible Qwen3-4B endpoint (OpenRouter, DashScope, Together AI),
because that's exactly what FlagScale exposes.

Token economy notes:
- `max_tokens` and `stop` are exposed per call so a binary-classification task
  can cap output at ~10 tokens and stop on </label>, while task 8 (kernel
  generation) gets ~4000.
- `complete_batch` parallelises N prompts via a ThreadPoolExecutor -- crucial
  when the full run is 3666 calls and any single call can take 5-30s.
- The shared per-task prefix (~30K of demos) is byte-identical across every
  query in a task, so vLLM/OpenRouter prefix caching kicks in automatically.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Sequence


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
    concurrency: int = 1  # raise to 8-16 against a hosted API


class StubBackend:
    """Returns canned text built from `responder(prompt)`, for tests/dev."""

    def __init__(self, cfg: BackendConfig, responder: Callable[[str], str] | None = None):
        self.cfg = cfg
        self._responder = responder or (lambda p: "<label>STUB</label>")

    def complete(
        self,
        prompt: str,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> list[str]:
        return [self._responder(prompt) for _ in range(n)]

    def complete_batch(
        self,
        prompts: list[str],
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> list[list[str]]:
        return [self.complete(p, n, temperature, max_tokens, stop) for p in prompts]


class OpenAIBackend:
    """OpenAI-compatible /v1/completions client.

    Works against FlagScale+vLLM and any other OpenAI-compatible Qwen3-4B
    endpoint (OpenRouter, DashScope, Together AI, self-hosted).
    """

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

    def complete(
        self,
        prompt: str,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> list[str]:
        temp = self.cfg.temperature if temperature is None else temperature
        mt = self.cfg.max_tokens if max_tokens is None else max_tokens
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self._client.completions.create(
                    model=self.cfg.model,
                    prompt=prompt,
                    max_tokens=mt,
                    temperature=temp,
                    top_p=self.cfg.top_p,
                    n=n,
                    stop=list(stop) if stop else None,
                )
                outputs = [c.text or "" for c in resp.choices]
                while len(outputs) < n:
                    outputs.append(outputs[-1] if outputs else "")
                return outputs[:n]
            except Exception as exc:
                last_err = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenAI backend failed after retries: {last_err}")

    def complete_batch(
        self,
        prompts: list[str],
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> list[list[str]]:
        """Issue prompts concurrently. Order of results matches order of prompts."""
        workers = max(1, self.cfg.concurrency)
        if workers == 1 or len(prompts) <= 1:
            return [self.complete(p, n, temperature, max_tokens, stop) for p in prompts]

        results: list[list[str] | None] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self.complete, p, n, temperature, max_tokens, stop): i
                for i, p in enumerate(prompts)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                results[i] = fut.result()
        # `results` is fully populated when all futures complete.
        return [r if r is not None else [""] * n for r in results]


def build_backend(cfg: BackendConfig, stub_responder: Callable[[str], str] | None = None):
    if cfg.type == "stub":
        return StubBackend(cfg, stub_responder)
    if cfg.type == "openai":
        return OpenAIBackend(cfg)
    raise ValueError(f"Unknown backend.type: {cfg.type!r}")
