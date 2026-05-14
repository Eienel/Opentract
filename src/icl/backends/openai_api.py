"""OpenAI-compatible backend -- the primary path.

Works against any OpenAI-compatible endpoint: hosted Qwen3-4B providers
(OpenRouter, Alibaba DashScope) or a self-hosted vLLM server on a rented GPU.
"""

from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ModelBackend


class OpenAIBackend(ModelBackend):
    def __init__(self, cfg):
        super().__init__(cfg)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("backend.type=openai_api requires `pip install openai`") from exc

        api_key = os.environ.get(cfg.api_key_env, "EMPTY")  # vLLM accepts any key
        self._client = OpenAI(base_url=cfg.base_url, api_key=api_key, timeout=cfg.request_timeout)
        self._extra_body = {}
        # Qwen3 exposes reasoning ("thinking") mode through chat_template_kwargs.
        if not cfg.thinking_mode:
            self._extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    def generate(self, prompt: str, system: str, n: int = 1, temperature: float | None = None) -> list[str]:
        temp = self.cfg.temperature if temperature is None else temperature

        @retry(
            stop=stop_after_attempt(self.cfg.max_retries),
            wait=wait_exponential(multiplier=2, min=2, max=30),
            reraise=True,
        )
        def _call() -> list[str]:
            resp = self._client.chat.completions.create(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.cfg.max_tokens,
                temperature=temp,
                n=n,
                extra_body=self._extra_body or None,
            )
            return [c.message.content or "" for c in resp.choices]

        outputs = _call()
        # Some providers ignore `n`; pad by repeating if needed.
        while len(outputs) < n:
            outputs.append(outputs[-1] if outputs else "")
        return outputs[:n]
