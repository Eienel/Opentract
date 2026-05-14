"""ModelBackend interface + factory.

Every way of running Qwen3-4B (hosted OpenAI-compatible API, self-hosted vLLM,
local transformers on a rented GPU, CPU GGUF via llama.cpp) sits behind this one
interface. Switching is a single config line: backend.type.
"""

from __future__ import annotations

import abc

from ..config import BackendConfig


class ModelBackend(abc.ABC):
    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg

    @abc.abstractmethod
    def generate(self, prompt: str, system: str, n: int = 1, temperature: float | None = None) -> list[str]:
        """Return `n` completions for the prompt.

        `temperature` overrides the config default (used by self-consistency).
        """

    def close(self) -> None:  # pragma: no cover - most backends need no teardown
        pass


def build_backend(cfg: BackendConfig) -> ModelBackend:
    if cfg.type == "stub":
        from .stub import StubBackend

        return StubBackend(cfg)
    if cfg.type == "openai_api":
        from .openai_api import OpenAIBackend

        return OpenAIBackend(cfg)
    if cfg.type == "transformers":
        from .transformers_be import TransformersBackend

        return TransformersBackend(cfg)
    if cfg.type == "llamacpp":
        from .llamacpp_be import LlamaCppBackend

        return LlamaCppBackend(cfg)
    raise ValueError(f"Unknown backend.type: {cfg.type!r}")
