"""CPU GGUF backend -- last-resort path with no GPU and no API budget.

Runs Qwen3-4B (e.g. Q4_K_M, ~2.5 GB) via llama.cpp. Slow but free.
Install with: pip install -r requirements.txt -r requirements-cpu.txt
"""

from __future__ import annotations

from .base import ModelBackend


class LlamaCppBackend(ModelBackend):
    def __init__(self, cfg):
        super().__init__(cfg)
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "backend.type=llamacpp requires `pip install -r requirements-cpu.txt`"
            ) from exc

        repo = cfg.extra.get("gguf_repo", "Qwen/Qwen3-4B-GGUF")
        filename = cfg.extra.get("gguf_file", "*Q4_K_M.gguf")
        self._llm = Llama.from_pretrained(
            repo_id=repo,
            filename=filename,
            n_ctx=cfg.extra.get("n_ctx", 32768),
            n_threads=cfg.extra.get("n_threads", 8),
            verbose=False,
        )

    def generate(self, prompt: str, system: str, n: int = 1, temperature: float | None = None) -> list[str]:
        temp = self.cfg.temperature if temperature is None else temperature
        outputs = []
        for _ in range(n):
            resp = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.cfg.max_tokens,
                temperature=temp,
            )
            outputs.append(resp["choices"][0]["message"]["content"] or "")
        return outputs
