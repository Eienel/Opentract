"""Local transformers backend -- for running Qwen3-4B on a rented cloud GPU.

Install with: pip install -r requirements.txt -r requirements-gpu.txt
"""

from __future__ import annotations

from .base import ModelBackend


class TransformersBackend(ModelBackend):
    def __init__(self, cfg):
        super().__init__(cfg)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "backend.type=transformers requires `pip install -r requirements-gpu.txt`"
            ) from exc

        device = cfg.extra.get("device", "cuda")
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        self._model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            torch_dtype=cfg.extra.get("dtype", "auto"),
            device_map=device,
        )

    def generate(self, prompt: str, system: str, n: int = 1, temperature: float | None = None) -> list[str]:
        temp = self.cfg.temperature if temperature is None else temperature
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.cfg.thinking_mode,
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        do_sample = temp > 0
        outputs = []
        for _ in range(n):
            with self._torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_tokens,
                    do_sample=do_sample,
                    temperature=temp if do_sample else None,
                )
            new_tokens = generated[0][inputs.input_ids.shape[1]:]
            outputs.append(self._tokenizer.decode(new_tokens, skip_special_tokens=True))
        return outputs
