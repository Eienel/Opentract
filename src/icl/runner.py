"""AnnotationRunner -- orchestrates retrieval, prompting, voting and chunking.

Per eval item:
  chunk (if oversized)
    -> retrieve exemplars
    -> build prompt
    -> generate N samples (self-consistency)
    -> parse each, bounded stricter re-prompt on parse failure
    -> majority vote over samples
  -> aggregate per-chunk votes into the final annotation

Results are cached to disk by item id so an interrupted run resumes for free --
important when paying per API token under deadline pressure.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from .backends.base import ModelBackend
from .chunking import aggregate, chunk_examples, majority_vote
from .config import Config
from .datasets import Example
from .parser import OutputParser
from .prompt import PromptBuilder
from .retriever import Retriever


@dataclass
class AnnotationResult:
    id: str
    prediction: Any | None
    raw_samples: list[str]
    n_chunks: int
    parse_failed: bool


class AnnotationRunner:
    def __init__(
        self,
        cfg: Config,
        backend: ModelBackend,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        parser: OutputParser,
        cache_path: str | None = None,
    ):
        self.cfg = cfg
        self.backend = backend
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.parser = parser
        self.cache_path = cache_path
        self._cache: dict[str, dict] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self._cache[rec["id"]] = rec

    def run(self, eval_examples: list[Example]) -> list[AnnotationResult]:
        results = []
        for ex in tqdm(eval_examples, desc="annotating"):
            if ex.id in self._cache:
                rec = self._cache[ex.id]
                results.append(
                    AnnotationResult(
                        id=rec["id"],
                        prediction=rec["prediction"],
                        raw_samples=rec.get("raw_samples", []),
                        n_chunks=rec.get("n_chunks", 1),
                        parse_failed=rec.get("parse_failed", False),
                    )
                )
                continue
            result = self._annotate_one(ex)
            results.append(result)
            self._append_cache(result)
        return results

    def _annotate_one(self, ex: Example) -> AnnotationResult:
        chunks = chunk_examples(ex, self.cfg.chunking)
        chunk_preds: list[Any] = []
        all_raw: list[str] = []
        parse_failed_any = False
        for chunk in chunks:
            pred, raw, failed = self._annotate_chunk(chunk)
            chunk_preds.append(pred)
            all_raw.extend(raw)
            parse_failed_any = parse_failed_any or failed
        final = chunk_preds[0] if len(chunks) == 1 else aggregate(chunk_preds, self.cfg.chunking)
        return AnnotationResult(
            id=ex.id,
            prediction=final,
            raw_samples=all_raw,
            n_chunks=len(chunks),
            parse_failed=parse_failed_any,
        )

    def _annotate_chunk(self, chunk: Example) -> tuple[Any | None, list[str], bool]:
        exemplars = self.retriever.select(chunk)
        prompt = self.prompt_builder.build(chunk, exemplars)
        n = max(1, self.cfg.self_consistency.n)
        temp = self.cfg.self_consistency.temperature if n > 1 else self.cfg.backend.temperature
        raw_samples = self.backend.generate(
            prompt, self.prompt_builder.system, n=n, temperature=temp
        )
        parsed = [self.parser.parse(r) for r in raw_samples]

        parse_failed = all(p is None for p in parsed)
        if parse_failed:
            # Bounded stricter re-prompt -- malformed output otherwise scores zero.
            strict_prompt = self.prompt_builder.build(chunk, exemplars, stricter=True)
            for _ in range(self.cfg.parser.max_retries):
                retry_raw = self.backend.generate(
                    strict_prompt, self.prompt_builder.system, n=1, temperature=0.0
                )
                raw_samples.extend(retry_raw)
                retry_parsed = self.parser.parse(retry_raw[0])
                if retry_parsed is not None:
                    parsed.append(retry_parsed)
                    parse_failed = False
                    break

        prediction = majority_vote(parsed)
        return prediction, raw_samples, parse_failed

    def _append_cache(self, result: AnnotationResult) -> None:
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "id": result.id,
                        "prediction": result.prediction,
                        "raw_samples": result.raw_samples,
                        "n_chunks": result.n_chunks,
                        "parse_failed": result.parse_failed,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
