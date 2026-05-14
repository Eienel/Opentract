"""Exemplar store + retriever for in-context demonstration selection.

Retrieval-based exemplar selection is the single highest-leverage ICL technique
here, so this module supports three strategies (similarity / random /
class_balanced) and three ordering policies, all config-driven for fast tuning.
"""

from __future__ import annotations

import random

import numpy as np

from .config import RetrievalConfig
from .datasets import Example
from .embedding import EmbeddingModel


class ExemplarStore:
    """Holds the unified pool and its cached embedding matrix."""

    def __init__(self, examples: list[Example], embeddings: np.ndarray):
        if len(examples) != embeddings.shape[0]:
            raise ValueError("examples / embeddings length mismatch")
        self.examples = examples
        self.embeddings = embeddings.astype(np.float32)

    @classmethod
    def build(cls, examples: list[Example], embedder: EmbeddingModel) -> "ExemplarStore":
        vecs = embedder.encode([e.input_text for e in examples])
        return cls(examples, vecs)

    def save(self, path: str) -> None:
        import json
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.save(path + ".npy", self.embeddings)
        with open(path + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump(
                [
                    {"id": e.id, "input_text": e.input_text, "label": e.label, "metadata": e.metadata}
                    for e in self.examples
                ],
                fh,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str) -> "ExemplarStore":
        import json

        embeddings = np.load(path + ".npy")
        with open(path + ".meta.json", "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        examples = [
            Example(id=m["id"], input_text=m["input_text"], label=m["label"], metadata=m.get("metadata", {}))
            for m in meta
        ]
        return cls(examples, embeddings)


class Retriever:
    def __init__(self, store: ExemplarStore, cfg: RetrievalConfig, embedder: EmbeddingModel):
        self.store = store
        self.cfg = cfg
        self.embedder = embedder
        self._rng = random.Random(cfg.seed)

    def _rank_by_similarity(self, query_text: str) -> list[int]:
        q = self.embedder.encode([query_text])[0]
        scores = self.store.embeddings @ q  # vectors are L2-normalised -> cosine
        return list(np.argsort(-scores))

    def _order(self, indices: list[int]) -> list[int]:
        """Apply the ordering policy. `indices` arrive most-similar-first."""
        if self.cfg.ordering == "similar_first":
            return indices
        if self.cfg.ordering == "similar_last":
            return list(reversed(indices))
        if self.cfg.ordering == "as_is":
            return sorted(indices)
        raise ValueError(f"Unknown retrieval.ordering: {self.cfg.ordering!r}")

    def select(self, query: Example) -> list[Example]:
        k = min(self.cfg.k_shots, len(self.store.examples))
        if k <= 0:
            return []

        if self.cfg.strategy == "similarity":
            chosen = self._rank_by_similarity(query.input_text)[:k]
        elif self.cfg.strategy == "random":
            chosen = self._rng.sample(range(len(self.store.examples)), k)
        elif self.cfg.strategy == "class_balanced":
            chosen = self._class_balanced(query, k)
        else:
            raise ValueError(f"Unknown retrieval.strategy: {self.cfg.strategy!r}")

        ordered = self._order(chosen)
        return [self.store.examples[i] for i in ordered]

    def _class_balanced(self, query: Example, k: int) -> list[int]:
        """Round-robin across labels, picking the most similar within each class.

        Keeps every label represented in the prompt -- valuable for skewed
        annotation tasks where rare classes would otherwise never be shown.
        """
        ranked = self._rank_by_similarity(query.input_text)
        by_label: dict[object, list[int]] = {}
        for idx in ranked:
            by_label.setdefault(self.store.examples[idx].label, []).append(idx)
        labels = list(by_label)
        chosen: list[int] = []
        cursor = {lbl: 0 for lbl in labels}
        while len(chosen) < k:
            progressed = False
            for lbl in labels:
                pool = by_label[lbl]
                if cursor[lbl] < len(pool):
                    chosen.append(pool[cursor[lbl]])
                    cursor[lbl] += 1
                    progressed = True
                    if len(chosen) == k:
                        break
            if not progressed:
                break
        return chosen
