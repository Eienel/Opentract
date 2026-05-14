"""Text embedding for retrieval-based exemplar selection.

Default: BAAI/bge-m3 (multilingual -- the FlagOS dataset may be Chinese/mixed).
Fallback: a dependency-free hashing embedder, so the pipeline and its tests run
even when sentence-transformers / torch are not installed.
"""

from __future__ import annotations

import abc
import hashlib

import numpy as np

from .config import EmbeddingConfig


class EmbeddingModel(abc.ABC):
    @abc.abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (len(texts), dim) float32 array of L2-normalised vectors."""


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedding(EmbeddingModel):
    """Deterministic bag-of-words hashing embedder. No external deps.

    Not competitive with a real model, but keeps retrieval functional offline
    and makes the test suite hermetic.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return _l2_normalise(out)


class SentenceTransformerEmbedding(EmbeddingModel):
    def __init__(self, model_name: str, cache_dir: str | None = None):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, cache_folder=cache_dir)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)


def build_embedding(cfg: EmbeddingConfig) -> EmbeddingModel:
    if cfg.type == "hashing":
        return HashingEmbedding(dim=cfg.dim)
    if cfg.type == "sentence_transformers":
        try:
            return SentenceTransformerEmbedding(cfg.model, cfg.cache_dir)
        except ImportError:
            # Graceful degradation: a working pipeline beats a hard crash when
            # the heavy dependency is missing on a given machine.
            print(
                "[embedding] sentence-transformers unavailable; "
                "falling back to HashingEmbedding."
            )
            return HashingEmbedding(dim=cfg.dim)
    raise ValueError(f"Unknown embedding.type: {cfg.type!r}")
