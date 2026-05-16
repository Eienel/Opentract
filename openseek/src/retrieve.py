"""Exemplar selection for ICL.

The official baseline packs the first N demos up to 8192 tokens. The task
actually allows ~30K context -- so the single highest-leverage thing we can do
is pack MORE demos, picked by similarity to the test input.

Two strategies:
- "first_n":   take demos in order, fill until token budget. Cheap and fast.
- "similarity": rank demos by embedding similarity to the test input, then pack.

`pack_examples` does the token-budget bin-packing using a tokenizer if provided,
otherwise a 4-chars-per-token char estimate (good enough offline; real run uses
the Qwen tokenizer for accuracy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .data import Example, render_io


@dataclass
class RetrievalConfig:
    strategy: str = "first_n"          # first_n | similarity
    max_demo_tokens: int = 28000       # leave headroom for system + query + completion
    demo_template: str = "# {input} <label> {output} </label>\n"
    embedding_model: str = "BAAI/bge-m3"
    embedding_cache_dir: str = ".cache/embeddings"
    seed: int = 0


def _tok_len(text: str, tokenizer: Any | None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return max(1, len(text) // 4)


def render_demo(ex: Example, template: str) -> str:
    return template.format(input=render_io(ex.input), output=render_io(ex.output))


def pack_examples(
    demos: list[Example],
    cfg: RetrievalConfig,
    tokenizer: Any | None = None,
) -> tuple[str, int]:
    """Greedy bin-pack demos into a single string under the token budget.

    Returns (rendered, n_packed).
    """
    pieces: list[str] = []
    n_tokens = 0
    for ex in demos:
        if ex.output is None:
            continue
        line = render_demo(ex, cfg.demo_template)
        cost = _tok_len(line, tokenizer)
        if n_tokens + cost > cfg.max_demo_tokens:
            if not pieces:  # at least one demo even if oversized
                pieces.append(line)
                n_tokens += cost
            break
        pieces.append(line)
        n_tokens += cost
    return "".join(pieces), len(pieces)


class _Embedder:
    """Lazy sentence-transformers wrapper with a tiny LRU disk cache by input id."""

    def __init__(self, model_name: str, cache_dir: str | None = None):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, cache_folder=cache_dir)

    def encode(self, texts: list[str]):
        import numpy as np

        vecs = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype("float32")


def rank_by_similarity(
    demos: list[Example],
    query_text: str,
    cfg: RetrievalConfig,
) -> list[Example]:
    """Return demos ordered most-similar-first relative to query_text."""
    import numpy as np

    texts = [render_io(d.input) for d in demos]
    embedder = _Embedder(cfg.embedding_model, cfg.embedding_cache_dir)
    demo_vecs = embedder.encode(texts)
    q = embedder.encode([query_text])[0]
    scores = demo_vecs @ q
    order = np.argsort(-scores)
    return [demos[int(i)] for i in order]


def select_demos(
    demos: list[Example],
    query: Example,
    cfg: RetrievalConfig,
    tokenizer: Any | None = None,
) -> tuple[str, int]:
    if cfg.strategy == "similarity":
        ranked = rank_by_similarity(demos, render_io(query.input), cfg)
    elif cfg.strategy == "first_n":
        ranked = demos
    else:
        raise ValueError(f"Unknown retrieval.strategy: {cfg.strategy!r}")
    return pack_examples(ranked, cfg, tokenizer)
