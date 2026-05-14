"""Long-context handling: split oversized inputs, aggregate per-chunk results.

Character-budget based (token-free) so it works without a tokenizer dependency.
max_input_chars should be set well below the model context window to leave room
for the system prompt and the in-context exemplars.
"""

from __future__ import annotations

from collections import Counter

from .config import ChunkingConfig
from .datasets import Example


def split_input(text: str, cfg: ChunkingConfig) -> list[str]:
    if len(text) <= cfg.max_input_chars:
        return [text]
    step = max(1, cfg.max_input_chars - cfg.overlap_chars)
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + cfg.max_input_chars])
        start += step
    return chunks


def chunk_examples(query: Example, cfg: ChunkingConfig) -> list[Example]:
    """Return one Example per chunk (single-element list when no split needed)."""
    pieces = split_input(query.input_text, cfg)
    if len(pieces) == 1:
        return [query]
    return [
        Example(
            id=f"{query.id}::chunk{i}",
            input_text=piece,
            label=query.label,
            metadata={**query.metadata, "parent_id": query.id, "chunk_index": i},
        )
        for i, piece in enumerate(pieces)
    ]


def aggregate(predictions: list[object], cfg: ChunkingConfig) -> object | None:
    """Collapse per-chunk predictions into one annotation for the parent item."""
    valid = [p for p in predictions if p is not None]
    if not valid:
        return None
    if cfg.aggregation == "first":
        return valid[0]
    if cfg.aggregation == "majority":
        return majority_vote(valid)
    raise ValueError(f"Unknown chunking.aggregation: {cfg.aggregation!r}")


def majority_vote(values: list[object]) -> object | None:
    """Most common value, preserving the original (non-stringified) object.

    Shared by chunk aggregation and self-consistency voting.
    """
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    original_by_key: dict[object, object] = {}
    counts: Counter = Counter()
    for v in valid:
        key = _hashable(v)
        counts[key] += 1
        original_by_key.setdefault(key, v)
    winner_key = counts.most_common(1)[0][0]
    return original_by_key[winner_key]


def _hashable(value: object) -> object:
    if isinstance(value, (list, dict)):
        import json

        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value
