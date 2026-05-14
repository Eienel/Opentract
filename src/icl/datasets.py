"""Format-agnostic dataset loading.

The committee's exact dataset schema is unknown until registration. This module
is the seam that absorbs that uncertainty: arbitrary CSV / JSON / JSONL columns
are mapped onto a uniform `Example` via the field names in DatasetConfig. When
the real data lands, only configs/default.yaml changes.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from .config import DatasetConfig


@dataclass
class Example:
    id: str
    input_text: str
    label: Any | None = None  # absent for unlabelled eval items
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_label(self) -> bool:
        return self.label is not None and self.label != ""


def _read_rows(path: str, fmt: str) -> list[dict[str, Any]]:
    if fmt == "jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if fmt == "json":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            # Allow {"data": [...]} or {"examples": [...]} wrappers.
            for key in ("data", "examples", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            raise ValueError(f"{path}: JSON object has no list under data/examples/items")
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON list or wrapped list")
        return data
    if fmt == "csv":
        with open(path, "r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    raise ValueError(f"Unsupported dataset format: {fmt!r}")


def _to_example(row: dict[str, Any], cfg: DatasetConfig, idx: int) -> Example:
    raw_id = row.get(cfg.id_field)
    ex_id = str(raw_id) if raw_id is not None else str(idx)
    if cfg.input_field not in row:
        raise KeyError(
            f"Row {ex_id}: input_field {cfg.input_field!r} not found. "
            f"Available keys: {sorted(row)}"
        )
    label = row.get(cfg.label_field)
    metadata = {k: row[k] for k in cfg.metadata_fields if k in row}
    return Example(
        id=ex_id,
        input_text=str(row[cfg.input_field]),
        label=label,
        metadata=metadata,
    )


def load_examples(path: str, cfg: DatasetConfig) -> list[Example]:
    rows = _read_rows(path, cfg.format)
    return [_to_example(row, cfg, i) for i, row in enumerate(rows)]


def load_unified(cfg: DatasetConfig) -> list[Example]:
    """The labelled exemplar pool used for in-context retrieval."""
    examples = load_examples(cfg.unified_path, cfg)
    unlabelled = [e.id for e in examples if not e.has_label]
    if unlabelled:
        raise ValueError(
            f"Unified pool has {len(unlabelled)} unlabelled examples "
            f"(e.g. id={unlabelled[0]}); check dataset.label_field."
        )
    return examples


def load_eval(cfg: DatasetConfig) -> list[Example]:
    """The items to annotate. Labels may or may not be present (held-out slice)."""
    return load_examples(cfg.eval_path, cfg)


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
