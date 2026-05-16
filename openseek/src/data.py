"""Load OpenSeek competition datasets.

Schema (per JSON file):
  task_id, task_name, Definition (list[str]), examples (labelled), test_samples (no labels), License
Each example:  {id, input, output (list[str])}
Each test:     {id, input}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

TASK_IDS = list(range(1, 9))


@dataclass
class Example:
    id: str
    input: Any
    output: str | None = None  # gold annotation for demos; None for test items

    @classmethod
    def demo(cls, raw: dict) -> "Example":
        out = raw.get("output")
        if isinstance(out, list):
            out = out[0] if out else None
        return cls(id=raw["id"], input=raw["input"], output=out)

    @classmethod
    def test(cls, raw: dict) -> "Example":
        return cls(id=raw["id"], input=raw["input"], output=None)


@dataclass
class Task:
    task_id: int
    task_name: str
    definition: str
    demos: list[Example]
    tests: list[Example]

    @property
    def slug(self) -> str:
        return f"openseek-{self.task_id}"


def _data_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def find_task_file(task_id: int) -> str:
    prefix = f"openseek-{task_id}_"
    for fn in os.listdir(_data_dir()):
        if fn.startswith(prefix) and fn.endswith(".json"):
            return os.path.join(_data_dir(), fn)
    raise FileNotFoundError(
        f"No file matching {prefix}*.json in {_data_dir()}. "
        f"Run: python openseek/scripts/fetch_data.py"
    )


def load_task(task_id: int) -> Task:
    path = find_task_file(task_id)
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    definitions = raw.get("Definition") or raw.get("definition") or [""]
    definition = definitions[0] if isinstance(definitions, list) else str(definitions)
    demos = [Example.demo(e) for e in raw.get("examples", [])]
    tests = [Example.test(e) for e in raw.get("test_samples", [])]
    return Task(
        task_id=int(raw["task_id"]) if str(raw.get("task_id", task_id)).isdigit() else task_id,
        task_name=raw.get("task_name", f"openseek-{task_id}"),
        definition=definition,
        demos=demos,
        tests=tests,
    )


def load_all_tasks() -> list[Task]:
    return [load_task(i) for i in TASK_IDS]


def render_io(value: Any) -> str:
    """Stringify a sample input or output for prompt embedding."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)
