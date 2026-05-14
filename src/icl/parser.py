"""Extract a structured annotation from raw model text.

Malformed outputs cost score directly, so the parser is strict but forgiving:
it strips Qwen3 <think> blocks, supports label / json / regex extraction, and
snaps to the closest valid label when a label space is defined. `parse` returns
None on failure, which the runner uses to trigger a bounded stricter re-prompt.
"""

from __future__ import annotations

import json
import re

from .config import ParserConfig, TaskConfig

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove Qwen3 reasoning blocks and leading 'Annotation:' echoes."""
    text = _THINK_RE.sub("", text)
    text = re.sub(r"^\s*annotation\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


class OutputParser:
    def __init__(self, parser_cfg: ParserConfig, task_cfg: TaskConfig):
        self.cfg = parser_cfg
        self.task_cfg = task_cfg
        self._label_lookup = {str(lbl).strip().lower(): lbl for lbl in task_cfg.label_space}
        self._regex = re.compile(parser_cfg.regex, re.DOTALL) if parser_cfg.regex else None

    def parse(self, raw: str) -> object | None:
        text = strip_thinking(raw)
        if not text:
            return None

        if self.cfg.type == "label":
            value = text.splitlines()[0].strip() if "\n" in text else text
        elif self.cfg.type == "json":
            value = self._parse_json(text)
            if value is None:
                return None
        elif self.cfg.type == "regex":
            if self._regex is None:
                raise ValueError("parser.type=regex requires parser.regex to be set")
            m = self._regex.search(text)
            if not m:
                return None
            value = (m.group(1) if m.groups() else m.group(0)).strip()
        else:
            raise ValueError(f"Unknown parser.type: {self.cfg.type!r}")

        return self._coerce_label(value)

    def _parse_json(self, text: str) -> object | None:
        # Tolerate prose around the JSON object.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict):
            return obj.get(self.cfg.json_key)
        return obj

    def _coerce_label(self, value: object) -> object | None:
        """Snap free-text to the closest valid label, if a label space exists."""
        if not self.task_cfg.label_space:
            return value if (value is not None and str(value).strip()) else None
        key = str(value).strip().lower()
        if key in self._label_lookup:
            return self._label_lookup[key]
        # Substring rescue: model wrapped the label in a sentence.
        for lk, original in self._label_lookup.items():
            if re.search(rf"\b{re.escape(lk)}\b", key):
                return original
        return None
