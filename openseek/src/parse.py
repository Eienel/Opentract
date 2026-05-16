"""Extract the final prediction from model output.

Matches the official evaluator's logic: pull the most common <label>...</label>
content. For Task 8 (kernel generation) we want the full block intact, including
newlines, so we do not collapse whitespace there.
"""

from __future__ import annotations

import re
from collections import Counter

_LABEL_RE_SINGLE_LINE = re.compile(r"<label>\s*(.+?)\s*</label>", re.DOTALL)


def extract_label(text: str, task_id: int) -> str | None:
    if not text:
        return None
    matches = _LABEL_RE_SINGLE_LINE.findall(text)
    if not matches:
        return None

    if task_id == 8:
        # Keep code blocks intact; pick the LAST match (the model's final code).
        return matches[-1]

    counts = Counter(m.strip() for m in matches if m.strip())
    if not counts:
        return None
    top, _ = counts.most_common(1)[0]
    if len(top) >= 200:  # baseline rejects unreasonably long labels for non-code tasks
        return None
    return top


def majority_vote(predictions: list[str | None]) -> str | None:
    valid = [p for p in predictions if p is not None]
    if not valid:
        return None
    counts = Counter(valid)
    return counts.most_common(1)[0][0]
