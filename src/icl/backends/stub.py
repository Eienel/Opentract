"""Offline stub backend -- no network, no model weights.

It implements a deterministic nearest-exemplar baseline: it echoes the label of
the last exemplar embedded in the prompt. With `retrieval.ordering: similar_last`
that is the most-similar retrieved exemplar, so the stub behaves like a 1-NN
classifier. This keeps the full pipeline (retrieval -> prompt -> parse -> vote ->
score) exercisable and CI-testable without any GPU or API budget.
"""

from __future__ import annotations

import re

from .base import ModelBackend

_ANNOTATION_RE = re.compile(r"Annotation:\s*(.+?)\s*(?:\n|$)")


class StubBackend(ModelBackend):
    def generate(self, prompt: str, system: str, n: int = 1, temperature: float | None = None) -> list[str]:
        matches = _ANNOTATION_RE.findall(prompt)
        # The trailing "Annotation:" for the query produces no group; findall
        # only captures non-empty labels, so the last match is the closest
        # exemplar's label.
        answer = matches[-1].strip() if matches else ""
        return [answer] * n
