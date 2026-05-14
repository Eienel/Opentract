"""Scoring + leaderboard submission writing.

The exact competition metric is unknown until registration, so `metric` is
pluggable (accuracy / macro_f1). Confirm the official metric on day 1 and, if it
is something else, add it here -- the rest of the pipeline is unaffected.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

from .config import Config
from .datasets import Example
from .runner import AnnotationResult


@dataclass
class EvalReport:
    metric: str
    score: float
    n_total: int
    n_scored: int  # items that had a gold label
    n_parse_failed: int
    n_missing_pred: int


def _norm(value: object) -> str:
    return str(value).strip().lower()


def accuracy(pred: list[object], gold: list[object]) -> float:
    if not gold:
        return 0.0
    correct = sum(1 for p, g in zip(pred, gold) if p is not None and _norm(p) == _norm(g))
    return correct / len(gold)


def macro_f1(pred: list[object], gold: list[object]) -> float:
    if not gold:
        return 0.0
    labels = {_norm(g) for g in gold} | {_norm(p) for p in pred if p is not None}
    f1s = []
    for lbl in labels:
        tp = sum(1 for p, g in zip(pred, gold) if p is not None and _norm(p) == lbl and _norm(g) == lbl)
        fp = sum(1 for p, g in zip(pred, gold) if p is not None and _norm(p) == lbl and _norm(g) != lbl)
        fn = sum(1 for p, g in zip(pred, gold) if _norm(g) == lbl and (p is None or _norm(p) != lbl))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


_METRICS = {"accuracy": accuracy, "macro_f1": macro_f1}


def evaluate(results: list[AnnotationResult], eval_examples: list[Example], cfg: Config) -> EvalReport:
    metric_name = cfg.evaluation.metric
    if metric_name not in _METRICS:
        raise ValueError(f"Unknown evaluation.metric: {metric_name!r} (have {sorted(_METRICS)})")
    gold_by_id = {e.id: e.label for e in eval_examples if e.has_label}
    pred, gold = [], []
    for r in results:
        if r.id in gold_by_id:
            pred.append(r.prediction)
            gold.append(gold_by_id[r.id])
    score = _METRICS[metric_name](pred, gold)
    return EvalReport(
        metric=metric_name,
        score=score,
        n_total=len(results),
        n_scored=len(gold),
        n_parse_failed=sum(1 for r in results if r.parse_failed),
        n_missing_pred=sum(1 for r in results if r.prediction is None),
    )


def write_submission(results: list[AnnotationResult], cfg: Config) -> str:
    """Write the leaderboard submission file in the configured format."""
    sub = cfg.submission
    os.makedirs(os.path.dirname(sub.path) or ".", exist_ok=True)
    rows = [{sub.id_field: r.id, sub.prediction_field: r.prediction} for r in results]
    if sub.format == "jsonl":
        import json

        with open(sub.path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif sub.format == "csv":
        with open(sub.path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[sub.id_field, sub.prediction_field])
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError(f"Unknown submission.format: {sub.format!r}")
    return sub.path
