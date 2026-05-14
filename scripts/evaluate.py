"""Score an existing submission file against gold labels in the eval set.

Usage: python scripts/evaluate.py --config configs/mock.yaml
       python scripts/evaluate.py --config configs/mock.yaml --submission outputs/mock_submission.jsonl
"""

import argparse
import csv
import json

import _bootstrap  # noqa: F401

from icl.config import Config
from icl.datasets import load_eval


def _norm(value: object) -> str:
    return str(value).strip().lower()


def _read_submission(path: str, fmt: str, id_field: str, pred_field: str) -> dict[str, object]:
    preds: dict[str, object] = {}
    if fmt == "jsonl":
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    preds[str(row[id_field])] = row[pred_field]
    elif fmt == "csv":
        with open(path, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                preds[str(row[id_field])] = row[pred_field]
    else:
        raise ValueError(f"Unknown submission.format: {fmt!r}")
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--submission", default=None, help="defaults to submission.path in config")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    sub_path = args.submission or cfg.submission.path
    preds = _read_submission(
        sub_path, cfg.submission.format, cfg.submission.id_field, cfg.submission.prediction_field
    )
    eval_examples = load_eval(cfg.dataset)
    gold = {e.id: e.label for e in eval_examples if e.has_label}
    if not gold:
        raise SystemExit("Eval set has no gold labels; cannot score offline.")

    missing = [eid for eid in gold if eid not in preds]
    if missing:
        print(f"WARNING: {len(missing)} gold items missing from submission (e.g. {missing[:3]})")

    correct = sum(1 for eid, g in gold.items() if eid in preds and _norm(preds[eid]) == _norm(g))
    acc = correct / len(gold)
    print(f"accuracy {acc:.4f}  ({correct}/{len(gold)} correct, submission={sub_path})")


if __name__ == "__main__":
    main()
