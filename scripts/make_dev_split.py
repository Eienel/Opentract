"""Carve a labelled dev split out of the unified pool for offline tuning.

The committee's eval set may ship without gold labels, so to tune ICL settings
you need your own held-out labelled slice. This splits the unified pool into a
tuning pool (still used for retrieval) and a dev set (scored against), both
written in the same schema as the input.

Usage:
  python scripts/make_dev_split.py --config configs/default.yaml \
      --tune-out data/real/unified_tune.jsonl --dev-out data/real/dev.jsonl \
      --dev-frac 0.2
"""

import argparse
import json
import random

import _bootstrap  # noqa: F401

from icl.config import Config
from icl.datasets import Example, load_examples


def _to_row(ex: Example, cfg: Config) -> dict:
    d = cfg.dataset
    row = {d.id_field: ex.id, d.input_field: ex.input_text, d.label_field: ex.label}
    row.update(ex.metadata)
    return row


def _write(path: str, examples: list[Example], cfg: Config) -> None:
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(_to_row(ex, cfg), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", default=None, help="labelled file to split (default: dataset.unified_path)")
    ap.add_argument("--tune-out", required=True)
    ap.add_argument("--dev-out", required=True)
    ap.add_argument("--dev-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    source = args.source or cfg.dataset.unified_path
    examples = load_examples(source, cfg.dataset)
    labelled = [e for e in examples if e.has_label]
    if len(labelled) != len(examples):
        print(f"WARNING: dropping {len(examples) - len(labelled)} unlabelled rows from {source}")

    rng = random.Random(args.seed)
    rng.shuffle(labelled)
    n_dev = max(1, int(len(labelled) * args.dev_frac))
    dev, tune = labelled[:n_dev], labelled[n_dev:]

    _write(args.tune_out, tune, cfg)
    _write(args.dev_out, dev, cfg)
    print(f"Split {len(labelled)} labelled rows -> tune={len(tune)} ({args.tune_out}), dev={len(dev)} ({args.dev_out})")
    print("Now point a sweep config at these two files (see configs/sweep.yaml).")


if __name__ == "__main__":
    main()
