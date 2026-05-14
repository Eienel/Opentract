"""Run the ICL annotation pipeline over the eval set and write a submission file.

Usage:
  python scripts/annotate.py --config configs/default.yaml
  python scripts/annotate.py --config configs/mock.yaml --cache outputs/mock_cache.jsonl

If the eval file carries gold labels (a held-out slice), a score is printed too.
"""

import argparse

import _bootstrap  # noqa: F401

from icl.config import Config
from icl.datasets import load_eval
from icl.evaluator import evaluate, write_submission
from icl.pipeline import build_runner


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default=None, help="resumable per-item result cache (jsonl)")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    eval_examples = load_eval(cfg.dataset)
    runner = build_runner(cfg, cache_path=args.cache)

    results = runner.run(eval_examples)
    path = write_submission(results, cfg)
    print(f"Wrote {len(results)} predictions -> {path}")

    if any(e.has_label for e in eval_examples):
        report = evaluate(results, eval_examples, cfg)
        print(
            f"[{report.metric}] {report.score:.4f}  "
            f"(scored {report.n_scored}/{report.n_total}, "
            f"parse_failed={report.n_parse_failed}, missing_pred={report.n_missing_pred})"
        )


if __name__ == "__main__":
    main()
