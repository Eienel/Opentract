"""Run all 8 tasks against the FlagScale Qwen3-4B serving endpoint and
produce the submission ZIP.

Prereq: FlagScale is serving Qwen3-4B at the configured base_url (default
http://0.0.0.0:2026/v1). See openseek/README.md for the GPU+FlagScale steps.

Usage:
  python openseek/scripts/run_all.py                                # all tasks, full
  python openseek/scripts/run_all.py --limit-tests 10               # smoke run
  python openseek/scripts/run_all.py --strategy similarity --n 3    # tuned run
  python openseek/scripts/run_all.py --base-url http://1.2.3.4:2026/v1
"""

import argparse
import os

import _bootstrap  # noqa: F401

from openseek.src.backend import BackendConfig
from openseek.src.orchestrator import RunAllConfig, run_all
from openseek.src.retrieve import RetrievalConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://0.0.0.0:2026/v1"))
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "../Qwen3-4B"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="fallback when no per-task override; per-task defaults live in prompts.TASK_GEN_PARAMS")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=8,
                    help="parallel in-flight requests against the API (default 8)")
    ap.add_argument("--strategy", choices=["first_n", "similarity"], default="first_n")
    ap.add_argument("--max-demo-tokens", type=int, default=28000)
    ap.add_argument("--n", "--self-consistency-n", type=int, default=1, dest="n")
    ap.add_argument("--sc-temp", type=float, default=0.7)
    ap.add_argument("--max-demos", type=int, default=1000)
    ap.add_argument("--limit-tests", type=int, default=None,
                    help="cap per-task test count; useful for smoke runs")
    ap.add_argument("--out-dir", default="openseek/outputs/run")
    ap.add_argument("--submission", default="openseek/outputs/submission.zip")
    args = ap.parse_args()

    cfg = RunAllConfig(
        out_dir=args.out_dir,
        submission_path=args.submission,
        backend=BackendConfig(
            type="openai",
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
        ),
        retrieval=RetrievalConfig(
            strategy=args.strategy,
            max_demo_tokens=args.max_demo_tokens,
        ),
        self_consistency_n=args.n,
        self_consistency_temp=args.sc_temp,
        max_demos=args.max_demos,
        limit_tests=args.limit_tests,
    )

    path = run_all(cfg)
    print(f"\nSubmission ready: {path}")


if __name__ == "__main__":
    main()
