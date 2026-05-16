"""Run a single OpenSeek task -- useful for fast iteration on one task at a time.

Usage:
  python openseek/scripts/run_one.py --task 5 --limit-tests 20 --base-url http://0.0.0.0:2026/v1
"""

import argparse

import _bootstrap  # noqa: F401

from openseek.src.backend import BackendConfig, build_backend
from openseek.src.data import load_task
from openseek.src.retrieve import RetrievalConfig
from openseek.src.runner import TaskRunConfig, run_task


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", type=int, required=True, choices=range(1, 9))
    ap.add_argument("--base-url", default="http://0.0.0.0:2026/v1")
    ap.add_argument("--model", default="../Qwen3-4B")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--strategy", choices=["first_n", "similarity"], default="first_n")
    ap.add_argument("--max-demo-tokens", type=int, default=28000)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--sc-temp", type=float, default=0.7)
    ap.add_argument("--max-demos", type=int, default=1000)
    ap.add_argument("--limit-tests", type=int, default=None)
    ap.add_argument("--out-dir", default="openseek/outputs/run")
    args = ap.parse_args()

    backend = build_backend(BackendConfig(
        type="openai",
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    ))
    task = load_task(args.task)
    cfg = TaskRunConfig(
        task_id=args.task,
        retrieval=RetrievalConfig(strategy=args.strategy, max_demo_tokens=args.max_demo_tokens),
        self_consistency_n=args.n,
        self_consistency_temp=args.sc_temp,
        max_demos=args.max_demos,
        limit_tests=args.limit_tests,
    )
    out = run_task(task, backend, cfg, args.out_dir)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
