"""Verify the configured Qwen3-4B endpoint is reachable, the API key is valid,
and the response parses to a <label>...</label>. Saves debugging time before
kicking off a 3-hour batch.

Usage:
  export OPENAI_API_KEY=sk-or-...
  export OPENAI_BASE_URL=https://openrouter.ai/api/v1
  export OPENAI_MODEL=qwen/qwen3-4b
  python openseek/scripts/check_endpoint.py
"""

import argparse
import os
import time

import _bootstrap  # noqa: F401

from openseek.src.backend import BackendConfig, build_backend
from openseek.src.parse import extract_label


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://0.0.0.0:2026/v1"))
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "../Qwen3-4B"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    args = ap.parse_args()

    print(f"endpoint: {args.base_url}")
    print(f"model:    {args.model}")
    print(f"api_key:  {'set' if (args.api_key and args.api_key != 'EMPTY') else 'EMPTY'}")

    backend = build_backend(BackendConfig(
        type="openai",
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        max_tokens=20,
        temperature=0.0,
        concurrency=1,
        max_retries=1,
    ))

    prompt = (
        "Output exactly: <label>OK</label>\n\n"
        "Your answer (must be wrapped in <label>...</label>):"
    )
    t0 = time.time()
    outs = backend.complete(prompt, n=1, max_tokens=20, stop=["</label>"])
    dt = time.time() - t0
    raw = outs[0]
    parsed = extract_label(raw, task_id=1)

    print(f"\nlatency: {dt:.2f}s")
    print(f"raw:     {raw!r}")
    print(f"parsed:  {parsed!r}")

    if parsed:
        print("\nendpoint OK -- safe to run the full pipeline")
    else:
        print("\nWARNING: did not parse a <label>. Endpoint works but the model "
              "may not follow format yet -- still OK to proceed; per-task "
              "prompts in src/prompts.py enforce the wrapping.")


if __name__ == "__main__":
    main()
