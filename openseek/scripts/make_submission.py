"""Pack existing per-task JSONL files into the submission ZIP.

Useful when you've run tasks individually with run_one.py and want to ship.
Verifies the ZIP has exactly 8 files matching the openseek-N-*.jsonl pattern.
"""

import argparse
import glob
import os
import re

import _bootstrap  # noqa: F401

from openseek.src.orchestrator import pack_submission


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", default="openseek/outputs/run")
    ap.add_argument("--out", default="openseek/outputs/submission.zip")
    args = ap.parse_args()

    pat = os.path.join(args.in_dir, "openseek-*.jsonl")
    paths = sorted(glob.glob(pat))
    by_task: dict[int, str] = {}
    for p in paths:
        m = re.search(r"openseek-(\d+)", os.path.basename(p))
        if m:
            by_task[int(m.group(1))] = p
    missing = [i for i in range(1, 9) if i not in by_task]
    if missing:
        raise SystemExit(f"Missing JSONLs for tasks {missing}; run run_one.py for those first.")

    final = [by_task[i] for i in range(1, 9)]
    out = pack_submission(final, args.out)
    print(f"Packed submission: {out}")
    print("Contents:")
    for p in final:
        print(f"  {os.path.basename(p)}  ({os.path.getsize(p)} bytes)")


if __name__ == "__main__":
    main()
