"""Download the official OpenSeek competition datasets into openseek/data/.

Run once after cloning. The 8 files total ~10MB and are .gitignored.
"""

from __future__ import annotations

import os
import sys
import urllib.request

RAW = (
    "https://raw.githubusercontent.com/FlagAI-Open/OpenSeek/main/"
    "openseek/competition/LongContext-ICL-Annotation/data"
)

FILES = [
    "openseek-1_closest_integers.json",
    "openseek-2_count_nouns_verbs.json",
    "openseek-3_collatz_conjecture.json",
    "openseek-4_conala_concat_strings.json",
    "openseek-5_semeval_2018_task1_tweet_sadness_detection.json",
    "openseek-6_mnli_same_genre_classification.json",
    "openseek-7_jeopardy_answer_generation_all.json",
    "openseek-8_kernel_generation.json",
]


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(here, "data")
    os.makedirs(out_dir, exist_ok=True)
    for fn in FILES:
        dst = os.path.join(out_dir, fn)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"  ok      {fn}")
            continue
        url = f"{RAW}/{fn}"
        print(f"  fetch   {fn}")
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as exc:
            print(f"  FAILED  {fn}: {exc}", file=sys.stderr)
            sys.exit(1)
    print(f"\nAll 8 datasets in {out_dir}")


if __name__ == "__main__":
    main()
