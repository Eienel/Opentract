"""Embed the unified exemplar pool and cache it to disk.

Usage: python scripts/build_index.py --config configs/default.yaml
"""

import argparse

import _bootstrap  # noqa: F401  (path setup)

from icl.config import Config
from icl.pipeline import build_index, index_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    store = build_index(cfg)
    print(f"Indexed {len(store.examples)} exemplars -> {index_path(cfg)}.npy")


if __name__ == "__main__":
    main()
