"""Grid-search ICL config settings on a held-out dev split.

Reads a sweep spec (see configs/sweep.yaml), runs the full pipeline for every
combination in the grid against the dev set, prints a ranked table, and writes
the best-scoring config to disk.

Cost note: every grid cell runs one full pass over the dev set, so each cell is
`len(dev)` model calls. Keep the dev set and grid modest with a paid API.

Usage:
  python scripts/sweep.py --sweep configs/sweep.yaml
"""

import argparse
import itertools

import yaml

import _bootstrap  # noqa: F401

from icl.config import Config, apply_overrides, dump_yaml
from icl.datasets import load_eval, load_unified
from icl.embedding import build_embedding
from icl.evaluator import evaluate
from icl.pipeline import assemble_runner
from icl.retriever import ExemplarStore


def _expand_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    combos = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combos.append(dict(zip(keys, values)))
    return combos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", required=True, help="sweep spec YAML")
    args = ap.parse_args()

    with open(args.sweep, "r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)

    base = Config.from_yaml(spec["base"])
    # Point the base config at the dev split for the duration of the sweep.
    base.dataset.unified_path = spec["dev_unified"]
    base.dataset.eval_path = spec["dev_eval"]
    out_best = spec.get("out_best", "configs/best.yaml")
    grid = spec["grid"]

    combos = _expand_grid(grid)
    print(f"Sweeping {len(combos)} config combinations on dev split...")

    # Embedder + exemplar store are config-invariant within the sweep -> build once.
    embedder = build_embedding(base.embedding)
    store = ExemplarStore.build(load_unified(base.dataset), embedder)
    dev = load_eval(base.dataset)
    if not any(e.has_label for e in dev):
        raise SystemExit(f"dev_eval {spec['dev_eval']} has no gold labels; cannot score a sweep.")

    rows = []
    for i, overrides in enumerate(combos, 1):
        cfg = apply_overrides(base, overrides)
        runner = assemble_runner(cfg, embedder, store, cache_path=None)
        results = runner.run(dev)
        report = evaluate(results, dev, cfg)
        rows.append((report.score, report.n_parse_failed, overrides))
        label = ", ".join(f"{k}={v}" for k, v in overrides.items())
        print(f"  [{i}/{len(combos)}] {report.metric}={report.score:.4f}  parse_fail={report.n_parse_failed}  {label}")

    rows.sort(key=lambda r: (-r[0], r[1]))
    print("\n=== Ranked (best first) ===")
    for score, pf, overrides in rows:
        print(f"  {score:.4f}  parse_fail={pf}  {overrides}")

    best_score, _, best_overrides = rows[0]
    best_cfg = apply_overrides(Config.from_yaml(spec["base"]), best_overrides)
    dump_yaml(best_cfg, out_best)
    print(f"\nBest {best_score:.4f} -> {out_best}")
    print(f"Best overrides: {best_overrides}")
    print("Review out_best, then run: python scripts/annotate.py --config", out_best)


if __name__ == "__main__":
    main()
