# Opentract — ICL Annotation Pipeline (FlagOS Challenge, Track 3)

A config-driven **In-Context Learning** pipeline for automatic data annotation
with **Qwen3-4B**, built for Track 3 of the FlagOS Open Computing Global
Challenge.

## Why this track

The team has **no GPU**. That rules out Track 1 (operator development — scored on
benchmarked kernel performance) and Track 2 (multi-chip inference cluster).
Track 3 is **inference-only**, prompt-engineering driven, and leaderboard-ranked
— it runs on a cheap rented GPU, a hosted API, or even CPU. It is also the most
winnable track for a first-time-winning team. Deadline: **May 20, 2026**.

## What it does

Given the committee's **unified dataset** (a labelled exemplar pool) and an
**evaluation dataset** of (possibly long-context) inputs, the pipeline:

1. embeds the exemplar pool for retrieval-based exemplar selection,
2. for each eval item: retrieves exemplars → builds a few/many-shot prompt →
   queries Qwen3-4B → parses a structured annotation → majority-votes over
   self-consistency samples → aggregates across chunks for long inputs,
3. writes a leaderboard submission file and scores any held-out labelled slice.

The committee dataset format/metric are unknown until registration, so the
pipeline is **format-agnostic** — adapting it is a one-file edit
(`configs/default.yaml`), no code changes.

## Quick start (hermetic, no GPU/API/network)

```bash
pip install -r requirements.txt
pytest -q                                            # 4 tests, ~0.2s
python scripts/build_index.py --config configs/mock.yaml
python scripts/annotate.py    --config configs/mock.yaml --cache outputs/mock_cache.jsonl
python scripts/evaluate.py    --config configs/mock.yaml
# -> accuracy 1.0000 on the mock sentiment set (offline stub backend)
```

## Layout

```
configs/        default.yaml (real run), mock.yaml (offline test),
                sweep.yaml + sweep_mock.yaml (tuning grids)
data/mock/      tiny separable dataset for hermetic E2E testing
data/real/      committee dataset goes here (gitignored)
src/icl/        config, datasets, backends/, embedding, retriever,
                prompt, parser, chunking, runner, evaluator, pipeline
scripts/        build_index.py, annotate.py, evaluate.py,
                make_dev_split.py, sweep.py
tests/          test_e2e_mock.py
```

## Model backends (`backend.type` in config)

| type           | use case                                         | extra deps                |
|----------------|--------------------------------------------------|---------------------------|
| `stub`         | offline tests, pipeline development               | none                      |
| `openai_api`   | **primary** — hosted Qwen3-4B (OpenRouter / DashScope) or self-hosted vLLM | `requirements.txt` |
| `transformers` | Qwen3-4B on a rented cloud GPU                     | `requirements-gpu.txt`    |
| `llamacpp`     | last-resort CPU GGUF (Q4_K_M)                     | `requirements-cpu.txt`    |

Set the API key via the env var named in `backend.api_key_env` (default
`OPENAI_API_KEY`).

## Adapting to the real dataset

After registering and downloading the committee data into `data/real/`, edit
**`configs/default.yaml`** only:

- `dataset.*` — file paths, format, and the `input_field` / `label_field` /
  `id_field` column names;
- `task.description` + `task.label_space` — the annotation task and its label set;
- `backend.*` — endpoint, model id, API key env var;
- `submission.*` — match the official leaderboard file schema;
- `evaluation.metric` — confirm the official metric (accuracy / macro_f1).

Then tune `retrieval.k_shots` / `strategy` / `ordering`, `self_consistency.n`,
and `backend.thinking_mode` on a held-out labelled slice (see below).

## Tuning workflow (Day 5)

The committee eval set may ship without gold labels, so tune on a held-out
slice carved from the unified pool:

```bash
# 1. split the labelled unified pool -> retrieval pool + scored dev set
python scripts/make_dev_split.py --config configs/default.yaml \
    --tune-out data/real/unified_tune.jsonl --dev-out data/real/dev.jsonl --dev-frac 0.2

# 2. grid-search ICL settings (edit the grid in configs/sweep.yaml first)
python scripts/sweep.py --sweep configs/sweep.yaml      # -> ranked table + configs/best.yaml

# 3. annotate the real eval set with the winning config
python scripts/annotate.py --config configs/best.yaml --cache outputs/run_cache.jsonl
```

`sweep.py` runs the full pipeline for every grid cell against the dev set and
writes the best-scoring config. Cost = `cells x len(dev)` model calls, so keep
the dev set (~50-100 rows) and grid modest on a paid API. Try it offline first:
`python scripts/sweep.py --sweep configs/sweep_mock.yaml`.

## ICL techniques (ranked by accuracy-per-hour)

1. **Retrieval-based exemplar selection** — `BAAI/bge-m3` similarity over the pool.
2. **Configurable shot count** — few-shot → many-shot sweep.
3. **Structured parsing + bounded stricter re-prompt** — kills malformed outputs.
4. **Self-consistency** — sample N, majority vote.
5. **Exemplar ordering** (`similar_last`) + **long-context chunking & aggregation**.
6. **Qwen3 thinking mode** — A/B flag.

## Status / next steps

- [x] End-to-end pipeline working on mock data, tests green (7/7).
- [x] Day-5 tuning tooling: dev-split maker + config sweep harness.
- [x] Registered for the challenge (Feishu form submitted).
- [ ] Join the Kaggle competition, lock in Track 3, download the dataset.
- [ ] Confirm official submission schema + metric.
- [ ] Wire `openai_api` backend to a hosted Qwen3-4B endpoint.
- [ ] Make dev split, run the sweep, pick best config.
- [ ] Final run on the real eval set + technical report.
