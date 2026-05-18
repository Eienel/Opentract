# FlagOS Open Computing Global Challenge — Track 3
## Qwen3-4B In-Context Learning Annotation Pipeline

**One-line summary.** A config-driven, format-agnostic in-context-learning pipeline that uses Qwen3-4B with retrieval-based exemplar selection, self-consistency, and long-context chunking to automatically annotate a competition-supplied dataset. Built for the FlagOS Open Computing Global Challenge (Track 3, leaderboard-ranked, deadline May 20 2026).

**Repo.** `github.com/eienel/opentract`

---

### The constraint that shaped the design

No GPU. That ruled out the two operator/inference-cluster tracks and forced a design that runs on whatever compute is actually affordable on the day — hosted APIs (OpenRouter / DashScope Qwen3-4B), a rented cloud GPU (vLLM, HuggingFace Transformers), or even CPU (llama.cpp Q4_K_M GGUF). Backend is a single config switch (`backend.type`); zero code changes to swap.

### What the pipeline does

For each evaluation item:

1. **Retrieve** — dense retrieval over the labeled exemplar pool using `BAAI/bge-m3` embeddings.
2. **Compose** — assemble a few-shot → many-shot prompt with configurable exemplar ordering (`similar_last` puts the most-similar example closest to the query).
3. **Generate** — Qwen3-4B inference, optional thinking-mode toggle, structured output target.
4. **Parse + repair** — strict structured-output parser; on malformed output, one bounded stricter re-prompt before giving up.
5. **Vote** — self-consistency: N samples per item, majority vote on the parsed label.
6. **Aggregate** — for over-length inputs, chunk and aggregate the chunk-level predictions.

A leaderboard submission file is written; any held-out labeled slice is scored automatically.

### Engineering choices that matter

- **Format-agnostic.** Adapting to a new dataset is a single-file edit to `configs/default.yaml` — `dataset.*`, `task.*`, `submission.*` schema, metric. No code changes. This is what makes the pipeline shippable on Day 5 with a never-before-seen committee dataset.
- **Hermetic offline tests.** Full E2E test on a mock sentiment dataset against an offline stub backend (4 tests, ~0.2s). The whole pipeline runs locally with `pip install -r requirements.txt && pytest -q`, no GPU/API/network required.
- **Sweep harness.** `scripts/sweep.py` grid-searches `k_shots × strategy × ordering × self_consistency.n × thinking_mode` against a dev split carved from the labeled pool. Writes the winning `configs/best.yaml`. Cost-bounded so paid-API tuning stays under budget.
- **Dev-split tooling.** `scripts/make_dev_split.py` carves a held-out slice from the unified pool when the committee eval set ships unlabeled — so tuning has a real metric to optimize.

### Stack

Python · PyTorch · HuggingFace Transformers · vLLM · llama.cpp · `BAAI/bge-m3` retrieval · pytest · YAML config-as-API.

### Skills demonstrated (mapped to AI training / annotation / LLM-evaluation work)

| Skill | Evidence in this project |
|---|---|
| LLM prompt engineering | Few-shot → many-shot sweep, `similar_last` exemplar ordering, structured-output target with bounded retry |
| Retrieval-augmented generation | Dense bge-m3 retrieval conditioning every prompt; retrieval pool / dev split separation |
| LLM serving | vLLM + OpenAI-compatible API + Transformers + llama.cpp behind one abstraction |
| Evaluation pipeline design | Dev splits, metric-agnostic scoring, leaderboard submission schema |
| Long-context handling | Chunk + aggregate strategy for over-length inputs |
| Inference-time techniques | Self-consistency with N-sample majority vote, thinking-mode A/B |
| Reproducible research engineering | Config-driven, hermetic offline tests, ablation sweeps, cost-aware tuning |

### Status

End-to-end pipeline green on mock data; tuning + sweep tooling in place; registered for the competition; final tuning + leaderboard submission in progress before the May 20 2026 deadline.
