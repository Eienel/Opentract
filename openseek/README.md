# OpenSeek Track 3 — Sprint Runbook

ICL data annotation on 8 long-context tasks with **Qwen3-4B**, served via
**FlagScale + vLLM**. Submission deadline: **May 20, 2026, 16:59 UTC+8**.

This module is purpose-built for the OpenSeek format and runs the full pipeline:
load → retrieve demos → prompt → call Qwen3-4B → parse `<label>` → write JSONL →
ZIP for upload. See `../hackathon-research.html` for the broader hackathon
strategy.

## 0. Local smoke test (no GPU, no model — instant)

```bash
pip install -r requirements.txt
python openseek/scripts/fetch_data.py                # 8 JSON files (~10 MB)
pytest openseek/tests/ -q                            # 2 tests, ~0.3s
```

The smoke test runs the full orchestrator with a stub backend on a tiny per-task
subset and asserts the submission ZIP has the correct shape (8 flat JSONLs,
right schema, no nesting).

## 1. Get a GPU

**Recommended:** 1× **A100 80GB** or **H100 80GB**.
- Qwen3-4B in bf16 fits easily; 80GB headroom is for **30K+ context** KV-cache.
- Vast.ai / RunPod community pricing: ~$2–3/hr. Budget ~$100–150 for the sprint.
- Alternative: 1× A100 40GB works for tasks 1–7 (30K context) but is tight on
  task 8 (16K, generous output) and any many-shot experiments. 80GB strongly preferred.

## 2. Install FlagScale + serve Qwen3-4B

Follow the official script (it's in the OpenSeek repo):
`OpenSeek/openseek/competition/LongContext-ICL-Annotation/src/create_env_nvidia.sh`

Highlights / gotchas:
- Python **3.11.11** in a fresh conda env
- Torch **2.6.0 + CUDA 12.4** (matches the FlashAttention wheel they pin)
- vLLM **0.8.5**, FlagGems @ `release_v1.0.0`
- The flash-attn wheel filename must match `cu124torch2.6 + cp311`

Then download the model and tweak its context config:

```bash
hf download Qwen/Qwen3-4B --local-dir Qwen3-4B
# edit Qwen3-4B/config.json -- replace rope_scaling with:
# "rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
```

Then bring up serving from the FlagScale directory using
`OpenSeek/.../src/llm_config.yaml` (port **2026**, OpenAI-compatible):

```bash
cd FlagScale
python run.py --config-path /path/to/openseek/llm_config --config-name llm_config action=run
# health check:
curl http://0.0.0.0:2026/v1/models
```

## 3. First leaderboard submission (Day 2)

Once serving is healthy:

```bash
# Tiny smoke first (~80 calls, a couple of minutes)
python openseek/scripts/run_all.py --limit-tests 10

# Full baseline run on all 8 tasks
python openseek/scripts/run_all.py \
    --strategy first_n --max-demo-tokens 28000 \
    --out-dir openseek/outputs/run-001

# -> openseek/outputs/submission.zip
```

Upload `submission.zip` via the **Submission > Prediction Result** tab at
[flagos.io/RaceDetail?id=296fmsd8](https://flagos.io/RaceDetail?id=296fmsd8&lang=en).
You get **up to 5 submissions per day** — system retains your best.

## 4. Iteration plan (Days 3–4)

In order of expected leverage:

| Lever | How | Expected lift |
|---|---|---|
| **Many-shot, full 30K context** | already on by default (`--max-demo-tokens 28000`). Verify each task fills the budget. | base → strong baseline |
| **Similarity retrieval** | `--strategy similarity` (pulls in BGE-m3 embeddings) | tasks where input shape matters (4, 5, 6, 7) |
| **Self-consistency** | `--n 3 --sc-temp 0.7` for classification/integer tasks (1, 2, 3, 5, 6) — skip for code (4, 8) which would explode cost | classification +1-3% |
| **Per-task prompts** | already tuned in `src/prompts.py` — refine when you see failure modes | format-error reduction |
| **Task 8 specifically** | scored by *code execution*. Look at example outputs, ensure the prompt produces a complete `import torch / import triton / @triton.jit / wrapper` block. Consider a structural validator before submission. | task-8 is the biggest single-task lever |

Run single tasks while iterating:

```bash
python openseek/scripts/run_one.py --task 5 --strategy similarity --n 3 --limit-tests 50
# inspect outputs/run/openseek-5-v1.jsonl, tweak src/prompts.py
```

When you've iterated each task to your satisfaction, repack:

```bash
python openseek/scripts/make_submission.py --in-dir openseek/outputs/run
```

## 5. Tech report + open-source PR (Days 5–7)

- **May 25** — `Technical_Report-[Team_Name].pdf`, 5+ pages, uploaded to the
  flagos.io Submission tab. Cover prompt design, long-context construction,
  reproducibility. Already mostly captured by this README + the code comments.
- **May 30** — full source code archive uploaded.
- **May 21–31** — open a PR to
  [`FlagAI-Open/OpenSeek`](https://github.com/FlagAI-Open/OpenSeek) with the
  same code, paste the PR link in the Submission > PR Link tab.

## 6. Module map

```
openseek/
  data/                 # 8 official JSONs (gitignored, fetch_data.py downloads)
  src/
    data.py             # Task + Example loaders
    backend.py          # OpenAI-compatible client (talks to FlagScale vLLM)
    retrieve.py         # token-budget bin-pack of demos, first_n or similarity
    prompts.py          # per-task prompt builders (the lever in plain sight)
    parse.py            # <label>...</label> extractor + majority vote
    runner.py           # per-task: prompt -> generate -> parse -> JSONL
    orchestrator.py     # run all 8 -> ZIP submission
  scripts/
    fetch_data.py       # download the 8 datasets
    run_all.py          # one command to produce submission.zip
    run_one.py          # iterate on a single task
    make_submission.py  # repack from existing per-task JSONLs
  tests/test_smoke.py   # end-to-end offline (stub backend) — 2 tests
```
