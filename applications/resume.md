# Nasiru Abubakar
elnasirulabaran212@gmail.com · Bauchi, Nigeria · github.com/Eienel

---

## Education

**Abubakar Tafawa Balewa University Bauchi**, Nigeria
B.Eng / BSc [DEGREE — e.g. Computer Engineering / Computer Science] · 3rd Year · Expected graduation: [20XX]

---

## Skills

**Languages:** Python (primary), Bash
**ML / DL:** PyTorch, HuggingFace Transformers, vLLM, llama.cpp
**LLM / Prompt Engineering:** Qwen3-4B, few-shot & many-shot ICL, self-consistency sampling, retrieval-augmented prompting, structured output parsing, chain-of-thought / thinking-mode
**Retrieval:** Dense embeddings (BAAI/bge-m3), FAISS-style nearest-neighbour search
**Data & Config:** JSONL, CSV, YAML config-as-API, dataset curation and dev-split tooling
**Dev:** pytest, Git, Linux

---

## Projects

### FlagOS Open Computing Global Challenge — Track 3 (2025–2026)
`github.com/Eienel/opentract`

Built a production-quality, config-driven In-Context Learning annotation pipeline for a global AI competition (leaderboard-ranked, deadline May 2026). The system annotates arbitrary evaluation datasets using Qwen3-4B with no code changes required to adapt to new tasks.

**What it does:**
- Dense retrieval over a labeled exemplar pool (BAAI/bge-m3) to select semantically relevant few-shot examples for each input
- Self-consistency inference: N samples per input + majority vote for label stability
- Long-context chunking and aggregation for over-length inputs
- Structured output parser with bounded re-prompt on malformed outputs
- Automated hyperparameter sweep across k-shots, retrieval strategy, exemplar ordering, and thinking-mode; writes the winning config automatically
- Multi-backend support: hosted API (OpenRouter / DashScope), cloud GPU (vLLM / HuggingFace Transformers), CPU-only (llama.cpp Q4_K_M GGUF)
- Hermetic offline test suite (pytest, mock dataset) — full end-to-end verifiable with zero GPU / API / network dependency

**Stack:** Python, PyTorch, HuggingFace Transformers, vLLM, llama.cpp, bge-m3, pytest, YAML

---

## About

Third-year engineering student at a Nigerian federal university. Self-directed in machine learning — built the FlagOS pipeline independently to compete in a global AI challenge. Comfortable with the full LLM development loop: dataset preparation, retrieval pipeline design, prompt engineering, inference-time techniques (self-consistency, thinking-mode), evaluation, and cost-aware tuning. Looking for AI training / RLHF / annotation work where I can apply and deepen these skills on real production tasks.
