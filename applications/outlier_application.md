# Outlier Application — Field-by-Field Draft

Apply at: https://outlier.ai/expert/apply
Estimated time: 20 min profile + 1-2 hr assessment test

---

## Profile Fields

**Full name:** Nasiru Abubakar
**Email:** elnasirulabaran212@gmail.com
**Country:** Nigeria
**GitHub / Portfolio:** https://github.com/Eienel/opentract

**Skills / areas of expertise to select:**
- Python
- Machine Learning
- Artificial Intelligence
- Data Science
- Software Engineering
- Natural Language Processing

**Education:**
- Abubakar Tafawa Balewa University Bauchi — B.Eng [DEGREE] — 3rd Year

**Resume:** Upload `resume.md` converted to PDF, or copy-paste the text from `resume.md`.

---

## Assessment Test Prep

Outlier's coding assessment (1-2 hr) typically has 3 sections:

### Section 1: Python Coding (usually 3-5 problems, 30-45 min)

Practice these exact problem types:

```python
# Type 1: Parse structured LLM output
def parse_label(response: str, label_space: list[str]) -> str | None:
    """Return the first label in label_space found in response, else None."""
    response_lower = response.lower()
    for label in label_space:
        if label.lower() in response_lower:
            return label
    return None

# Type 2: Majority vote
from collections import Counter
def majority_vote(votes: list[str]) -> str:
    return Counter(votes).most_common(1)[0][0]

# Type 3: Cosine similarity
import numpy as np
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

# Type 4: JSONL reader
import json
def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
```

### Section 2: AI/ML Conceptual Questions

**Q: What is the difference between few-shot and zero-shot prompting?**
> Zero-shot: give the model only the instruction, no examples — relies entirely on the model's pretrained knowledge. Few-shot: provide labeled input→output examples in the prompt so the model infers the pattern. Few-shot consistently outperforms zero-shot on classification and annotation tasks because it anchors the model to the exact output format and task framing you want.

**Q: What is RLHF and why is it used?**
> RLHF (Reinforcement Learning from Human Feedback) aligns LLMs to human preferences after pretraining. A reward model is trained on human preference rankings of model outputs, then the base model is fine-tuned with RL (PPO or DPO) to maximise that reward. It's how ChatGPT/Claude/Gemini went from autocomplete models to helpful assistants. Human annotators are the core input that makes RLHF work — their preference ratings and labeled outputs train the reward model.

**Q: How would you evaluate whether a model's output is correct?**
> For classification/annotation tasks: compare against a gold-label dev set using accuracy or macro F1. For open-ended generation: human preference ratings (pairwise: "which response is better?"), plus automatic metrics like ROUGE for summarization or exact-match for structured output. I've built evaluation pipelines that automate the dev-set comparison and report per-class accuracy breakdowns.

**Q: What is self-consistency in LLMs?**
> Sample the model N times on the same input (temperature > 0), then take the majority vote over the N outputs. It improves accuracy on reasoning and classification tasks because the model's correct chain of thought is more likely to dominate the vote than any individual error. Works especially well when the model is uncertain — a single sample might flip between classes, but 5-10 samples will cluster on the correct one.

### Section 3: AI Output Evaluation (common in Outlier code-review tasks)

You'll be shown AI-generated code and asked to rate it on dimensions like:
- **Correctness** — does it do what the prompt asked?
- **Style** — is it clean, idiomatic Python?
- **Safety** — any obvious bugs, off-by-ones, division by zero?
- **Helpfulness** — does the explanation match the code?

**Tip:** Be specific in your ratings. Don't write "this code is fine." Write "the loop correctly iterates over items but will raise IndexError on an empty list because there's no guard on line 3." Outlier raters who give specific, actionable feedback get assigned to higher-paying tasks.

---

## Payout Setup for Nigeria

Outlier pays primarily via **Airwallex**. Nigeria's Airwallex support has been inconsistent — verify during account setup that your country is supported for bank transfer. If not, check if Payoneer is offered as an alternative.

**Backup plan:** Create a **Grey** account (grey.co — Nigeria-focused, supported USD virtual account) which can receive Airwallex/ACH transfers as USD and convert to NGN. Grey is widely used by Nigerian freelancers on international platforms.

---

## Timeline Expectation

- Day 0: Submit application
- Days 1-5: Assessment test invite arrives (check spam folder)
- Days 3-10: Assessment reviewed, ID verification if passed
- Days 7-14: First project assignment
- Day 14+: First payout (Airwallex / Grey)

---

## What to write in the "Why do you want to work with Outlier?" field

> "I've been building LLM systems independently — currently a Qwen3-4B annotation pipeline for a global AI competition. I want to work on AI training tasks because I understand the pipeline end-to-end: how prompt design affects output quality, how to evaluate labels consistently, and how human feedback shapes model behaviour. I'm looking for work where that understanding translates to better annotation — not just following a rubric, but reasoning about why a response is good or bad."
