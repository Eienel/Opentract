# Mercor Application — Field-by-Field Draft

Apply at: https://mercor.com/apply
Estimated time: 30 min to fill profile + 60-90 min AI interview

---

## Profile Fields

**Full name:** Nasiru Abubakar
**Email:** elnasirulabaran212@gmail.com
**Country:** Nigeria
**City:** Bauchi
**GitHub:** https://github.com/Eienel
**LinkedIn:** (leave blank or skip if optional)
**Portfolio / website:** https://github.com/Eienel/opentract

**Skills to select (check all that apply):**
- Python
- Machine Learning
- Deep Learning
- Natural Language Processing
- Large Language Models / LLM
- Prompt Engineering
- PyTorch
- HuggingFace
- Data Annotation / Labeling
- AI Evaluation / RLHF
- Git / Version Control

**Years of experience:** 0-1 (select "Student" or "Less than 1 year" if available)

**Education:**
- University: Abubakar Tafawa Balewa University Bauchi
- Degree: B.Eng [DEGREE] / Computer Engineering
- Year: 3rd year (expected graduation [20XX])
- GPA: [Add if ≥ 3.5/4.0 or equivalent]

**Expected hourly rate:** Start at $15-20/hr. Do NOT undersell below $15 — Mercor routes you to higher-quality tasks if you set a minimum above their floor.

**Bio (200 characters max — copy this exactly):**
```
3rd-year engineering student at ATBU Bauchi. Built a Qwen3-4B ICL annotation pipeline for a global AI challenge. Prompt eng, RAG, self-consistency.
```
(149 chars — paste as-is)

---

## AI Interview Prep (Mercor uses an AI interviewer, ~60-90 min)

The interview has 3 parts: introductory, technical, and a live task. Prepare these answers now.

### 1. Tell me about yourself

> "I'm a third-year engineering student at Abubakar Tafawa Balewa University in Bauchi, Nigeria. I've been building in machine learning independently — most recently I built a production-ready In-Context Learning pipeline that uses Qwen3-4B with dense retrieval and self-consistency to automatically annotate datasets. It's live on GitHub as part of a global AI competition I'm competing in right now. I'm specifically looking for AI training and annotation work because I want to apply what I've built to real production tasks."

### 2. Describe a project you're proud of

Use the FlagOS project. Key beats to hit:
- **Problem:** Needed to annotate an unknown-format evaluation dataset for a leaderboard competition, with no GPU budget.
- **Approach:** Built a retrieval-augmented few-shot pipeline — embed the labeled exemplar pool with bge-m3, retrieve the most relevant examples for each input, assemble a few-shot prompt, inference with Qwen3-4B, majority-vote over N self-consistency samples.
- **Engineering decisions:** Format-agnostic config-driven design (one YAML edit to adapt to new dataset), multi-backend support (hosted API / vLLM / llama.cpp), hermetic offline tests.
- **Result:** End-to-end pipeline green, sweep harness built, competing now.

### 3. How do you handle a model giving inconsistent or wrong outputs?

> "Two approaches depending on the source. For format/structure issues, I add a bounded re-prompt — parse the output, and if it fails validation, feed the error back to the model once and try again with a stricter format instruction. For label instability, I use self-consistency — sample N outputs at higher temperature and take the majority vote. Both are implemented in the FlagOS pipeline. I also tune on a held-out dev split so I can measure whether a change actually improves accuracy before running the full eval."

### 4. What do you know about RLHF / AI training?

> "RLHF — Reinforcement Learning from Human Feedback — is how modern LLMs are aligned to be helpful and safe after pretraining. The key loop is: generate model responses, have human raters rank them (preference data), train a reward model on those preferences, then fine-tune the policy with PPO or DPO to maximise the reward signal. What I do in annotation work directly feeds this pipeline — the ranked examples and labeled outputs I produce become the human feedback that drives alignment. I'm familiar with DPO specifically as a cleaner alternative to PPO for preference optimization."

### 5. Walk me through a prompt you've engineered

Use the ICL prompt from the project:
> "In the FlagOS pipeline, each prompt has three parts. First, a task description block that says exactly what annotation is needed and what the valid label set is — this is injected from config so it's always consistent. Second, a retrieved few-shot block — I use bge-m3 to find the k most similar labeled examples from the pool and format them as input→output pairs, ordered with the most similar one last (proximity to the query matters for attention). Third, the actual input to annotate. I found that 'similar-last' ordering and 4-8 shots gave the best accuracy on the mock dev set, though the right k varies by task — that's what the sweep harness finds automatically."

### 6. "Tell me about other projects you've built" — talking points

**BlockBuilders:** "I built a 3D browser game that teaches crypto concepts using Three.js for instanced 60fps rendering and Gemini 2.5 Flash for an in-game AI tutor agent. Players can ask the AI anything about wallets, DeFi, or ZK proofs in natural language during gameplay. It mints a completion NFT on Sui blockchain."

**ShadowPay:** "I built a confidential payroll protocol using Fully Homomorphic Encryption — specifically iExec Nox with Zama's FHE coprocessor — so companies can pay employees in stablecoins on-chain without exposing salaries publicly. I also added a natural language interface so you can type 'Pay Alice $5K monthly' and it generates the contract parameters."

**Bequest:** "I built a crypto dead-man's switch. You designate a beneficiary; if you miss a check-in window, your assets auto-transfer. The tricky parts were the state machine (draft → armed → triggered enforced by minutely cron jobs), per-wallet AES-256 key management, and multi-channel check-in via web/Telegram/email."

**Why these matter for Mercor:** These show you can architect non-trivial systems, integrate AI APIs into real products, and ship end-to-end — not just write isolated scripts.

### 7. Live coding task (common Mercor format)

Likely a Python task: string processing, data transformation, or a small ML function. Practice these:
- Write a function to parse structured output from an LLM response
- Write a function to compute majority vote over a list of labels
- Implement a simple cosine similarity function over numpy vectors
- Basic pandas: group, filter, aggregate a JSONL file

---

## After the interview

- Mercor typically responds within 2-5 business days
- If accepted, you're matched to a project — say yes to the first one even if the rate is $15/hr. First 2-4 weeks build your review score; rates go up.
- **Payment:** Set up Wise (wise.com) before your first payment lands. Mercor pays USD weekly via Wise. Nigeria is fully supported.
