"""Per-task runner -- ICL selection, prompting, concurrent inference, parsing, JSONL write.

Resume-safe: if the output JSONL already exists, previously-completed
test_sample_ids are skipped, so an interrupted run can be re-launched.

Concurrency: backend.complete_batch fans out prompts in parallel (configured
via BackendConfig.concurrency). For a hosted API and 500 long-context queries
per task, this is the difference between a 30-minute run and a 5-hour one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Any

from tqdm import tqdm

from .data import Task, render_io
from .parse import extract_label, majority_vote
from .prompts import build_prompt, task_gen_params
from .retrieve import RetrievalConfig, select_demos


@dataclass
class TaskRunConfig:
    task_id: int
    retrieval: RetrievalConfig
    self_consistency_n: int = 1
    self_consistency_temp: float = 0.7
    max_demos: int = 1000  # cap before packing (saves embedding time on similarity)
    limit_tests: int | None = None
    # Per-task generation overrides; falls back to prompts.task_gen_params(task_id).
    max_tokens: int | None = None
    stop: list[str] | None = None
    # Mini-batch size for incremental JSONL flushes -- protects against losing
    # all progress if a long batch crashes mid-way.
    flush_every: int = 50
    # Hard ceiling on the prompt+completion that the backend will accept. Used
    # to compute a per-task safe demo budget so a long test input + a generous
    # completion budget (task 8: 4000 tok) can't push the prompt past the model
    # context window. Default mirrors --max-model-len in the Kaggle notebook.
    model_max_len: int = 32768


def _tok_len(text: str, tokenizer: Any | None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return max(1, len(text) // 4)


def _adjust_demo_budget(
    task: Task,
    remaining: list,
    retrieval: RetrievalConfig,
    max_tokens: int,
    model_max_len: int,
    tokenizer: Any | None,
    safety_margin: int = 256,
) -> RetrievalConfig:
    """Return a retrieval config whose ``max_demo_tokens`` is small enough that
    even the worst-case prompt (longest test input + full completion budget +
    measured system/template overhead) fits inside ``model_max_len``.

    Only narrows the budget; never widens it past ``retrieval.max_demo_tokens``.
    """
    # Measure the fixed overhead by building a real prompt with empty examples
    # and the first test input, then subtract that input's token cost.
    sample_prompt = build_prompt(task, remaining[0].input, "")
    sample_input_rendered = render_io(remaining[0].input)
    fixed_overhead = _tok_len(sample_prompt, tokenizer) - _tok_len(sample_input_rendered, tokenizer)

    longest_input_tokens = max(_tok_len(render_io(t.input), tokenizer) for t in remaining)

    safe = model_max_len - max_tokens - longest_input_tokens - fixed_overhead - safety_margin
    safe = max(1024, safe)  # always pack at least one or two demos

    if safe < retrieval.max_demo_tokens:
        print(
            f"  task {task.task_id}: trimmed demo budget "
            f"{retrieval.max_demo_tokens} -> {safe} "
            f"(longest input {longest_input_tokens} tok, completion {max_tokens} tok, "
            f"system overhead {fixed_overhead} tok)"
        )
        return replace(retrieval, max_demo_tokens=safe)
    return retrieval


def _output_path(out_dir: str, task: Task) -> str:
    return os.path.join(out_dir, f"{task.slug}-v1.jsonl")


def _load_existing(out_path: str) -> set[str]:
    if not os.path.exists(out_path):
        return set()
    done: set[str] = set()
    with open(out_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["test_sample_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_task(
    task: Task,
    backend: Any,
    cfg: TaskRunConfig,
    out_dir: str,
    tokenizer: Any | None = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = _output_path(out_dir, task)

    done = _load_existing(out_path)
    if done:
        print(f"  task {task.task_id}: resuming, {len(done)} already done")

    demos_pool = task.demos[: cfg.max_demos]
    tests = task.tests if cfg.limit_tests is None else task.tests[: cfg.limit_tests]
    remaining = [t for t in tests if t.id not in done]
    if not remaining:
        print(f"  task {task.task_id}: nothing to do")
        return out_path

    # Resolve generation params: explicit > per-task default > backend default.
    defaults = task_gen_params(task.task_id)
    max_tokens = cfg.max_tokens if cfg.max_tokens is not None else defaults["max_tokens"]
    stop = cfg.stop if cfg.stop is not None else defaults.get("stop")

    # Trim the demo budget so the WORST-CASE prompt (longest test input + full
    # completion budget + system/template overhead) still fits in the model
    # context. Without this, fixed MAX_DEMO_TOK + a long test input blows past
    # --max-model-len and the backend returns 400.
    adjusted_retrieval = _adjust_demo_budget(
        task=task,
        remaining=remaining,
        retrieval=cfg.retrieval,
        max_tokens=max_tokens,
        model_max_len=cfg.model_max_len,
        tokenizer=tokenizer,
    )

    # Demos are reused across queries in the task -- this is what makes
    # provider-side prefix caching effective.
    examples_str, n_packed = select_demos(demos_pool, remaining[0], adjusted_retrieval, tokenizer)
    print(f"  task {task.task_id}: packed {n_packed} demos into the prefix")

    n = max(1, cfg.self_consistency_n)
    temp = cfg.self_consistency_temp if n > 1 else None

    mode = "a" if done else "w"
    with open(out_path, mode, encoding="utf-8") as fout:
        # Process in mini-batches so progress is flushed often and the tqdm
        # bar moves smoothly under concurrency.
        for start in tqdm(
            range(0, len(remaining), cfg.flush_every),
            desc=f"task {task.task_id} {task.task_name}",
        ):
            batch = remaining[start : start + cfg.flush_every]
            prompts = [build_prompt(task, t.input, examples_str) for t in batch]
            outputs = backend.complete_batch(
                prompts, n=n, temperature=temp, max_tokens=max_tokens, stop=stop
            )

            for test, samples in zip(batch, outputs):
                parsed = [extract_label(s, task.task_id) for s in samples]
                prediction = majority_vote(parsed) if n > 1 else parsed[0]
                row = {"test_sample_id": test.id, "prediction": prediction}
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

    return out_path
