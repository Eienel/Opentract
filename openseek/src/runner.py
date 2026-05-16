"""Per-task runner -- handles ICL selection, prompting, sampling, parsing, JSONL write."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from .data import Example, Task
from .parse import extract_label, majority_vote
from .prompts import build_prompt
from .retrieve import RetrievalConfig, select_demos


@dataclass
class TaskRunConfig:
    task_id: int
    retrieval: RetrievalConfig
    self_consistency_n: int = 1
    self_consistency_temp: float = 0.7
    max_demos: int = 1000  # cap before packing (saves embedding time)
    reuse_demos_across_queries: bool = True  # baseline behaviour: same demo set
    limit_tests: int | None = None  # None = all; useful for smoke/dev


def _output_path(out_dir: str, task: Task) -> str:
    return os.path.join(out_dir, f"{task.slug}-v1.jsonl")


def run_task(
    task: Task,
    backend: Any,
    cfg: TaskRunConfig,
    out_dir: str,
    tokenizer: Any | None = None,
    cache_path: str | None = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = _output_path(out_dir, task)

    seen: set[str] = set()
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    seen.add(json.loads(line)["test_sample_id"])

    demos_pool = task.demos[: cfg.max_demos]
    tests = task.tests if cfg.limit_tests is None else task.tests[: cfg.limit_tests]

    examples_str: str | None = None  # reused when reuse_demos_across_queries is true

    with open(out_path, "w", encoding="utf-8") as fout:
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as fh:
                fout.write(fh.read())

        for test in tqdm(tests, desc=f"task {task.task_id} {task.task_name}"):
            if test.id in seen:
                continue

            if examples_str is None or not cfg.reuse_demos_across_queries:
                examples_str, _ = select_demos(demos_pool, test, cfg.retrieval, tokenizer)

            prompt = build_prompt(task, test.input, examples_str)

            n = max(1, cfg.self_consistency_n)
            temp = cfg.self_consistency_temp if n > 1 else None
            samples = backend.complete(prompt, n=n, temperature=temp)
            parsed = [extract_label(s, task.task_id) for s in samples]
            prediction = majority_vote(parsed) if n > 1 else parsed[0]

            row = {"test_sample_id": test.id, "prediction": prediction}
            line = json.dumps(row, ensure_ascii=False)
            fout.write(line + "\n")
            fout.flush()
            if cache_path:
                with open(cache_path, "a", encoding="utf-8") as cf:
                    cf.write(line + "\n")

    return out_path
