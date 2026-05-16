"""Run all 8 tasks and pack the submission ZIP."""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass, field
from typing import Any

from .backend import BackendConfig, build_backend
from .data import TASK_IDS, load_task
from .retrieve import RetrievalConfig
from .runner import TaskRunConfig, run_task


@dataclass
class RunAllConfig:
    out_dir: str = "openseek/outputs/run"
    submission_path: str = "openseek/outputs/submission.zip"
    backend: BackendConfig = field(default_factory=BackendConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    self_consistency_n: int = 1
    self_consistency_temp: float = 0.7
    max_demos: int = 1000
    limit_tests: int | None = None
    task_overrides: dict[int, dict[str, Any]] = field(default_factory=dict)


def _build_task_cfg(task_id: int, cfg: RunAllConfig) -> TaskRunConfig:
    overrides = cfg.task_overrides.get(task_id, {})
    return TaskRunConfig(
        task_id=task_id,
        retrieval=cfg.retrieval,
        self_consistency_n=overrides.get("self_consistency_n", cfg.self_consistency_n),
        self_consistency_temp=overrides.get("self_consistency_temp", cfg.self_consistency_temp),
        max_demos=overrides.get("max_demos", cfg.max_demos),
        limit_tests=overrides.get("limit_tests", cfg.limit_tests),
    )


def run_all(cfg: RunAllConfig, tokenizer: Any | None = None, stub_responder=None) -> str:
    backend = build_backend(cfg.backend, stub_responder=stub_responder)
    os.makedirs(cfg.out_dir, exist_ok=True)

    jsonl_paths: list[str] = []
    for tid in TASK_IDS:
        task = load_task(tid)
        tcfg = _build_task_cfg(tid, cfg)
        cache_path = os.path.join(cfg.out_dir, f"{task.slug}.cache.jsonl")
        path = run_task(task, backend, tcfg, cfg.out_dir, tokenizer=tokenizer, cache_path=cache_path)
        jsonl_paths.append(path)

    return pack_submission(jsonl_paths, cfg.submission_path)


def pack_submission(jsonl_paths: list[str], submission_path: str) -> str:
    """Zip the 8 jsonl files flat (no nested directories) per the official spec."""
    if len(jsonl_paths) != 8:
        raise ValueError(f"Expected 8 jsonl files, got {len(jsonl_paths)}")
    os.makedirs(os.path.dirname(submission_path) or ".", exist_ok=True)
    with zipfile.ZipFile(submission_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in jsonl_paths:
            zf.write(p, arcname=os.path.basename(p))
    return submission_path
