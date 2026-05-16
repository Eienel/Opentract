"""End-to-end smoke test using the stub backend on a few real test samples per
task. Asserts that the orchestrator produces a valid 8-file submission ZIP.

The stub responder echoes the last <label>X</label> seen in the prompt, mimicking
a 1-NN classifier. That makes the test deterministic without a GPU or API.
"""

import json
import os
import re
import zipfile

from openseek.src.backend import BackendConfig
from openseek.src.data import TASK_IDS, find_task_file, load_all_tasks
from openseek.src.orchestrator import RunAllConfig, run_all
from openseek.src.retrieve import RetrievalConfig

_LAST_LABEL = re.compile(r"<label>\s*(.+?)\s*</label>", re.DOTALL)


def _last_label_responder(prompt: str) -> str:
    matches = _LAST_LABEL.findall(prompt)
    if matches:
        return f"<label>{matches[-1]}</label>"
    return "<label>STUB</label>"


def _has_datasets() -> bool:
    try:
        for tid in TASK_IDS:
            find_task_file(tid)
        return True
    except FileNotFoundError:
        return False


def test_orchestrator_produces_valid_submission(tmp_path):
    if not _has_datasets():
        import pytest

        pytest.skip("Datasets not fetched. Run: python openseek/scripts/fetch_data.py")

    out_dir = tmp_path / "run"
    sub_path = tmp_path / "submission.zip"

    cfg = RunAllConfig(
        out_dir=str(out_dir),
        submission_path=str(sub_path),
        backend=BackendConfig(type="stub"),
        retrieval=RetrievalConfig(strategy="first_n", max_demo_tokens=2000),
        self_consistency_n=1,
        max_demos=50,
        limit_tests=2,  # tiny per-task smoke
    )
    final_zip = run_all(cfg, stub_responder=_last_label_responder)

    assert os.path.exists(final_zip)
    with zipfile.ZipFile(final_zip) as zf:
        names = sorted(zf.namelist())
        assert len(names) == 8
        for i, name in enumerate(names, 1):
            assert name.startswith(f"openseek-{i}-"), f"unexpected file: {name}"
            assert "/" not in name and "\\" not in name, f"nested file forbidden: {name}"
            with zf.open(name) as fh:
                lines = [ln for ln in fh.read().decode("utf-8").splitlines() if ln.strip()]
            assert len(lines) == 2, f"{name}: expected 2 predictions, got {len(lines)}"
            for ln in lines:
                row = json.loads(ln)
                assert set(row.keys()) == {"test_sample_id", "prediction"}
                assert isinstance(row["test_sample_id"], str)


def test_task_loader_covers_all_eight():
    if not _has_datasets():
        import pytest

        pytest.skip("Datasets not fetched")
    tasks = load_all_tasks()
    assert [t.task_id for t in tasks] == list(range(1, 9))
    counts = {t.task_id: len(t.tests) for t in tasks}
    # spec from data/README: 500 per task except task 8 which is 166
    for tid in range(1, 8):
        assert counts[tid] == 500, f"task {tid}: {counts[tid]}"
    assert counts[8] == 166
