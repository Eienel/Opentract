"""Hermetic end-to-end test: retrieval -> prompt -> stub backend -> parse ->
vote -> score -> submission. No GPU, no API, no network.

The mock dataset is lexically separable and the stub backend is a 1-NN
classifier, so a correct pipeline must score well above chance.
"""

import os

from icl.config import Config
from icl.datasets import load_eval
from icl.evaluator import evaluate, write_submission
from icl.parser import OutputParser, strip_thinking
from icl.pipeline import build_runner

CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "mock.yaml")


def _cfg() -> Config:
    return Config.from_yaml(CONFIG)


def test_pipeline_runs_and_scores_above_chance(tmp_path):
    cfg = _cfg()
    cfg.submission.path = str(tmp_path / "submission.jsonl")

    eval_examples = load_eval(cfg.dataset)
    runner = build_runner(cfg, cache_path=str(tmp_path / "cache.jsonl"))
    results = runner.run(eval_examples)

    assert len(results) == len(eval_examples)
    # Every item gets a prediction inside the label space (no parse failures).
    valid = set(cfg.task.label_space)
    for r in results:
        assert r.prediction in valid, f"{r.id}: {r.prediction!r} not a valid label"

    report = evaluate(results, eval_examples, cfg)
    assert report.n_scored == len(eval_examples)
    # 1-NN over separable mock data; chance is 0.33.
    assert report.score >= 0.8, f"accuracy too low: {report.score}"

    path = write_submission(results, cfg)
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    assert len(lines) == len(eval_examples)


def test_cache_resumes(tmp_path):
    cfg = _cfg()
    cache = str(tmp_path / "cache.jsonl")
    eval_examples = load_eval(cfg.dataset)

    first = build_runner(cfg, cache_path=cache).run(eval_examples)
    assert os.path.exists(cache)
    # Second run must read from cache and produce identical predictions.
    second = build_runner(cfg, cache_path=cache).run(eval_examples)
    assert [r.prediction for r in first] == [r.prediction for r in second]


def test_parser_strips_thinking_and_coerces_label():
    cfg = _cfg()
    parser = OutputParser(cfg.parser, cfg.task)
    assert strip_thinking("<think>hmm</think>\nAnnotation: positive") == "positive"
    assert parser.parse("<think>reasoning here</think> negative") == "negative"
    assert parser.parse("The sentiment is clearly Positive.") == "positive"
    assert parser.parse("") is None
    assert parser.parse("not-a-label-at-all") is None


def test_unknown_config_key_fails_loudly(tmp_path):
    import pytest

    bad = tmp_path / "bad.yaml"
    bad.write_text("retrieval:\n  k_shots: 3\n  bogus_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config keys"):
        Config.from_yaml(str(bad))
