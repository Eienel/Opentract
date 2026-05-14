"""Typed, YAML-backed configuration for the ICL annotation pipeline.

The whole pipeline is driven by a single config file. When the real committee
dataset arrives, adapting the pipeline should mean editing one YAML file -- not
touching code. See configs/default.yaml for the documented schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, get_type_hints

import yaml


@dataclass
class DatasetConfig:
    unified_path: str = "data/mock/unified.jsonl"
    eval_path: str = "data/mock/eval.jsonl"
    format: str = "jsonl"  # jsonl | json | csv
    id_field: str = "id"
    input_field: str = "text"
    label_field: str = "label"  # may be absent in the eval file
    metadata_fields: list[str] = field(default_factory=list)


@dataclass
class TaskConfig:
    description: str = "Annotate the input."
    # Closed label set enables constrained parsing + class-balanced retrieval.
    # Leave null/empty for open-ended annotation.
    label_space: list[str] = field(default_factory=list)


@dataclass
class BackendConfig:
    type: str = "stub"  # stub | openai_api | transformers | llamacpp
    model: str = "Qwen/Qwen3-4B"
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 512
    temperature: float = 0.0
    thinking_mode: bool = False  # Qwen3 dual-mode reasoning toggle
    request_timeout: int = 120
    max_retries: int = 4
    # Backend-specific extras (e.g. gguf_repo, n_threads, device).
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    type: str = "sentence_transformers"  # sentence_transformers | hashing
    model: str = "BAAI/bge-m3"
    cache_dir: str = ".index_cache"
    dim: int = 512  # only used by the hashing fallback


@dataclass
class RetrievalConfig:
    k_shots: int = 5
    strategy: str = "similarity"  # similarity | random | class_balanced
    ordering: str = "similar_last"  # similar_last | similar_first | as_is
    seed: int = 0


@dataclass
class SelfConsistencyConfig:
    n: int = 1
    temperature: float = 0.7  # applied only when n > 1


@dataclass
class ChunkingConfig:
    max_input_chars: int = 24000  # rough budget; longer inputs are split
    overlap_chars: int = 500
    aggregation: str = "majority"  # majority | first


@dataclass
class ParserConfig:
    type: str = "label"  # label | json | regex
    json_key: str = "label"
    regex: str | None = None
    max_retries: int = 2


@dataclass
class PromptConfig:
    system: str = (
        "You are a precise data annotation engine. Follow the instructions "
        "exactly and output only the requested annotation."
    )
    template: str = (
        "{{ task_description }}\n\n"
        "{% if label_space %}Valid labels: {{ label_space | join(', ') }}\n\n{% endif %}"
        "{% for ex in exemplars %}"
        "Input: {{ ex.input_text }}\nAnnotation: {{ ex.label }}\n\n"
        "{% endfor %}"
        "Input: {{ query.input_text }}\nAnnotation:"
    )


@dataclass
class SubmissionConfig:
    format: str = "jsonl"  # jsonl | csv
    path: str = "outputs/submission.jsonl"
    id_field: str = "id"
    prediction_field: str = "label"


@dataclass
class EvaluationConfig:
    metric: str = "accuracy"  # accuracy | macro_f1


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    self_consistency: SelfConsistencyConfig = field(default_factory=SelfConsistencyConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    submission: SubmissionConfig = field(default_factory=SubmissionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return _build(cls, raw)


def _build(dc_type: type, raw: dict[str, Any]) -> Any:
    """Recursively construct a (possibly nested) dataclass from a plain dict.

    Unknown keys raise immediately -- a typo in the config should fail loudly
    rather than silently using a default during a time-boxed competition.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"Expected mapping for {dc_type.__name__}, got {type(raw).__name__}")
    kwargs: dict[str, Any] = {}
    # get_type_hints resolves the string annotations produced by
    # `from __future__ import annotations` back into real types.
    hints = get_type_hints(dc_type)
    known = {f.name for f in fields(dc_type)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys for {dc_type.__name__}: {sorted(unknown)}")
    for name in known:
        if name not in raw:
            continue
        value = raw[name]
        field_type = hints.get(name)
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[name] = _build(field_type, value)
        else:
            kwargs[name] = value
    return dc_type(**kwargs)
