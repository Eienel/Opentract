"""Wiring helpers shared by the CLI scripts.

Keeps build_index / annotate / evaluate thin and consistent.
"""

from __future__ import annotations

import os

from .backends.base import build_backend
from .config import Config
from .datasets import load_unified
from .embedding import EmbeddingModel, build_embedding
from .parser import OutputParser
from .prompt import PromptBuilder
from .retriever import ExemplarStore, Retriever
from .runner import AnnotationRunner


def index_path(cfg: Config) -> str:
    return os.path.join(cfg.embedding.cache_dir, "exemplar_store")


def build_index(cfg: Config) -> ExemplarStore:
    """Embed the unified pool and cache it to disk."""
    embedder = build_embedding(cfg.embedding)
    examples = load_unified(cfg.dataset)
    store = ExemplarStore.build(examples, embedder)
    store.save(index_path(cfg))
    return store


def load_index(cfg: Config) -> ExemplarStore:
    path = index_path(cfg)
    if not os.path.exists(path + ".npy"):
        return build_index(cfg)
    return ExemplarStore.load(path)


def build_runner(cfg: Config, cache_path: str | None = None) -> AnnotationRunner:
    embedder = build_embedding(cfg.embedding)
    store = load_index(cfg)
    return assemble_runner(cfg, embedder, store, cache_path=cache_path)


def assemble_runner(
    cfg: Config,
    embedder: EmbeddingModel,
    store: ExemplarStore,
    cache_path: str | None = None,
) -> AnnotationRunner:
    """Build a runner from prebuilt components.

    Lets the sweep harness reuse one embedder + exemplar store across many
    config variations instead of reloading them each iteration.
    """
    retriever = Retriever(store, cfg.retrieval, embedder)
    prompt_builder = PromptBuilder(cfg.prompt, cfg.task)
    parser = OutputParser(cfg.parser, cfg.task)
    backend = build_backend(cfg.backend)
    return AnnotationRunner(cfg, backend, retriever, prompt_builder, parser, cache_path=cache_path)
