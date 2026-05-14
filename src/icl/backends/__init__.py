"""Swappable model backends behind a single ModelBackend interface."""

from .base import ModelBackend, build_backend

__all__ = ["ModelBackend", "build_backend"]
