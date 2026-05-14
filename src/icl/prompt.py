"""Prompt assembly from a config-supplied Jinja2 template."""

from __future__ import annotations

from jinja2 import Environment

from .config import PromptConfig, TaskConfig
from .datasets import Example


class PromptBuilder:
    def __init__(self, prompt_cfg: PromptConfig, task_cfg: TaskConfig):
        self.prompt_cfg = prompt_cfg
        self.task_cfg = task_cfg
        self._env = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
        self._template = self._env.from_string(prompt_cfg.template)

    @property
    def system(self) -> str:
        return self.prompt_cfg.system

    def build(self, query: Example, exemplars: list[Example], stricter: bool = False) -> str:
        """Render the prompt. `stricter` appends a format reminder on retry."""
        text = self._template.render(
            task_description=self.task_cfg.description,
            label_space=self.task_cfg.label_space,
            exemplars=exemplars,
            query=query,
        )
        if stricter:
            reminder = "\n\nRespond with ONLY the annotation value -- no explanation, no extra text."
            if self.task_cfg.label_space:
                reminder = (
                    f"\n\nRespond with EXACTLY one of: "
                    f"{', '.join(map(str, self.task_cfg.label_space))}. Nothing else."
                )
            text += reminder
        return text
