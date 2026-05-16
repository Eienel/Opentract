"""Per-task prompt builders.

The official baseline uses a single generic prompt for all 8 tasks, which leaves
score on the table -- a kernel-generation task and a binary-sentiment task want
different output shapes. Each task has a tuned variant here.

All variants use the same <label>...</label> wrapping that the evaluator's
parser expects, since the baseline's `count_answer` extracts the most common
<label>-enclosed string. Task 8 (kernel generation) is special: the metric runs
the generated CODE, so the prompt insists on a complete executable Triton block
inside <label>.
"""

from __future__ import annotations

from .data import Task, render_io


_SHARED_RULES = (
    "### Critical Output Rule\n"
    "After any reasoning, your FINAL answer MUST be wrapped in <label>...</label> tags. "
    "Only the content inside the LAST <label>...</label> pair will be scored. "
    "Do not put extra spaces or commentary inside the tags. "
    "Do not omit either tag.\n\n"
)


_DEFAULT_TEMPLATE = (
    "### Role\n"
    "You are a precise data annotation engine. Follow the task definition exactly.\n\n"
    "### Task\n"
    "{definition}\n\n"
    "{shared_rules}"
    "### Examples\n"
    "{examples}\n"
    "### Now annotate this input\n"
    "{input}\n\n"
    "### Answer (wrap final answer in <label>...</label>)\n"
)


_INTEGER_HINT = (
    "Output a single integer inside <label>...</label>, with no units or words. "
    "Example: <label>42</label>\n\n"
)

_BINARY_LABEL_HINT = (
    "Output exactly one of the labels shown in the examples, copied verbatim, "
    "inside <label>...</label>. Example: <label>Sad</label>\n\n"
)

_KERNEL_HINT = (
    "Output a COMPLETE, EXECUTABLE Triton kernel implementation inside one "
    "<label>...</label> block. Include all imports (torch, triton, triton.language as tl), "
    "the @triton.jit kernel function, and a Python wrapper. The code inside the tags must "
    "run as-is when copied to a file and called by the evaluator. Do not include "
    "explanatory prose inside the tags.\n\n"
)


TASK_HINTS: dict[int, str] = {
    1: _INTEGER_HINT,   # closest_integers
    2: _INTEGER_HINT,   # count_nouns_verbs
    3: _INTEGER_HINT,   # collatz_conjecture
    4: "",              # conala_concat_strings -- output is short code string
    5: _BINARY_LABEL_HINT,  # semeval sadness
    6: _BINARY_LABEL_HINT,  # mnli classification
    7: "",              # jeopardy answer generation
    8: _KERNEL_HINT,    # kernel generation
}


def build_prompt(task: Task, query_input: object, examples_str: str) -> str:
    hint = TASK_HINTS.get(task.task_id, "")
    return _DEFAULT_TEMPLATE.format(
        definition=task.definition,
        shared_rules=_SHARED_RULES + hint,
        examples=examples_str,
        input=render_io(query_input),
    )
