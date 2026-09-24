"""Default letter temperatures for questions that no fit calibrates.

Letter probabilities read at temperature 1 are overconfident on most tasks. For the model and prompt template listed
in DEFAULTS, one temperature per question type was fitted on 13,610 labelled decisions from 19 tasks and checked
leave-one-task-out (results/calibration/default_temperature.md): mean expected calibration error 0.218 -> 0.132 and
NLL 2.16 -> 0.92, with no answer changed. Easy tasks read under-confident until they are fitted. Any other model or
template keeps temperature 1, since nothing was measured for it.

A choice question's bucket is the number of options it actually showed the model, __none__ included; a tournament
counts the finalists of its last round.
"""
from __future__ import annotations

import re
from typing import Any

CHOICE_SPLIT = 10          # choice questions showing more options than this use "choice_many"

# (template, model-name pattern, temperature per question type)
DEFAULTS: tuple[tuple[str, re.Pattern, dict[str, float]], ...] = (
    ("gemma4", re.compile(r"gemma[-_ ]?4\b.*\b12b\b.*\bq8_0\b", re.I),
     {"noul": 6.01, "choice": 4.71, "choice_many": 2.66, "score": 5.18}),
)


def parse(value: Any) -> str | float:
    """'auto' (the table below, when the model and template match), 'off' (temperature 1) or a positive number used
    for every unfitted question."""
    if value is None:
        return "auto"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        t = float(value)
    else:
        text = str(value).strip().lower()
        if text in ("auto", "off"):
            return text
        try:
            t = float(text)
        except ValueError:
            raise ValueError(f"default temperature must be auto, off or a positive number, got {value!r}") from None
    if not (t > 0 and t < float("inf")):
        raise ValueError(f"default temperature must be auto, off or a positive number, got {value!r}")
    return t


def table_for(template: str, model_name: str | None) -> dict[str, float] | None:
    """The measured temperatures for this model and template, or None."""
    for tpl, pattern, temps in DEFAULTS:
        if tpl == template and model_name and pattern.search(model_name):
            return temps
    return None


def pick(temps: dict[str, float] | None, qtype: str, n_shown: int) -> float:
    """The temperature for one question from a table (1.0 without one)."""
    if not temps:
        return 1.0
    if qtype == "choice":
        return temps["choice_many" if n_shown > CHOICE_SPLIT else "choice"]
    return temps.get(qtype, 1.0)
