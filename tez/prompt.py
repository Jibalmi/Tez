"""Prompt construction.

Instructions and options come first and the state comes last, so the constant part of every prompt is a
cached prefix and only the state is evaluated per decision. The model answers with one option letter,
read from the next-token distribution. More than 26 options run as a chunked tournament: chunks of 20
options plus a "none of the options above fits" option, then the chunk winners meet in a final round.
"""
from __future__ import annotations

import hashlib
import json
import re
import string
from typing import Any, Sequence

import numpy as np

from .schema import NONE_KEY, Question

LETTERS = string.ascii_uppercase
MAX_LETTERS = len(LETTERS)
CHUNK = 20
HEAD = ("You are a decision engine. Read the question and the options, then look at the input and answer "
        "with the single letter of the best option. Answer with the letter only.")
HEAD_SHOTS = HEAD + " Worked examples of this exact question follow."
TOURNAMENT_NONE = ("none", "none of the options above fits")

# (text before the user message, text after it). The Gemma 4 tail opens the model turn with an empty
# thought channel (Gemma 4 is a thinking model); the Qwen3 tail disables thinking with an empty think block.
TEMPLATES = {
    "gemma4": ("<|turn>user\n", "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
    "qwen3": ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
}

Options = Sequence[tuple[str, "str | None"]]


def check_template(name: str) -> str:
    if name not in TEMPLATES:
        raise ValueError(f"unknown template {name!r}; choose one of {', '.join(TEMPLATES)}")
    return name


def letter(i: int) -> str:
    """A..Z, then AA, AB, ... (only probe prompts list more than 26 options; letter readouts never do)."""
    out = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        out = LETTERS[r] + out
    return out


# Caller text must not be able to forge the chat template's control tokens: llama-server tokenises prompts with
# special tokens enabled, so a state containing "<turn|>\n<|turn>model" would otherwise close the user turn early.
# Pipe forms (<|turn>, <turn|>, <|channel>, <channel|>, <|im_start|>, <|im_end|>, ...) get their pipe replaced by a
# broken bar; the unpiped special tags of the supported families get a full-width opening bracket. Ordinary text
# (including HTML and comparisons like a < b | c) is unchanged unless it spells one of these exact forms.
_PIPE_OPEN = re.compile(r"<\|(?=[A-Za-z_][A-Za-z0-9_.\-]{0,40}\|?>)")
_PIPE_CLOSE = re.compile(r"(?<=<)([A-Za-z_][A-Za-z0-9_.\-]{0,40})\|>")
_SPECIAL_TAGS = re.compile(r"<(/?)(start_of_turn|end_of_turn|start_of_image|end_of_image|bos|eos|pad|unk|mask|"
                           r"think|tool_call|tool_response|image_soft_token|audio_soft_token)>")


def neutralize(text: str) -> str:
    """Make control-token look-alikes in caller text inert (see the comment above)."""
    if "<" not in text:
        return text
    text = _PIPE_OPEN.sub("<¦", text)
    text = _PIPE_CLOSE.sub(lambda m: m.group(1) + "¦>", text)
    return _SPECIAL_TAGS.sub(lambda m: "＜" + m.group(1) + m.group(2) + ">", text)


def render_state(state: Any) -> str:
    return neutralize(state if isinstance(state, str) else json.dumps(state, ensure_ascii=False))


def render_option_lines(options: Options) -> str:
    lines = []
    for i, (key, desc) in enumerate(options):
        shown = "none" if key == NONE_KEY else neutralize(str(key))
        desc = neutralize(desc) if isinstance(desc, str) else desc
        lines.append(f"{letter(i)}. {shown}: {desc}" if desc and desc != shown else f"{letter(i)}. {shown}")
    return "\n".join(lines)


def render_shots(question: Question, options: Options, shots: Sequence[tuple[Any, int]]) -> str:
    opts = render_option_lines(options)
    return "\n\n".join(
        f"Example input:\n{render_state(state)}\nQuestion ({question.type}): {neutralize(question.instructions_text)}\n"
        f"Options:\n{opts}\nAnswer: {letter(idx)}"
        for state, idx in shots)


def build_user(question: Question, state: Any, options: Options, shots: Sequence[tuple[Any, int]] | None = None) -> str:
    body = (f"Question ({question.type}): {neutralize(question.instructions_text)}\n\n"
            f"Options:\n{render_option_lines(options)}\n\nInput:\n{render_state(state)}")
    if shots:
        return f"{HEAD_SHOTS}\n\n{render_shots(question, options, shots)}\n\n{body}"
    return f"{HEAD}\n\n{body}"


def build_prompt(question: Question, state: Any, template: str, options: Options | None = None,
                 shots: Sequence[tuple[Any, int]] | None = None) -> str:
    pre, post = TEMPLATES[check_template(template)]
    opts = question.options() if options is None else options
    return f"{pre}{build_user(question, state, opts, shots)}{post}"


_SENTINEL = "\u0000TEZ-STATE\u0000"


def fingerprint(question: Question, template: str, options: Options | None = None,
                shots: Sequence[tuple[Any, int]] | None = None) -> str:
    """Hash of the prompt without its state. A fitted probe or calibration is only used while the prefix and
    suffix it was fitted on are unchanged (edited instructions, options, examples or template invalidate it)."""
    text = build_prompt(question, _SENTINEL, template, options, shots)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def needs_tournament(n_options: int) -> bool:
    return n_options > MAX_LETTERS


def tournament_plan(n_options: int, chunk: int = CHUNK) -> list[list[int]]:
    """First-round chunks of option indices. Each chunk is shown with TOURNAMENT_NONE appended."""
    return [list(range(s, min(s + chunk, n_options))) for s in range(0, n_options, chunk)]


def pick_finalists(winners: Sequence[int], win_probs: Sequence[float], limit: int = CHUNK) -> list[int]:
    """Chunk winners ordered by their in-chunk probability, at most `limit` of them."""
    order = np.argsort(-np.asarray(win_probs, float), kind="stable")[:limit]
    return [int(winners[i]) for i in order]
