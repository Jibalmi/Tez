"""Prompt construction.

Two layouts, both answered with one option letter read from the next-token distribution:

  question_first  instructions, the question and its options, then `Input:` and the state last. The constant
                  part of a question's prompt is a cached prefix, so a stream of states against one question only
                  evaluates each state (the voice loop, one question per request).
  state_first     instructions, `Input:` and the state, then the question and its options. Every question about
                  one state shares the state as a cached prefix, so a request with several questions reads the
                  state once and then only each question's tail. Measured on typed-decisions with Gemma 4 12B:
                  the same accuracy as question_first (0.7015 vs 0.705, McNemar p = 0.74) for about half the
                  evaluated tokens.

`auto` (the default) picks per request: state_first for two or more questions, question_first for one question and
for streaming. More than 26 options run as a chunked tournament: chunks of 20 options plus a "none of the options
above fits" option, then the chunk winners meet in a final round.
"""
from __future__ import annotations

import hashlib
import json
import re
import string
from typing import Any, Sequence

import numpy as np

from .schema import LAYOUTS, NONE_KEY, Question

LETTERS = string.ascii_uppercase
MAX_LETTERS = len(LETTERS)
CHUNK = 20
QUESTION_FIRST, STATE_FIRST = "question_first", "state_first"
HEAD = ("You are a decision engine. Read the question and the options, then look at the input and answer "
        "with the single letter of the best option. Answer with the letter only.")
HEAD_SHOTS = HEAD + " Worked examples of this exact question follow."
HEAD_STATE_FIRST = ("You are a decision engine. Read the input, then the question and its options, and answer "
                    "with the single letter of the best option. Answer with the letter only.")
HEAD_STATE_FIRST_SHOTS = HEAD_STATE_FIRST + " Worked examples of this exact question follow."
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


def check_layout(name: str) -> str:
    """A concrete prompt layout (question_first or state_first); `auto` must be resolved first (resolve_layout)."""
    if name not in (QUESTION_FIRST, STATE_FIRST):
        raise ValueError(f"unknown prompt layout {name!r}; choose question_first or state_first")
    return name


def resolve_layout(layout: str | None, n_questions: int, streaming: bool = False) -> str:
    """The concrete layout for a request: an explicit layout wins; `auto` (or None) is state_first when a request
    has two or more questions about its state and question_first for one question and for streaming (a state that
    grows word by word shares nothing with its previous version once it comes first)."""
    if layout in (QUESTION_FIRST, STATE_FIRST):
        return layout
    if layout not in (None, "auto"):
        raise ValueError(f"unknown prompt layout {layout!r}; choose one of {', '.join(LAYOUTS)}")
    return STATE_FIRST if n_questions >= 2 and not streaming else QUESTION_FIRST


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


def question_block(question: Question, options: Options) -> str:
    return f"Question ({question.type}): {neutralize(question.instructions_text)}\n\nOptions:\n{render_option_lines(options)}"


def build_user(question: Question, state: Any, options: Options, shots: Sequence[tuple[Any, int]] | None = None,
               layout: str = QUESTION_FIRST) -> str:
    """The user message. question_first: [examples,] question, options, input. state_first: [examples,] input,
    question, options. Worked examples come before the input in both layouts, so the question that closes a
    state-first prompt is always about the input right above it (a question with examples then shares only the
    instructions with the other questions about its state)."""
    if check_layout(layout) == STATE_FIRST:
        parts = [HEAD_STATE_FIRST_SHOTS if shots else HEAD_STATE_FIRST]
        if shots:
            parts.append(render_shots(question, options, shots))
        parts += [f"Input:\n{render_state(state)}", question_block(question, options)]
        return "\n\n".join(parts)
    body = f"{question_block(question, options)}\n\nInput:\n{render_state(state)}"
    if shots:
        return f"{HEAD_SHOTS}\n\n{render_shots(question, options, shots)}\n\n{body}"
    return f"{HEAD}\n\n{body}"


def build_prompt(question: Question, state: Any, template: str, options: Options | None = None,
                 shots: Sequence[tuple[Any, int]] | None = None, layout: str = QUESTION_FIRST) -> str:
    pre, post = TEMPLATES[check_template(template)]
    opts = question.options() if options is None else options
    return f"{pre}{build_user(question, state, opts, shots, layout)}{post}"


def state_prefix(state: Any, template: str) -> str:
    """The part every state-first prompt of one state shares (without worked examples): the backend's prompt cache
    keeps it between consecutive questions about that state."""
    pre, _ = TEMPLATES[check_template(template)]
    return f"{pre}{HEAD_STATE_FIRST}\n\nInput:\n{render_state(state)}\n\n"


_SENTINEL = "\u0000TEZ-STATE\u0000"


def fingerprint(question: Question, template: str, options: Options | None = None,
                shots: Sequence[tuple[Any, int]] | None = None, layout: str = QUESTION_FIRST) -> str:
    """Hash of the prompt without its state. A fitted probe or calibration is only used while the prefix and
    suffix it was fitted on are unchanged (edited instructions, options, examples, template or layout invalidate
    it). The layout is part of the hash: a fit made under one layout is stale under the other. question_first
    hashes exactly as before layouts existed, so earlier fits stay valid."""
    text = build_prompt(question, _SENTINEL, template, options, shots, layout)
    if layout != QUESTION_FIRST:
        text = f"layout:{layout}\n{text}"
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
