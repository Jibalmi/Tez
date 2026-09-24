"""Hooks: observe or shape every decision without forking Tez. The rules are in docs/HOOKS.md.

A hook is an object with any of these methods (subclass BaseHook and override only the ones you need):

    on_decide_start(ctx)                     before any backend call; may rewrite ctx.state, or answer with
                                             ctx.skip(response) (the backend is then not called)
    on_question_end(ctx, qid, answer, meta)  after each question, in the order questions are read
    on_decide_end(ctx)                       after the response is built (ctx.response), also after a skip
    on_error(ctx)                            when the decision failed (ctx.error); it never masks that error
    on_feedback(row)                         before a feedback row is written; may edit the row in place

Hooks of one call run in this order: the process-wide defaults (set_default_hooks), the engine's
(Tez(hooks=...), add_hook), then the call's own (decide(..., hooks=...)). Built in: DecisionLog, Redact, Cache,
Metrics and OTelHook (the `otel` extra). `tez serve --hook package.module:object` loads one by name.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import logging
import math
import random
import re
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ._version import __version__
from .schema import state_key

log = logging.getLogger("tez")
EVENTS = ("on_decide_start", "on_question_end", "on_decide_end", "on_error", "on_feedback")


def new_run_id() -> str:
    """A run id: 32 hex characters (a random UUID). Sent back as X-Tez-Run-Id and x-typesafe-request-id."""
    return uuid.uuid4().hex


@dataclass(eq=False)
class DecisionContext:
    """What one decision's hooks see and share. Fields are filled as the decision proceeds: after a request that
    could not be parsed only run_id, request and error are set."""

    run_id: str
    request: Any = None                 # the wire-format request body
    engine: Any = None                  # the tez.Tez deciding it
    state: Any = None                   # on_decide_start may replace it (Redact does)
    questions: dict | None = None       # question id -> tez.schema.Question
    schema: Any = None                  # the named tez.schema.Schema, or None
    readout: str | None = None
    abstain: bool = False
    alpha: float | None = None
    layout: str | None = None           # question_first | state_first | mixed (after resolution)
    requested_layout: str | None = None  # auto | question_first | state_first, as asked
    model: str | None = None            # the request's model alias
    response: dict | None = None
    traces: dict = field(default_factory=dict)      # question id -> {readout, layout, tokens, ms, calls}
    batches: list = field(default_factory=list)     # batched backend calls (in-process): {kind, group, prompts, ...}
    usage: dict | None = None
    latency_ms: float | None = None
    error: BaseException | None = None
    skipped: bool = False
    parent_run_id: str | None = None    # batch items: the batch's run id (the item's run id is <parent>.<index>)
    index: int | None = None            # batch items: position in the batch
    data: dict = field(default_factory=dict)        # scratch space shared by the hooks of this decision
    hook_errors: list = field(default_factory=list)  # (hook name, event, exception) swallowed with hooks_raise=False
    started: float = field(default_factory=time.perf_counter)

    def skip(self, response: dict) -> None:
        """Answer from on_decide_start without calling the backend: `response` becomes the decision's response
        (its tez.latency_ms is set to the time actually spent). on_decide_end hooks still run, with skipped=True."""
        if not isinstance(response, dict):
            raise TypeError("skip() takes the response dict the decision should return")
        self.response = response
        self.skipped = True


class BaseHook:
    """No-op hook: subclass it and override the events you need."""

    def on_decide_start(self, ctx: DecisionContext) -> None:
        pass

    def on_question_end(self, ctx: DecisionContext, qid: str, answer: dict, meta: dict) -> None:
        pass

    def on_decide_end(self, ctx: DecisionContext) -> None:
        pass

    def on_error(self, ctx: DecisionContext) -> None:
        pass

    def on_feedback(self, row: dict) -> None:
        pass


def _hook_name(hook: Any) -> str:
    return type(hook).__name__


def normalise_hooks(hooks: Any) -> list:
    """None, one hook or a sequence of hooks, validated: instances with at least one event method."""
    if hooks is None:
        return []
    items = list(hooks) if isinstance(hooks, (list, tuple)) else [hooks]
    for hook in items:
        if isinstance(hook, type):
            raise TypeError(f"hooks must be instances, not classes: instantiate {hook.__name__} first")
        methods = [e for e in EVENTS if getattr(hook, e, None) is not None]
        if not methods:
            raise TypeError(f"{_hook_name(hook)} is not a hook: it has none of {', '.join(EVENTS)}")
        for e in methods:
            if not callable(getattr(hook, e)):
                raise TypeError(f"{_hook_name(hook)}.{e} must be callable")
    return items


_DEFAULTS: list = []
_DEFAULTS_LOCK = threading.Lock()


def set_default_hooks(hooks: Any = None) -> None:
    """Replace the process-wide hooks that run first in every Tez engine (None clears them)."""
    items = normalise_hooks(hooks)
    with _DEFAULTS_LOCK:
        _DEFAULTS[:] = items


def default_hooks() -> list:
    """The process-wide hooks, a copy."""
    with _DEFAULTS_LOCK:
        return list(_DEFAULTS)


class HookSet:
    """The hooks of one call with the engine's failure rule. raise_errors=True: an exception from a hook fails the
    decision (or the feedback). raise_errors=False: it is logged, recorded in ctx.hook_errors and skipped. An
    exception in on_error is always logged and skipped, so the original error is what the caller sees."""

    def __init__(self, hooks: Sequence, raise_errors: bool = True):
        self.hooks = list(hooks)
        self.raise_errors = raise_errors

    def __bool__(self) -> bool:
        return bool(self.hooks)

    def emit(self, event: str, *args: Any) -> None:
        ctx = args[0] if args and isinstance(args[0], DecisionContext) else None
        for hook in self.hooks:
            method = getattr(hook, event, None)
            if method is None:
                continue
            try:
                method(*args)
            except Exception as exc:
                if self.raise_errors and event != "on_error":
                    raise
                log.warning("hook %s.%s failed: %s: %s", _hook_name(hook), event, exc.__class__.__name__, exc)
                if ctx is not None:
                    ctx.hook_errors.append((_hook_name(hook), event, exc))


def load_hook(spec: str) -> Any:
    """A hook from `package.module:object` (tez serve --hook): a class is instantiated without arguments, a hook
    instance is used as it is, any other callable is called and must return a hook."""
    if not isinstance(spec, str):
        raise ValueError(f"a hook is named package.module:object, got {spec!r}")
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name.strip() or not attr.strip():
        raise ValueError(f"a hook is named package.module:object, got {spec!r}")
    try:
        obj: Any = importlib.import_module(module_name.strip())
    except ImportError as exc:
        raise ValueError(f"cannot import {module_name!r} for hook {spec!r}: {exc}") from exc
    for part in attr.strip().split("."):
        if not hasattr(obj, part):
            raise ValueError(f"{module_name!r} has no {attr!r} (hook {spec!r})")
        obj = getattr(obj, part)
    if isinstance(obj, type):
        obj = obj()
    elif not any(getattr(obj, e, None) is not None for e in EVENTS) and callable(obj):
        obj = obj()
    return normalise_hooks(obj)[0]


# ---------------------------------------------------------------------------------------------- answers
def predicted_label(answer: dict) -> Any:
    """An answer's label in the stored form tez fit reads: "true"/"false", the choice, or the most likely level."""
    kind = answer.get("type")
    if kind == "noul":
        return "true" if float(answer.get("noul", 0.0)) >= 0.5 else "false"
    if kind == "choice":
        return answer.get("choice")
    probs = answer.get("probabilities") or {}
    if probs:
        return int(max(probs, key=lambda k: probs[k]))
    return int(round(float(answer.get("score", 0.0))))


def answer_confidence(answer: dict) -> float:
    if answer.get("type") == "noul":
        p = float(answer.get("noul", 0.0))
        return max(p, 1.0 - p)
    probs = answer.get("probabilities") or {}
    return float(max(probs.values())) if probs else float(answer.get("confidence") or 0.0)


# ---------------------------------------------------------------------------------------------- DecisionLog
class DecisionLog(BaseHook):
    """Append one JSON line per decision to `path`, in the row format tez fit and tez eval read:

        {"state": ..., "labels": {}, "predicted": {question: label}, "decisions": {question: {...}},
         "schema": ..., "run_id": ..., "ts": ..., "model": ..., "cached": false}

    `labels` is left empty on purpose: fill it in after review (the model's answers are in `predicted`) and pass the
    file to `tez fit --labels`. A fit on the model's own answers would measure the model against itself, and the
    gate's error guarantee would rest on nothing. sample: the share of decisions written (random, seedable)."""

    def __init__(self, path: str | Path, sample: float = 1.0, seed: int | None = None):
        if isinstance(sample, bool) or not isinstance(sample, (int, float)) or not 0.0 <= float(sample) <= 1.0:
            raise ValueError(f"sample must be a share between 0 and 1, got {sample!r}")
        self.path = Path(path)
        self.sample = float(sample)
        self.written = 0
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"DecisionLog({str(self.path)!r}, sample={self.sample:g})"

    def row(self, ctx: DecisionContext) -> dict:
        res = ctx.response or {}
        answers = res.get("answers") or {}
        metas = (res.get("tez") or {}).get("questions") or {} if isinstance(res.get("tez"), dict) else {}
        decisions = {}
        for qid, a in answers.items():
            m = metas.get(qid) or {}
            d = {"confidence": round(answer_confidence(a), 4), "readout": m.get("readout")}
            for k in ("decision", "p_correct", "calibration_id", "layout"):
                if m.get(k) is not None:
                    d[k] = m[k]
            decisions[qid] = d
        row = {"state": ctx.state, "labels": {}, "predicted": {qid: predicted_label(a) for qid, a in answers.items()},
               "decisions": decisions, "schema": ctx.schema.name if ctx.schema is not None else None,
               "run_id": ctx.run_id, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "model": res.get("model"), "cached": bool(ctx.skipped)}
        if isinstance(res.get("tez"), dict) and res["tez"].get("values") is not None:
            row["values"] = res["tez"]["values"]
        return row

    def on_decide_end(self, ctx: DecisionContext) -> None:
        if ctx.response is None:
            return
        with self._lock:
            if self.sample < 1.0 and self._rng.random() >= self.sample:
                return
            line = json.dumps(self.row(ctx), ensure_ascii=False, default=str)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self.written += 1


# ---------------------------------------------------------------------------------------------- Redact
def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


class _Found:
    """What a redaction rule's replacement sees: group(0) is the matched text."""

    def __init__(self, text: str):
        self.text = text

    def group(self, index: int = 0) -> str:
        if index != 0:
            raise IndexError("no such group")
        return self.text


class _Email:
    """The built-in email pattern, found from its @ so it stays linear on any text: a regular expression finds the @
    and the domain, then the local part is read backwards from the @, at most 64 characters (an address's local part is
    at most 64). A regular expression that had to start at the local part could either start anywhere inside a long run
    of address characters (64 steps at every position) or only at the start of a run, and then an address glued to a
    longer run ("------...john@example.com") was not redacted. Here the last 64 characters before the @ go with it."""

    pattern = "email: [A-Za-z0-9._%+-]{1,64}@domain, found from the @"
    _domain = re.compile(r"@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63}){0,8}\.[A-Za-z]{2,24}(?![A-Za-z])")
    _local = re.compile(r"[A-Za-z0-9._%+-]{1,64}")        # matched on the characters before the @, reversed

    def sub(self, repl: Any, text: str) -> str:
        out, last = [], 0
        for m in self._domain.finditer(text):
            at = m.start()
            back = self._local.match(text[max(last, at - 64):at][::-1])
            if back is None:
                continue
            start = at - back.end()
            out += [text[last:start], repl(_Found(text[start:m.end()])) if callable(repl) else repl]
            last = m.end()
        out.append(text[last:])
        return "".join(out)


# Every pattern runs over untrusted text (up to tez serve's 50,000-character states), so each is linear: a match can only
# start at the beginning of a run (the lookbehinds; the email is found from its @) and every repeat is bounded (an
# address's local part is at most 64 characters, a domain label 63). Without that, a long run of letters costs
# quadratic backtracking.
_BUILTIN = {
    "email": (_Email(), None),
    "iban": (re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b"), None),
    "card": (re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])"), lambda s: _luhn(re.sub(r"\D", "", s))),
    "phone": (re.compile(r"(?<![\w+])\+?\(?\d[\d ().-]{7,}\d(?!\w)"), lambda s: 9 <= len(re.sub(r"\D", "", s)) <= 15),
}


class Redact(BaseHook):
    """Replace personal data in the state before the model reads it, and in feedback rows before they are written,
    so the decision log, traces and fits never see it either.

    patterns     built-in names ("email", "iban", "card", "phone"; default: all four) and/or regular expressions
                 (strings or compiled). A card number must pass the Luhn check, a phone number have 9 to 15 digits.
    replacement  None (default): "[EMAIL]", "[IBAN]", "[CARD]", "[PHONE]" for the built-ins and "[REDACTED]" for your
                 own patterns; or one fixed string for all of them.
    Strings inside object and array states are redacted; keys are kept. Put Redact first in the hook list."""

    def __init__(self, patterns: Iterable[Any] | str | None = None, replacement: str | None = None):
        if patterns is None:
            patterns = list(_BUILTIN)
        elif isinstance(patterns, (str, re.Pattern)):
            patterns = [patterns]
        self.rules: list[tuple[Any, Any, str]] = []        # (pattern with .sub, check, replacement)
        for p in patterns:
            if isinstance(p, str) and p in _BUILTIN:
                rx, check = _BUILTIN[p]
                self.rules.append((rx, check, f"[{p.upper()}]" if replacement is None else replacement))
            elif isinstance(p, re.Pattern):
                self.rules.append((p, None, "[REDACTED]" if replacement is None else replacement))
            elif isinstance(p, str):
                try:
                    rx = re.compile(p)
                except re.error as exc:
                    raise ValueError(f"invalid redaction pattern {p!r}: {exc}") from exc
                self.rules.append((rx, None, "[REDACTED]" if replacement is None else replacement))
            else:
                raise ValueError(f"a redaction pattern is a built-in name, a regex string or a compiled regex, got {p!r}")
        if not self.rules:
            raise ValueError("Redact needs at least one pattern")
        self.count = 0
        self._lock = threading.Lock()

    def text(self, s: str) -> str:
        for rx, check, repl in self.rules:
            def sub(m: re.Match, check=check, repl=repl) -> str:
                if check is not None and not check(m.group(0)):
                    return m.group(0)
                with self._lock:
                    self.count += 1
                return repl
            s = rx.sub(sub, s)
        return s

    def redact(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {k: self.redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        return value

    def on_decide_start(self, ctx: DecisionContext) -> None:
        ctx.state = self.redact(ctx.state)

    def on_feedback(self, row: dict) -> None:
        row["state"] = self.redact(row.get("state"))


# ---------------------------------------------------------------------------------------------- Cache
def _model_name(backend: Any) -> str | None:
    read = getattr(backend, "model_name", None)
    return read() if callable(read) else None


class Cache(BaseHook):
    """Answer a decision seen before from memory (least recently used, at most `maxsize` entries). The key is the
    state, every question's definition, the schema and its calibration id, readout, abstain, the gate's alpha, the
    requested layout, the model alias, and the engine's settings: the backend's URL, template, model name and n_probs,
    the embedding backend's URL, template and model name, and the default temperature. So a new fit, another engine
    setting or a model name that differs never serves an old answer, even with one Cache shared by several engines.

    The model name is the one the engine read from the backend, and an engine reads it once. A model swapped behind the
    same URL while the engine runs keeps the old name (the engine itself does not notice either, see its fits' model
    check): call clear() after such a swap, or restart."""

    def __init__(self, maxsize: int = 1024):
        if isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize < 1:
            raise ValueError(f"maxsize must be a positive integer, got {maxsize!r}")
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._slot = f"cache:{id(self)}"

    def __len__(self) -> int:
        return len(self._store)

    def key(self, ctx: DecisionContext) -> str:
        engine = ctx.engine
        schema = ctx.schema
        cal = None
        if engine is not None and schema is not None and schema.name in getattr(engine, "fitted", {}):
            cal = engine.fitted[schema.name].calibration_id
        backend = getattr(engine, "backend", None)
        embedder = getattr(engine, "embedder", None)
        parts = [state_key(ctx.state), [[qid, q.signature()] for qid, q in (ctx.questions or {}).items()],
                 schema.name if schema is not None else None, cal, ctx.readout, ctx.abstain, ctx.alpha,
                 ctx.requested_layout, ctx.model, getattr(backend, "url", None), getattr(backend, "template", None),
                 getattr(ctx.request, "get", lambda k: None)("json_schema"),
                 _model_name(backend), getattr(backend, "n_probs", None),
                 getattr(embedder, "url", None), getattr(embedder, "template", None),
                 _model_name(embedder) if embedder is not backend else None,
                 getattr(engine, "default_temperature", None)]
        return hashlib.sha256(json.dumps(parts, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()

    def on_decide_start(self, ctx: DecisionContext) -> None:
        if ctx.skipped:
            return
        k = self.key(ctx)
        ctx.data[self._slot] = k
        with self._lock:
            hit = self._store.get(k)
            if hit is None:
                self.misses += 1
                return
            self._store.move_to_end(k)
            self.hits += 1
            response = copy.deepcopy(hit)
        response["usage"] = {"input_tokens": 0, "output_tokens": 0}
        if isinstance(response.get("tez"), dict):
            response["tez"]["cached"] = True
        ctx.skip(response)

    def on_decide_end(self, ctx: DecisionContext) -> None:
        k = ctx.data.get(self._slot)
        if ctx.skipped or ctx.response is None or k is None:
            return
        with self._lock:
            self._store[k] = copy.deepcopy(ctx.response)
            self._store.move_to_end(k)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict:
        with self._lock:
            return {"size": len(self._store), "maxsize": self.maxsize, "hits": self.hits, "misses": self.misses}


# ---------------------------------------------------------------------------------------------- Metrics
def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    i = min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))
    return round(s[i], 3)


class Metrics(BaseHook):
    """Counters for monitoring: decisions, errors (by type), skipped (cached) decisions, questions by readout, layout
    and gate decision, input tokens, feedback rows, and latency percentiles over the last `window` decisions.
    snapshot() returns them as a dict, prometheus() in the Prometheus text format."""

    def __init__(self, window: int = 10_000):
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window!r}")
        self.window = window
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.decisions = self.errors = self.skipped = self.questions = self.input_tokens = self.feedback = 0
            self.errors_by_type: dict[str, int] = defaultdict(int)
            self.readouts: dict[str, int] = defaultdict(int)
            self.layouts: dict[str, int] = defaultdict(int)
            self.gate: dict[str, int] = defaultdict(int)
            self._latency: deque = deque(maxlen=self.window)

    def on_question_end(self, ctx: DecisionContext, qid: str, answer: dict, meta: dict) -> None:
        with self._lock:
            self.questions += 1
            self.readouts[str(meta.get("readout"))] += 1
            layout = (ctx.traces.get(qid) or {}).get("layout")
            if layout:
                self.layouts[layout] += 1
            if meta.get("decision"):
                self.gate[meta["decision"]] += 1

    def on_decide_end(self, ctx: DecisionContext) -> None:
        with self._lock:
            self.decisions += 1
            self.skipped += bool(ctx.skipped)
            self.input_tokens += int((ctx.usage or {}).get("input_tokens") or 0)
            if ctx.latency_ms is not None:
                self._latency.append(float(ctx.latency_ms))

    def on_error(self, ctx: DecisionContext) -> None:
        with self._lock:
            self.errors += 1
            self.errors_by_type[getattr(ctx.error, "type", None) or type(ctx.error).__name__] += 1

    def on_feedback(self, row: dict) -> None:
        with self._lock:
            self.feedback += 1

    def snapshot(self) -> dict:
        with self._lock:
            lat = list(self._latency)
            return {"decisions": self.decisions, "errors": self.errors, "errors_by_type": dict(self.errors_by_type),
                    "skipped": self.skipped, "questions": self.questions, "readouts": dict(self.readouts),
                    "layouts": dict(self.layouts), "gate": dict(self.gate), "input_tokens": self.input_tokens,
                    "feedback": self.feedback,
                    "latency_ms": {"count": len(lat), "mean": round(sum(lat) / len(lat), 3) if lat else None,
                                   "p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95), "max": round(max(lat), 3) if lat else None}}

    def prometheus(self, prefix: str = "tez") -> str:
        s = self.snapshot()
        out = [f"# TYPE {prefix}_decisions_total counter", f"{prefix}_decisions_total {s['decisions']}",
               f"# TYPE {prefix}_errors_total counter"]
        out += [f'{prefix}_errors_total{{type="{t}"}} {n}' for t, n in sorted(s["errors_by_type"].items())] or \
               [f"{prefix}_errors_total 0"]
        out += [f"# TYPE {prefix}_skipped_total counter", f"{prefix}_skipped_total {s['skipped']}",
                f"# TYPE {prefix}_questions_total counter"]
        out += [f'{prefix}_questions_total{{readout="{r}"}} {n}' for r, n in sorted(s["readouts"].items())] or \
               [f"{prefix}_questions_total 0"]
        out += [f"# TYPE {prefix}_gate_total counter"]
        out += [f'{prefix}_gate_total{{decision="{d}"}} {n}' for d, n in sorted(s["gate"].items())] or [f"{prefix}_gate_total 0"]
        out += [f"# TYPE {prefix}_input_tokens_total counter", f"{prefix}_input_tokens_total {s['input_tokens']}",
                f"# TYPE {prefix}_feedback_total counter", f"{prefix}_feedback_total {s['feedback']}",
                f"# TYPE {prefix}_latency_ms summary"]
        for q in ("p50", "p95"):
            if s["latency_ms"][q] is not None:
                out.append(f'{prefix}_latency_ms{{quantile="{0.5 if q == "p50" else 0.95}"}} {s["latency_ms"][q]}')
        out.append(f"{prefix}_latency_ms_count {s['latency_ms']['count']}")
        return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------- OTelHook
def _otel_trace():
    try:
        from opentelemetry import trace
    except ImportError as exc:
        raise ImportError('OTelHook needs the OpenTelemetry API: pip install "tez-decisions[otel]"') from exc
    return trace


def _attrs(d: dict) -> dict:
    return {k: v for k, v in d.items() if isinstance(v, (str, bool, int, float))}


class OTelHook(BaseHook):
    """One OpenTelemetry span per decision ("tez.decide", attributes tez.run_id, tez.questions, tez.schema,
    tez.readout, tez.layout, tez.latency_ms, tez.input_tokens, tez.model) with an event per question. The state is
    never recorded. tracer: an OpenTelemetry tracer (default: the global provider's tracer "tez"); needs the `otel`
    extra unless you pass your own."""

    def __init__(self, tracer: Any = None, span_name: str = "tez.decide"):
        self.tracer = tracer if tracer is not None else _otel_trace().get_tracer("tez", __version__)
        self.span_name = span_name
        self._slot = f"otel:{id(self)}"

    def _start(self, ctx: DecisionContext) -> Any:
        attrs = _attrs({"tez.run_id": ctx.run_id, "tez.questions": len(ctx.questions or {}),
                        "tez.schema": ctx.schema.name if ctx.schema is not None else None, "tez.readout": ctx.readout,
                        "tez.layout": ctx.layout, "tez.parent_run_id": ctx.parent_run_id, "tez.index": ctx.index})
        span = self.tracer.start_span(self.span_name, attributes=attrs)
        ctx.data[self._slot] = span
        return span

    def on_decide_start(self, ctx: DecisionContext) -> None:
        self._start(ctx)

    def on_question_end(self, ctx: DecisionContext, qid: str, answer: dict, meta: dict) -> None:
        span = ctx.data.get(self._slot)
        if span is None:
            return
        trace = ctx.traces.get(qid) or {}
        span.add_event("tez.question", _attrs({"tez.question": qid, "tez.readout": meta.get("readout"),
                                               "tez.layout": trace.get("layout"), "tez.decision": meta.get("decision"),
                                               "tez.confidence": round(answer_confidence(answer), 4),
                                               "tez.tokens": trace.get("tokens")}))

    def on_decide_end(self, ctx: DecisionContext) -> None:
        span = ctx.data.pop(self._slot, None)
        if span is None:
            return
        for k, v in _attrs({"tez.latency_ms": ctx.latency_ms, "tez.input_tokens": (ctx.usage or {}).get("input_tokens"),
                            "tez.model": (ctx.response or {}).get("model"), "tez.skipped": ctx.skipped,
                            "tez.layout": ctx.layout}).items():
            span.set_attribute(k, v)
        span.end()

    def on_error(self, ctx: DecisionContext) -> None:
        span = ctx.data.pop(self._slot, None) or self._start(ctx)
        ctx.data.pop(self._slot, None)
        if ctx.error is not None:
            span.record_exception(ctx.error)
            try:
                from opentelemetry.trace import Status, StatusCode
                span.set_status(Status(StatusCode.ERROR, str(ctx.error)))
            except ImportError:
                pass
        span.end()
