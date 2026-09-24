"""Framework-free logic behind the LangChain / LangGraph components (tez.integrations.langchain): what a component
decides on, which question it asks, and what an answer means for a router, a guard, a triage step or an evaluator.

Nothing here imports LangChain. Messages and documents are recognised by shape (a `content` + `type` pair, a
`page_content` attribute, an OpenAI-style {"role", "content"} dict or a ("user", "text") tuple), so the same functions
serve LangChain objects, plain dicts and strings, and can be tested without the optional dependency.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..engine import READOUTS
from ..errors import InvalidRequest
from ..schema import NONE_KEY, parse_alpha, parse_question
from .remote import DEFAULT_URL, RemoteTez

GUARD_ACTIONS = ("raise", "filter", "annotate")
_ROLES = {"human", "user", "ai", "assistant", "system", "tool", "function", "developer", "chat"}


# ---------------------------------------------------------------------------------------------- engine
def make_decider(tez: Any = None, base_url: str | None = None, api_key: str | None = None, timeout: float = 30.0) -> Any:
    """The engine a component talks to: `tez` in-process (a tez.Tez, or anything with the same decide and
    schema_detail methods, such as a RemoteTez), otherwise a RemoteTez for `base_url` (default http://127.0.0.1:8787)."""
    if tez is not None:
        if base_url is not None or api_key is not None:
            raise ValueError("pass tez=... (in-process) or base_url/api_key (over HTTP), not both")
        if not callable(getattr(tez, "decide", None)):
            raise TypeError(f"tez must be a tez.Tez engine or a RemoteTez, got {type(tez).__name__}")
        return tez
    return RemoteTez(base_url or DEFAULT_URL, api_key=api_key, timeout=timeout)


def decide_options(readout: str = "auto", abstain: bool = False, alpha: Any = None) -> dict:
    """Keyword arguments for decide(): only what differs from the defaults. alpha=False turns a schema's default
    gate off (the wire format's `tez.gate: false`)."""
    opts: dict[str, Any] = {}
    if readout != "auto":
        opts["readout"] = readout
    if abstain:
        opts["abstain"] = True
    if alpha is False:
        opts["gate"] = False
    elif alpha is not None:
        opts["alpha"] = alpha
    return opts


def check_options(readout: str, abstain: bool, alpha: Any) -> None:
    if readout not in READOUTS:
        raise ValueError(f"readout must be one of {', '.join(READOUTS)}, got {readout!r}")
    if not isinstance(abstain, bool):
        raise ValueError("abstain must be True or False")
    if alpha is not None and alpha is not False:
        try:
            parse_alpha(alpha, "alpha")
        except InvalidRequest as exc:
            raise ValueError(exc.message) from exc


# ---------------------------------------------------------------------------------------------- state
def _is_document(x: Any) -> bool:
    return hasattr(x, "page_content") and not isinstance(x, (str, bytes, Mapping))


def _is_message(x: Any) -> bool:
    return (not isinstance(x, (str, bytes, Mapping, list, tuple)) and hasattr(x, "content") and hasattr(x, "type")
            and not _is_document(x))


def _is_message_dict(x: Any) -> bool:
    return isinstance(x, Mapping) and "content" in x and (x.get("role") in _ROLES or x.get("type") in _ROLES)


def _is_role_tuple(x: Any) -> bool:
    return isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) and x[0] in _ROLES


def _is_message_like(x: Any) -> bool:
    return _is_message(x) or _is_message_dict(x) or _is_role_tuple(x)


def content_text(content: Any) -> str:
    """Message content as text: strings pass through, content blocks contribute their text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return "" if content is None else str(content)


def jsonable(value: Any) -> Any:
    """A JSON-compatible copy: messages become their text, documents their page_content, models their dump."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if _is_document(value):
        return value.page_content
    if _is_message(value):
        return content_text(value.content)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return jsonable(dump())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def wire_state(value: Any) -> str | dict | list:
    """A value as a wire-format state (string, object or array). A conversation (a list of messages) is decided on
    its latest message."""
    if value is None:
        raise ValueError("the state is empty (None)")
    if isinstance(value, str):
        return value
    if _is_document(value):
        return value.page_content
    if _is_message(value):
        return content_text(value.content)
    if _is_message_dict(value):
        return content_text(value["content"])
    if _is_role_tuple(value):
        return content_text(value[1])
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if value and all(_is_message_like(v) for v in value):
            return wire_state(value[-1])
        return [jsonable(v) for v in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return wire_state(dump())
    s = jsonable(value)
    return s if isinstance(s, (str, dict, list)) else str(s)


def get_key(state: Any, key: str) -> Any:
    """state[key] for a mapping, state.key for a model (a LangGraph state may be either)."""
    if isinstance(state, Mapping):
        if key not in state:
            raise ValueError(f"the state has no key {key!r} (keys: {', '.join(map(str, state)) or 'none'})")
        return state[key]
    if not isinstance(state, (str, bytes)) and hasattr(state, key):
        return getattr(state, key)
    raise ValueError(f"the input ({type(state).__name__}) has no key {key!r}; state_key needs a mapping or a model")


def messages_of(state: Any) -> list | None:
    """The `messages` list of a LangGraph state (MessagesState or any mapping / model with one), else None."""
    if isinstance(state, Mapping):
        m = state.get("messages")
    elif isinstance(state, (str, bytes, list, tuple)) or _is_message(state) or _is_document(state):
        return None
    else:
        m = getattr(state, "messages", None)
    return list(m) if isinstance(m, (list, tuple)) else None


def is_state(value: Any) -> bool:
    """A LangGraph-style state: a mapping, or a model with fields (not a message, a document or a string)."""
    if isinstance(value, Mapping):
        return not _is_message_dict(value)
    return (callable(getattr(value, "model_dump", None)) and not _is_message(value) and not _is_document(value))


def pick_state(state: Any, state_key: str | None = None) -> str | dict | list:
    """What a router or a triage step decides on: state[state_key] when a key is given; otherwise the latest message
    of a `messages` list (LangGraph's MessagesState); otherwise the input itself."""
    if state_key is not None:
        return wire_state(get_key(state, state_key))
    messages = messages_of(state)
    if messages is not None:
        if not messages:
            raise ValueError("the state's messages list is empty")
        return wire_state(messages[-1])
    return wire_state(state)


# ---------------------------------------------------------------------------------------------- questions
def inline_question(question: Any = None, criteria: Any = None, instructions: Any = None) -> dict | None:
    """A wire-format question from the ways a component accepts one; None when none was given.

    question      a question object {type, instructions, criteria} (or a tez.Question), or the instructions text
    instructions  the instructions text, as a keyword
    criteria      with instructions text: a mapping label -> description (or None), or a list of labels, makes a
                  choice question; without criteria the question is yes/no (noul)
    """
    if question is not None and instructions is not None:
        raise ValueError("give the question or instructions=..., not both")
    if question is not None and not isinstance(question, (str, Mapping, list)) and hasattr(question, "to_wire"):
        question = question.to_wire()
    if isinstance(question, Mapping):
        if criteria is not None:
            raise ValueError("criteria belongs inside the question object")
        return dict(question)
    text = question if question is not None else instructions
    if text is None:
        if criteria is not None:
            raise ValueError("criteria needs a question (its instructions)")
        return None
    if criteria is None:
        return {"type": "noul", "instructions": text}
    if isinstance(criteria, Mapping):
        return {"type": "choice", "instructions": text, "criteria": dict(criteria)}
    if isinstance(criteria, (list, tuple)):
        return {"type": "choice", "instructions": text, "criteria": {str(c): None for c in criteria}}
    raise ValueError("criteria must be a mapping of label -> description (or None), or a list of labels")


def validate_question(qid: str, question: Any, where: str = "question") -> dict:
    """The question in canonical wire form; ValueError naming the offending field when it breaks the contract."""
    try:
        return parse_question(qid, question, where=where).to_wire()
    except InvalidRequest as exc:
        raise ValueError(exc.message) from exc


def validate_questions(questions: Mapping) -> dict[str, dict]:
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("questions must be a non-empty mapping of question id -> question")
    return {str(qid): validate_question(str(qid), q, where="questions") for qid, q in questions.items()}


def schema_questions(decider: Any, schema: str) -> dict[str, dict]:
    """A loaded schema's questions in wire form (GET /v1/schemas/{name}, or the engine's schema_detail)."""
    detail = decider.schema_detail(schema)
    qs = detail.get("questions") if isinstance(detail, Mapping) else None
    if not isinstance(qs, Mapping) or not qs:
        raise ValueError(f"schema {schema!r} returned no questions")
    return {str(k): dict(v) for k, v in qs.items()}


def pick_questions(available: Mapping[str, dict], ids: Sequence[str], schema: str) -> dict[str, dict]:
    missing = [q for q in ids if q not in available]
    if missing:
        raise ValueError(f"schema {schema!r} has no question {', '.join(map(repr, missing))} "
                         f"(questions: {', '.join(available)})")
    return {q: available[q] for q in ids}


# ---------------------------------------------------------------------------------------------- answers
def answer_label(answer: Mapping) -> str:
    """The label an answer picks: "true"/"false" for noul (P(yes) >= 0.5), the choice, or the most likely score level."""
    kind = answer.get("type")
    if kind == "noul":
        return "true" if float(answer["noul"]) >= 0.5 else "false"
    if kind == "choice":
        return str(answer["choice"])
    if kind == "score":
        probs = answer.get("probabilities")
        if isinstance(probs, Mapping) and probs:
            return str(max(probs, key=lambda k: probs[k]))
        return str(int(round(float(answer["score"]))))
    raise ValueError(f"unknown answer type {kind!r}")


def answer_confidence(answer: Mapping) -> float:
    """Jev's confidence (k * p_max - 1) / (k - 1); for noul (which carries none) the same formula with k = 2."""
    if answer.get("type") == "noul":
        p = float(answer["noul"])
        return max(0.0, min(1.0, 2.0 * max(p, 1.0 - p) - 1.0))
    conf = answer.get("confidence")
    if isinstance(conf, (int, float)) and not isinstance(conf, bool):
        return float(conf)
    probs = list((answer.get("probabilities") or {}).values())
    k = len(probs)
    if k < 2:
        return 1.0
    return max(0.0, min(1.0, (k * max(probs) - 1.0) / (k - 1.0)))


def merge_answers(response: Mapping) -> dict[str, dict]:
    """{question id: answer} with each question's `tez` block (readout, decision, p_correct, calibration_id) folded
    into its answer under "tez" when the server sent one."""
    metas = ((response.get("tez") or {}).get("questions") or {}) if isinstance(response.get("tez"), Mapping) else {}
    out: dict[str, dict] = {}
    for qid, answer in (response.get("answers") or {}).items():
        a = dict(answer)
        meta = metas.get(qid)
        if meta:
            a["tez"] = dict(meta)
        out[qid] = a
    return out


def question_meta(response: Mapping, qid: str) -> dict:
    tez = response.get("tez")
    if not isinstance(tez, Mapping):
        return {}
    return dict((tez.get("questions") or {}).get(qid) or {})


# ---------------------------------------------------------------------------------------------- router
@dataclass
class Route:
    """One routing decision. `label` is what the router returns: the answer's label, or the fallback (see `reason`)."""

    label: str
    answer_label: str
    confidence: float
    decision: str | None          # "act" / "escalate" when a gate applied, else None
    reason: str | None            # why the fallback was returned; None when the answer's own label was
    answer: dict
    meta: dict = field(default_factory=dict)
    response: dict = field(default_factory=dict)

    @property
    def fell_back(self) -> bool:
        return self.reason is not None


def route_from_answer(answer: Mapping, meta: Mapping | None, fallback: str, min_confidence: float | None = None,
                      response: Mapping | None = None) -> Route:
    """The fallback when the gate escalated, when abstain picked `__none__`, or when confidence < min_confidence."""
    meta = dict(meta or {})
    label = answer_label(answer)
    confidence = answer_confidence(answer)
    decision = meta.get("decision")
    reason = None
    if decision == "escalate":
        reason = "the gate escalated" + ("" if meta.get("p_correct") is not None else " (no fitted calibration for this question)")
    elif answer.get("type") == "choice" and label == NONE_KEY:
        reason = "none of the options fits"
    elif min_confidence is not None and confidence < min_confidence:
        reason = f"confidence {confidence:.3f} is below min_confidence {min_confidence:g}"
    return Route(label=fallback if reason else label, answer_label=label, confidence=confidence, decision=decision,
                 reason=reason, answer=dict(answer), meta=meta, response=dict(response or {}))


class Router:
    """See tez.integrations.langchain.TezRouter."""

    def __init__(self, question: Any = None, *, criteria: Any = None, instructions: Any = None, schema: str | None = None,
                 question_id: str | None = None, alpha: Any = None, min_confidence: float | None = None,
                 abstain: bool = False, readout: str = "auto", fallback: str = "escalate", state_key: str | None = None,
                 decider: Any = None):
        q = inline_question(question, criteria, instructions)
        if q is None and schema is None:
            raise ValueError("a router needs a question (instructions, plus criteria for a choice), or schema=... "
                             "and question_id=... for a schema's question")
        if not isinstance(fallback, str) or not fallback:
            raise ValueError("fallback must be a non-empty string (the label returned when the router does not decide)")
        if min_confidence is not None and (isinstance(min_confidence, bool) or not 0.0 <= float(min_confidence) <= 1.0):
            raise ValueError(f"min_confidence must be between 0 and 1, got {min_confidence!r}")
        check_options(readout, abstain, alpha)
        self.question_id = question_id if question_id is not None else (None if q is None else "route")
        self.schema = schema
        self.fallback = fallback
        self.min_confidence = None if min_confidence is None else float(min_confidence)
        self.state_key = state_key
        self.options = decide_options(readout, abstain, alpha)
        self.decider = decider if decider is not None else make_decider()
        self._question = validate_question(self.question_id, q) if q is not None else None
        self._lock = threading.Lock()

    def question(self) -> tuple[str, dict]:
        """(question id, wire question); a schema's question is fetched once, on first use."""
        with self._lock:
            if self._question is None:
                available = schema_questions(self.decider, self.schema)
                qid = self.question_id
                if qid is None:
                    if len(available) != 1:
                        raise ValueError(f"schema {self.schema!r} has {len(available)} questions "
                                         f"({', '.join(available)}): pass question_id=...")
                    qid = next(iter(available))
                self._question = pick_questions(available, [qid], self.schema)[qid]
                self.question_id = qid
            return self.question_id, self._question

    def decide(self, state: Any) -> Route:
        qid, q = self.question()
        res = self.decider.decide(pick_state(state, self.state_key), questions={qid: q}, schema=self.schema,
                                  **self.options)
        answers = res.get("answers") or {}
        if qid not in answers:
            raise ValueError(f"the server's response has no answer for question {qid!r}")
        return route_from_answer(answers[qid], question_meta(res, qid), self.fallback, self.min_confidence, res)

    def route(self, state: Any) -> str:
        return self.decide(state).label


# ---------------------------------------------------------------------------------------------- guard
class TezGuardError(ValueError):
    """Raised by TezGuard(action="raise") when an input is flagged.

    violations  one dict per (item, question) over the threshold: index, question, probability, threshold, and the
                question's tez block (decision, p_correct) when the server sent one
    response    the wire-format response for the first flagged item; `responses` holds one per scored item
    """

    def __init__(self, violations: Sequence[Mapping], response: Mapping | None = None,
                 responses: Sequence[Mapping] | None = None):
        self.violations = [dict(v) for v in violations]
        self.response = dict(response) if response is not None else None
        self.responses = [dict(r) for r in responses] if responses is not None else (
            [self.response] if self.response is not None else [])
        if self.violations:
            v = self.violations[0]
            more = len(self.violations) - 1
            msg = (f"Tez guard flagged item {v['index']}: {v['question']} P(yes) = {v['probability']:.3f} "
                   f">= {v['threshold']:g}" + (f" (and {more} more)" if more else ""))
        else:
            msg = "Tez guard flagged the input"
        super().__init__(msg)


@dataclass
class GuardReport:
    """What the guard saw: one {question id: P(yes)} per scored item and the violations among them."""

    flagged: bool
    threshold: float
    questions: list[str]
    scores: list[dict]
    violations: list[dict]
    responses: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"flagged": self.flagged, "threshold": self.threshold, "questions": list(self.questions),
                "scores": [dict(s) for s in self.scores], "violations": [dict(v) for v in self.violations]}


@dataclass
class _Target:
    kind: str                 # "single" | "list" | "state"
    items: list               # what is scored
    offset: int = 0           # index of items[0] in the original list (the last message of a conversation)
    key: str | None = None    # the state key, for kind == "state"
    value: Any = None         # the value at `key`
    single: bool = True       # the value at `key` is one item (not a list)


class Guard:
    """See tez.integrations.langchain.TezGuard. `remover` turns flagged messages into state updates that delete them
    (the LangChain layer returns RemoveMessage objects); without one a filtered state key gets the filtered list."""

    def __init__(self, schema: str = "prompt-injection-guard", *, question_id: Any = None, action: str = "raise",
                 threshold: float = 0.5, state_key: str | None = None, out_key: str = "tez_guard", decider: Any = None,
                 remover: Callable[[list], list | None] | None = None):
        if action not in GUARD_ACTIONS:
            raise ValueError(f"action must be one of {', '.join(GUARD_ACTIONS)}, got {action!r}")
        if not isinstance(schema, str) or not schema:
            raise ValueError("a guard needs schema=... (a loaded schema with yes/no questions)")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be a probability between 0 and 1, got {threshold!r}")
        if question_id is not None and not isinstance(question_id, (str, list, tuple)):
            raise ValueError("question_id must be a question id or a list of them")
        if not isinstance(out_key, str) or not out_key:
            raise ValueError("out_key must be a non-empty string")
        self.schema = schema
        self.question_ids = [question_id] if isinstance(question_id, str) else (list(question_id) if question_id else None)
        self.action = action
        self.threshold = float(threshold)
        self.state_key = state_key
        self.out_key = out_key
        self.decider = decider if decider is not None else make_decider()
        self.remover = remover
        self._questions: dict[str, dict] | None = None
        self._lock = threading.Lock()

    def questions(self) -> dict[str, dict]:
        """The yes/no questions the guard asks: `question_id` (one or several), or every noul question of the schema."""
        with self._lock:
            if self._questions is None:
                available = schema_questions(self.decider, self.schema)
                if self.question_ids is None:
                    picked = {q: d for q, d in available.items() if d.get("type") == "noul"}
                    if not picked:
                        raise ValueError(f"schema {self.schema!r} has no yes/no (noul) question to guard with")
                else:
                    picked = pick_questions(available, self.question_ids, self.schema)
                    other = [q for q, d in picked.items() if d.get("type") != "noul"]
                    if other:
                        raise ValueError(f"a guard reads yes/no (noul) questions; {', '.join(other)} in schema "
                                         f"{self.schema!r} is not one")
                self._questions = picked
            return self._questions

    # -- what to score
    def target(self, value: Any) -> _Target:
        if is_state(value):
            key = self.state_key
            if key is None:
                messages = messages_of(value)
                if messages is None:
                    raise ValueError("the guard got a state without `messages`: pass state_key=... to say what to screen")
                if not messages:
                    return _Target("state", [], key="messages", value=messages, single=False)
                return _Target("state", [messages[-1]], offset=len(messages) - 1, key="messages", value=messages,
                               single=False)
            inner = get_key(value, key)
            if isinstance(inner, (list, tuple)):
                return _Target("state", list(inner), key=key, value=inner, single=False)
            return _Target("state", [inner], key=key, value=inner, single=True)
        if isinstance(value, (list, tuple)) and not _is_role_tuple(value):
            return _Target("list", list(value), value=value, single=False)
        return _Target("single", [value], value=value)

    def check(self, items: Sequence[Any], offset: int = 0) -> GuardReport:
        qs = self.questions()
        scores, violations, responses = [], [], []
        for i, item in enumerate(items):
            res = self.decider.decide(wire_state(item), questions=qs, schema=self.schema)
            responses.append(res)
            answers = res.get("answers") or {}
            row = {}
            for qid in qs:
                if qid not in answers or "noul" not in answers[qid]:
                    raise ValueError(f"the server's response has no yes/no answer for question {qid!r}")
                p = float(answers[qid]["noul"])
                row[qid] = p
                if p >= self.threshold:
                    v = {"index": offset + i, "question": qid, "probability": p, "threshold": self.threshold}
                    meta = question_meta(res, qid)
                    if meta:
                        v["tez"] = meta
                    violations.append(v)
            scores.append(row)
        return GuardReport(flagged=bool(violations), threshold=self.threshold, questions=list(qs), scores=scores,
                           violations=violations, responses=responses)

    def check_value(self, value: Any) -> GuardReport:
        t = self.target(value)
        return self.check(t.items, t.offset)

    # -- what to return
    def apply(self, value: Any) -> Any:
        """raise: the input unchanged (a state: {}), or TezGuardError. filter: flagged items removed (a flagged single
        input becomes None; a state gets {key: filtered}). annotate: {"input": value, out_key: report}, or for a state
        {out_key: report}. A state (a mapping or model) always gets a partial update, as a LangGraph node returns."""
        t = self.target(value)
        report = self.check(t.items, t.offset)
        if self.action == "raise":
            if report.flagged:
                first = report.violations[0]["index"] - t.offset
                raise TezGuardError(report.violations, report.responses[first], report.responses)
            return {} if t.kind == "state" else value
        if self.action == "annotate":
            ann = report.as_dict()
            return {self.out_key: ann} if t.kind == "state" else {"input": value, self.out_key: ann}
        flagged = sorted({v["index"] for v in report.violations})
        if t.kind == "single":
            return None if flagged else value
        if t.kind == "list":
            kept = [x for i, x in enumerate(t.value) if i not in flagged]
            return tuple(kept) if isinstance(t.value, tuple) else kept
        if not flagged:
            return {}
        if t.single:
            return {t.key: None}
        dropped = [t.value[i] for i in flagged]
        if self.remover is not None:
            update = self.remover(dropped)
            if update is not None:
                return {t.key: update}
        return {t.key: [x for i, x in enumerate(t.value) if i not in flagged]}


# ---------------------------------------------------------------------------------------------- triage
class Triage:
    """See tez.integrations.langchain.TezTriage."""

    def __init__(self, schema: str | None = None, *, state_key: str | None = None, out_key: str | None = "triage",
                 questions: Any = None, readout: str = "auto", abstain: bool = False, alpha: Any = None,
                 decider: Any = None):
        if schema is None and questions is None:
            raise ValueError("triage needs schema=... (its questions), questions=..., or both")
        if out_key is not None and (not isinstance(out_key, str) or not out_key):
            raise ValueError("out_key must be a non-empty string, or None to return the answers themselves")
        check_options(readout, abstain, alpha)
        self.schema = schema
        self.state_key = state_key
        self.out_key = out_key
        self.options = decide_options(readout, abstain, alpha)
        self.decider = decider if decider is not None else make_decider()
        self._inline: dict[str, dict] | None = None
        self._ids: list[str] | None = None
        if isinstance(questions, Mapping):
            self._inline = validate_questions(questions)
        elif isinstance(questions, (list, tuple)):
            if schema is None:
                raise ValueError("a list of question ids needs schema=... (the schema they belong to)")
            if not questions:
                raise ValueError("questions must not be empty")
            self._ids = [str(q) for q in questions]
        elif questions is not None:
            raise ValueError("questions must be a mapping of wire-format questions or a list of the schema's question ids")
        self._lock = threading.Lock()

    def questions(self) -> dict[str, dict] | None:
        """The questions sent with each request; None lets the server use the schema's own questions."""
        with self._lock:
            if self._inline is None and self._ids is not None:
                self._inline = pick_questions(schema_questions(self.decider, self.schema), self._ids, self.schema)
            return self._inline

    def answers(self, value: Any) -> dict[str, dict]:
        res = self.decider.decide(pick_state(value, self.state_key), questions=self.questions(), schema=self.schema,
                                  **self.options)
        return merge_answers(res)

    def run(self, value: Any) -> dict:
        answers = self.answers(value)
        return {self.out_key: answers} if self.out_key else answers


# ---------------------------------------------------------------------------------------------- evaluator
def score_value(answer: Mapping) -> tuple[float | None, str]:
    """(score in [0, 1] or None, label): noul -> P(yes); score -> expected level / (levels - 1); choice -> no score."""
    kind = answer.get("type")
    label = answer_label(answer)
    if kind == "noul":
        return float(answer["noul"]), label
    if kind == "score":
        levels = len(answer.get("legend") or answer.get("probabilities") or {})
        return (float(answer["score"]) / (levels - 1) if levels >= 2 else None), label
    return None, label


class Evaluator:
    """See tez.integrations.langchain.TezEvaluator."""

    requires_input = False
    requires_reference = False
    evaluation_name = "tez"

    def __init__(self, questions: Any, *, schema: str | None = None, readout: str = "auto", decider: Any = None):
        check_options(readout, False, None)
        self.schema = schema
        self.options = decide_options(readout)
        self.decider = decider if decider is not None else make_decider()
        self._ids: list[str] | None = None
        self._questions: dict[str, dict] | None = None
        if isinstance(questions, Mapping):
            self._questions = validate_questions(questions)
        elif isinstance(questions, (list, tuple)) and questions and schema is not None:
            self._ids = [str(q) for q in questions]
        else:
            raise ValueError("questions must be a mapping of wire-format questions (or a list of question ids with schema=...)")
        self._lock = threading.Lock()

    def questions(self) -> dict[str, dict]:
        with self._lock:
            if self._questions is None:
                self._questions = pick_questions(schema_questions(self.decider, self.schema), self._ids or [], self.schema)
            return self._questions

    def evaluate_strings(self, prediction: Any = None, *, reference: Any = None, input: Any = None, **kwargs: Any) -> dict:
        """Score one prediction. The state is {"input", "output", "reference"} (absent parts left out). Returns
        {"results": [{"key", "score", "value", "answer"}, ...]} plus top-level "score" and "value" when there is
        exactly one question."""
        if prediction is None:
            raise ValueError("prediction is required")
        state = {"input": input, "output": prediction, "reference": reference}
        state = {k: jsonable(v) for k, v in state.items() if v is not None}
        res = self.decider.decide(state, questions=self.questions(), schema=self.schema, **self.options)
        results = []
        for qid, answer in merge_answers(res).items():
            score, value = score_value(answer)
            results.append({"key": qid, "score": score, "value": value, "answer": answer})
        out: dict[str, Any] = {"results": results}
        if len(results) == 1:
            out["score"], out["value"] = results[0]["score"], results[0]["value"]
        return out
