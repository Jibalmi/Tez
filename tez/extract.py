"""Structured extraction: a JSON schema (or a pydantic model) as Tez questions, and the answers back as typed values.

    tez.extract("Charged twice for invoice 4411, fix it today!", Ticket)      # -> Ticket(department='billing', ...)
    POST /v1/systemone {"state": ..., "json_schema": {...}}                    # values in response["tez"]["values"]

The mapping is a documented subset; anything else is refused with an error that names the path:

  enum, Literal[...], an Enum class, oneOf/anyOf of consts   choice (2 to 255 options); values come back as given
                                                             (strings, numbers, booleans, null); a const's
                                                             description becomes its option's description
  boolean                                                    noul -> True / False
  integer with minimum and maximum, 2 to 10 values           score -> the most likely value
  one possible value (const, a one-value enum or range)      fixed: filled in, not asked
  object with properties (nested models)                     its properties, as dotted question ids ("address.city")
  optional (anyOf with null, type [..., "null"])             a choice gets a "null" option (value None); booleans and
                                                             integer scales always get a value
  description                                                the question's instructions

Refused: free strings and numbers without an enum, arrays, integer ranges over 10 values, more than 255 options,
unions of different types, $ref outside the document and recursive schemas. pydantic is never imported by Tez:
from_pydantic reads the model's own JSON schema (pydantic v2 model_json_schema, v1 schema).

Size: $refs can make a small schema expand without end (ten properties that each refer to the next level, a few levels
deep). The conversion counts as it goes and stops with 413 payload_too_large at the first question past the caller's
question budget (tez serve's --max-questions), and at MAX_REFS references followed or MAX_REF_CHARS characters of schema
copied in by them, whatever the budget.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from .errors import InvalidRequest, PayloadTooLarge
from .schema import CHOICE_MAX, NONE_KEY, SCORE_MAX, Question, Schema, parse_question

NULL_LABEL = "null"
NULL_TEXT = "not stated, or does not apply"
MAX_REFS = 10_000             # $refs followed while converting one JSON schema
MAX_REF_CHARS = 1_000_000     # characters of schema (about its JSON length) those $refs copy in, all told


class EscalationRequired(Exception):
    """Tez.extract(..., alpha=...) without return_details: at least one field's gate decision was `escalate`, so the
    extracted object is not certified. `fields` names them; `values` holds the model's best guess and `response` the
    whole decision. Hand the case to a person or a larger model, or pass return_details=True to handle it yourself."""

    def __init__(self, fields: list[str], values: dict, response: dict):
        self.fields = list(fields)
        self.values = values
        self.response = response
        super().__init__(f"the gate escalated {', '.join(self.fields)}: no certified value (hand the case to a person "
                         "or a larger model, or use return_details=True)")


@dataclass
class Field:
    """One field of the extracted object: how its answer becomes a value."""

    qid: str                     # question id (the dotted path)
    path: tuple[str, ...]        # where the value goes in the (nested) object
    kind: str                    # choice | noul | score | const | label (a plain schema's choice: its label)
    values: list[Any] = field(default_factory=list)   # choice: value per option; score: value per level; const: [value]
    labels: list[str] = field(default_factory=list)   # choice: option labels, in order


@dataclass
class Extraction:
    """How a schema's answers become values: `fields` per question id, `consts` filled in without asking. `source` is
    the JSON schema it came from (sent on the wire as json_schema), `model` the pydantic model, when there is one."""

    fields: dict[str, Field]
    consts: list[Field] = field(default_factory=list)
    source: dict | None = None
    model: Any = None

    def values(self, answers: dict) -> dict:
        """The (nested) object from wire-format answers. Missing answers are left out."""
        out: dict = {}
        for f in self.consts:
            _put(out, f.path, f.values[0])
        for qid, f in self.fields.items():
            a = answers.get(qid)
            if a is None:
                continue
            _put(out, f.path, _value(f, a))
        return out

    def build(self, values: dict) -> Any:
        """A model instance when this came from a pydantic model (validated by it), else the values themselves."""
        if self.model is None:
            return values
        if hasattr(self.model, "model_validate"):
            return self.model.model_validate(values)
        if hasattr(self.model, "parse_obj"):
            return self.model.parse_obj(values)
        return self.model(**values)


@dataclass
class ExtractResult:
    """Tez.extract(..., return_details=True): the value (dict or model instance), the plain values, the gate decisions
    per field (when a gate applied), the escalated fields and the whole wire-format response."""

    value: Any
    values: dict
    response: dict
    decisions: dict = field(default_factory=dict)
    escalated: list = field(default_factory=list)


def _put(out: dict, path: tuple[str, ...], value: Any) -> None:
    d = out
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value


def _value(f: Field, answer: dict) -> Any:
    kind = answer.get("type")
    if f.kind == "noul" or kind == "noul":
        return float(answer.get("noul", 0.0)) >= 0.5
    probs = answer.get("probabilities") or {}
    if f.kind == "score":
        level = int(max(probs, key=lambda k: probs[k])) if probs else int(round(float(answer.get("score", 0.0))))
        return f.values[level]
    label = answer.get("choice")
    if label == NONE_KEY:
        return None
    if f.kind == "label":
        return label
    return f.values[f.labels.index(label)] if label in f.labels else label


def default_extraction(schema: Schema) -> Extraction:
    """For a schema not made from a JSON schema: noul -> bool, choice -> its label, score -> the level number."""
    fields = {}
    for qid, q in schema.questions.items():
        kind = "noul" if q.type == "noul" else ("score" if q.type == "score" else "label")
        values = list(range(len(q.criteria))) if q.type == "score" else []
        fields[qid] = Field(qid=qid, path=(qid,), kind=kind, values=values)
    return Extraction(fields=fields)


# ---------------------------------------------------------------------------------------------- JSON schema -> questions
def _label(v: Any) -> str:
    if v is None:
        return NULL_LABEL
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return json.dumps(v)
    raise ValueError(v)


def _is_null(node: dict) -> bool:
    return node.get("type") == "null" or ("const" in node and node["const"] is None) or node.get("enum") == [None]


def _integral(v: Any) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) and float(v).is_integer()


def _scalar_size(v: Any) -> int:
    return len(v) + 2 if isinstance(v, str) else 4


class _Converter:
    def __init__(self, root: dict, name: str, max_questions: int | None = None):
        self.root = root
        self.name = name
        self.max_questions = max_questions or None       # 0 or None: no budget
        self.questions: dict[str, Question] = {}
        self.fields: dict[str, Field] = {}
        self.consts: list[Field] = []
        self.refs = 0                                     # $refs followed so far
        self.ref_chars = 0                                # schema text they copied in
        self._sizes: dict[int, int] = {}

    def fail(self, where: str, message: str) -> InvalidRequest:
        return InvalidRequest(f"{where}: {message}")

    def size(self, node: Any) -> int:
        """About how many characters of JSON `node` is: what following a $ref to it adds to the schema. Each object
        is measured once, without recursion (a Python object that contains itself counts once)."""
        if not isinstance(node, (dict, list)):
            return _scalar_size(node)
        memo, pending, stack = self._sizes, set(), [node]
        while stack:
            n = stack[-1]
            if id(n) in memo:
                stack.pop()
                continue
            kids = list(n.values()) if isinstance(n, dict) else n
            if id(n) not in pending:
                pending.add(id(n))
                stack.extend(c for c in kids if isinstance(c, (dict, list)) and id(c) not in memo and id(c) not in pending)
                continue
            stack.pop()
            total = 2 + sum(memo.get(id(c), 0) if isinstance(c, (dict, list)) else _scalar_size(c) for c in kids)
            memo[id(n)] = total + (sum(len(str(k)) + 3 for k in n) if isinstance(n, dict) else len(n))
        return memo[id(node)]

    def count_ref(self, target: Any, where: str) -> None:
        self.refs += 1
        self.ref_chars += self.size(target)
        if self.refs > MAX_REFS or self.ref_chars > MAX_REF_CHARS:
            raise PayloadTooLarge(f"{where}: the schema's $refs expand past {MAX_REFS:,} references or "
                                  f"{MAX_REF_CHARS:,} characters of schema; a schema this large cannot be decided")

    def resolve(self, node: Any, where: str, seen: tuple = (), followed: list | None = None) -> dict:
        """The node with local $refs followed (siblings of a $ref win) and a one-element allOf merged; every $ref
        followed is appended to `followed`."""
        if not isinstance(node, dict):
            raise self.fail(where, f"a schema must be an object, got {type(node).__name__}")
        if "$ref" in node:
            ref = node["$ref"]
            if not isinstance(ref, str) or not ref.startswith("#"):
                raise self.fail(where, f"only references inside the document are supported, got {ref!r}")
            if ref in seen:
                raise self.fail(where, f"{ref} refers to itself: a recursive schema cannot be decided in one pass")
            target: Any = self.root
            for part in [p for p in ref[1:].split("/") if p]:
                part = part.replace("~1", "/").replace("~0", "~")
                if not isinstance(target, dict) or part not in target:
                    raise self.fail(where, f"{ref} does not resolve inside the schema")
                target = target[part]
            self.count_ref(target, where)
            if followed is not None:
                followed.append(ref)
            merged = {**self.resolve(target, where, seen + (ref,), followed),
                      **{k: v for k, v in node.items() if k != "$ref"}}
            return merged
        if isinstance(node.get("allOf"), list) and len(node["allOf"]) == 1:
            inner = self.resolve(node["allOf"][0], f"{where}.allOf[0]", seen, followed)
            return {**inner, **{k: v for k, v in node.items() if k != "allOf"}}
        return node

    def enter(self, refs: tuple, followed: list, where: str) -> tuple:
        """The $refs open along a path after following `followed`: meeting an open one again means the schema contains
        itself (siblings that share a definition are fine)."""
        for ref in followed:
            if ref in refs:
                raise self.fail(where, f"{ref} contains itself: a recursive schema cannot be decided in one pass")
        return refs + tuple(dict.fromkeys(followed))

    def instructions(self, node: dict, default: str) -> Any:
        desc = node.get("description")
        return desc if isinstance(desc, str) and desc.strip() else default

    def add(self, qid: str, raw: dict, f: Field, where: str) -> None:
        if qid in self.questions:
            raise self.fail(where, f"two fields map to the question id {qid!r}")
        if self.max_questions and len(self.questions) >= self.max_questions:
            raise PayloadTooLarge(f"{self.name}: too many questions: more than {self.max_questions} (the limit is "
                                  f"{self.max_questions}; tez serve --max-questions)")
        try:
            self.questions[qid] = parse_question(qid, raw, where=self.name)
        except InvalidRequest as exc:
            raise self.fail(where, exc.message) from exc
        self.fields[qid] = f

    def const(self, path: tuple, value: Any) -> None:
        self.consts.append(Field(qid=".".join(path), path=path, kind="const", values=[value]))

    def convert(self, node: Any, path: tuple[str, ...], where: str, nullable: bool = False, refs: tuple = ()) -> None:
        """Add the questions for one field. `refs` are the $refs being expanded along this path (through $ref, a
        one-element allOf or a union's branch): meeting one again means the schema contains itself."""
        followed: list[str] = []
        node = self.resolve(node, where, followed=followed)
        refs = self.enter(refs, followed, where)
        dotted = ".".join(path)
        for key in ("anyOf", "oneOf"):
            if isinstance(node.get(key), list):
                in_branches: list[str] = []
                branches = [self.resolve(b, f"{where}.{key}[{i}]", followed=in_branches) for i, b in enumerate(node[key])]
                real = [b for b in branches if not _is_null(b)]
                has_null = len(real) < len(branches)
                if real and all("const" in b for b in real):
                    options = [(b["const"], b.get("description") or b.get("title")) for b in real]
                    return self.choice(node, path, where, options, nullable or has_null)
                if len(real) == 1:
                    rest = {k: v for k, v in node.items() if k not in ("anyOf", "oneOf")}
                    return self.convert({**real[0], **rest}, path, where, nullable or has_null,
                                        self.enter(refs, in_branches, where))
                raise self.fail(where, "a union of different types cannot be decided in one pass; use one enum")
        jtype = node.get("type")
        if isinstance(jtype, list):
            kinds = [t for t in jtype if t != "null"]
            nullable = nullable or len(kinds) < len(jtype)
            if len(kinds) != 1:
                raise self.fail(where, f"a field must have one type, got {jtype}")
            jtype = kinds[0]
        if "const" in node:
            return self.const(path, node["const"])
        if "enum" in node:
            enum = node["enum"]
            if not isinstance(enum, list) or not enum:
                raise self.fail(where, "enum must be a non-empty list")
            has_null = any(v is None for v in enum)
            return self.choice(node, path, where, [(v, None) for v in enum if v is not None], nullable or has_null)
        if jtype == "boolean":
            if nullable:
                return self.choice(node, path, where, [(True, "yes"), (False, "no")], True,
                                   default=f"Is `{dotted}` true for this input?")
            raw = {"type": "noul", "instructions": self.instructions(node, f"Is `{dotted}` true for this input?")}
            return self.add(dotted, raw, Field(qid=dotted, path=path, kind="noul"), where)
        if jtype == "integer":
            return self.scale(node, path, where)
        if jtype == "object" or "properties" in node:
            props = node.get("properties")
            if not isinstance(props, dict) or not props:
                raise self.fail(where, "an object needs properties (a free-form object cannot be decided)")
            for key, sub in props.items():
                self.convert(sub, path + (str(key),), f"{where}.properties.{key}", refs=refs)
            return None
        if jtype == "string":
            raise self.fail(where, "a free-text string cannot be decided in one pass; give it an enum (or use a "
                                   "boolean per option)")
        if jtype == "number":
            raise self.fail(where, "a free number cannot be decided in one pass; use an integer with minimum and maximum "
                                   f"(at most {SCORE_MAX} values) or an enum")
        if jtype == "array":
            raise self.fail(where, "arrays are not supported; ask one boolean per possible element")
        raise self.fail(where, "cannot tell what this field is: give it a type, an enum or a const")

    def choice(self, node: dict, path: tuple, where: str, options: list[tuple[Any, Any]], nullable: bool,
               default: str | None = None) -> None:
        dotted = ".".join(path)
        values = [v for v, _ in options]
        if all(isinstance(v, bool) for v in values) and set(values) == {True, False} and not nullable:
            raw = {"type": "noul", "instructions": self.instructions(node, default or f"Is `{dotted}` true for this input?")}
            return self.add(dotted, raw, Field(qid=dotted, path=path, kind="noul"), where)
        if len(values) == 1 and not nullable:
            return self.const(path, values[0])
        try:
            labels = [_label(v) for v in values]
        except ValueError as exc:
            raise self.fail(where, f"enum values must be strings, numbers, booleans or null, got {exc.args[0]!r}") from exc
        descriptions = [d if isinstance(d, str) and d.strip() else None for _, d in options]
        if nullable:
            labels.append(NULL_LABEL)
            values.append(None)
            descriptions.append(NULL_TEXT)
        if len(set(labels)) != len(labels):
            raise self.fail(where, f"enum values must be distinct as text, got {labels}")
        if NONE_KEY in labels:
            raise self.fail(where, f"{NONE_KEY!r} is reserved for abstain")
        if len(labels) > CHOICE_MAX:
            raise self.fail(where, f"{len(labels)} options; a choice takes at most {CHOICE_MAX}")
        raw = {"type": "choice", "instructions": self.instructions(node, default or f"What is `{dotted}` for this input?"),
               "criteria": dict(zip(labels, descriptions))}
        return self.add(dotted, raw, Field(qid=dotted, path=path, kind="choice", values=values, labels=labels), where)

    def scale(self, node: dict, path: tuple, where: str) -> None:
        dotted = ".".join(path)
        lo, hi = node.get("minimum"), node.get("maximum")
        emin, emax = node.get("exclusiveMinimum"), node.get("exclusiveMaximum")
        if isinstance(emin, bool):             # draft 4: a flag on minimum
            lo = lo + 1 if emin and _integral(lo) else lo
        elif _integral(emin):
            lo = emin + 1 if lo is None else max(lo, emin + 1)
        if isinstance(emax, bool):
            hi = hi - 1 if emax and _integral(hi) else hi
        elif _integral(emax):
            hi = emax - 1 if hi is None else min(hi, emax - 1)
        if not (_integral(lo) and _integral(hi)):
            raise self.fail(where, f"an integer needs minimum and maximum (at most {SCORE_MAX} values) to be decided; "
                                   "or give it an enum")
        lo, hi = int(lo), int(hi)
        if hi < lo:
            raise self.fail(where, f"maximum {hi} is below minimum {lo}")
        n = hi - lo + 1
        if n == 1:
            return self.const(path, lo)
        if n > SCORE_MAX:
            raise self.fail(where, f"{n} values from {lo} to {hi}; a score takes at most {SCORE_MAX} (narrow the range or "
                                   "use an enum)")
        levels = list(range(lo, hi + 1))
        raw = {"type": "score",
               "instructions": self.instructions(node, f"What is `{dotted}` for this input, from {lo} to {hi}?"),
               "criteria": [f"{dotted} = {v}" for v in levels]}
        return self.add(dotted, raw, Field(qid=dotted, path=path, kind="score", values=levels), where)


def schema_from_json_schema(json_schema: Any, name: str = "extract", *, max_questions: int | None = None) -> Schema:
    """A Schema whose questions are the JSON schema's fields (see the module docstring), with the extraction attached
    (schema.extraction.values(answers) builds the object). max_questions (0 or None: none) stops the conversion with
    413 payload_too_large as soon as the schema has more questions than that."""
    if not isinstance(json_schema, dict):
        raise InvalidRequest(f"{name}: a JSON schema must be an object, got {type(json_schema).__name__}")
    conv = _Converter(json_schema, name, max_questions)
    try:
        followed: list[str] = []
        root = conv.resolve(json_schema, name, followed=followed)
        if root.get("type") not in (None, "object") or not isinstance(root.get("properties"), dict) or not root["properties"]:
            raise InvalidRequest(f"{name}: the top level must be an object with properties")
        top = conv.enter((), followed, name)
        for key, sub in root["properties"].items():
            conv.convert(sub, (str(key),), f"{name}.properties.{key}", refs=top)
    except RecursionError as exc:
        raise InvalidRequest(f"{name}: the schema is nested too deeply to be decided") from exc
    if not conv.questions:
        raise InvalidRequest(f"{name}: every field has a single possible value; there is nothing to decide")
    desc = root.get("description") if isinstance(root.get("description"), str) else ""
    schema = Schema(name=name, questions=conv.questions, description=desc)
    schema.extraction = Extraction(fields=conv.fields, consts=conv.consts, source=json_schema)
    return schema


def json_schema_of(model: Any) -> dict:
    """A pydantic model's JSON schema (v2 model_json_schema, v1 schema)."""
    if isinstance(model, type) and callable(getattr(model, "model_json_schema", None)):
        return model.model_json_schema()
    if isinstance(model, type) and callable(getattr(model, "schema", None)) and hasattr(model, "parse_obj"):
        return model.schema()
    raise TypeError(f"expected a pydantic model class, got {model!r}")


def schema_from_pydantic(model: Any, name: str | None = None) -> Schema:
    """A Schema from a pydantic model class (its JSON schema); extraction.build(values) returns a model instance."""
    js = json_schema_of(model)
    schema = schema_from_json_schema(js, name or getattr(model, "__name__", "model"))
    schema.extraction.model = model
    return schema


def is_pydantic_model(obj: Any) -> bool:
    try:
        json_schema_of(obj)
    except TypeError:
        return False
    return True


# ---------------------------------------------------------------------------------------------- extract() for Tez and RemoteTez
def prepare(schema_or_model: Any, loaded: Any) -> tuple[Extraction, dict]:
    """What an extract call sends: (the extraction, the request fields). `loaded(name)` returns a loaded Schema (or
    None) and is how a schema name, or a Schema object that is loaded, keeps its fitted probes and calibration."""
    fields: dict[str, Any] = {}
    if isinstance(schema_or_model, str):
        schema = loaded(schema_or_model)
        if schema is None:
            raise InvalidRequest(f"unknown schema '{schema_or_model}'")
        fields["schema"] = schema_or_model
    elif isinstance(schema_or_model, Schema):
        schema = schema_or_model
        if loaded(schema.name) is schema:
            fields["schema"] = schema.name
    elif isinstance(schema_or_model, dict):
        schema = schema_from_json_schema(schema_or_model)
    elif is_pydantic_model(schema_or_model):
        schema = schema_from_pydantic(schema_or_model)
    else:
        raise TypeError("extract takes a JSON schema (dict), a pydantic model class, a Schema or a schema name")
    extraction = schema.extraction or default_extraction(schema)
    if extraction.source is not None:
        fields["json_schema"] = extraction.source
    elif "schema" not in fields:
        fields["questions"] = {qid: q.to_wire() for qid, q in schema.questions.items()}
    return extraction, fields


def finish(extraction: Extraction, response: dict, alpha: float | None, return_details: bool) -> Any:
    """The value an extract call returns for a wire-format response (see Tez.extract)."""
    values = extraction.values(response.get("answers") or {})
    metas = (response.get("tez") or {}).get("questions") or {} if isinstance(response.get("tez"), dict) else {}
    decisions = {qid: m["decision"] for qid, m in metas.items() if isinstance(m, dict) and m.get("decision")}
    escalated = [qid for qid, d in decisions.items() if d == "escalate"]
    if return_details:
        return ExtractResult(value=extraction.build(values), values=values, response=response, decisions=decisions,
                             escalated=escalated)
    if alpha is not None and escalated:
        raise EscalationRequired(escalated, values, response)
    return extraction.build(values)
