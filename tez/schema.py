"""Questions and schemas: validation against the wire contract (docs/API.md), YAML loading, wire conversion.

A question is Jev's {type, instructions, criteria}:
  noul    yes/no; criteria optional {"true": text|null, "false": text|null}
  choice  2-255 options; criteria maps label -> description or null
  score   2-10 ordered levels; criteria is a list, level i = index i
With `tez.abstain` every choice question gets an implicit `__none__` option ("none of these fits").
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from .errors import InvalidRequest

QUESTION_TYPES = ("noul", "choice", "score")
LAYOUTS = ("auto", "question_first", "state_first")      # prompt layouts (tez/prompt.py)
NONE_KEY = "__none__"
NONE_TEXT = "none of these fits"
CHOICE_MIN, CHOICE_MAX = 2, 255
SCORE_MIN, SCORE_MAX = 2, 10
NOUL_DEFAULT = {"false": "the statement does not hold", "true": "the statement holds"}
MAX_SHOTS = 4            # worked examples per question taken from a schema's `examples`
MAX_SHOT_OPTIONS = 26    # shots are only shown when the whole option list fits one letter readout
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}


def as_text(value: Any) -> str:
    """Strings pass through; objects and arrays are serialised as JSON (the wire format allows all three)."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def state_key(state: Any) -> str:
    """Canonical text of a state, used to deduplicate labelled rows (objects with sorted keys)."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)


def _key(k: Any) -> str:
    """Mapping keys as strings. YAML turns unquoted `true:` into a boolean and `1:` into an int."""
    if isinstance(k, bool):
        return "true" if k else "false"
    return k if isinstance(k, str) else str(k)


def parse_alpha(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
        raise InvalidRequest(f"{where} must be a number between 0 and 1 (exclusive), got {value!r}")
    return float(value)


def parse_layout(value: Any, where: str) -> str:
    if value not in LAYOUTS:
        raise InvalidRequest(f"{where} must be one of {', '.join(LAYOUTS)}; got {value!r}")
    return value


@dataclass
class Question:
    id: str
    type: str
    instructions: Any          # string, object or array, as given
    criteria: Any = None       # noul: dict or None; choice: dict label -> str|None; score: list[str]

    @property
    def instructions_text(self) -> str:
        return as_text(self.instructions)

    def keys(self, abstain: bool = False) -> list[str]:
        """Answer keys in option order: noul false/true, choice labels, score "0".."n-1"."""
        if self.type == "noul":
            return ["false", "true"]
        if self.type == "score":
            return [str(i) for i in range(len(self.criteria))]
        keys = list(self.criteria)
        if abstain and NONE_KEY not in self.criteria:
            keys.append(NONE_KEY)
        return keys

    def options(self, abstain: bool = False) -> list[tuple[str, str | None]]:
        """(key, description) pairs in the order they are shown to the model."""
        if self.type == "noul":
            crit = self.criteria or {}
            no = crit.get("false") or NOUL_DEFAULT["false"]
            yes = crit.get("true") or NOUL_DEFAULT["true"]
            return [("false", f"no, {no}"), ("true", f"yes, {yes}")]
        if self.type == "score":
            return [(f"level {i}", c) for i, c in enumerate(self.criteria)]
        opts = list(self.criteria.items())
        if abstain and NONE_KEY not in self.criteria:
            opts.append((NONE_KEY, NONE_TEXT))
        return opts

    def label_index(self, label: Any, allow_none: bool = False) -> int:
        """Option index of a label from a labelled row or feedback. `__none__` maps past the last option when allowed."""
        if self.type == "noul":
            if isinstance(label, bool):
                return int(label)
            if isinstance(label, (int, float)) and label in (0, 1):
                return int(label)
            if isinstance(label, str) and label.strip().lower() in _TRUE | _FALSE:
                return int(label.strip().lower() in _TRUE)
            raise InvalidRequest(f"label for noul question '{self.id}' must be true or false, got {label!r}")
        if self.type == "score":
            n = len(self.criteria)
            if isinstance(label, bool):
                idx = None
            elif isinstance(label, int):
                idx = label
            elif isinstance(label, float) and label.is_integer():
                idx = int(label)
            elif isinstance(label, str) and re.fullmatch(r"\s*\d+\s*", label):
                idx = int(label)
            else:
                idx = None
            if idx is None or not 0 <= idx < n:
                raise InvalidRequest(f"label for score question '{self.id}' must be a level from 0 to {n - 1}, got {label!r}")
            return idx
        if label is None or isinstance(label, bool):
            raise InvalidRequest(f"label for choice question '{self.id}' must be one of its options, got {label!r}")
        key = _key(label)
        if key in self.criteria:
            return list(self.criteria).index(key)
        if allow_none and key == NONE_KEY:
            return len(self.criteria)
        shown = ", ".join(list(self.criteria)[:12]) + (", ..." if len(self.criteria) > 12 else "")
        raise InvalidRequest(f"label {label!r} is not an option of choice question '{self.id}' (options: {shown})")

    def canonical_label(self, label: Any, allow_none: bool = False) -> Any:
        """The stored form of a label: "true"/"false" for noul, the level int for score, the option key for choice."""
        idx = self.label_index(label, allow_none=allow_none)
        if self.type == "noul":
            return "true" if idx else "false"
        if self.type == "score":
            return idx
        return NONE_KEY if idx == len(self.criteria) else list(self.criteria)[idx]

    def to_wire(self) -> dict:
        out: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            out["criteria"] = dict(self.criteria) if isinstance(self.criteria, dict) else list(self.criteria)
        return out

    def signature(self) -> str:
        """Order-preserving identity of the definition (option order matters for prompts and probes)."""
        return json.dumps(self.to_wire(), ensure_ascii=False)


def parse_question(qid: Any, raw: Any, where: str = "questions") -> Question:
    """Validate one wire-format question. Raises InvalidRequest with the path of the offending field."""
    if not isinstance(qid, str) or not qid.strip():
        raise InvalidRequest(f"{where}: question ids must be non-empty strings")
    here = f"{where}.{qid}"
    if not isinstance(raw, dict):
        raise InvalidRequest(f"{here} must be an object with type, instructions and criteria")
    qtype = raw.get("type")
    if qtype not in QUESTION_TYPES:
        raise InvalidRequest(f"{here}.type must be one of noul, choice, score; got {qtype!r}")
    instructions = raw.get("instructions")
    if instructions is None or (isinstance(instructions, str) and not instructions.strip()):
        raise InvalidRequest(f"{here}.instructions is required")
    if not isinstance(instructions, (str, dict, list)):
        raise InvalidRequest(f"{here}.instructions must be a string, object or array")
    crit = raw.get("criteria")
    if qtype == "noul":
        if crit is not None:
            if not isinstance(crit, dict):
                raise InvalidRequest(f"{here}.criteria must be an object with optional \"true\" and \"false\" descriptions")
            crit = {_key(k): v for k, v in crit.items()}
            extra = sorted(set(crit) - {"true", "false"})
            if extra:
                raise InvalidRequest(f"{here}.criteria keys must be \"true\" and/or \"false\"; got {extra}")
            for k, v in crit.items():
                if v is not None and not isinstance(v, str):
                    raise InvalidRequest(f"{here}.criteria.{k} must be a string or null")
    elif qtype == "choice":
        if not isinstance(crit, dict):
            raise InvalidRequest(f"{here}.criteria must be an object mapping each option label to a description or null")
        norm = {_key(k): v for k, v in crit.items()}
        if len(norm) != len(crit):
            raise InvalidRequest(f"{here}.criteria has duplicate labels")
        if not CHOICE_MIN <= len(norm) <= CHOICE_MAX:
            raise InvalidRequest(f"{here}.criteria must have {CHOICE_MIN} to {CHOICE_MAX} options, got {len(norm)}")
        for k, v in norm.items():
            if not k.strip():
                raise InvalidRequest(f"{here}.criteria labels must be non-empty strings")
            if v is not None and not isinstance(v, str):
                raise InvalidRequest(f"{here}.criteria.{k} must be a string or null")
        crit = norm
    else:
        if not isinstance(crit, list):
            raise InvalidRequest(f"{here}.criteria must be a list of level descriptions (level i = index i)")
        if not SCORE_MIN <= len(crit) <= SCORE_MAX:
            raise InvalidRequest(f"{here}.criteria must have {SCORE_MIN} to {SCORE_MAX} levels, got {len(crit)}")
        for i, v in enumerate(crit):
            if not isinstance(v, str):
                raise InvalidRequest(f"{here}.criteria[{i}] must be a string")
        crit = list(crit)
    return Question(id=qid, type=qtype, instructions=instructions, criteria=crit)


def parse_questions(raw: Any, where: str = "questions") -> dict[str, Question]:
    if not isinstance(raw, dict):
        raise InvalidRequest(f"{where} must be an object mapping question ids to questions")
    if not raw:
        raise InvalidRequest(f"{where} must not be empty")
    out: dict[str, Question] = {}
    for qid, q in raw.items():
        key = _key(qid)
        out[key] = parse_question(key, q, where)
    return out


@dataclass
class Schema:
    """A named, reusable set of questions (docs/API.md, "Schema files")."""

    name: str
    questions: dict[str, Question]
    description: str = ""
    state: str = ""
    gate_alpha: float | None = None
    examples: list[dict] = field(default_factory=list)    # {"state": ..., "labels": {qid: canonical label}}
    path: Path | None = None
    layout: str | None = None         # auto | question_first | state_first; None inherits the server's default
    builtin: bool = False             # a preset shipped with Tez (tez/presets/): zero-shot, never loads artefacts
    extraction: Any = None            # set by from_json_schema / from_pydantic: how answers become values

    @property
    def base_dir(self) -> Path:
        return self.path.parent if self.path is not None else Path(".")

    @property
    def artifact_dir(self) -> Path:
        """Trained artefacts live next to the schema file: schemas/.tez/<name>/."""
        return self.base_dir / ".tez" / self.name

    @classmethod
    def from_json_schema(cls, json_schema: Any, name: str = "extract") -> Schema:
        """Questions from a JSON schema: enum -> choice (up to 255 options), boolean -> noul, an integer with at most
        10 levels between minimum and maximum -> score, description -> instructions (tez/extract.py)."""
        from .extract import schema_from_json_schema
        return schema_from_json_schema(json_schema, name)

    @classmethod
    def from_pydantic(cls, model: Any, name: str | None = None) -> Schema:
        """Questions from a pydantic model (its JSON schema; Literal and Enum fields become choices). pydantic is
        not a dependency of Tez: pass a model from your own code."""
        from .extract import schema_from_pydantic
        return schema_from_pydantic(model, name)

    def shots(self, qid: str, n_options: int | None = None) -> list[tuple[Any, int]]:
        """Few-shot worked examples for one question: the first MAX_SHOTS schema examples labelled for it,
        as (state, option index). Empty when the option list is too long for a single letter readout
        (the tournament shows no examples)."""
        q = self.questions[qid]
        n = len(q.options()) if n_options is None else n_options
        if n > MAX_SHOT_OPTIONS:
            return []
        out = []
        for ex in self.examples:
            label = ex["labels"].get(qid)
            if label is None or label == NONE_KEY:
                continue
            out.append((ex["state"], q.label_index(label)))
            if len(out) == MAX_SHOTS:
                break
        return out

    def to_wire(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "state": self.state,
            "questions": {qid: q.to_wire() for qid, q in self.questions.items()},
            "gate": {"alpha": self.gate_alpha} if self.gate_alpha is not None else None,
            "examples": len(self.examples),
            "layout": self.layout,
            "builtin": self.builtin,
        }


def parse_labels(raw: Any, questions: dict[str, Question], where: str, allow_none: bool = True,
                 unknown: str = "error") -> dict[str, Any]:
    """Validate a {qid: label} mapping; returns canonical labels. unknown='error' | 'skip' for foreign ids."""
    if not isinstance(raw, dict):
        raise InvalidRequest(f"{where}.labels must be an object mapping question ids to labels")
    out = {}
    for qid, label in raw.items():
        key = _key(qid)
        if key not in questions:
            if unknown == "skip":
                continue
            raise InvalidRequest(f"{where}.labels: unknown question '{key}'")
        out[key] = questions[key].canonical_label(label, allow_none=allow_none)
    return out


def parse_schema(raw: Any, source: str = "schema", default_name: str | None = None) -> Schema:
    if not isinstance(raw, dict):
        raise InvalidRequest(f"{source}: a schema must be a mapping with name and questions")
    name = raw.get("name", default_name)
    name = _key(name) if name is not None else None
    if not name or not _NAME_RE.match(name):
        raise InvalidRequest(f"{source}: name must be letters, digits, '.', '_' or '-' (got {name!r})")
    questions = parse_questions(raw.get("questions"), where=f"{source}: questions")
    gate_alpha = None
    gate = raw.get("gate")
    if gate is not None:
        if not isinstance(gate, dict):
            raise InvalidRequest(f"{source}: gate must be a mapping like {{alpha: 0.05}}")
        if gate.get("alpha") is not None:
            gate_alpha = parse_alpha(gate["alpha"], f"{source}: gate.alpha")
    examples = []
    for i, ex in enumerate(raw.get("examples") or []):
        here = f"{source}: examples[{i}]"
        if not isinstance(ex, dict) or "state" not in ex:
            raise InvalidRequest(f"{here} must be a mapping with state and labels")
        if not isinstance(ex["state"], (str, dict, list)):
            raise InvalidRequest(f"{here}.state must be a string, mapping or list")
        examples.append({"state": ex["state"], "labels": parse_labels(ex.get("labels") or {}, questions, here)})
    description = raw.get("description") or ""
    state = raw.get("state") or ""
    layout = parse_layout(raw["layout"], f"{source}: layout") if raw.get("layout") is not None else None
    return Schema(name=name, questions=questions, description=as_text(description), state=as_text(state),
                  gate_alpha=gate_alpha, examples=examples, layout=layout)


class _Loader(yaml.SafeLoader):
    """YAML 1.2 booleans: only true/false. PyYAML's YAML 1.1 default turns yes/no/on/off into booleans,
    which would silently rename choice options called `yes` or `no`."""


_Loader.yaml_implicit_resolvers = {
    ch: [r for r in rs if r[0] != "tag:yaml.org,2002:bool"] for ch, rs in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Loader.add_implicit_resolver("tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))


def load_yaml(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)  # noqa: S506 - a SafeLoader subclass


def load_schema(path: str | Path) -> Schema:
    path = Path(path)
    try:
        raw = load_yaml(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InvalidRequest(f"cannot read schema {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise InvalidRequest(f"{path}: invalid YAML: {exc}") from exc
    schema = parse_schema(raw, source=str(path), default_name=path.stem)
    schema.path = path.resolve()
    return schema


def schema_files(directory: str | Path) -> list[Path]:
    d = Path(directory)
    if not d.is_dir():
        raise InvalidRequest(f"schema directory not found: {d}")
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in (".yaml", ".yml"))


def load_schemas(source: str | Path | Iterable) -> dict[str, Schema]:
    """Load every *.yaml / *.yml in a directory (or an iterable of files / Schema objects). Names must be unique."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        items: list = schema_files(p) if p.is_dir() else [p]
    else:
        items = list(source)
    out: dict[str, Schema] = {}
    origin: dict[str, str] = {}
    for item in items:
        schema = item if isinstance(item, Schema) else load_schema(item)
        where = str(schema.path or schema.name)
        if schema.name in out:
            raise InvalidRequest(f"schema name '{schema.name}' is defined twice ({origin[schema.name]} and {where})")
        out[schema.name] = schema
        origin[schema.name] = where
    return out
