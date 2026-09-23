"""Question / schema validation, YAML loading and wire conversion."""
from __future__ import annotations

from pathlib import Path

import pytest

from tez import InvalidRequest, Tez
from tez.schema import NONE_KEY, NONE_TEXT, load_schema, load_schemas, parse_question, parse_questions


def choice(n):
    return {"type": "choice", "instructions": "Pick one", "criteria": {f"o{i}": None for i in range(n)}}


@pytest.mark.parametrize("raw, fragment", [
    ({"type": "boolean", "instructions": "x"}, "type must be one of"),
    ({"type": "noul"}, "instructions is required"),
    ({"type": "noul", "instructions": "  "}, "instructions is required"),
    ({"type": "noul", "instructions": 3}, "string, object or array"),
    ({"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}}, "\"true\" and/or \"false\""),
    ({"type": "noul", "instructions": "x", "criteria": ["yes", "no"]}, "must be an object"),
    ({"type": "noul", "instructions": "x", "criteria": {"true": 1}}, "string or null"),
    (choice(1), "2 to 255 options"),
    (choice(256), "2 to 255 options"),
    ({"type": "choice", "instructions": "x", "criteria": ["a", "b"]}, "must be an object mapping"),
    ({"type": "choice", "instructions": "x", "criteria": {"a": "ok", "b": {"nested": True}}}, "string or null"),
    ({"type": "choice", "instructions": "x", "criteria": {"a": None, " ": None}}, "non-empty"),
    ({"type": "score", "instructions": "x", "criteria": ["only"]}, "2 to 10 levels"),
    ({"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]}, "2 to 10 levels"),
    ({"type": "score", "instructions": "x", "criteria": {"0": "a", "1": "b"}}, "must be a list"),
    ({"type": "score", "instructions": "x", "criteria": ["a", None]}, "must be a string"),
    ("not an object", "must be an object"),
])
def test_question_validation_errors(raw, fragment):
    with pytest.raises(InvalidRequest) as err:
        parse_question("q", raw)
    assert fragment in err.value.message
    assert "questions.q" in err.value.message


def test_limits_are_inclusive():
    assert len(parse_question("q", choice(255)).keys()) == 255
    assert len(parse_question("q", choice(2)).keys()) == 2
    assert len(parse_question("q", {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(10)]}).keys()) == 10


def test_questions_container_errors():
    with pytest.raises(InvalidRequest, match="must not be empty"):
        parse_questions({})
    with pytest.raises(InvalidRequest, match="must be an object"):
        parse_questions([{"type": "noul"}])


def test_wire_round_trip():
    raw = {"type": "choice", "instructions": {"k": "v"}, "criteria": {"b": "second", "a": None}}
    q = parse_question("q", raw)
    assert q.to_wire() == raw
    assert list(q.to_wire()["criteria"]) == ["b", "a"]          # option order is preserved
    assert parse_question("q", q.to_wire()).signature() == q.signature()
    noul = parse_question("n", {"type": "noul", "instructions": "x"})
    assert noul.to_wire() == {"type": "noul", "instructions": "x"}


def test_options_and_keys():
    q = parse_question("t", {"type": "choice", "instructions": "x", "criteria": {"a": "alpha", "b": None}})
    assert q.options() == [("a", "alpha"), ("b", None)]
    assert q.options(abstain=True)[-1] == (NONE_KEY, NONE_TEXT)
    assert q.keys(abstain=True) == ["a", "b", NONE_KEY]
    n = parse_question("n", {"type": "noul", "instructions": "x", "criteria": {"true": "Yes it is", "false": None}})
    assert n.options() == [("false", "no, the statement does not hold"), ("true", "yes, Yes it is")]
    assert n.keys(abstain=True) == ["false", "true"]
    s = parse_question("s", {"type": "score", "instructions": "x", "criteria": ["low", "high"]})
    assert s.options() == [("level 0", "low"), ("level 1", "high")]
    assert s.keys() == ["0", "1"]


@pytest.mark.parametrize("label, idx", [(True, 1), (False, 0), ("true", 1), ("False", 0), ("yes", 1), ("no", 0), (1, 1), (0, 0), ("1", 1)])
def test_noul_labels(label, idx):
    assert parse_question("n", {"type": "noul", "instructions": "x"}).label_index(label) == idx


def test_score_and_choice_labels():
    s = parse_question("s", {"type": "score", "instructions": "x", "criteria": ["a", "b", "c"]})
    assert s.label_index(2) == 2 and s.label_index("1") == 1 and s.label_index(0.0) == 0
    for bad in (3, -1, True, "high", 1.5):
        with pytest.raises(InvalidRequest):
            s.label_index(bad)
    c = parse_question("c", {"type": "choice", "instructions": "x", "criteria": {"x": None, "1": None}})
    assert c.label_index("x") == 0 and c.label_index(1) == 1
    with pytest.raises(InvalidRequest, match="not an option"):
        c.label_index("z")
    with pytest.raises(InvalidRequest):
        c.label_index(NONE_KEY)
    assert c.label_index(NONE_KEY, allow_none=True) == 2
    assert c.canonical_label(NONE_KEY, allow_none=True) == NONE_KEY


def test_load_docs_schema(schema_dir: Path):
    s = load_schema(schema_dir / "support-triage.yaml")
    assert s.name == "support-triage"
    assert [q.type for q in s.questions.values()] == ["choice", "noul", "score"]
    assert s.questions["is_urgent"].criteria == {"true": "Explicitly time-sensitive or blocking", "false": "No urgency expressed"}
    assert s.gate_alpha == 0.05
    assert s.examples == [{"state": "I was charged twice this month.", "labels": {"topic": "billing", "is_urgent": "false", "anger": 1}}]
    assert s.artifact_dir == schema_dir.resolve() / ".tez" / "support-triage"
    assert s.shots("topic") == [("I was charged twice this month.", 0)]
    assert s.shots("anger") == [("I was charged twice this month.", 1)]


def test_yaml_keeps_yes_no_as_strings(tmp_path: Path):
    (tmp_path / "a.yaml").write_text(
        "name: a\nquestions:\n  q:\n    type: choice\n    instructions: pick\n    criteria:\n      yes: agree\n      no: disagree\n"
        "      on: toggled\n  n:\n    type: noul\n    instructions: ok?\n    criteria:\n      true: fine\n      false: bad\n",
        encoding="utf-8")
    s = load_schema(tmp_path / "a.yaml")
    assert list(s.questions["q"].criteria) == ["yes", "no", "on"]
    assert s.questions["n"].criteria == {"true": "fine", "false": "bad"}


def test_schema_errors(tmp_path: Path):
    (tmp_path / "bad.yaml").write_text("name: ../evil\nquestions: {q: {type: noul, instructions: x}}\n", encoding="utf-8")
    with pytest.raises(InvalidRequest, match="name must be"):
        load_schema(tmp_path / "bad.yaml")
    (tmp_path / "bad.yaml").write_text("name: ok\nquestions: {q: {type: noul, instructions: x}}\nexamples:\n  - state: s\n    labels: {zz: 1}\n", encoding="utf-8")
    with pytest.raises(InvalidRequest, match="unknown question 'zz'"):
        load_schema(tmp_path / "bad.yaml")
    (tmp_path / "bad.yaml").write_text("name: ok\nquestions: {q: {type: noul, instructions: x}}\ngate: {alpha: 2}\n", encoding="utf-8")
    with pytest.raises(InvalidRequest, match="gate.alpha"):
        load_schema(tmp_path / "bad.yaml")
    (tmp_path / "bad.yaml").write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(InvalidRequest, match="invalid YAML"):
        load_schema(tmp_path / "bad.yaml")


def test_load_schema_directory_and_duplicates(schema_dir: Path):
    (schema_dir / "other.yml").write_text("questions: {q: {type: noul, instructions: x}}\n", encoding="utf-8")
    (schema_dir / "notes.txt").write_text("ignored", encoding="utf-8")
    loaded = load_schemas(schema_dir)
    assert sorted(loaded) == ["other", "support-triage"]           # name defaults to the file stem
    (schema_dir / "dup.yaml").write_text("name: other\nquestions: {q: {type: noul, instructions: x}}\n", encoding="utf-8")
    with pytest.raises(InvalidRequest, match="defined twice"):
        load_schemas(schema_dir)


@pytest.mark.parametrize("body, fragment", [
    ([], "JSON object"),
    ({"questions": {"q": {"type": "noul", "instructions": "x"}}}, "state is required"),
    ({"state": None, "questions": {"q": {"type": "noul", "instructions": "x"}}}, "state is required"),
    ({"state": 5, "questions": {"q": {"type": "noul", "instructions": "x"}}}, "string, object or array"),
    ({"state": "s"}, "questions is required"),
    ({"state": "s", "schema": "nope"}, "unknown schema 'nope'"),
    ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}, "model": 3}, "model must be a string"),
    ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}, "tez": {"readout": "logits"}}, "tez.readout"),
    ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}, "tez": {"abstain": "yes"}}, "tez.abstain"),
    ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}, "tez": {"gate": {"alpha": 0}}}, "tez.gate.alpha"),
    ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}, "tez": {"gate": {"alpha": 1.5}}}, "tez.gate.alpha"),
    ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}, "tez": {"gate": {"alpha": True}}}, "tez.gate.alpha"),
    ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}, "tez": {"gate": 0.05}}, "tez.gate must be an object"),
    ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}, "tez": {"gate": {}}}, "tez.gate.alpha is required"),
    ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}, "tez": "auto"}, "tez must be an object"),
    ({"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}, "tez": {"readout": "probe"}}, "no usable probe"),
])
def test_request_validation(body, fragment):
    with pytest.raises(InvalidRequest) as err:
        Tez(backend="fake").handle(body)
    assert fragment in err.value.message


def test_schema_request_uses_schema_questions_and_default_gate(schema_dir: Path):
    t = Tez(backend="fake", schemas=schema_dir)
    res = t.handle({"state": "Where is my refund?", "schema": "support-triage"})
    assert list(res["answers"]) == ["topic", "is_urgent", "anger"]
    assert set(res["answers"]["topic"]["probabilities"]) == {"billing", "technical", "account", "sales"}
    assert all(m["decision"] == "escalate" for m in res["tez"]["questions"].values())   # schema default alpha, no fit yet
    res = t.handle({"state": "x", "schema": "support-triage", "tez": {"gate": None}})
    assert all("decision" not in m for m in res["tez"]["questions"].values())


def test_schema_examples_become_few_shot(schema_dir: Path):
    from tez import FakeBackend
    fb = FakeBackend()
    t = Tez(backend=fb, schemas=schema_dir)
    t.handle({"state": "The app crashes on start", "schema": "support-triage"})
    topic_prompt = fb.prompts[0]
    assert "Worked examples of this exact question follow." in topic_prompt
    assert "Example input:\nI was charged twice this month." in topic_prompt and "Answer: A" in topic_prompt
    assert topic_prompt.index("Answer: A") < topic_prompt.index("Input:\nThe app crashes on start")
    # a request-defined question with the same id but another definition is ad hoc: no examples
    fb.prompts.clear()
    t.handle({"state": "s", "schema": "support-triage",
              "questions": {"topic": {"type": "choice", "instructions": "Other?", "criteria": {"x": None, "y": None}}}})
    assert "Example input" not in fb.prompts[0]
