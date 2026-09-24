"""Structured extraction: JSON schema / pydantic -> questions, answers -> values, Tez.extract, the wire's json_schema."""
from __future__ import annotations

import subprocess
import sys
from enum import Enum
from typing import Literal, Optional

import pytest
from fastapi.testclient import TestClient

from stub_http import StubServer
from tez import EscalationRequired, ExtractResult, FakeBackend, InvalidRequest, Schema, Tez
from tez.config import Limits
from tez.errors import PayloadTooLarge
from tez.extract import NULL_LABEL, default_extraction, schema_from_json_schema
from tez.integrations import RemoteTez
from tez.server import create_app

TICKET = {
    "title": "Ticket",
    "description": "A support ticket, triaged.",
    "type": "object",
    "properties": {
        "department": {"enum": ["billing", "technical", "sales"], "description": "Which team should handle it?"},
        "urgent": {"type": "boolean", "description": "Does it need a reply today?"},
        "priority": {"type": "integer", "minimum": 1, "maximum": 3},
        "channel": {"oneOf": [{"const": "email", "description": "Sent by email"},
                              {"const": "chat", "description": "Came through the chat widget"}]},
        "customer": {"type": "object", "properties": {
            "tier": {"$ref": "#/$defs/Tier"},
            "kind": {"const": "person"}}},
        "refund": {"anyOf": [{"enum": ["full", "partial"]}, {"type": "null"}], "description": "Refund asked for, if any"},
    },
    "$defs": {"Tier": {"enum": [1, 2, 3], "type": "integer", "description": "Support tier"}},
}


def fixed(scores: dict):
    """letters_fn: the option at index scores[k] wins for a question with k options."""
    def fn(prompt, k):
        out = [0.0] * k
        out[scores.get(k, 0)] = 5.0
        return out
    return fn


# ---------------------------------------------------------------------------------------------- mapping
def test_json_schema_mapping():
    s = schema_from_json_schema(TICKET, "ticket")
    q = s.questions
    assert list(q) == ["department", "urgent", "priority", "channel", "customer.tier", "refund"]
    assert q["department"].to_wire() == {"type": "choice", "instructions": "Which team should handle it?",
                                         "criteria": {"billing": None, "technical": None, "sales": None}}
    assert q["urgent"].to_wire() == {"type": "noul", "instructions": "Does it need a reply today?"}
    assert q["priority"].type == "score" and q["priority"].criteria == ["priority = 1", "priority = 2", "priority = 3"]
    assert "from 1 to 3" in q["priority"].instructions
    assert q["channel"].criteria == {"email": "Sent by email", "chat": "Came through the chat widget"}
    assert q["customer.tier"].criteria == {"1": None, "2": None, "3": None}
    assert q["customer.tier"].instructions == "Support tier"                 # the $ref target's description
    assert list(q["refund"].criteria) == ["full", "partial", NULL_LABEL]      # optional: a null option
    assert s.description == "A support ticket, triaged." and s.extraction.source is TICKET


def test_answers_become_values():
    s = schema_from_json_schema(TICKET)
    answers = {
        "department": {"type": "choice", "choice": "sales", "probabilities": {}, "confidence": 1.0},
        "urgent": {"type": "noul", "noul": 0.8},
        "priority": {"type": "score", "score": 1.2, "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2}, "confidence": 0.5},
        "channel": {"type": "choice", "choice": "chat", "probabilities": {}, "confidence": 1.0},
        "customer.tier": {"type": "choice", "choice": "3", "probabilities": {}, "confidence": 1.0},
        "refund": {"type": "choice", "choice": "null", "probabilities": {}, "confidence": 1.0},
    }
    assert s.extraction.values(answers) == {"department": "sales", "urgent": True, "priority": 2, "channel": "chat",
                                            "customer": {"tier": 3, "kind": "person"}, "refund": None}


@pytest.mark.parametrize("prop, fragment", [
    ({"type": "string"}, "free-text string"),
    ({"type": "number"}, "free number"),
    ({"type": "array", "items": {"type": "boolean"}}, "arrays are not supported"),
    ({"type": "integer", "minimum": 0}, "needs minimum and maximum"),
    ({"type": "integer", "minimum": 0, "maximum": 20}, "21 values from 0 to 20"),
    ({"type": "integer", "minimum": 5, "maximum": 1}, "below minimum"),
    ({"enum": [f"o{i}" for i in range(256)]}, "256 options"),
    ({"enum": []}, "non-empty"),
    ({"enum": [1, "1"]}, "distinct as text"),
    ({"enum": [{"a": 1}, "b"]}, "strings, numbers, booleans or null"),
    ({"anyOf": [{"type": "boolean"}, {"type": "integer", "minimum": 0, "maximum": 2}]}, "union of different types"),
    ({"$ref": "https://example.com/s.json"}, "inside the document"),
    ({"$ref": "#/$defs/Missing"}, "does not resolve"),
    ({"$ref": "#/$defs/Node"}, "recursive"),
    ({"$ref": "#/$defs/Wrapped"}, "recursive"),                           # pydantic v1: nested models inside allOf
    ({"allOf": [{"$ref": "#/$defs/Wrapped"}]}, "recursive"),
    ({"anyOf": [{"$ref": "#/$defs/Maybe"}, {"type": "null"}]}, "recursive"),   # Optional[Self]
    ({"type": "object"}, "needs properties"),
    ({}, "give it a type"),
    ({"type": ["string", "integer"]}, "one type"),
])
def test_errors_name_the_path(prop, fragment):
    js = {"type": "object", "properties": {"field": prop},
          "$defs": {"Node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/Node"}}},
                    "Wrapped": {"type": "object", "properties": {"flag": {"type": "boolean"},
                                                                 "child": {"allOf": [{"$ref": "#/$defs/Wrapped"}]}}},
                    "Maybe": {"type": "object", "properties": {"next": {"anyOf": [{"$ref": "#/$defs/Maybe"},
                                                                                  {"type": "null"}]}}}}}
    with pytest.raises(InvalidRequest) as err:
        schema_from_json_schema(js, "my_schema")
    assert err.value.message.startswith("my_schema.properties.field") and fragment in err.value.message


def test_a_schema_nested_past_the_recursion_limit_is_an_invalid_request():
    node: dict = {"type": "boolean"}
    for _ in range(3000):
        node = {"type": "object", "properties": {"x": node}}
    with pytest.raises(InvalidRequest, match="deep: the schema is nested too deeply"):
        schema_from_json_schema({"type": "object", "properties": {"root": node}}, "deep")
    node = {"type": "boolean"}
    for _ in range(40):                                                      # deep, but fine
        node = {"allOf": [{"type": "object", "properties": {"x": node}}]}
    assert list(schema_from_json_schema({"properties": {"root": node}}).questions) == ["root" + ".x" * 40]


def test_shared_definitions_are_not_recursion():
    js = {"properties": {"billing": {"$ref": "#/$defs/Addr"}, "shipping": {"allOf": [{"$ref": "#/$defs/Addr"}]},
                         "either": {"anyOf": [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/B"}]},
                         "maybe": {"anyOf": [{"$ref": "#/$defs/Addr"}, {"type": "null"}]}},
          "$defs": {"Addr": {"type": "object", "properties": {"country": {"enum": ["PT", "ES"]}}},
                    "A": {"const": "a", "description": "the first"}, "B": {"const": "b"}}}
    s = schema_from_json_schema(js)
    assert list(s.questions) == ["billing.country", "shipping.country", "either", "maybe.country"]
    assert s.questions["either"].criteria == {"a": "the first", "b": None}


OPTIONAL_ADDRESS = {
    "$defs": {"Address": {"type": "object", "properties": {
                  "country": {"enum": ["PT", "ES"]},
                  "verified": {"type": "boolean"},
                  "floor": {"type": "integer", "minimum": 0, "maximum": 3},
                  "kind": {"const": "postal"},
                  "geo": {"type": "object", "properties": {"zone": {"enum": ["north", "south"]}}},
                  "note": {"anyOf": [{"$ref": "#/$defs/Note"}, {"type": "null"}]}}},
              "Note": {"type": "object", "properties": {"tone": {"enum": ["calm", "angry"]}}}},
    "type": "object",
    "properties": {"urgent": {"type": "boolean"},
                   "address": {"anyOf": [{"$ref": "#/$defs/Address"}, {"type": "null"}], "default": None},
                   "maybe": {"type": ["object", "null"], "properties": {"x": {"type": "boolean"}}}},
}


def test_an_optional_object_is_none_when_none_of_its_fields_has_a_value():
    s = schema_from_json_schema(OPTIONAL_ADDRESS)
    q = s.questions
    assert list(q["address.country"].criteria) == ["PT", "ES", "null"]     # every field inside can say "not there"
    assert list(q["address.verified"].criteria) == ["true", "false", "null"]
    assert q["address.floor"].type == "choice" and list(q["address.floor"].criteria) == ["0", "1", "2", "3", "null"]
    assert list(q["address.geo.zone"].criteria) == ["north", "south", "null"]
    assert list(q["address.note.tone"].criteria) == ["calm", "angry", "null"]
    assert list(q["maybe.x"].criteria) == ["true", "false", "null"]
    assert q["urgent"].type == "noul"                                        # required fields are unchanged

    def said(label):
        return {"type": "choice", "choice": label}
    nothing = {qid: said("null") for qid in q if qid != "urgent"}
    assert s.extraction.values({"urgent": {"type": "noul", "noul": 0.9}, **nothing}) == \
        {"urgent": True, "address": None, "maybe": None}                     # not fabricated: no consts, no fields
    assert s.extraction.values({}) == {"address": None, "maybe": None}
    some = s.extraction.values({**nothing, "address.country": said("PT"), "address.floor": said("2")})
    assert some["address"] == {"country": "PT", "verified": None, "floor": 2, "kind": "postal", "geo": {"zone": None},
                               "note": None} and some["maybe"] is None
    with pytest.raises(InvalidRequest, match="js.properties.meta: an optional object needs a field to decide"):
        schema_from_json_schema({"properties": {"ok": {"type": "boolean"}, "meta": {
            "anyOf": [{"type": "object", "properties": {"v": {"const": 1}}}, {"type": "null"}]}}}, "js")


def test_extract_an_optional_nested_model():
    pydantic = pytest.importorskip("pydantic")
    if not hasattr(pydantic.BaseModel, "model_json_schema"):
        pytest.skip("pydantic v1")
    from pydantic import BaseModel

    class Address(BaseModel):
        country: Literal["PT", "ES"]

    class Ticket(BaseModel):
        urgent: bool
        address: Optional[Address] = None

    assert list(Schema.from_pydantic(Ticket).questions["address.country"].criteria) == ["PT", "ES", "null"]
    last = Tez(backend=FakeBackend(letters_fn=lambda prompt, k: [0.0] * (k - 1) + [5.0]))     # "null" wins
    assert last.extract("Please call me back, it's urgent.", Ticket).address is None
    first = Tez(backend=FakeBackend(letters_fn=fixed({3: 0})))
    assert first.extract("Ship it to Lisbon, urgently.", Ticket).address == Address(country="PT")


def fan_out(depth: int, leaf: dict | None = None, fan: int = 10) -> dict:
    """A small schema that expands to fan ** (depth + 1) leaves: each level has `fan` properties that $ref the next."""
    defs = {f"L{depth}": leaf or {"type": "boolean"}}
    for i in range(depth - 1, -1, -1):
        defs[f"L{i}"] = {"type": "object", "properties": {f"p{j}": {"$ref": f"#/$defs/L{i + 1}"} for j in range(fan)}}
    return {"type": "object", "$defs": defs, "properties": {f"r{j}": {"$ref": "#/$defs/L0"} for j in range(fan)}}


@pytest.fixture
def parsed(monkeypatch):
    """Counts the questions the JSON schema conversion builds."""
    import tez.extract as extract
    calls = []
    real = extract.parse_question

    def counting(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)
    monkeypatch.setattr(extract, "parse_question", counting)
    return calls


def test_the_question_budget_is_counted_while_a_json_schema_is_expanded(parsed):
    js = fan_out(3)                                          # 10,000 questions from a 1.4 KB schema
    tez = Tez(backend="fake")
    runs = {"systemone": lambda: tez.handle({"state": "x", "json_schema": js}, limits=Limits()),
            "plan": lambda: tez.plan({"json_schema": js}, Limits()),
            "batch": lambda: tez.handle_batch({"states": ["x", "y"], "json_schema": js}, limits=Limits()),
            "budget of 5": lambda: schema_from_json_schema(js, "js", max_questions=5)}
    for what, run in runs.items():
        parsed.clear()
        with pytest.raises(PayloadTooLarge, match="too many questions: more than"):
            run()
        assert len(parsed) <= (5 if what == "budget of 5" else 64), what        # stopped at the first one past it
    assert len(schema_from_json_schema(fan_out(1), max_questions=0).questions) == 100    # 0: no budget
    c = TestClient(create_app(Tez(backend="fake")))
    for path, body in (("/v1/systemone", {"state": "x", "json_schema": js}), ("/v1/plan", {"json_schema": js}),
                       ("/v1/systemone/batch", {"states": ["x"], "json_schema": js})):
        parsed.clear()
        r = c.post(path, json=body)
        assert r.status_code == 413 and r.json()["error"]["type"] == "payload_too_large", path
        assert len(parsed) <= 64, path


def test_ref_expansion_is_capped_whatever_the_question_budget(parsed):
    few_questions = {**fan_out(4, leaf={"const": 1}), "properties": {"ask": {"type": "boolean"},
                                                                     "tree": {"$ref": "#/$defs/L0"}}}
    with pytest.raises(PayloadTooLarge, match=r"tree.*\$refs expand past 10,000 references"):
        schema_from_json_schema(few_questions)                               # 10,000 consts, one question
    big = {"$defs": {"Big": {"const": "x" * 200_000}},
           "properties": {"ask": {"type": "boolean"}, **{f"c{i}": {"$ref": "#/$defs/Big"} for i in range(10)}}}
    with pytest.raises(PayloadTooLarge, match="1,000,000 characters of schema"):
        schema_from_json_schema(big)
    looped: dict = {"type": "boolean"}
    looped["self"] = looped                                  # a Python object that contains itself is measured once
    s = schema_from_json_schema({"$defs": {"B": looped}, "properties": {"a": {"$ref": "#/$defs/B"}}})
    assert list(s.questions) == ["a"]
    assert len(schema_from_json_schema(fan_out(2)).questions) == 1000           # within the caps: fine


def test_top_level_errors():
    for bad in ([], {"type": "array"}, {"type": "object", "properties": {}}):
        with pytest.raises(InvalidRequest, match="object with properties|must be an object"):
            schema_from_json_schema(bad, "js")
    with pytest.raises(InvalidRequest, match="nothing to decide"):
        schema_from_json_schema({"properties": {"k": {"const": 1}}}, "js")


def test_other_shapes():
    s = schema_from_json_schema({"properties": {
        "flag": {"type": ["boolean", "null"]},
        "yes": {"enum": [True, False]},
        "level": {"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 4},
        "old": {"type": "integer", "minimum": 0, "maximum": 3, "exclusiveMaximum": True},
        "one": {"type": "integer", "minimum": 7, "maximum": 7},
        "wrapped": {"allOf": [{"enum": ["a", "b"]}], "description": "Pick"},
        "maybe": {"type": ["string", "null"], "enum": ["x", "y", None]},
    }})
    q = s.questions
    assert list(q["flag"].criteria) == ["true", "false", "null"] and q["yes"].type == "noul"
    assert q["level"].criteria == ["level = 1", "level = 2", "level = 3"]
    assert q["old"].criteria == ["old = 0", "old = 1", "old = 2"]
    assert "one" not in q and s.extraction.values({}) == {"one": 7}
    assert q["wrapped"].instructions == "Pick" and list(q["maybe"].criteria) == ["x", "y", "null"]
    assert s.extraction.values({"flag": {"type": "choice", "choice": "false"}})["flag"] is False


# ---------------------------------------------------------------------------------------------- Tez.extract
def test_extract_with_a_json_schema():
    tez = Tez(backend=FakeBackend(letters_fn=fixed({3: 2, 2: 1, 4: 0})))
    values = tez.extract("I was charged twice", TICKET)
    assert values == {"department": "sales", "urgent": True, "priority": 3, "channel": "chat",
                      "customer": {"tier": 3, "kind": "person"}, "refund": None}
    details = tez.extract("I was charged twice", TICKET, return_details=True)
    assert isinstance(details, ExtractResult) and details.value == values and details.escalated == []
    assert details.response["tez"]["values"] == values                       # the wire carries them too


def test_extract_gate():
    tez = Tez(backend="fake")
    with pytest.raises(EscalationRequired) as err:                            # nothing is fitted: the gate escalates
        tez.extract("I was charged twice", TICKET, alpha=0.1)
    assert set(err.value.fields) == set(schema_from_json_schema(TICKET).questions)
    assert err.value.values["customer"]["kind"] == "person" and "answers" in err.value.response
    details = tez.extract("I was charged twice", TICKET, alpha=0.1, return_details=True)
    assert details.decisions and set(details.escalated) == set(details.decisions)


def test_extract_with_a_schema_name_or_object(schema_dir):
    tez = Tez(backend=FakeBackend(letters_fn=fixed({4: 1, 2: 1, 3: 2})), schemas=schema_dir)
    values = tez.extract("The app crashes", "support-triage")
    assert values == {"topic": "technical", "is_urgent": True, "anger": 2}
    assert default_extraction(tez.schemas["support-triage"]).fields["anger"].kind == "score"
    made = Schema.from_json_schema({"properties": {"ok": {"type": "boolean"}}}, "ok")
    assert tez.extract("x", made) == {"ok": True}
    with pytest.raises(InvalidRequest, match="unknown schema 'nope'"):
        tez.extract("x", "nope")
    with pytest.raises(TypeError, match="extract takes"):
        tez.extract("x", 42)


def test_extract_with_pydantic():
    pydantic = pytest.importorskip("pydantic")
    if not hasattr(pydantic.BaseModel, "model_json_schema"):
        pytest.skip("pydantic v1")
    from pydantic import BaseModel, Field

    class Tier(Enum):
        free = "free"
        pro = "pro"

    class Customer(BaseModel):
        tier: Tier = Field(description="Which plan is the customer on?")

    class Ticket(BaseModel):
        department: Literal["billing", "technical", "sales"] = Field(description="Which team should handle it?")
        urgent: bool
        priority: int = Field(ge=1, le=3)
        customer: Customer
        refund: Optional[Literal["full", "partial"]] = None

    s = Schema.from_pydantic(Ticket)
    assert s.name == "Ticket" and list(s.questions) == ["department", "urgent", "priority", "customer.tier", "refund"]
    assert s.questions["customer.tier"].instructions == "Which plan is the customer on?"
    tez = Tez(backend=FakeBackend(letters_fn=fixed({3: 0, 2: 1})))
    t = tez.extract("I was charged twice", Ticket)
    assert isinstance(t, Ticket) and t.department == "billing" and t.urgent is True and t.priority == 1
    assert t.customer.tier is Tier.pro and t.refund == "full"
    with pytest.raises(InvalidRequest, match="Bad.properties.note: a free-text string"):
        class Bad(BaseModel):
            note: str
        Schema.from_pydantic(Bad)
    with pytest.raises(TypeError, match="pydantic model class"):
        Schema.from_pydantic(dict)


def test_import_tez_does_not_import_optional_sdks():
    """pydantic stays optional for extraction, and the MCP SDK and OpenTelemetry load only when used."""
    code = ("import sys, tez, tez.extract, tez.hooks, tez.mcp_server, tez.state, tez.presets; "
            "print(sorted({m.split('.')[0] for m in sys.modules} & {'pydantic', 'mcp', 'opentelemetry', 'langchain_core'}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


# ---------------------------------------------------------------------------------------------- the wire
def test_json_schema_on_the_wire():
    c = TestClient(create_app(Tez(backend=FakeBackend(letters_fn=fixed({3: 2, 2: 1, 4: 0})))))
    r = c.post("/v1/systemone", json={"state": "I was charged twice", "json_schema": TICKET})
    assert r.status_code == 200
    body = r.json()
    assert body["tez"]["values"]["department"] == "sales" and body["answers"]["customer.tier"]["choice"] == "3"
    r = c.post("/v1/systemone", json={"state": "x", "json_schema": TICKET, "questions": {"q": {"type": "noul", "instructions": "x"}}})
    assert r.status_code == 422 and "not both" in r.json()["error"]["message"]
    r = c.post("/v1/systemone", json={"state": "x", "json_schema": {"properties": {"note": {"type": "string"}}}})
    assert r.status_code == 422 and r.json()["error"]["message"].startswith("json_schema.properties.note:")
    plain = c.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "noul", "instructions": "x"}}})
    assert "values" not in plain.json()["tez"]
    batch = c.post("/v1/systemone/batch", json={"states": ["a", "b"], "json_schema": TICKET}).json()
    assert [r["tez"]["values"]["department"] for r in batch["results"]] == ["sales", "sales"]


def test_remote_extract(schema_dir):
    engine = Tez(backend=FakeBackend(letters_fn=fixed({3: 2, 2: 1, 4: 1})), schemas=schema_dir)
    with StubServer(engine) as srv:
        remote = RemoteTez(srv.url)
        assert remote.extract("I was charged twice", TICKET) == engine.extract("I was charged twice", TICKET)
        assert remote.extract("The app crashes", "support-triage") == {"topic": "technical", "is_urgent": True, "anger": 2}
    sent = [r["body"] for r in srv.requests if r["path"] == "/v1/systemone"]
    assert sent[0]["json_schema"] == TICKET and "questions" not in sent[0]
    assert sent[1] == {"state": "The app crashes", "schema": "support-triage"}
