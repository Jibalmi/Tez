"""tez plan / POST /v1/plan (what a request would do, no backend call) and tez doctor (against a stub llama-server)."""
from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import DOCS_REQUEST, SYNTH_SCHEMA_YAML, synth_rows, write_jsonl
from stub_llama import QWEN_TEMPLATE, StubLlama
from tez import FakeBackend, LlamaCppBackend, Tez
from tez.cli import main
from tez.doctor import detect_template, run_doctor
from tez.fit import fit
from tez.prompt import fingerprint
from tez.server import create_app

QUESTIONS = DOCS_REQUEST["questions"]
STATE = DOCS_REQUEST["state"]


def by_name(checks):
    return {c.name: c for c in checks}


# ---------------------------------------------------------------------------------------------- plan
def test_plan_of_a_state_first_request():
    tez = Tez(backend="fake")
    plan = tez.plan({"state": STATE, "questions": QUESTIONS})
    assert plan["layout"] == "state_first" and plan["requested_layout"] == "auto" and plan["order"] == list(QUESTIONS)
    first, second = plan["questions"]["is_urgent"], plan["questions"]["topic"]
    assert first["readout"] == "letters" and first["calls"] == {"letters": 1, "embed": 0} and first["layout"] == "state_first"
    assert first["fit"] == {"status": "none", "reason": "no schema named: questions are read zero-shot"}
    assert first["cached_tokens"] == 0 and second["cached_tokens"] >= plan["state_tokens"]    # the state is cached
    q = tez.parse_request({"state": STATE, "questions": QUESTIONS}).questions["topic"]
    assert second["prompt_sha"] == fingerprint(q, "gemma4", q.options(), [], "state_first") and second["options"] == 3
    t = plan["totals"]
    assert t["calls"] == {"letters": 3, "embed": 0} and t["evaluated_tokens"] == t["prompt_tokens"] - t["cached_tokens"]
    assert any("state first" in n for n in plan["notes"])
    one = tez.plan({"questions": {"q": QUESTIONS["topic"]}})                     # no state: estimates leave it out
    assert one["layout"] == "question_first" and one["state_tokens"] is None


def test_plan_shows_fits_and_their_layout(tmp_path: Path):
    (tmp_path / "synth.yaml").write_text(SYNTH_SCHEMA_YAML, encoding="utf-8")
    tez = Tez(backend=FakeBackend(), schemas=tmp_path)
    fit(tez, tez.schemas["synth"], [write_jsonl(tmp_path / "l.jsonl", synth_rows(60, seed=3, anger_every=0))],
        say=lambda m: None)
    plan = tez.plan({"state": "invoice refund", "schema": "synth"})
    topic, anger = plan["questions"]["topic"], plan["questions"]["anger"]
    assert topic["readout"] == "probe" and topic["calls"] == {"letters": 1, "embed": 1}      # blended with the letters
    assert topic["fit"]["status"] == "ready" and topic["fit"]["fit_layout"] == "question_first" and topic["layout"] == "question_first"
    assert anger["fit"]["status"] == "none" and anger["layout"] == "state_first"
    assert plan["order"] == ["anger", "topic", "is_urgent"]                        # state-first questions run first
    stale = tez.plan({"state": "x", "schema": "synth", "tez": {"layout": "state_first", "readout": "probe"}})
    assert stale["questions"]["topic"]["fit"]["status"] == "stale"
    assert "question_first layout" in stale["questions"]["topic"]["fit"]["reason"]
    assert stale["questions"]["topic"]["readout"] is None and "no usable probe" in stale["questions"]["topic"]["error"]


def test_plan_of_a_tournament():
    q = {"type": "choice", "instructions": "Which?", "criteria": {f"o{i}": None for i in range(45)}}
    plan = Tez(backend="fake").plan({"state": "x", "questions": {"w": q}})
    assert plan["questions"]["w"]["calls"] == {"letters": 4, "embed": 0} and "tournament of 3 chunks" in plan["notes"][0]


def test_plan_never_calls_the_backend(monkeypatch):
    backend = LlamaCppBackend("http://127.0.0.1:9")

    def refuse(*a, **k):
        raise AssertionError("the plan called the backend")

    monkeypatch.setattr(backend._session, "request", refuse)
    monkeypatch.setattr(backend._session, "get", refuse)
    plan = Tez(backend=backend).plan({"state": STATE, "questions": QUESTIONS})
    assert plan["model"] is None and any("model names not checked" in n for n in plan["notes"])


def test_plan_endpoint_and_cli(tmp_path: Path, capsys):
    c = TestClient(create_app(Tez(backend="fake"), api_key="k"))
    r = c.post("/v1/plan", json={"state": STATE, "questions": QUESTIONS}, headers={"Authorization": "Bearer k"})
    assert r.status_code == 200 and r.json() == Tez(backend="fake").plan({"state": STATE, "questions": QUESTIONS})
    assert c.post("/v1/plan", json={"state": "x", "questions": QUESTIONS}).status_code == 401
    bad = c.post("/v1/plan", json={"questions": {"q": {"type": "maybe"}}}, headers={"Authorization": "Bearer k"})
    assert bad.status_code == 422
    qfile = tmp_path / "q.json"
    qfile.write_text(json.dumps(QUESTIONS), encoding="utf-8")
    assert main(["plan", STATE, "--backend", "fake", "--questions", str(qfile)]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].split()[:4] == ["question", "type", "options", "readout"] and "state_first" in out
    assert "to evaluate with the prompt cache" in out
    assert main(["plan", "--backend", "fake", "--preset", "support-triage", "--json"]) == 0
    assert set(json.loads(capsys.readouterr().out)["questions"]) == {"topic", "urgent", "frustration"}


# ---------------------------------------------------------------------------------------------- doctor
def test_detect_template():
    assert detect_template("<|turn>user") == "gemma4" and detect_template(QWEN_TEMPLATE) == "qwen3"
    assert detect_template("<start_of_turn>user") == "gemma3" and detect_template(None) is None


def test_doctor_on_a_healthy_server():
    with StubLlama() as srv:
        checks = by_name(run_doctor(srv.url, template="gemma4"))
    for name in ("health", "model", "template", "n_probs", "cache", "prefix", "embeddings", "slots"):
        assert checks[name].status == "ok", (name, checks[name])
    assert "gemma-4-12b-it-q8_0" in checks["model"].detail and "context 4096" in checks["model"].detail
    assert checks["cache-ram"].status == "info" and checks["cache-ram"].fix == "start llama-server with --cache-ram 0"


@pytest.mark.parametrize("kwargs, template, name, status, fix", [
    ({"template": QWEN_TEMPLATE}, "gemma4", "template", "fail", "tez serve --template qwen3"),
    ({"template": "<start_of_turn>x"}, "gemma4", "template", "fail", "serve Gemma 4"),
    ({"cache": "exact"}, "gemma4", "prefix", "warn", "--swa-full"),
    ({"cache": "exact", "model_path": "/m/Qwen3.5-4B-Q8_0.gguf", "template": QWEN_TEMPLATE}, "qwen3", "prefix", "warn",
     "question_first"),
    ({"cache": "off"}, "gemma4", "cache", "fail", "--no-cache-prompt"),
    ({"top": 20}, "gemma4", "n_probs", "warn", "--n-probs 20"),
    ({"embeddings": False}, "gemma4", "embeddings", "info", "--embeddings --pooling last"),
    ({"slots": 2}, "gemma4", "slots", "warn", "-np 1"),
    ({"n_ctx": 1024}, "gemma4", "context", "warn", "-c 4096"),
    ({"timings": False}, "gemma4", "cache", "info", None),
])
def test_doctor_finds_problems_and_names_the_fix(kwargs, template, name, status, fix):
    with StubLlama(**kwargs) as srv:
        checks = by_name(run_doctor(srv.url, template=template))
    assert checks[name].status == status
    if fix:
        assert fix in checks[name].fix


def test_doctor_loading_and_unreachable():
    with StubLlama(health=503) as srv:
        checks = run_doctor(srv.url)
    assert [(c.name, c.status) for c in checks] == [("health", "warn")]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    (only,) = run_doctor(f"127.0.0.1:{port}", timeout=5)
    assert (only.name, only.status) == ("health", "fail") and "--swa-full --cache-ram 0" in only.fix


def test_doctor_cli(capsys):
    with StubLlama(template=QWEN_TEMPLATE) as srv:
        assert main(["doctor", "--backend", srv.url]) == 1                         # a failed check: exit 1
        out = capsys.readouterr().out
        assert "fail  template" in out and "fix: tez serve --template qwen3" in out and "1 failed" in out
        assert main(["doctor", "--backend", srv.url, "--template", "qwen3", "--json"]) == 0
        rows = json.loads(capsys.readouterr().out)
    assert {r["name"] for r in rows} >= {"health", "template", "cache", "cache-ram"}
