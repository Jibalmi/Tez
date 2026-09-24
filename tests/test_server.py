"""Every endpoint of docs/API.md through FastAPI's TestClient, CORS included."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import SYNTH_SCHEMA_YAML, synth_rows, write_jsonl
from tez import FakeBackend, Tez, __version__
from tez.fit import fit
from tez.server import create_app

ORIGIN = {"Origin": "http://localhost:5173"}           # allowed by the default CORS list (http://localhost:*)


@pytest.fixture
def client(schema_dir: Path) -> TestClient:
    return TestClient(create_app(Tez(backend="fake", schemas=schema_dir)))


@pytest.fixture
def fitted_client(tmp_path: Path) -> TestClient:
    d = tmp_path / "schemas"
    d.mkdir()
    (d / "synth.yaml").write_text(SYNTH_SCHEMA_YAML, encoding="utf-8")
    tez = Tez(backend=FakeBackend(), schemas=d)
    fit(tez, tez.schemas["synth"], [write_jsonl(tmp_path / "l.jsonl", synth_rows(90, seed=4, anger_every=0))], say=lambda m: None)
    return TestClient(create_app(tez))


def test_systemone_docs_example(client, docs_request):
    r = client.post("/v1/systemone", json=docs_request)
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == f"tez-{__version__} (fake, letters)"
    assert set(body["answers"]) == {"is_urgent", "topic", "anger"}
    assert set(body["answers"]["is_urgent"]) == {"type", "noul"}
    assert set(body["answers"]["topic"]) == {"type", "choice", "probabilities", "confidence"}
    assert set(body["answers"]["anger"]) == {"type", "score", "legend", "probabilities", "confidence"}
    assert body["usage"]["output_tokens"] == 0 and body["usage"]["input_tokens"] > 0
    assert body["tez"]["questions"]["topic"] == {"readout": "letters"}


def test_systemone_with_schema_only(client):
    r = client.post("/v1/systemone", json={"state": "The app crashes", "schema": "support-triage", "tez": {"gate": None}})
    assert r.status_code == 200
    assert list(r.json()["answers"]) == ["topic", "is_urgent", "anger"]


def test_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json() == {"models": [{"name": "tez-latest", "description": "Tez local decision engine (fake, letters)",
                                    "release_date": "2026-09-24"}]}


def test_healthz(client):
    body = client.get("/healthz").json()
    for key, value in {"status": "ok", "version": __version__, "backend": "fake", "template": "gemma4",
                       "embed_backend": None, "schemas": ["support-triage"], "probes": {"support-triage": []}}.items():
        assert body[key] == value


def test_schemas_list_and_detail(client):
    listed = client.get("/v1/schemas").json()
    assert listed["schemas"][0]["name"] == "support-triage"
    assert listed["schemas"][0]["questions"] == {"topic": "choice", "is_urgent": "noul", "anger": "score"}
    detail = client.get("/v1/schemas/support-triage").json()
    assert detail["questions"]["anger"] == {"type": "score", "instructions": "How upset is the writer?", "criteria": ["Calm", "Frustrated", "Very angry"]}
    assert detail["gate"] == {"alpha": 0.05} and detail["examples"] == 1
    assert detail["probes"]["topic"]["probe"] == "none" and detail["calibration"] is None
    r = client.get("/v1/schemas/nope")
    assert r.status_code == 404 and r.json() == {"error": {"type": "not_found", "message": "unknown schema 'nope'"}}


def test_fitted_schema_endpoints_and_gate(fitted_client):
    health = fitted_client.get("/healthz").json()
    assert health["probes"] == {"synth": ["topic", "is_urgent"]}
    assert "letters+probe" in fitted_client.get("/v1/models").json()["models"][0]["description"]
    detail = fitted_client.get("/v1/schemas/synth").json()
    assert detail["probes"]["topic"]["probe"] == "ready" and detail["calibration_id"].startswith("synth@")
    assert detail["calibration"]["questions"]["topic"]["probe"]["n_train"] == 63
    assert "train_hashes" not in detail["manifest"]
    r = fitted_client.post("/v1/systemone", json={"state": "invoice refund charged thanks", "schema": "synth",
                                                  "tez": {"readout": "auto", "gate": {"alpha": 0.1}}})
    metas = r.json()["tez"]["questions"]
    assert metas["topic"]["readout"] == "probe" and metas["topic"]["decision"] in ("act", "escalate")
    assert metas["topic"]["calibration_id"] == detail["calibration_id"] and 0 <= metas["topic"]["p_correct"] <= 1
    assert metas["anger"] == {"readout": "letters", "decision": "escalate"}      # never labelled: nothing certifies it


def test_feedback(client, schema_dir):
    r = client.post("/v1/feedback", json={"schema": "support-triage", "question": "is_urgent", "state": "Need it by 5pm", "label": True})
    assert r.status_code == 200 and r.json() == {"ok": True, "schema": "support-triage", "question": "is_urgent", "label": "true"}
    client.post("/v1/feedback", json={"schema": "support-triage", "question": "topic", "state": {"text": "hi"}, "label": "__none__"})
    rows = [json.loads(line) for line in (schema_dir / ".tez" / "feedback" / "support-triage.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(x["question"], x["label"]) for x in rows] == [("is_urgent", "true"), ("topic", "__none__")]
    assert rows[1]["state"] == {"text": "hi"} and "ts" in rows[0]
    for bad, fragment in [({"schema": "zzz", "question": "topic", "state": "s", "label": "billing"}, "unknown schema"),
                          ({"schema": "support-triage", "question": "zzz", "state": "s", "label": "billing"}, "question must be"),
                          ({"schema": "support-triage", "question": "topic", "state": "s", "label": "refunds"}, "not an option"),
                          ({"schema": "support-triage", "question": "anger", "state": "s", "label": 7}, "level from 0 to 2"),
                          ({"schema": "support-triage", "question": "anger", "state": "s"}, "label is required"),
                          ({"schema": "support-triage", "question": "anger", "label": 1}, "state is required")]:
        r = client.post("/v1/feedback", json=bad)
        assert r.status_code == 422 and r.json()["error"]["type"] == "invalid_request" and fragment in r.json()["error"]["message"]


@pytest.mark.parametrize("payload, fragment", [
    (b"{not json", "not valid JSON"),
    (b"", "must be a JSON object"),
    (b"[1, 2]", "must be a JSON object"),
    (json.dumps({"questions": {"q": {"type": "noul", "instructions": "x"}}}).encode(), "state is required"),
    (json.dumps({"state": "s", "questions": {"q": {"type": "maybe", "instructions": "x"}}}).encode(), "type must be one of"),
    (json.dumps({"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": None}}}}).encode(), "2 to 255"),
    (json.dumps({"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}, "tez": {"readout": "probe"}}).encode(), "no usable probe"),
    (json.dumps({"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}, "tez": {"gate": {"alpha": 2}}}).encode(), "tez.gate.alpha"),
])
def test_systemone_422(client, payload, fragment):
    r = client.post("/v1/systemone", content=payload, headers={"content-type": "application/json", **ORIGIN})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["type"] == "invalid_request" and fragment in err["message"]
    assert r.headers["access-control-allow-origin"] == ORIGIN["Origin"]       # errors carry CORS headers too


def test_backend_down_is_503(docs_request):
    c = TestClient(create_app(Tez(backend=FakeBackend(fail=True))))
    r = c.post("/v1/systemone", json=docs_request)
    assert r.status_code == 503
    assert r.json() == {"error": {"type": "backend_unavailable", "message": "fake backend is set to fail"}}
    assert c.get("/healthz").json()["status"] == "degraded"


def test_api_key():
    c = TestClient(create_app(Tez(backend="fake"), api_key="s3cret"))
    body = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}}
    r = c.post("/v1/systemone", json=body)
    assert r.status_code == 401 and r.json()["error"]["type"] == "unauthorized"
    assert c.post("/v1/systemone", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.post("/v1/systemone", json=body, headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert c.get("/v1/models", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert c.get("/v1/models").status_code == 401
    assert c.get("/healthz").status_code == 200                                # liveness stays open
    pre = c.options("/v1/systemone", headers={**ORIGIN, "Access-Control-Request-Method": "POST",
                                              "Access-Control-Request-Headers": "authorization,content-type"})
    assert pre.status_code == 200                                              # preflight never needs the key


def test_any_bearer_accepted_without_key(client, docs_request):
    assert client.post("/v1/systemone", json=docs_request, headers={"Authorization": "Bearer anything"}).status_code == 200


def test_cors_simple_and_preflight(client):
    r = client.get("/v1/models", headers=ORIGIN)
    assert r.headers["access-control-allow-origin"] == ORIGIN["Origin"]
    pre = client.options("/v1/systemone", headers={**ORIGIN, "Access-Control-Request-Method": "POST",
                                                   "Access-Control-Request-Headers": "content-type,authorization",
                                                   "Access-Control-Request-Private-Network": "true"})
    assert pre.status_code == 200
    assert pre.headers["access-control-allow-origin"] == ORIGIN["Origin"]
    assert "POST" in pre.headers["access-control-allow-methods"]
    assert "authorization" in pre.headers["access-control-allow-headers"].lower()
    assert pre.headers["access-control-allow-private-network"] == "true"


PLAYGROUND = "https://jibalmi.github.io"


@pytest.mark.parametrize("path", ["/v1/systemone", "/v1/models", "/healthz", "/v1/schemas", "/v1/feedback"])
def test_playground_private_network_preflight(path):
    """The website playground (a public https page) calling http://127.0.0.1:8787: Chrome's Private Network Access
    preflight must be answered with Access-Control-Allow-Private-Network: true, with or without an API key."""
    for app in (create_app(Tez(backend="fake")), create_app(Tez(backend="fake"), api_key="k")):
        r = TestClient(app).options(path, headers={
            "Origin": PLAYGROUND, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
            "Access-Control-Request-Private-Network": "true"})
        assert r.status_code == 200
        assert r.headers["access-control-allow-private-network"] == "true"
        assert r.headers["access-control-allow-origin"] == PLAYGROUND
        methods = {m.strip() for m in r.headers["access-control-allow-methods"].split(",")}
        assert {"GET", "POST", "OPTIONS"} <= methods
        allowed = {h.strip().lower() for h in r.headers["access-control-allow-headers"].split(",")}
        assert {"authorization", "content-type"} <= allowed


def test_every_preflight_allows_private_network(client):
    r = client.options("/v1/systemone", headers={"Origin": PLAYGROUND, "Access-Control-Request-Method": "POST"})
    assert r.status_code == 200 and r.headers["access-control-allow-private-network"] == "true"
    assert r.headers.get_list("access-control-allow-private-network") == ["true"]     # exactly once
    r = client.post("/v1/systemone", json={"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}},
                    headers={"Origin": PLAYGROUND})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == PLAYGROUND


def test_cors_can_be_switched_off(docs_request):
    c = TestClient(create_app(Tez(backend="fake"), cors=False))
    assert "access-control-allow-origin" not in c.get("/v1/models", headers=ORIGIN).headers


def test_unknown_route_and_method(client):
    r = client.get("/v2/nothing")
    assert r.status_code == 404 and r.json()["error"]["type"] == "not_found"
    r = client.get("/v1/systemone")
    assert r.status_code == 405 and r.json()["error"]["type"] == "method_not_allowed"
    assert client.get("/").json()["name"] == "tez"
