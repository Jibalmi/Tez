"""Safe server defaults: the CORS allowlist, Private Network Access only for allowed origins, feedback refused from other
origins, and the TEZ_* environment the CLI reads (with VAR_FILE secrets)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tez import Tez
from tez.cli import build_parser, cors_origins, main
from tez.config import DEFAULT_CORS_ORIGINS, Limits
from tez.server import OriginPolicy, create_app

PLAYGROUND = "https://jibalmi.github.io"
PREFLIGHT = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type",
             "Access-Control-Request-Private-Network": "true"}
Q = {"q": {"type": "noul", "instructions": "x"}}


# ---------------------------------------------------------------------------------------------- the policy
@pytest.mark.parametrize("origin, allowed", [
    ("http://127.0.0.1:8000", True), ("http://127.0.0.1", True), ("http://localhost:3000", True),
    ("http://LOCALHOST:3000", True), (PLAYGROUND, True),
    ("https://jibalmi.github.io.evil.com", False), ("https://evil.jibalmi.github.io", False),
    ("http://127.0.0.1.evil.com:80", False), ("https://localhost:8000", False), ("http://localhost:8000/path", False),
    ("null", False), ("https://tez.example", False), ("", False), (None, False),
])
def test_default_policy(origin, allowed):
    assert OriginPolicy(DEFAULT_CORS_ORIGINS).allows(origin) is allowed


def test_policy_patterns():
    p = OriginPolicy(["https://*.example.com", "http://10.0.0.5:8080", "null"])
    assert p.allows("https://app.example.com") and p.allows("https://a.b.example.com") and not p.allows("https://example.com")
    assert p.allows("http://10.0.0.5:8080") and not p.allows("http://10.0.0.5:8081") and p.allows("null")
    anyone = OriginPolicy(["*"])
    assert anyone.allows("https://whatever.test") and anyone.allow_origin_value("https://whatever.test") == "*"
    assert not OriginPolicy([]).allows("http://localhost:1")
    for bad in ("localhost:8000", "https://a*b.com", "https://*.*.com", "ftp//x"):
        with pytest.raises(ValueError, match="invalid CORS origin"):
            OriginPolicy([bad])
    assert cors_origins("http://a.test, https://b.test  null,*") == ["http://a.test", "https://b.test", "null", "*"]


# ---------------------------------------------------------------------------------------------- the server
def test_other_origins_get_no_cors_and_no_private_network_access():
    c = TestClient(create_app(Tez(backend="fake")))
    r = c.post("/v1/systemone", json={"state": "s", "questions": Q}, headers={"Origin": "https://evil.test"})
    assert r.status_code == 200 and "access-control-allow-origin" not in r.headers   # the browser cannot read it
    pre = c.options("/v1/systemone", headers={"Origin": "https://evil.test", **PREFLIGHT})
    assert pre.status_code == 403 and pre.json()["error"]["type"] == "forbidden"
    assert "access-control-allow-origin" not in pre.headers and "access-control-allow-private-network" not in pre.headers
    ok = c.options("/v1/systemone", headers={"Origin": PLAYGROUND, **PREFLIGHT})
    assert ok.status_code == 200 and ok.headers["access-control-allow-private-network"] == "true"
    assert ok.headers["access-control-allow-origin"] == PLAYGROUND and ok.headers["vary"] == "Origin"
    r = c.post("/v1/systemone", json={"state": "s", "questions": Q}, headers={"Origin": "http://localhost:8000"})
    exposed = {h.strip() for h in r.headers["access-control-expose-headers"].split(",")}
    assert {"x-tez-run-id", "x-typesafe-request-id", "server-timing"} <= exposed
    assert r.headers["timing-allow-origin"] == "http://localhost:8000" and "Origin" in r.headers["vary"]


def test_star_allows_every_origin_explicitly():
    c = TestClient(create_app(Tez(backend="fake"), cors_origins=["*"]))
    r = c.get("/v1/models", headers={"Origin": "https://anywhere.test"})
    assert r.headers["access-control-allow-origin"] == "*"
    pre = c.options("/v1/systemone", headers={"Origin": "https://anywhere.test", **PREFLIGHT})
    assert pre.status_code == 200 and pre.headers["access-control-allow-private-network"] == "true"


def test_feedback_refuses_other_origins(schema_dir: Path):
    body = {"schema": "support-triage", "question": "topic", "state": "s", "label": "billing"}
    c = TestClient(create_app(Tez(backend="fake", schemas=schema_dir)))
    r = c.post("/v1/feedback", json=body, headers={"Origin": "https://evil.test"})
    assert r.status_code == 403 and r.json()["error"]["type"] == "forbidden" and "may not record feedback" in r.json()["error"]["message"]
    path = schema_dir / ".tez" / "feedback" / "support-triage.jsonl"
    assert not path.exists()
    # a "simple" request (text/plain, no preflight) from another site is refused too
    r = c.post("/v1/feedback", content=json.dumps(body), headers={"Origin": "https://evil.test", "Content-Type": "text/plain"})
    assert r.status_code == 403 and not path.exists()
    assert c.post("/v1/feedback", json=body, headers={"Origin": PLAYGROUND}).status_code == 200
    assert c.post("/v1/feedback", json=body).status_code == 200                      # curl, SDKs: no Origin
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    closed = TestClient(create_app(Tez(backend="fake", schemas=schema_dir), cors=False))
    assert closed.post("/v1/feedback", json=body, headers={"Origin": PLAYGROUND}).status_code == 403
    open_ = TestClient(create_app(Tez(backend="fake", schemas=schema_dir), cors_origins=["*"]))
    assert open_.post("/v1/feedback", json=body, headers={"Origin": "https://evil.test"}).status_code == 200


# ---------------------------------------------------------------------------------------------- the environment
def test_serve_reads_the_environment(tmp_path: Path):
    key = tmp_path / "key.txt"
    key.write_text("s3cret\n", encoding="utf-8")
    env = {"TEZ_BACKEND": "fake", "TEZ_TEMPLATE": "qwen3", "TEZ_HOST": "0.0.0.0", "TEZ_PORT": "9001",
           "TEZ_SCHEMAS": "/schemas", "TEZ_DATA_DIR": "/data", "TEZ_API_KEY_FILE": str(key), "TEZ_LOG_LEVEL": "WARNING",
           "TEZ_CORS_ORIGINS": "https://app.test", "TEZ_PRESETS": "1", "TEZ_LAYOUT": "state_first"}
    a = build_parser(env).parse_args(["serve"])
    assert (a.backend, a.template, a.host, a.port, a.schemas, a.data_dir) == ("fake", "qwen3", "0.0.0.0", 9001, "/schemas", "/data")
    assert (a.api_key, a.log_level, a.cors_origins, a.presets, a.layout) == ("s3cret", "warning", "https://app.test", True, "state_first")
    flags = build_parser(env).parse_args(["serve", "--port", "7000", "--host", "127.0.0.1", "--layout", "auto"])
    assert (flags.port, flags.host, flags.layout) == (7000, "127.0.0.1", "auto")           # flags win
    d = build_parser({}).parse_args(["serve"])
    assert (d.host, d.port, d.api_key, d.presets, d.layout, d.log_level) == ("127.0.0.1", 8787, None, False, None, "info")
    assert d.cors_origins == ",".join(DEFAULT_CORS_ORIGINS)
    for bad in ({"TEZ_LOG_LEVEL": "loud"}, {"TEZ_TEMPLATE": "llama"}, {"TEZ_LAYOUT": "sideways"}, {"TEZ_PORT": "http"}):
        with pytest.raises(SystemExit):
            build_parser(bad).parse_args(["serve"])
    assert build_parser({"TEZ_BACKEND": "fake"}).parse_args(["decide", "x", "--questions", "q.json"]).backend == "fake"


def test_bad_secret_file_is_a_clean_error(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("TEZ_API_KEY_FILE", str(tmp_path / "missing"))
    assert main(["serve"]) == 2
    assert "cannot read TEZ_API_KEY_FILE" in capsys.readouterr().err


def test_cmd_serve_builds_the_app_from_the_environment(tmp_path: Path, monkeypatch, schema_dir: Path):
    seen = {}

    def fake_run(app, host, port, log_level):
        seen.update(app=app, host=host, port=port, log_level=log_level)

    import uvicorn
    monkeypatch.setattr(uvicorn, "run", fake_run)
    for k, v in {"TEZ_BACKEND": "fake", "TEZ_SCHEMAS": str(schema_dir), "TEZ_PRESETS": "yes", "TEZ_API_KEY": "k",
                 "TEZ_CORS_ORIGINS": "https://app.test", "TEZ_PORT": "9123"}.items():
        monkeypatch.setenv(k, v)
    assert main(["serve", "--max-questions", "2"]) == 0
    assert (seen["host"], seen["port"], seen["log_level"]) == ("127.0.0.1", 9123, "info")
    c = TestClient(seen["app"])
    auth = {"Authorization": "Bearer k"}
    names = {s["name"]: s["builtin"] for s in c.get("/v1/schemas", headers=auth).json()["schemas"]}
    assert names["support-triage"] is False and names["topic-news"] is True
    assert c.get("/v1/models", headers={"Origin": "https://app.test", **auth}).headers["access-control-allow-origin"] == "https://app.test"
    assert "access-control-allow-origin" not in c.get("/v1/models", headers={"Origin": PLAYGROUND, **auth}).headers
    r = c.post("/v1/systemone", json={"state": "s", "questions": {**Q, "r": Q["q"], "t": Q["q"]}}, headers=auth)
    assert r.status_code == 413 and seen["app"].state.limits == Limits(max_questions=2)
    monkeypatch.setenv("TEZ_CORS_ORIGINS", "not an origin")
    assert main(["serve"]) == 2
