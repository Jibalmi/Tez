"""RemoteTez: the engine's API over HTTP against a stub wire-format server (no LangChain needed)."""
from __future__ import annotations

import socket
import time

import pytest

from conftest import DOCS_REQUEST
from stub_http import StubServer
from tez import FakeBackend, Tez
from tez.errors import BackendUnavailable, InvalidRequest, NotFound, TezError, Unauthorized
from tez.integrations import RemoteTez


def scripted(prompt: str, k: int) -> list[float]:
    return [float(k - i) for i in range(k)]          # always the first option, deterministic


@pytest.fixture
def engine(schema_dir):
    return Tez(backend=FakeBackend(letters_fn=scripted), schemas=schema_dir)


def test_decide_matches_the_in_process_engine(engine, docs_request):
    with StubServer(engine) as srv:
        remote = RemoteTez(srv.url).decide(docs_request["state"], questions=docs_request["questions"])
    local = engine.decide(docs_request["state"], questions=docs_request["questions"])
    assert remote["answers"] == local["answers"]
    assert remote["model"] == local["model"] and remote["usage"] == local["usage"]
    assert set(remote["tez"]) == {"latency_ms", "questions"}


def test_request_body_is_minimal_and_extensions_only_when_set(engine):
    q = {"q": {"type": "noul", "instructions": "Is it?"}}
    with StubServer(engine) as srv:
        client = RemoteTez(srv.url)
        client.decide("s", questions=q)
        client.decide("s", schema="support-triage", readout="letters", abstain=True, alpha=0.1, model="tez-latest")
        client.decide("s", schema="support-triage", gate=False)
    bodies = [r["body"] for r in srv.requests]
    assert bodies[0] == {"state": "s", "questions": q}                       # a plain Jev request
    assert bodies[1] == {"state": "s", "model": "tez-latest", "schema": "support-triage",
                         "tez": {"readout": "letters", "abstain": True, "gate": {"alpha": 0.1}}}
    assert bodies[2] == {"state": "s", "schema": "support-triage", "tez": {"gate": False}}
    assert all(r["path"] == "/v1/systemone" and r["headers"]["content-type"] == "application/json" for r in srv.requests)
    assert "authorization" not in srv.requests[0]["headers"]


def test_other_endpoints(engine):
    with StubServer(engine) as srv:
        client = RemoteTez(srv.url + "/")                                    # trailing slash is fine
        assert client.health()["status"] == "ok"
        assert client.models()["models"][0]["name"] == "tez-latest"
        assert client.schema_summaries()["schemas"][0]["name"] == "support-triage"
        detail = client.schema_detail("support-triage")
        assert detail["questions"]["topic"]["type"] == "choice"
        fb = client.record_feedback({"schema": "support-triage", "question": "topic", "state": "x", "label": "billing"})
        assert fb == {"ok": True, "schema": "support-triage", "question": "topic", "label": "billing"}
        with pytest.raises(NotFound) as e:
            client.schema_detail("a b/c")
    assert srv.requests[-1]["path"] == "/v1/schemas/a%20b%2Fc"               # one path segment, encoded
    assert e.value.status == 404 and e.value.type == "not_found" and "a b/c" in e.value.message


def test_api_key_is_sent_as_bearer(engine):
    with StubServer(engine, api_key="s3cret") as srv:
        with pytest.raises(Unauthorized) as e:
            RemoteTez(srv.url).models()
        assert e.value.status == 401
        assert RemoteTez(srv.url, api_key="s3cret").models()["models"]
        assert repr(RemoteTez(srv.url, api_key="s3cret")).endswith("api_key=***)")
    assert srv.requests[-1]["headers"]["authorization"] == "Bearer s3cret"


def test_error_bodies_map_to_engine_errors(engine):
    routes = {
        ("GET", "/v1/models"): lambda b, h: (503, {"error": {"type": "backend_unavailable", "message": "loading"}}),
        ("GET", "/healthz"): lambda b, h: (502, b"<html>bad gateway</html>"),
        ("GET", "/v1/schemas"): lambda b, h: (405, {"error": {"type": "method_not_allowed", "message": "nope"}}),
        ("POST", "/v1/feedback"): lambda b, h: (422, {"detail": "field required"}),
        ("GET", "/v1/schemas/x"): lambda b, h: (200, b"not json"),
    }
    with StubServer(engine, routes=routes) as srv:
        client = RemoteTez(srv.url)
        with pytest.raises(InvalidRequest) as e:
            client.decide("s", questions={"q": {"type": "maybe", "instructions": "x"}})
        assert e.value.status == 422 and "type must be one of" in e.value.message
        with pytest.raises(BackendUnavailable, match="loading") as e:
            client.models()
        assert e.value.status == 503
        with pytest.raises(BackendUnavailable, match="bad gateway") as e:
            client.health()
        assert e.value.status == 502
        with pytest.raises(TezError) as e:
            client.schema_summaries()
        assert (e.value.status, e.value.type, e.value.message) == (405, "method_not_allowed", "nope")
        with pytest.raises(InvalidRequest, match="field required"):
            client.record_feedback({"schema": "s"})
        with pytest.raises(BackendUnavailable) as e:
            client.schema_detail("x")
        assert e.value.type == "invalid_response"


def test_same_origin_redirect_is_followed_with_its_body(engine, docs_request):
    with StubServer(engine) as srv:
        srv.routes[("POST", "/old/v1/systemone")] = lambda b, h: (308, b"", {"Location": "/v1/systemone"})
        res = RemoteTez(srv.url + "/old").handle(docs_request)
    assert res["answers"]["topic"]["choice"] == "billing"
    assert [r["path"] for r in srv.requests] == ["/old/v1/systemone", "/v1/systemone"]
    assert srv.requests[1]["body"] == docs_request


def test_cross_origin_redirect_is_refused_before_sending(engine, docs_request):
    with StubServer(engine) as target, StubServer() as front:
        front.routes[("POST", "/v1/systemone")] = lambda b, h: (307, b"", {"Location": target.url + "/v1/systemone"})
        with pytest.raises(TezError) as e:
            RemoteTez(front.url, api_key="k").handle(docs_request)
    assert e.value.type == "redirect_refused" and e.value.status == 307 and "another origin" in e.value.message
    assert target.requests == []                     # neither the state nor the key reached the other host


def test_redirect_that_would_drop_a_post_body_is_refused(engine):
    with StubServer(engine) as srv:
        srv.routes[("POST", "/v1/systemone")] = lambda b, h: (302, b"", {"Location": "/elsewhere"})
        with pytest.raises(TezError, match="drop the POST body"):
            RemoteTez(srv.url).decide("s", questions={"q": {"type": "noul", "instructions": "x"}})


def test_timeout_and_unreachable_raise_backend_unavailable(engine):
    with StubServer(engine) as srv:
        srv.routes[("GET", "/v1/models")] = lambda b, h: (time.sleep(1.5), (200, {"models": []}))[1]
        t0 = time.perf_counter()
        with pytest.raises(BackendUnavailable, match="timed out after 0.3s"):
            RemoteTez(srv.url, timeout=0.3).models()
        assert time.perf_counter() - t0 < 1.4
    with socket.socket() as s:                       # a port that nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(BackendUnavailable, match="unreachable"):
        RemoteTez(f"127.0.0.1:{port}", timeout=10).health()      # Windows retries a refused local connect for ~2 s


def test_constructor_validation_and_local_proxy_bypass():
    assert RemoteTez("localhost:8787").base_url == "http://localhost:8787"
    assert RemoteTez("http://127.0.0.1:8787")._session.trust_env is False     # never through a corporate proxy
    assert RemoteTez("https://tez.example.com")._session.trust_env is True
    for bad in ("", "ftp://host", None):
        with pytest.raises(ValueError):
            RemoteTez(bad)
    for bad in (0, -1, True, "30"):
        with pytest.raises(ValueError):
            RemoteTez("http://127.0.0.1:8787", timeout=bad)


def test_docs_request_through_handle(engine):
    with StubServer(engine) as srv:
        res = RemoteTez(srv.url).handle(DOCS_REQUEST)
    assert list(res["answers"]) == ["is_urgent", "topic", "anger"]
    assert set(res["answers"]["anger"]) == {"type", "score", "legend", "probabilities", "confidence"}
