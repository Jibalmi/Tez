"""llama.cpp response parsing, the EOS retry, token counts, model names and transport errors."""
from __future__ import annotations

import json as _json
import math
import socket
import time

import numpy as np
import pytest
import requests

from tez import Tez
from tez.backends import (FakeBackend, LlamaCppBackend, common_prefix, derive_model_name, letters_from_response,
                          make_backend, prompt_tokens)
from tez.errors import BackendRequestError, BackendUnavailable


def completion(tops, **extra):
    return {"completion_probabilities": [{"id": 1, "token": tops[0][0], "logprob": tops[0][1], "bytes": [],
                                          "top_logprobs": [{"id": i, "token": t, "logprob": lp, "bytes": []} for i, (t, lp) in enumerate(tops)]}],
            **extra}


def test_letters_from_top_logprobs_with_floor():
    data = completion([("A", -0.1), (" B", -2.5), ("C", -3.0), ("A", -9.0), ("<eos>", -6.0)])
    z = letters_from_response(data, 3)
    assert z[0] == -0.1                     # first occurrence wins
    assert z[1] == -9.0 - 2.0               # " B" is a different token: B missing -> min logprob - 2
    assert z[2] == -3.0


def test_letters_from_old_probs_format():
    data = {"completion_probabilities": [{"content": "A", "probs": [{"tok_str": "A", "prob": 0.7}, {"tok_str": "B", "prob": 0.2}]}]}
    z = letters_from_response(data, 2)
    assert np.allclose(z, [math.log(0.7), math.log(0.2)])


def test_letters_missing_probabilities():
    assert letters_from_response({"content": ""}, 2) is None
    assert letters_from_response({"completion_probabilities": []}, 2) is None


def test_prompt_tokens_precedence():
    assert prompt_tokens({"tokens_evaluated": 94, "timings": {"prompt_n": 1, "cache_n": 93}}, "x") == 94
    assert prompt_tokens({"timings": {"prompt_n": 18, "cache_n": 73}}, "x") == 91
    assert prompt_tokens({}, "x" * 400) == 100


def test_model_names():
    blob = "C:/Users/me/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795"
    assert derive_model_name(blob, blob, "Q8_0", 11907350576, "gemma4") == "gemma-4-12b-q8_0"
    assert derive_model_name("C:\\models\\Qwen3.5-4B-Q8_0-L24.gguf", None, "Q8_0", 4e9, "qwen3") == "qwen3.5-4b-q8_0-l24"
    assert derive_model_name("/m/x.gguf", "my-alias", None, None, "gemma4") == "my-alias"
    assert derive_model_name(None, None, None, 6e8, "qwen3") == "qwen3-0.6b"


class Scripted:
    """Stands in for requests.Session.request: returns queued (status, json) pairs and records bodies."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.bodies = []

    def __call__(self, method, url, json=None, timeout=None):
        self.bodies.append((method, url, json))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        status, payload = item
        r = requests.Response()
        r.status_code = status
        r._content = _json.dumps(payload).encode()
        r.headers["content-type"] = "application/json"
        return r


def test_eos_retry_and_request_body(monkeypatch):
    b = LlamaCppBackend("http://127.0.0.1:1", template="gemma4", cache_prompt=False)
    s = Scripted((200, {"content": "", "stop_type": "eos"}), (200, completion([("B", -0.2), ("A", -1.9)], tokens_evaluated=42)))
    monkeypatch.setattr(b._session, "request", s)
    r = b.letters("prompt", 2)
    assert np.allclose(r.logits, [-1.9, -0.2]) and r.tokens == 42
    (_, url1, body1), (_, _, body2) = s.bodies
    assert url1.endswith("/completion")
    assert body1 == {"prompt": "prompt", "n_predict": 1, "n_probs": 200, "temperature": 0, "samplers": [], "cache_prompt": False}
    assert body2 == dict(body1, ignore_eos=True)


def test_http_errors_map_to_tez_errors(monkeypatch):
    b = LlamaCppBackend("http://127.0.0.1:1", cache_prompt=False, retries=0)
    monkeypatch.setattr(b._session, "request", Scripted((400, {"error": {"code": 400, "message": "the request exceeds the available context size"}})))
    with pytest.raises(BackendRequestError, match="exceeds the available context"):
        b.letters("p", 2)
    monkeypatch.setattr(b._session, "request", Scripted((501, {"error": {"message": "This server does not support embeddings."}})))
    with pytest.raises(BackendUnavailable, match="does not support /embedding"):
        b.embed("p")
    monkeypatch.setattr(b._session, "request", Scripted((500, {"error": {"message": "boom"}})))
    with pytest.raises(BackendUnavailable, match="boom"):
        b.letters("p", 2)


def test_connection_errors_retry_then_503(monkeypatch):
    b = LlamaCppBackend("http://127.0.0.1:1", cache_prompt=False, retries=2)
    monkeypatch.setattr("tez.backends.time.sleep", lambda s: None)
    s = Scripted(requests.ConnectionError("refused"), requests.ConnectionError("refused"), (200, completion([("A", -0.1)])))
    monkeypatch.setattr(b._session, "request", s)
    assert b.letters("p", 1).logits.shape == (1,)
    assert len(s.bodies) == 3
    s = Scripted(*[requests.ConnectionError("refused")] * 3)
    monkeypatch.setattr(b._session, "request", s)
    with pytest.raises(BackendUnavailable, match="unreachable"):
        b.letters("p", 1)


def test_real_closed_port_is_unavailable():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    b = LlamaCppBackend(f"http://127.0.0.1:{port}", cache_prompt=False, retries=0)
    with pytest.raises(BackendUnavailable):
        b.letters("p", 2)
    assert b.health()["ok"] is False
    assert b.model_name() == "unknown"


def test_embedding_takes_last_vector(monkeypatch):
    b = LlamaCppBackend("http://127.0.0.1:1")
    monkeypatch.setattr(b._session, "request", Scripted((200, [{"index": 0, "embedding": [[0.0, 1.0], [2.0, 3.0]]}])))
    assert b.embed("p").vector.tolist() == [2.0, 3.0]
    monkeypatch.setattr(b._session, "request", Scripted((200, {"embedding": [4.0, 5.0]})))
    assert b.embed("p").vector.tolist() == [4.0, 5.0]


def test_prompt_cache_switched_off_for_qwen35(monkeypatch):
    b = LlamaCppBackend("http://127.0.0.1:1", template="qwen3", cache_prompt=True)
    s = Scripted((200, {"model_path": "/models/Qwen3.5-4B-Q8_0-L24.gguf", "model_ftype": "Q8_0"}), (200, {"data": []}),
                 (200, completion([("A", -0.1), ("B", -3.0)])))
    monkeypatch.setattr(b._session, "request", s)
    b.letters("p", 2)
    assert b.cache_prompt is False
    assert s.bodies[-1][2]["cache_prompt"] is False
    assert b.model_name() == "qwen3.5-4b-q8_0-l24"


def test_loopback_bypasses_proxies():
    assert LlamaCppBackend("http://127.0.0.1:8091")._session.trust_env is False
    assert LlamaCppBackend("http://localhost:8091")._session.trust_env is False
    assert LlamaCppBackend("http://models.internal:8091")._session.trust_env is True


def test_make_backend():
    assert isinstance(make_backend("fake"), FakeBackend)
    b = make_backend("127.0.0.1:8091", template="qwen3", cache_prompt=False)
    assert isinstance(b, LlamaCppBackend) and b.url == "http://127.0.0.1:8091" and b.template == "qwen3" and not b.cache_prompt
    with pytest.raises(ValueError):
        make_backend("fake", template="llama")


def test_fake_backend_is_deterministic_and_class_dependent():
    fb = FakeBackend()
    p1 = "<|turn>user\nOptions:\nA. x\n\nInput:\ninvoice refund charged<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    assert np.array_equal(fb.embed(p1).vector, fb.embed(p1).vector)
    p2 = p1.replace("invoice refund charged", "crash error broken")
    v1, v2 = fb.embed(p1).vector, fb.embed(p2).vector
    assert np.linalg.norm(v1 - v2) > 1.0
    with pytest.raises(BackendUnavailable):
        FakeBackend(fail=True).letters("p", 2)


def _naive_prefix(a: str, b: str) -> int:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


@pytest.mark.parametrize("a, b", [("", ""), ("abc", ""), ("abc", "abc"), ("abc", "abd"), ("abc", "abcdef"), ("xbc", "abc"),
                                  ("héllo wörld", "héllo world"), ("a" * 1000 + "x", "a" * 1000 + "y")])
def test_common_prefix(a, b):
    assert common_prefix(a, b) == common_prefix(b, a) == _naive_prefix(a, b)


def test_common_prefix_on_random_strings():
    rng = np.random.default_rng(0)
    for _ in range(300):
        a = "".join(rng.choice(list("ab"), size=int(rng.integers(0, 40))))
        b = a[: int(rng.integers(0, len(a) + 1))] + "".join(rng.choice(list("ab"), size=int(rng.integers(0, 5))))
        assert common_prefix(a, b) == _naive_prefix(a, b)


def test_common_prefix_compares_in_c_and_plan_stays_fast():
    a = "w " * 2_500_000
    t0 = time.perf_counter()
    assert common_prefix(a, a + "x") == len(a) and common_prefix(a[:-1] + "yx", a) == len(a) - 1
    assert time.perf_counter() - t0 < 0.25                  # one Python step per character took seconds here
    crit = {f"o{i}": None for i in range(255)}
    body = {"state": "w " * 24_999, "questions": {f"q{j}": {"type": "choice", "instructions": f"Which? {j}",
                                                            "criteria": crit} for j in range(16)}}
    t0 = time.perf_counter()
    plan = Tez(backend="fake").plan(body)
    assert plan["totals"]["calls"]["letters"] == 16 * 14 and time.perf_counter() - t0 < 1.5
