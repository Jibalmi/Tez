"""Live smoke test against a llama-server with Gemma 4 12B on http://127.0.0.1:8091 (template gemma4, started with
--embeddings --pooling last). Skipped automatically when nothing answers there. Read-only: it never restarts or
reconfigures the server."""
from __future__ import annotations

import os

import pytest
import requests

from conftest import DOCS_REQUEST
from tez import LlamaCppBackend, Tez

URL = os.environ.get("TEZ_LIVE_BACKEND", "http://127.0.0.1:8091")


def _reachable() -> bool:
    try:
        s = requests.Session()
        s.trust_env = False
        return s.get(f"{URL}/health", timeout=2).json().get("status") == "ok"
    except (requests.RequestException, ValueError):
        return False


pytestmark = [pytest.mark.live, pytest.mark.skipif(not _reachable(), reason=f"no llama-server at {URL}")]


@pytest.fixture(scope="module")
def tez():
    return Tez(backend=LlamaCppBackend(URL, template="gemma4"))


def test_docs_request_live(tez):
    res = tez.handle(DOCS_REQUEST)
    a = res["answers"]
    assert a["topic"]["choice"] == "billing"
    assert a["is_urgent"]["noul"] > 0.5
    assert 0.0 <= a["anger"]["score"] <= 2.0
    assert abs(sum(a["topic"]["probabilities"].values()) - 1) < 1e-9
    assert res["usage"]["input_tokens"] > 100
    assert res["model"].startswith("tez-0.1.0 (") and res["model"].endswith(", letters)")


def test_forty_options_live(tez):
    names = ["refund", "shipping", "password", "invoice", "cancel", "upgrade", "downgrade", "address", "damaged",
             "missing", "warranty", "gift card", "coupon", "subscription", "login", "2fa", "email change", "phone change",
             "delete account", "export data", "api key", "webhook", "outage", "slow app", "crash", "bug report",
             "feature request", "pricing", "enterprise", "education discount", "nonprofit", "tax", "receipt",
             "chargeback", "fraud", "privacy", "gdpr", "accessibility", "partnership", "press"]
    q = {"type": "choice", "instructions": "Which topic is the message about?", "criteria": {n.replace(" ", "_"): None for n in names}}
    res = tez.decide("I forgot my password and cannot sign in, please help me reset it.", questions={"t": q})
    a = res["answers"]["t"]
    assert len(a["probabilities"]) == 40 and abs(sum(a["probabilities"].values()) - 1) < 1e-9
    assert a["choice"] in {"password", "login"}


def test_embedding_live(tez):
    v = tez.embedder.embed(tez.probe_prompt(Tez(backend="fake").parse_request(DOCS_REQUEST).questions["topic"], "hello"))
    assert v.vector.ndim == 1 and v.vector.shape[0] >= 256
