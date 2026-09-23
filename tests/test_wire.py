"""Wire-format conformance: Jev's exact answer shapes and the response envelope (docs/API.md)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tez import FakeBackend, Tez, __version__
from tez.readout import confidence, softmax


def scripted(scores_by_k):
    """letters_fn returning fixed scores for each option count k."""
    return lambda prompt, k: scores_by_k[k]


def jev_confidence(p):
    k = len(p)
    return (k * max(p) - 1) / (k - 1)


def test_choice_shape(docs_request):
    fb = FakeBackend(letters_fn=scripted({2: [0.0, 1.0], 3: [2.0, 0.5, -1.0]}))
    res = Tez(backend=fb).handle(docs_request)
    a = res["answers"]["topic"]
    assert list(a) == ["type", "choice", "probabilities", "confidence"]
    assert a["type"] == "choice"
    assert list(a["probabilities"]) == ["billing", "technical", "sales"]
    p = list(a["probabilities"].values())
    assert math.isclose(sum(p), 1.0, abs_tol=1e-9)
    assert np.allclose(p, softmax([2.0, 0.5, -1.0]))
    assert a["choice"] == "billing"
    assert math.isclose(a["confidence"], jev_confidence(p), rel_tol=1e-9)
    assert all(isinstance(v, float) for v in p)


def test_score_shape_and_expectation(docs_request):
    fb = FakeBackend(letters_fn=scripted({2: [0.0, 1.0], 3: [-1.0, 1.5, 0.5]}))
    a = Tez(backend=fb).handle(docs_request)["answers"]["anger"]
    assert list(a) == ["type", "score", "legend", "probabilities", "confidence"]
    assert a["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert list(a["probabilities"]) == ["0", "1", "2"]
    p = np.array(list(a["probabilities"].values()))
    assert math.isclose(p.sum(), 1.0, abs_tol=1e-9)
    assert math.isclose(a["score"], float(np.dot([0, 1, 2], p)), rel_tol=1e-9)
    assert math.isclose(a["confidence"], jev_confidence(p), rel_tol=1e-9)


def test_noul_shape_has_no_confidence(docs_request):
    fb = FakeBackend(letters_fn=scripted({2: [-0.5, 1.5], 3: [0.0, 0.0, 0.0]}))
    a = Tez(backend=fb).handle(docs_request)["answers"]["is_urgent"]
    assert a == {"type": "noul", "noul": pytest.approx(float(softmax([-0.5, 1.5])[1]))}
    assert "confidence" not in a and "probabilities" not in a


def test_noul_is_probability_of_true_option_b():
    """Options are shown false first (A), true second (B): noul = P(B)."""
    fb = FakeBackend(letters_fn=lambda p, k: [3.0, -3.0])
    res = Tez(backend=fb).decide("x", questions={"q": {"type": "noul", "instructions": "Is it?"}})
    assert res["answers"]["q"]["noul"] < 0.01
    prompt = fb.prompts[-1]
    assert "A. false: no, the statement does not hold" in prompt and "B. true: yes, the statement holds" in prompt


def test_envelope(docs_request):
    res = Tez(backend="fake").handle(docs_request)
    assert list(res) == ["model", "answers", "usage", "tez"]
    assert res["model"] == f"tez-{__version__} (fake, letters)"
    assert list(res["answers"]) == ["is_urgent", "topic", "anger"]
    assert res["usage"]["output_tokens"] == 0
    assert isinstance(res["usage"]["input_tokens"], int) and res["usage"]["input_tokens"] > 0
    assert isinstance(res["tez"]["latency_ms"], float)
    assert res["tez"]["questions"] == {q: {"readout": "letters"} for q in ("is_urgent", "topic", "anger")}


def test_state_is_last_and_objects_are_json():
    fb = FakeBackend()
    Tez(backend=fb).decide({"customer": "Ana", "text": "hi"},
                           questions={"q": {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None}}})
    prompt = fb.prompts[-1]
    assert prompt.index("Question (choice): Pick") < prompt.index("Options:") < prompt.index("Input:\n")
    assert '{"customer": "Ana", "text": "hi"}' in prompt
    assert prompt.endswith('{"customer": "Ana", "text": "hi"}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>')


def test_confidence_formula_edges():
    assert confidence([0.25, 0.25, 0.25, 0.25]) == 0.0
    assert confidence([1.0, 0.0, 0.0]) == 1.0
    assert confidence([0.91, 0.08, 0.01]) == pytest.approx((3 * 0.91 - 1) / 2)


def test_instructions_object_and_array_are_serialised():
    fb = FakeBackend()
    Tez(backend=fb).decide("s", questions={
        "q1": {"type": "noul", "instructions": {"ask": "is it", "note": "strict"}},
        "q2": {"type": "noul", "instructions": ["step one", "step two"]}})
    assert 'Question (noul): {"ask": "is it", "note": "strict"}' in fb.prompts[0]
    assert 'Question (noul): ["step one", "step two"]' in fb.prompts[1]


def test_abstain_adds_none_option(docs_request):
    docs_request["tez"] = {"abstain": True}
    fb = FakeBackend(letters_fn=lambda p, k: [0.0] * (k - 1) + [2.0] if k == 4 else [0.0] * k)
    res = Tez(backend=fb).handle(docs_request)
    topic = res["answers"]["topic"]
    assert list(topic["probabilities"]) == ["billing", "technical", "sales", "__none__"]
    assert topic["choice"] == "__none__"
    assert math.isclose(sum(topic["probabilities"].values()), 1.0, abs_tol=1e-9)
    assert math.isclose(topic["confidence"], jev_confidence(list(topic["probabilities"].values())), rel_tol=1e-9)
    assert "D. none: none of these fits" in fb.prompts[1]
    # noul and score questions are unchanged by abstain
    assert list(res["answers"]["anger"]["probabilities"]) == ["0", "1", "2"]
    assert "noul" in res["answers"]["is_urgent"]


def test_abstain_with_gate_escalates_none(docs_request):
    docs_request["tez"] = {"abstain": True, "gate": {"alpha": 0.1}}
    fb = FakeBackend(letters_fn=lambda p, k: [0.0] * (k - 1) + [5.0] if k == 4 else [0.0] * k)
    res = Tez(backend=fb).handle(docs_request)
    assert res["answers"]["topic"]["choice"] == "__none__"
    assert res["tez"]["questions"]["topic"]["decision"] == "escalate"


def test_gate_without_calibration_escalates(docs_request):
    docs_request["tez"] = {"gate": {"alpha": 0.2}}
    res = Tez(backend="fake").handle(docs_request)
    for meta in res["tez"]["questions"].values():
        assert meta == {"readout": "letters", "decision": "escalate"}
