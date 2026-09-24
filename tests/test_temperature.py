"""Default temperatures for letters answers that no fit calibrates (tez.temperature)."""
from __future__ import annotations

import numpy as np
import pytest

from tez import FakeBackend, Tez
from tez.cli import build_parser
from tez.readout import fit_temperature_logits
from tez.temperature import CHOICE_SPLIT, parse, pick, table_for

GEMMA = "gemma-4-12b-q8_0"
QUESTIONS = {
    "topic": {"type": "choice", "instructions": "What is the message about?",
              "criteria": {"billing": "Payments, payouts, invoices", "technical": "Something is broken", "sales": None}},
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
}
STATE = "Help! My payouts have been failing for 3 days."


def test_table_matches_only_the_measured_model_and_template():
    assert table_for("gemma4", GEMMA) is not None
    assert table_for("gemma4", "gemma-4-12b-it-Q8_0") is not None           # the unsloth file name
    assert table_for("gemma4", "gemma-4-12b-q4_k_m") is None                # another quant was not measured
    assert table_for("gemma4", "gemma-4-27b-q8_0") is None
    assert table_for("qwen3", GEMMA) is None
    assert table_for("gemma4", "fake") is None and table_for("gemma4", None) is None


def test_pick_by_type_and_options_shown():
    t = table_for("gemma4", GEMMA)
    assert pick(t, "noul", 2) == 6.01 and pick(t, "score", 5) == 5.18
    assert pick(t, "choice", 3) == pick(t, "choice", CHOICE_SPLIT) == 4.71
    assert pick(t, "choice", CHOICE_SPLIT + 1) == 2.66
    assert pick(None, "choice", 3) == 1.0


@pytest.mark.parametrize("value, expected", [("auto", "auto"), ("OFF", "off"), (None, "auto"), ("1.5", 1.5), (2, 2.0)])
def test_parse(value, expected):
    assert parse(value) == expected


@pytest.mark.parametrize("value", ["0", "-1", "hot", "inf", True])
def test_parse_refuses(value):
    with pytest.raises(ValueError):
        parse(value)


def probs(tez: Tez) -> dict:
    return tez.decide(STATE, questions=QUESTIONS)


def test_unfitted_answers_are_tempered_for_the_measured_model_only():
    raw = probs(Tez(backend="fake", model_name=GEMMA, default_temperature="off"))
    tempered = probs(Tez(backend="fake", model_name=GEMMA))
    other = probs(Tez(backend="fake"))                                       # model "fake": no table, temperature 1

    assert other["answers"] == raw["answers"] and "temperature" not in other["tez"]["questions"]["topic"]
    a, b = raw["answers"]["topic"], tempered["answers"]["topic"]
    assert a["choice"] == b["choice"]                                        # the argmax never changes
    assert max(b["probabilities"].values()) < max(a["probabilities"].values())
    assert b["confidence"] < a["confidence"]
    assert tempered["tez"]["questions"]["topic"]["temperature"] == 4.71
    assert tempered["tez"]["questions"]["urgent"]["temperature"] == 6.01
    # yes/no moves towards 0.5 without crossing it
    assert (raw["answers"]["urgent"]["noul"] - 0.5) * (tempered["answers"]["urgent"]["noul"] - 0.5) > 0
    assert abs(tempered["answers"]["urgent"]["noul"] - 0.5) <= abs(raw["answers"]["urgent"]["noul"] - 0.5)


def test_a_score_keeps_its_most_probable_level_while_its_expected_level_moves_towards_the_middle():
    """What docs/API.md ("Default temperature") says: the argmax stays, `score` (the expected level over the tempered
    probabilities it is returned with) moves towards the middle of the scale."""
    q = {"severity": {"type": "score", "instructions": "How severe is the incident?",
                      "criteria": ["none", "low", "medium", "high", "critical"]}}

    def engine(mode):
        backend = FakeBackend(model=GEMMA, letters_fn=lambda prompt, k: [-1.0 * i for i in range(k)])
        return Tez(backend=backend, default_temperature=mode)

    raw = engine("off").decide(STATE, questions=q)["answers"]["severity"]
    tempered = engine("auto").decide(STATE, questions=q)["answers"]["severity"]
    top = [max(a["probabilities"], key=a["probabilities"].get) for a in (raw, tempered)]
    assert top == ["0", "0"]                                                  # the most probable level stays
    assert raw["score"] < tempered["score"] < 2                               # the expected level moves to the middle
    assert tempered["score"] == pytest.approx(sum(int(k) * p for k, p in tempered["probabilities"].items()))


def test_a_fixed_default_temperature_applies_to_any_model():
    res = probs(Tez(backend="fake", default_temperature=2.0))
    assert res["tez"]["questions"]["topic"]["temperature"] == 2.0


def test_cli_flag_and_environment():
    args = build_parser({}).parse_args(["decide", "--backend", "fake", "--state", "x", "--default-temperature", "off"])
    assert args.default_temperature == "off"
    args = build_parser({"TEZ_DEFAULT_TEMPERATURE": "3"}).parse_args(["decide", "--backend", "fake", "--state", "x"])
    assert args.default_temperature == 3.0
    with pytest.raises(SystemExit):
        build_parser({"TEZ_DEFAULT_TEMPERATURE": "warm"}).parse_args(["decide", "--backend", "fake", "--state", "x"])


def test_fit_prior_is_centred_on_the_default():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(40, 3)) * 6.0
    y = Z.argmax(1)
    assert fit_temperature_logits(Z[:3], y[:3], prior_center=4.71) == 4.71    # too few rows: the centre itself
    assert fit_temperature_logits(Z[:3], y[:3]) == 1.0
    # with little evidence the estimate stays nearer its centre
    few = fit_temperature_logits(Z[:6], y[:6], prior_center=4.71)
    assert few > fit_temperature_logits(Z[:6], y[:6])
