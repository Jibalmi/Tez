"""tez fit -> probes.npz / calibration.json / manifest.json -> served probe, on synthetic data where the
FakeBackend's hashed bag-of-words embeddings are class dependent."""
from __future__ import annotations

import json
import random
import shutil

import numpy as np
import pytest

from conftest import FILLER, SYNTH_SCHEMA_YAML, TOPICS, VOCAB, synth_rows, synth_state, write_jsonl
from stub_llama import StubLlama
from tez import FakeBackend, InvalidRequest, Tez
from tez.artifacts import load_artifacts, save_artifacts
from tez.fit import collect_rows, evaluate, fit, suggest
from tez.readout import Probe, softmax, temper
from tez.schema import load_schema

QUIET = dict(say=lambda msg: None)


def topic_of(state: str) -> str:
    words = set(state.split())
    return max(TOPICS, key=lambda t: len(words & set(VOCAB[t])))


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    d = tmp_path_factory.mktemp("fitted")
    (d / "synth.yaml").write_text(SYNTH_SCHEMA_YAML, encoding="utf-8")
    schema = load_schema(d / "synth.yaml")
    labels = write_jsonl(d / "labels.jsonl", synth_rows(150, seed=1, anger_every=12))    # anger: 13 labels only
    tez = Tez(backend=FakeBackend(), schemas=[schema])
    cal = fit(tez, schema, [labels], **QUIET)
    return tez, schema, cal, labels


def test_fit_writes_artifacts(fitted):
    tez, schema, cal, _ = fitted
    d = schema.artifact_dir
    assert (d / "probes.npz").exists() and (d / "calibration.json").exists() and (d / "manifest.json").exists()
    assert d == schema.path.parent / ".tez" / "synth"
    q = cal["questions"]
    assert q["topic"]["probe"]["n_train"] == 105 and q["topic"]["probe"]["n_held_out"] == 45
    assert q["topic"]["probe"]["blend"] is True
    assert q["topic"]["probe"]["weight"] == pytest.approx(105 / 115, abs=1e-4)
    assert q["topic"]["probe"]["dim"] == 64 and q["topic"]["probe"]["model"] == "fake"
    assert "probe" in q["is_urgent"]
    assert "probe" not in q["anger"] and "only 13 labels (< 20)" in q["anger"]["note"]
    assert q["anger"]["letters"]["temperature"] > 0                   # letters still calibrated
    assert set(q["topic"]["probe"]["thresholds"]) == {"0.01", "0.02", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3"}
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["probes"] == ["topic", "is_urgent"]
    assert manifest["label_counts"]["topic"]["total"] == 150
    assert manifest["label_counts"]["topic"]["by_label"] == {"billing": 50, "technical": 50, "sales": 50}
    assert manifest["embed_dim"] == 64 and manifest["template"] == "gemma4"
    assert len(manifest["train_hashes"]["topic"]) == 105
    assert cal["calibration_id"].startswith("synth@")
    assert tez.probe_index() == {"synth": ["topic", "is_urgent"]}


def test_probe_decides_fresh_states(fitted):
    tez, schema, cal, _ = fitted
    rows = synth_rows(90, seed=99, anger_every=0)
    probed = {qid: schema.questions[qid] for qid in ("topic", "is_urgent")}
    ok_topic = ok_urgent = 0
    for r in rows:
        res = tez.decide(r["state"], questions=probed, schema="synth", readout="probe", gate=False)
        ok_topic += res["answers"]["topic"]["choice"] == r["labels"]["topic"]
        ok_urgent += (res["answers"]["is_urgent"]["noul"] > 0.5) == (r["labels"]["is_urgent"] == "true")
        meta = res["tez"]["questions"]["topic"]
        assert meta["readout"] == "probe" and meta["calibration_id"] == cal["calibration_id"]
        assert meta["p_correct"] == pytest.approx(max(res["answers"]["topic"]["probabilities"].values()), abs=1e-4)
    assert ok_topic / len(rows) >= 0.9
    assert ok_urgent / len(rows) >= 0.85


def test_auto_readout_and_letters_opt_out(fitted):
    tez, *_ = fitted
    state = synth_state(random.Random(5), "sales", False)
    res = tez.decide(state, schema="synth")
    assert res["tez"]["questions"]["topic"]["readout"] == "probe"
    assert res["tez"]["questions"]["anger"]["readout"] == "letters"
    assert "p_correct" in res["tez"]["questions"]["anger"]               # letters calibration, no probe
    assert res["model"].endswith("(fake, letters+probe)")
    res = tez.decide(state, schema="synth", readout="letters")
    assert all(m["readout"] == "letters" and "p_correct" in m for m in res["tez"]["questions"].values())
    with pytest.raises(InvalidRequest, match="no usable probe"):
        tez.decide(state, schema="synth", readout="probe", questions={"anger": tez.schemas["synth"].questions["anger"]})


def test_blend_is_exactly_what_is_served(fitted):
    tez, schema, cal, _ = fitted
    state = synth_state(random.Random(7), "technical", True)
    q = schema.questions["topic"]
    fq = tez.fitted["synth"].questions["topic"]
    shots = schema.shots("topic")
    z, _ = tez.letter_logits(q, state, None, shots)
    prior = softmax(z, fq.letters.temperature)
    pp = fq.probe.predict(tez.embedder.embed(tez.probe_prompt(q, state, shots)).vector)
    n = fq.probe.n
    assert n == 105
    expected = temper((n / (n + 10)) * pp + (10 / (n + 10)) * prior, fq.probe_cal.temperature)
    got = tez.decide(state, questions={"topic": q}, schema="synth", readout="probe")["answers"]["topic"]["probabilities"]
    assert np.allclose(list(got.values()), expected, atol=1e-12)


def test_gate_decisions(fitted):
    tez, *_ = fitted
    acts = 0
    for r in synth_rows(30, seed=123, anger_every=0):
        metas = tez.decide(r["state"], schema="synth", alpha=0.1)["tez"]["questions"]
        assert {m["decision"] for m in metas.values()} <= {"act", "escalate"}
        assert metas["anger"]["decision"] == "escalate"                 # 4 held-out rows cannot certify 10 %
        acts += metas["topic"]["decision"] == "act"
    assert acts > 0


def test_probe_npz_round_trip(tmp_path):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X, y = rng.standard_normal((60, 8)), np.repeat([0, 1, 2], 20)
    probe = Probe.from_sklearn(LogisticRegression(max_iter=1000).fit(X, y), k=4, n=60)
    save_artifacts(tmp_path, {"schema": "s", "calibration_id": "s@x", "questions": {"q": {"type": "choice", "options": ["a", "b", "c", "d"], "probe": {"blend": False}}}},
                   {"schema": "s"}, {"q": probe})
    loaded = load_artifacts(tmp_path).questions["q"].probe
    assert loaded.k == 4 and loaded.n == 60 and loaded.classes.tolist() == [0, 1, 2]
    assert np.allclose(loaded.predict_many(X), probe.predict_many(X), atol=1e-5)


def test_edited_schema_makes_the_fit_stale(fitted, tmp_path):
    _, schema, _, _ = fitted
    d = tmp_path / "edited"
    shutil.copytree(schema.path.parent, d)
    text = (d / "synth.yaml").read_text(encoding="utf-8").replace("Which team should handle the message?", "Which team owns it?")
    (d / "synth.yaml").write_text(text, encoding="utf-8")
    tez = Tez(backend=FakeBackend(), schemas=d)
    assert tez.probe_index() == {"synth": ["is_urgent"]}
    detail = tez.schema_detail("synth")
    assert detail["probes"]["topic"]["probe"] == "stale" and "prompt changed" in detail["probes"]["topic"]["note"]
    res = tez.decide("invoice refund", schema="synth")
    # the stale fit is not used: uncalibrated letters, read with auto's state_first (is_urgent keeps its fit's
    # question_first, so the request is mixed and every question names its layout)
    assert res["tez"]["questions"]["topic"] == {"readout": "letters", "layout": "state_first"}
    with pytest.raises(InvalidRequest, match="prompt changed"):
        tez.decide("invoice refund", schema="synth", readout="probe")


def test_a_fit_made_at_one_n_probs_is_stale_at_another(tmp_path):
    d = tmp_path / "schemas"
    d.mkdir()
    (d / "tri.yaml").write_text('name: tri\nquestions:\n  topic: {type: choice, instructions: "Topic?", '
                                'criteria: {billing: null, technical: null, sales: null}}\n', encoding="utf-8")
    labels = write_jsonl(tmp_path / "labels.jsonl", [{"state": f"s{i}", "labels": {"topic": ["billing", "technical"][i % 2]}}
                                                     for i in range(30)])
    with StubLlama() as stub:
        schema = load_schema(d / "tri.yaml")
        cal = fit(Tez(backend=stub.url, schemas=[schema]), schema, [labels], min_labels=1000, **QUIET)
        assert cal["questions"]["topic"]["letters"]["n_probs"] == 200
        same = Tez(backend=stub.url, schemas=d)
        assert same.schema_detail("tri")["probes"]["topic"]["letters_calibrated"] is True
        assert "calibration_id" in same.decide("anything", schema="tri")["tez"]["questions"]["topic"]

        other = Tez(backend=stub.url, schemas=d, n_probs=1)          # letters past the first read the floor
        detail = other.schema_detail("tri")["probes"]["topic"]
        assert detail["letters_calibrated"] is False and "n_probs" in detail["note"]
        fit_status = other.plan({"schema": "tri"})["questions"]["topic"]["fit"]
        assert fit_status["status"] == "stale" and "n_probs" in fit_status["reason"]
        meta = other.decide("anything", schema="tri", alpha=0.3)["tez"]["questions"]["topic"]
        assert "calibration_id" not in meta and meta["decision"] == "escalate"

        path = d / ".tez" / "tri" / "calibration.json"                   # a fit from before n_probs was recorded
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["questions"]["topic"]["letters"]["n_probs"]
        path.write_text(json.dumps(data), encoding="utf-8")
        assert Tez(backend=stub.url, schemas=d).schema_detail("tri")["probes"]["topic"]["letters_calibrated"] is True
        assert Tez(backend=stub.url, schemas=d, n_probs=1).schema_detail("tri")["probes"]["topic"]["letters_calibrated"] \
            is False


def test_other_model_does_not_use_the_calibration(fitted):
    _, schema, _, _ = fitted
    tez = Tez(backend=FakeBackend(model="another-model"), schemas=[schema])
    metas = tez.decide("invoice refund payment", schema="synth")["tez"]["questions"]
    assert all(m == {"readout": "letters"} for m in metas.values())


def test_no_blend_serves_the_probe_alone(tmp_path):
    (tmp_path / "synth.yaml").write_text(SYNTH_SCHEMA_YAML, encoding="utf-8")
    schema = load_schema(tmp_path / "synth.yaml")
    labels = write_jsonl(tmp_path / "labels.jsonl", synth_rows(60, seed=2, anger_every=0))
    fb = FakeBackend()
    tez = Tez(backend=fb, schemas=[schema])
    cal = fit(tez, schema, [labels], use_blend=False, **QUIET)
    assert cal["questions"]["topic"]["probe"]["blend"] is False
    before = dict(fb.calls)
    res = tez.decide("refund invoice payment", schema="synth", questions={"topic": schema.questions["topic"]})
    assert res["tez"]["questions"]["topic"]["readout"] == "probe"
    assert fb.calls["letters"] == before["letters"] and fb.calls["embed"] == before["embed"] + 1


def test_fit_from_feedback_and_override_order(tmp_path):
    (tmp_path / "synth.yaml").write_text(SYNTH_SCHEMA_YAML, encoding="utf-8")
    schema = load_schema(tmp_path / "synth.yaml")
    tez = Tez(backend=FakeBackend(), schemas=[schema])
    rows = synth_rows(40, seed=3, anger_every=0)
    for r in rows:
        tez.record_feedback({"schema": "synth", "question": "topic", "state": r["state"], "label": r["labels"]["topic"]})
    path = tez.feedback_path(schema)
    assert path.resolve() == (tmp_path / ".tez" / "feedback" / "synth.jsonl").resolve()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 40
    labels = write_jsonl(tmp_path / "l.jsonl", [{"state": rows[0]["state"], "labels": {"topic": "sales" if rows[0]["labels"]["topic"] != "sales" else "billing"}}])
    data = collect_rows(schema, [labels], path)
    assert data.rows["topic"][rows[0]["state"]][1] == TOPICS.index(rows[0]["labels"]["topic"])   # feedback wins
    cal = fit(tez, schema, [], **QUIET)
    assert cal["questions"]["topic"]["n_labels"] == 40 and "probe" in cal["questions"]["topic"]
    assert "no labels" in cal["questions"]["anger"]["note"]


def test_fit_errors_and_problem_counts(tmp_path, schema_dir):
    (tmp_path / "synth.yaml").write_text(SYNTH_SCHEMA_YAML, encoding="utf-8")
    schema = load_schema(tmp_path / "synth.yaml")
    tez = Tez(backend=FakeBackend(), schemas=[schema])
    with pytest.raises(InvalidRequest, match="no labelled rows"):
        fit(tez, schema, [], **QUIET)
    bad = write_jsonl(tmp_path / "bad.jsonl", [{"state": "x", "labels": {"topic": "nope", "zz": 1, "is_urgent": "true"}}])
    data = collect_rows(schema, [bad])
    assert data.problems["invalid_label"] == 1 and data.problems["unknown_question"] == 1 and data.total() == 1
    docs = load_schema(schema_dir / "support-triage.yaml")          # its example is a few-shot row: never trained on
    shot = write_jsonl(tmp_path / "shot.jsonl", [{"state": "I was charged twice this month.", "labels": {"topic": "billing"}}])
    data = collect_rows(docs, [shot])
    assert data.rows["topic"] == {} and data.problems["few_shot_rows_excluded"] == 3


def test_suggest_picks_one_state_per_cluster(fitted):
    tez, schema, *_ = fitted
    rng = random.Random(11)
    states = [" ".join(rng.sample(VOCAB[t], 5) + rng.sample(FILLER, 1)) for t in TOPICS for _ in range(30)]
    picks = suggest(tez, schema, states, n=3, **QUIET)
    assert len(picks) == 3 and [p["_suggest"]["rank"] for p in picks] == [1, 2, 3]
    assert sorted(topic_of(p["state"]) for p in picks) == sorted(TOPICS)
    assert all(p["labels"] == {} for p in picks)
    one = suggest(tez, schema, states, n=3, question="topic", **QUIET)
    assert sorted(topic_of(p["state"]) for p in one) == sorted(TOPICS)


def test_evaluate(fitted, tmp_path):
    tez, schema, _, train_labels = fitted
    fresh = write_jsonl(tmp_path / "fresh.jsonl", synth_rows(60, seed=77, anger_every=0))
    res = evaluate(tez, schema, [fresh], **QUIET)
    assert res["questions"]["topic"]["n"] == 60 and res["questions"]["topic"]["accuracy"] >= 0.9
    assert res["questions"]["topic"]["readouts"] == {"probe": 60}
    assert 0 <= res["questions"]["topic"]["ece"] <= 1 and res["overall"]["n"] == 120
    seen = evaluate(tez, schema, [train_labels], **QUIET)
    assert seen["questions"]["topic"]["skipped_seen_in_training"] == 105
    assert seen["questions"]["topic"]["n"] == 45
    gated = evaluate(tez, schema, [fresh], alpha=0.1, **QUIET)
    assert 0 <= gated["questions"]["topic"]["acted"] <= 1
