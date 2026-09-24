"""Presets: the ten use-case schemas shipped as package data (tez/presets/*.yaml)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import tez.presets as presets
from tez import InvalidRequest, Tez
from tez.cli import build_parser, main
from tez.fit import fit
from tez.schema import load_schema
from tez.server import create_app

ROOT = Path(__file__).resolve().parents[1]
USECASES = ROOT / "examples" / "usecases"


def test_the_ten_use_cases_are_presets():
    if not USECASES.is_dir():
        pytest.skip("examples/ is not in this checkout")
    expected = sorted(p.name for p in USECASES.iterdir() if (p / "schema.yaml").is_file())
    assert len(expected) == 10 and presets.names() == expected
    for name in expected:
        example = (USECASES / name / "schema.yaml").read_text(encoding="utf-8")
        assert presets.text(name).replace("\r\n", "\n") == example.replace("\r\n", "\n"), f"{name} drifted from examples/"
        shipped = presets.load(name)
        assert shipped.builtin is True and shipped.path is None
        assert {**shipped.to_wire(), "builtin": False} == load_schema(USECASES / name / "schema.yaml").to_wire()


def test_unknown_preset():
    with pytest.raises(InvalidRequest, match="unknown preset 'nope' .*support-triage"):
        presets.load("nope")


def test_add_presets_and_the_schema_listing(schema_dir: Path):
    tez = Tez(backend="fake", schemas=schema_dir)             # its own support-triage
    added = tez.add_presets()
    assert "support-triage" not in added and len(added) == 9       # the user's schema of that name wins
    assert tez.schemas["support-triage"].builtin is False and tez.schemas["topic-news"].builtin is True
    c = TestClient(create_app(tez))
    listed = {s["name"]: s for s in c.get("/v1/schemas").json()["schemas"]}
    assert listed["support-triage"]["builtin"] is False and listed["prompt-injection-guard"]["builtin"] is True
    assert c.get("/v1/schemas/voice-commands").json()["builtin"] is True
    r = c.post("/v1/systemone", json={"state": "open spotify", "schema": "voice-commands", "tez": {"readout": "letters"}})
    assert r.status_code == 200 and r.json()["answers"]["action"]["choice"] == "open_spotify"
    assert Tez(backend="fake").add_presets(["topic-news"]) == ["topic-news"]


def test_preset_feedback_goes_where_the_copied_preset_will_find_it(schema_dir: Path, tmp_path: Path, monkeypatch):
    """A preset has no file of its own: its feedback goes to --data-dir, else the --schemas directory's .tez (where tez
    fit reads it once the preset is copied there to be fitted), else ./.tez."""
    body = {"schema": "voice-commands", "question": "action", "state": "open spotify", "label": "open_spotify"}
    tez = Tez(backend="fake", schemas=schema_dir)
    tez.add_presets(["voice-commands"])
    assert TestClient(create_app(tez)).post("/v1/feedback", json=body).status_code == 200
    written = schema_dir / ".tez" / "feedback" / "voice-commands.jsonl"
    assert json.loads(written.read_text(encoding="utf-8"))["label"] == "open_spotify"
    (schema_dir / "voice-commands.yaml").write_text(presets.text("voice-commands"), encoding="utf-8")
    copied = Tez(backend="fake", schemas=schema_dir)                     # the copy is an ordinary schema now
    assert copied.schemas["voice-commands"].builtin is False
    assert copied.feedback_path(copied.schemas["voice-commands"]) == written
    with_data = Tez(backend="fake", schemas=schema_dir / "support-triage.yaml", data_dir=tmp_path / "data")
    with_data.add_presets(["topic-news"])
    assert with_data.feedback_path(with_data.schemas["topic-news"]) == tmp_path / "data" / "feedback" / "topic-news.jsonl"
    one_file = Tez(backend="fake", schemas=schema_dir / "support-triage.yaml")     # a schema file: its directory
    one_file.add_presets(["topic-news"])
    assert one_file.feedback_path(one_file.schemas["topic-news"]) == schema_dir / ".tez" / "feedback" / "topic-news.jsonl"
    monkeypatch.chdir(tmp_path)
    bare = Tez(backend="fake")
    bare.add_presets(["voice-commands"])
    bare.record_feedback(body)
    assert (tmp_path / ".tez" / "feedback" / "voice-commands.jsonl").is_file()


def test_presets_never_load_artefacts_and_cannot_be_fitted(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stray = tmp_path / ".tez" / "support-triage"
    stray.mkdir(parents=True)
    (stray / "calibration.json").write_text(json.dumps({"schema": "support-triage", "calibration_id": "x", "questions": {}}),
                                            encoding="utf-8")
    tez = Tez(backend="fake")
    tez.add_presets(["support-triage"])
    assert tez.fitted == {}
    with pytest.raises(InvalidRequest, match="built-in preset: copy it into a schema file first"):
        fit(tez, tez.schemas["support-triage"], [], say=lambda m: None)


def test_cli_decide_with_a_preset(capsys):
    assert main(["decide", "My card payment was declined twice", "--backend", "fake", "--preset", "intent-router",
                 "--readout", "letters", "--compact"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert list(out["answers"]) == ["intent", "needs_account_access"]
    assert main(["decide", "x", "--backend", "fake", "--preset", "nope"]) == 2
    assert "unknown preset" in capsys.readouterr().err
    assert main(["decide", "x", "--backend", "fake", "--preset", "topic-news", "--schema", "s.yaml"]) == 2
    assert "not both" in capsys.readouterr().err


def test_cli_presets_command(capsys):
    assert main(["presets"]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in presets.names()) and "topic (choice)" in out
    assert main(["presets", "--show", "topic-news"]) == 0
    assert capsys.readouterr().out == presets.text("topic-news")
    assert build_parser().parse_args(["serve", "--presets"]).presets is True
