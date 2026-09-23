"""The command line against the FakeBackend."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from conftest import DOCS_REQUEST, SYNTH_SCHEMA_YAML, synth_rows, write_jsonl
from tez import __version__
from tez.cli import main


def test_decide_with_schema(schema_dir: Path, capsys):
    code = main(["decide", "--backend", "fake", "--schema", str(schema_dir / "support-triage.yaml"),
                 "--state", "Help! My payouts have been failing for 3 days."])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert list(out["answers"]) == ["topic", "is_urgent", "anger"]
    assert out["answers"]["topic"]["choice"] == "billing"          # the fake backend matches "payouts"
    assert out["tez"]["questions"]["topic"]["decision"] == "escalate"   # schema default gate, not fitted yet


def test_decide_with_questions_file_and_json_state(tmp_path: Path, capsys):
    qfile = tmp_path / "q.json"
    qfile.write_text(json.dumps(DOCS_REQUEST), encoding="utf-8")         # a whole request is accepted too
    sfile = tmp_path / "state.json"
    sfile.write_text(json.dumps({"message": "refund please", "customer": "Ana"}), encoding="utf-8")
    assert main(["decide", "--backend", "fake", "--questions", str(qfile), "--state-file", str(sfile),
                 "--abstain", "--readout", "letters", "--compact"]) == 0
    out = capsys.readouterr().out
    assert "\n" not in out.strip()
    res = json.loads(out)
    assert "__none__" in res["answers"]["topic"]["probabilities"]


def test_decide_errors(tmp_path: Path, capsys):
    assert main(["decide", "--backend", "fake", "--schema", str(tmp_path / "missing.yaml"), "--state", "x"]) == 2
    assert "error:" in capsys.readouterr().err
    assert main(["decide", "--backend", "fake", "--state", "x"]) == 2
    assert "give --schema" in capsys.readouterr().err


def test_fit_eval_suggest(tmp_path: Path, capsys):
    (tmp_path / "synth.yaml").write_text(SYNTH_SCHEMA_YAML, encoding="utf-8")
    labels = write_jsonl(tmp_path / "labels.jsonl", synth_rows(80, seed=8, anger_every=10))
    assert main(["fit", "--backend", "fake", "--schema", str(tmp_path / "synth.yaml"), "--labels", str(labels)]) == 0
    out = capsys.readouterr().out
    assert "trained on 56, blend w=0.85" in out and "only 8 labels (< 20)" in out
    assert (tmp_path / ".tez" / "synth" / "probes.npz").exists()
    fresh = write_jsonl(tmp_path / "fresh.jsonl", synth_rows(30, seed=9, anger_every=0))
    assert main(["eval", "--backend", "fake", "--schema", str(tmp_path / "synth.yaml"), "--labels", str(fresh),
                 "--alpha", "0.1", "--json", str(tmp_path / "eval.json")]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].split()[:4] == ["question", "n", "readout", "accuracy"]
    assert json.loads((tmp_path / "eval.json").read_text(encoding="utf-8"))["questions"]["topic"]["n"] == 30
    unl = tmp_path / "unlabelled.txt"
    unl.write_text("\n".join(r["state"] for r in synth_rows(40, seed=10, anger_every=0)), encoding="utf-8")
    assert main(["suggest", "--backend", "fake", "--schema", str(tmp_path / "synth.yaml"), "--unlabelled", str(unl), "--n", "5"]) == 0
    picks = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(picks) == 5 and all(p["labels"] == {} and isinstance(p["state"], str) for p in picks)
    res = subprocess.run([sys.executable, "-m", "tez", "decide", "--backend", "fake", "--schema", str(tmp_path / "synth.yaml"),
                          "--state", "invoice refund payment", "--compact"], capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout)["tez"]["questions"]["topic"]["readout"] == "probe"


def test_version_and_module_entry_point():
    res = subprocess.run([sys.executable, "-m", "tez", "--version"], capture_output=True, text=True)
    assert res.returncode == 0 and res.stdout.strip() == f"tez {__version__}"


def test_truncate_reports_missing_file(tmp_path: Path, capsys):
    code = main(["truncate", str(tmp_path / "in.gguf"), "20", str(tmp_path / "out.gguf")])
    assert code == 2
    assert "error:" in capsys.readouterr().err
