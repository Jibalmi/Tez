"""Trained artefacts of a schema, next to the schema file in schemas/.tez/<name>/:

  probes.npz        per-question logistic weights on the embedding backend's features
  calibration.json  per question: letters and probe temperature, conformal thresholds per alpha, fingerprints
  manifest.json     backend, template, model, layer/cut, label counts, settings, date
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .readout import Probe

log = logging.getLogger("tez")
PROBES, CALIBRATION, MANIFEST = "probes.npz", "calibration.json", "manifest.json"


@dataclass
class ReadoutCal:
    model: str
    template: str
    prompt_sha: str
    temperature: float
    thresholds: dict
    info: dict = field(default_factory=dict)
    layout: str = "question_first"      # the prompt layout the fit read (fits made before layouts: question_first)

    @classmethod
    def from_json(cls, d: dict | None) -> "ReadoutCal | None":
        if not d:
            return None
        return cls(model=str(d.get("model", "")), template=str(d.get("template", "")), prompt_sha=str(d.get("prompt_sha", "")),
                   temperature=float(d.get("temperature", 1.0)), thresholds=dict(d.get("thresholds") or {}), info=d,
                   layout=str(d.get("layout") or "question_first"))


@dataclass
class FittedQuestion:
    qid: str
    type: str
    options: list[str]
    letters: ReadoutCal | None
    probe_cal: ReadoutCal | None
    probe: Probe | None
    blend: bool
    note: str | None = None

    @property
    def layout(self) -> str:
        """The prompt layout this question was fitted under (letters and probe are fitted in one run)."""
        cal = self.letters or self.probe_cal
        return cal.layout if cal is not None else "question_first"


@dataclass
class Fitted:
    name: str
    calibration_id: str
    dir: Path
    questions: dict[str, FittedQuestion]
    calibration: dict
    manifest: dict


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def save_artifacts(directory: Path, calibration: dict, manifest: dict, probes: dict[str, Probe]) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {"questions": np.array(list(probes), dtype=str)}
    for i, pr in enumerate(probes.values()):
        arrays[f"q{i}_W"] = np.asarray(pr.W, np.float32)
        arrays[f"q{i}_b"] = np.asarray(pr.b, np.float64)
        arrays[f"q{i}_classes"] = np.asarray(pr.classes, np.int32)
        arrays[f"q{i}_meta"] = np.array([pr.k, pr.n], dtype=np.int64)
    tmp = directory / (PROBES + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, directory / PROBES)
    _write_atomic(directory / CALIBRATION, json.dumps(calibration, indent=1, ensure_ascii=False).encode("utf-8"))
    _write_atomic(directory / MANIFEST, json.dumps(manifest, indent=1, ensure_ascii=False).encode("utf-8"))


def load_probes(path: Path) -> dict[str, Probe]:
    out: dict[str, Probe] = {}
    with np.load(path, allow_pickle=False) as z:
        for i, qid in enumerate(str(q) for q in z["questions"]):
            k, n = (int(v) for v in z[f"q{i}_meta"])
            out[qid] = Probe(W=z[f"q{i}_W"].astype(np.float64), b=z[f"q{i}_b"].astype(np.float64),
                             classes=z[f"q{i}_classes"].astype(int), k=k, n=n)
    return out


def load_artifacts(directory: Path) -> Fitted | None:
    """The fitted state of a schema, or None when `tez fit` has not been run (or the files are unreadable)."""
    directory = Path(directory)
    cal_path = directory / CALIBRATION
    if not cal_path.exists():
        return None
    try:
        calibration = json.loads(cal_path.read_text(encoding="utf-8"))
        manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8")) if (directory / MANIFEST).exists() else {}
        probes = load_probes(directory / PROBES) if (directory / PROBES).exists() else {}
    except (OSError, ValueError, KeyError) as exc:
        log.warning("ignoring unreadable artefacts in %s: %s", directory, exc)
        return None
    questions = {}
    for qid, qc in (calibration.get("questions") or {}).items():
        pc = ReadoutCal.from_json(qc.get("probe"))
        questions[qid] = FittedQuestion(
            qid=qid, type=str(qc.get("type", "")), options=[str(o) for o in qc.get("options") or []],
            letters=ReadoutCal.from_json(qc.get("letters")), probe_cal=pc,
            probe=probes.get(qid) if pc is not None else None, blend=bool((qc.get("probe") or {}).get("blend", False)),
            note=qc.get("note"))
    return Fitted(name=str(calibration.get("schema", directory.name)), calibration_id=str(calibration.get("calibration_id", "")),
                  dir=directory, questions=questions, calibration=calibration, manifest=manifest)
