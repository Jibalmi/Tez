"""The tables of docs/plans/voice-phase1-results.md, from the replay results in results/voice/ (nothing is re-run).

  python experiments/voice_phase1_tables.py            # prints markdown tables; writes results/voice/phase1_tables.json

ASR arms (replay_*_asr_*.json): WER, the recogniser's busy time per chunk of audio, and, for the decisive word of every
actionable command, how long after the word's end (its forced-alignment end) the recogniser's output first held
  complete  the whole word
  prefix    a prefix of it of at least two letters ("spot" for "spotify"), which the session already reads
measured in audio time (the audio position of the step that emitted it), so recogniser compute is not included.
Decision and end-to-end arms (replay_*_oracle_*.json, replay_*_e2e_*.json): the harness's own summary.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from voice_asr import normalise  # noqa: E402

RES = ROOT / "results" / "voice"


def load(path: str) -> tuple[dict, list[dict]]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    rows_path = Path(path).with_name(Path(path).stem + "_rows.jsonl")
    rows = [json.loads(x) for x in rows_path.read_text(encoding="utf-8").splitlines() if x.strip()] if rows_path.exists() else []
    return d, rows


def q(v, p):
    return round(float(np.percentile(v, p)), 1) if len(v) else None


def emission(rows: list[dict]) -> dict:
    full, part = [], []
    for r in rows:
        cw = r.get("commit_word", -1)
        if cw is None or cw <= 0:
            continue
        w = r["words"][cw - 1]
        ref = normalise(w["w"])
        fp = ff = None
        for e in r["emissions"]:
            h = e["text"].split()
            if len(h) >= cw:
                x = h[cw - 1]
                if fp is None and len(x) >= 2 and ref.startswith(x):
                    fp = e["audio_s"]
                if ff is None and x == ref:
                    ff = e["audio_s"]
        if ff is not None:
            full.append((ff - w["end"]) * 1000.0)
        if fp is not None:
            part.append((fp - w["end"]) * 1000.0)
    f, p = np.array(full), np.array(part)
    return {"complete": {"n": len(f), "mean": round(float(f.mean()), 1) if len(f) else None, "p50": q(f, 50), "p95": q(f, 95)},
            "prefix": {"n": len(p), "mean": round(float(p.mean()), 1) if len(p) else None, "p50": q(p, 50), "p95": q(p, 95),
                       "before_end": round(float((p <= 0).mean()), 3) if len(p) else None}}


def thermal(m: dict) -> str:
    tb = m.get("thermal_before") or {}
    ld = m.get("load") or {}
    s = f"{tb.get('c')}->{ld.get('gpu_c_max')} C"
    return s + (" [throttled]" if m.get("throttled") else "")


def asr_table() -> list[dict]:
    out = []
    for path in sorted(glob.glob(str(RES / "replay_*_asr_*.json"))):
        d, rows = load(path)
        m, s = d["manifest"], d["summary"]
        a = s["asr"]
        em = emission(rows)
        out.append({"tag": Path(path).stem.replace("replay_", ""), "set": m["set"], "asr": m["asr"]["name"],
                    "chunk_ms": round(1000 * (m["asr"].get("chunk_s") or 0)), "wer": a["wer"],
                    "busy_per_chunk_p50": a["busy_ms_per_chunk"]["p50"], "busy_per_chunk_p95": a["busy_ms_per_chunk"]["p95"],
                    "complete": em["complete"], "prefix": em["prefix"],
                    "lag_beyond_chunk_mean": a["emission_lag_beyond_chunk_ms"]["mean"], "cpu": m["load"].get("cpu_pct_mean"),
                    "thermal": thermal(m), "throttled": m.get("throttled"), "vram_used_mb": (m.get("gpu_loaded") or {}).get("used_mb")})
    return out


def decision_table(pattern: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(str(RES / pattern))):
        if path.endswith("_rows.jsonl"):
            continue
        d, _ = load(path)
        m, s = d["manifest"], d["summary"]
        dec = m.get("decision") or {}
        procs = (m.get("gpu_loaded") or {}).get("processes") or []
        out.append({"tag": Path(path).stem.replace("replay_", ""), "set": m["set"], "asr": m["asr"]["name"],
                    "model": dec.get("model"), "backend": "http" if str(dec.get("backend", "")).startswith("http") else "inproc",
                    "commit": m["args"].get("commit"), "readout": dec.get("readout"), "n": s["rows"],
                    "decide_p50": (s.get("decide_ms_per_update") or {}).get("p50"),
                    "decide_p95": (s.get("decide_ms_per_update") or {}).get("p95"),
                    "commit_p50": (s.get("commit_ms_per_update") or {}).get("p50"),
                    "word_to_decision_p50": (s.get("word_end_to_decision_ms") or {}).get("p50"),
                    "word_to_decision_p95": (s.get("word_end_to_decision_ms") or {}).get("p95"),
                    "all_words_p50": (s.get("all_words_end_to_decision_ms") or {}).get("p50"),
                    "first_action_p50": (s.get("word_end_to_first_action_ms") or {}).get("p50"),
                    "first_action_p95": (s.get("word_end_to_first_action_ms") or {}).get("p95"),
                    "projected_p50": (s.get("projected_word_end_to_action_ms") or {}).get("p50"),
                    "projected_p95": (s.get("projected_word_end_to_action_ms") or {}).get("p95"),
                    "dispatch_p50": (s.get("dispatch_ms") or {}).get("p50"),
                    "acc": s.get("intent_accuracy"), "acc_then_alt": s.get("intent_accuracy_then_alt"),
                    "none_recall": s.get("none_recall"), "harmful": s.get("harmful"), "actionable": s.get("actionable"),
                    "oos_false": s.get("oos_false_actions"), "oos": s.get("oos"),
                    "first_word": s.get("first_action_word_mean"),
                    "at_or_before": s.get("first_action_at_or_before_decisive"),
                    "wer": (s.get("asr") or {}).get("wer"), "parity": m.get("parity"), "cpu": m["load"].get("cpu_pct_mean"),
                    "thermal": thermal(m), "throttled": m.get("throttled"),
                    "vram_used_mb": (m.get("gpu_loaded") or {}).get("used_mb"),
                    "vram_python_mb": sum(p["mb"] for p in procs if p["name"] in ("python", "llama-server") and p["kind"] == "dedicated")})
    return out


def md(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(c for c, _ in cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for r in rows:
        vals = []
        for _, k in cols:
            v = r
            for part in k.split("."):
                v = v.get(part) if isinstance(v, dict) else None
            vals.append("-" if v is None else str(v))
        body += "| " + " | ".join(vals) + " |\n"
    return head + body


def main() -> None:
    asr = asr_table()
    dec = decision_table("replay_*_oracle_*.json")
    e2e = decision_table("replay_*_e2e_*.json")
    (RES / "phase1_tables.json").write_text(json.dumps({"asr": asr, "decision": dec, "e2e": e2e}, indent=1), encoding="utf-8")
    print(md(asr, [("arm", "tag"), ("chunk ms", "chunk_ms"), ("WER", "wer"), ("busy ms/chunk p50", "busy_per_chunk_p50"),
                   ("p95", "busy_per_chunk_p95"), ("word complete: mean", "complete.mean"), ("p50", "complete.p50"),
                   ("p95", "complete.p95"), ("prefix: mean", "prefix.mean"), ("p50", "prefix.p50"),
                   ("before end", "prefix.before_end"), ("GPU", "thermal")]))
    cols = [("arm", "tag"), ("decide p50", "decide_p50"), ("p95", "decide_p95"), ("word end->decision p50", "word_to_decision_p50"),
            ("p95", "word_to_decision_p95"), ("->first action p50", "first_action_p50"), ("p95", "first_action_p95"),
            ("acc", "acc"), ("then/alt", "acc_then_alt"), ("none recall", "none_recall"), ("harmful", "harmful"),
            ("oos false", "oos_false"), ("1st word", "first_word"), ("GPU", "thermal")]
    print(md(dec, cols))
    if e2e:
        print(md(e2e, cols[:1] + [("WER", "wer")] + cols[1:]))


if __name__ == "__main__":
    main()
