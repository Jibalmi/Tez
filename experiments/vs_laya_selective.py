"""Selective automation on typed-decisions, the reliability diagram and the risk-coverage curve
(results/vs_laya/selective.json). No model is run: everything comes from the per-row probabilities of the
head-to-head run on byte-identical rows (results/h2h/rows_typed_decisions_<model>.jsonl, written by
experiments/bench_h2h.py; 400 cases / 2,000 decisions, LocalLLaMA/typed-decisions test split).

  selective automation   automate only the most confident X % of decisions (X = 30, 40, ..., 100) and report the
                         accuracy on the automated ones. Confidence = max probability as shipped. Ties in confidence
                         (Laya's API rounds probabilities to 4 decimals) are broken at random in expectation: the
                         tied group at the cut contributes its mean accuracy pro rata, so the number does not depend
                         on row order.
  reliability            15 equal-width confidence bins (lo, hi] (the ECE-15 bins of bench_h2h / Laya): mean
                         confidence, accuracy and count per bin, for Tez letters as shipped (temperature 1) and after
                         the 2-fold out-of-fold temperature of BENCHMARKS.md section 1 (bench_h2h.summarise: folds by
                         row parity, one temperature fitted by NLL on each fold's letter logits and applied to the other).
  risk-coverage          answer the most confident first; error rate among the answered decisions at every coverage
                         i / n (same tie rule), plus AURC.

Usage (repo root):  python experiments/vs_laya_selective.py
"""
from __future__ import annotations

import datetime
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "vs_laya"
H2H_ROWS = ROOT / "results" / "h2h"
SYSTEMS = {"tez letters": "tez", "laya": "laya-en", "laya-multilingual": "laya-ml", "laya-typed-decisions": "laya-td"}
COVERAGE = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BINS = 15


def _load_h2h():
    spec = importlib.util.spec_from_file_location("bench_h2h", ROOT / "experiments" / "bench_h2h.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


H2H = _load_h2h()


def load_rows(model):
    p = H2H_ROWS / f"rows_typed_decisions_{model}.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def conf_corr(rows, probs=None):
    P = probs if probs is not None else [np.asarray(r["probabilities"], float) for r in rows]
    conf = np.array([float(np.max(p)) for p in P])
    corr = np.array([float(int(np.argmax(p)) == r["gold"]) for p, r in zip(P, rows)])
    return conf, corr


def expected_top_correct(conf, corr):
    """E[#correct among the m most confident], m = 1..n, with ties in confidence broken uniformly at random."""
    n = len(conf)
    order = np.argsort(-conf, kind="stable")
    c, k = conf[order], corr[order]
    out = np.zeros(n)
    i, done_correct = 0, 0.0
    while i < n:
        j = i
        while j < n and c[j] == c[i]:
            j += 1
        g_rate = k[i:j].mean()
        for m in range(i + 1, j + 1):          # m items taken, the last (m - i) from the tied group
            out[m - 1] = done_correct + (m - i) * g_rate
        done_correct += k[i:j].sum()
        i = j
    return out


def selective(conf, corr, coverage):
    n = len(conf)
    top = expected_top_correct(conf, corr)
    res = []
    for cov in coverage:
        m = max(1, int(round(cov * n)))
        res.append(round(float(top[m - 1] / m), 4))
    return res


def risk_coverage(conf, corr):
    n = len(conf)
    top = expected_top_correct(conf, corr)
    m = np.arange(1, n + 1)
    err = 1.0 - top / m
    return [round(float(x), 5) for x in m / n], [round(float(x), 5) for x in err], round(float(err.mean()), 4)


def reliability(conf, corr, bins=BINS):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        n = int(s.sum())
        out.append({"lo": round(float(lo), 4), "hi": round(float(hi), 4),
                    "conf": round(float(conf[s].mean()), 4) if n else None,
                    "acc": round(float(corr[s].mean()), 4) if n else None, "n": n})
    return out


def oof_temperature(rows):
    """bench_h2h.summarise's refit: 2 folds by row parity, T fitted by NLL on one fold, applied to the other."""
    probs = [None] * len(rows)
    temps = {}
    lg = [(r["gold"], np.asarray(r["logits"], float)) for r in rows]
    for fold in (0, 1):
        fit = [rw for i, rw in enumerate(lg) if i % 2 == fold]
        T = H2H.fit_temperature(fit)
        temps[f"fitted_on_parity_{fold}"] = round(T, 4)
        for i, (_, z) in enumerate(lg):
            if i % 2 != fold:
                probs[i] = H2H.softmax_t(z, T)
    return probs, temps


def probe_rows_available():
    """Per-row probabilities of a Tez probe on typed-decisions, if any run saved them (none of the repo's probe
    experiments do: they write summaries and hidden-state caches only)."""
    hits = []
    for p in (ROOT / "results").rglob("*"):
        if p.is_file() and p.suffix in (".jsonl", ".json") and "probe" in p.name.lower() and p.stat().st_size > 200_000:
            hits.append(str(p.relative_to(ROOT)))
    return hits


def main():
    doc = {"meta": {}, "coverage": COVERAGE, "systems": {}, "reliability": {}, "risk_coverage": {}, "summary": {}}
    for name, model in SYSTEMS.items():
        rows = load_rows(model)
        conf, corr = conf_corr(rows)
        doc["systems"][name] = selective(conf, corr, COVERAGE)
        cov, err, aurc = risk_coverage(conf, corr)
        doc["risk_coverage"][name] = {"coverage": cov, "error": err, "aurc": aurc}
        doc["summary"][name] = {"n": len(rows), "accuracy": round(float(corr.mean()), 4), "ece": round(H2H.ece15(conf, corr), 4),
                                "mean_confidence": round(float(conf.mean()), 4),
                                "distinct_confidence_values": int(len(np.unique(conf))),
                                "rows": f"results/h2h/rows_typed_decisions_{model}.jsonl"}
        if name == "tez letters":
            probs_t, temps = oof_temperature(rows)
            conf_t, corr_t = conf_corr(rows, probs_t)
            doc["reliability"][name] = {"shipped": reliability(conf, corr), "temperature": reliability(conf_t, corr_t)}
            cov_t, err_t, aurc_t = risk_coverage(conf_t, corr_t)
            doc["risk_coverage"]["tez letters (temperature)"] = {"coverage": cov_t, "error": err_t, "aurc": aurc_t}
            doc["systems_extra"] = {"tez letters (temperature)": selective(conf_t, corr_t, COVERAGE)}
            doc["summary"][name].update(ece_after_temperature=round(H2H.ece15(conf_t, corr_t), 4),
                                        accuracy_after_temperature=round(float(corr_t.mean()), 4),
                                        mean_confidence_after_temperature=round(float(conf_t.mean()), 4),
                                        temperatures=temps)
        else:
            doc["reliability"].setdefault("extras", {})[name] = {"shipped": reliability(conf, corr)}
    probe = probe_rows_available()
    doc["meta"] = {
        "title": "typed-decisions (400 cases, 2,000 decisions): selective automation, reliability, risk-coverage",
        "date": datetime.date.today().isoformat(),
        "script": "experiments/vs_laya_selective.py",
        "source_rows": {k: f"results/h2h/rows_typed_decisions_{v}.jsonl" for k, v in SYSTEMS.items()},
        "tez": "Gemma 4 12B Q8_0 zero-shot letters (bench_h2h.py), probabilities as shipped (temperature 1)",
        "laya": ("laya 0.3.5 Agent.predict with the checkpoints' shipped temperatures (cuda bf16), probabilities as returned "
                 "by the API (rounded to 4 decimals, hence ties; see tie rule)"),
        "selective_rule": "accuracy on the round(X * 2000) most confident decisions; ties at the cut counted pro rata (random tie-breaking in expectation)",
        "reliability_bins": "15 equal-width bins (lo, hi]; conf = mean max probability in the bin, acc = accuracy, n = count; empty bins have null conf/acc",
        "temperature": ("2-fold out-of-fold temperature exactly as bench_h2h.summarise (results/h2h/summary.json ece_refit "
                        "0.0478 for Tez typed-decisions; BENCHMARKS.md section 1's 'after one temperature per task (2-fold OOF)'). "
                        "BENCHMARKS.md section 4c lists 0.067 for the same readout from the probe-lab tables, whose fold "
                        "split is not in the repo's code; this file uses the bench_h2h procedure."),
        "tez_probe": ("absent: no per-row probe probabilities for typed-decisions exist in results/ (the probe experiments "
                      "saved summaries and hidden-state caches such as results/probe_cache_qwen35-4b.npz, not per-row "
                      "predictions)" if not probe else f"candidate files: {probe}"),
    }
    order = ["meta", "coverage", "systems", "systems_extra", "reliability", "risk_coverage", "summary"]
    doc = {k: doc[k] for k in order if k in doc}
    (OUT / "selective.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'selective.json'}")
    for k, v in doc["systems"].items():
        print(f"  {k:22s} {v}  AURC {doc['risk_coverage'][k]['aurc']}")
    print("  tez letters (temperature)", doc["systems_extra"]["tez letters (temperature)"])
    print("  summary", json.dumps(doc["summary"]["tez letters"]))


if __name__ == "__main__":
    main()
