"""MASSIVE intent in all 51 languages Laya reports: Tez vs laya, laya-multilingual and routed Laya, same rows
(results/vs_laya/massive51.json).

Laya's protocol (research/scripts/bench_local.py part A, research/eval/laya_eval.py): mteb/amazon_massive_intent,
split test, the first 100 rows per language, 20 options = the gold intent + 19 sampled with a fresh
random.Random(13) per language, then shuffled; question "What is the user asking for in `utterance`?"; option text
= key with "_" -> " " and "." -> ": ". Random guessing = 0.05; "usable" = accuracy above 3x random (> 0.15), Laya's
languages_above_random. The cases come from bench_h2h.build_task("massive:<lang>"), which is checked row by row
against Laya's own sampling (laya_eval.build_suite logic) before anything is scored.

Systems
  tez                 zero-shot letters, bench_h2h.tez_decide against llama-server :8091. The 11 languages already
                      in results/h2h are reused only when their rows are identical: same ids, gold and option count,
                      and a re-score of the first --check rows gives the same probabilities (median max |dp| < 0.01;
                      a different prompt moves them by tenths). Otherwise the language is re-run. Full re-runs of ar,
                      fr, hi and km are kept as *_tez_rerun_check.jsonl (run-to-run agreement 97-100 % of rows).
  laya                convaiinnovations/laya (English), Laya's batched harness, CPU fp32
  laya-multilingual   convaiinnovations/laya subfolder multilingual, same
  laya-routed         Laya's router rule with the language given (laya/router.py: explicit lang "en" -> english
                      checkpoint, any other code -> multilingual): laya on en, laya-multilingual elsewhere.
                      (extras: the router's own per-utterance detection, Router().route(state), for reference)

Usage (repo root)
  python experiments/vs_laya_massive51.py --system tez
  python experiments/vs_laya_massive51.py --system laya --device cpu
  python experiments/vs_laya_massive51.py --system laya-multilingual --device cpu
  python experiments/vs_laya_massive51.py --aggregate
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "vs_laya"
ROWS = OUT / "massive51_rows"
H2H_ROWS = ROOT / "results" / "h2h"
PUBLISHED = OUT / "laya_published" / "cpu_51_language_sweep.json"
SEED, N_OPTS, PER_LANG = 13, 20, 100
RANDOM = 1.0 / N_OPTS
USABLE = 3.0 / N_OPTS
# Laya's 51 languages, in the order of research/results/cpu_51_language_sweep.json
LANGS51 = ["af", "am", "ar", "az", "bn", "cy", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he", "hi", "hu", "hy",
           "id", "is", "it", "ja", "jv", "ka", "km", "kn", "ko", "lv", "ml", "mn", "ms", "my", "nb", "nl", "pl", "pt",
           "ro", "ru", "sl", "sq", "sv", "sw", "ta", "te", "th", "tl", "tr", "ur", "vi", "zh-CN", "zh-TW"]
H2H_LANGS = ["en", "de", "fr", "es", "ja", "zh-CN", "ar", "hi", "th", "ko", "km"]
LAYA = {"laya": None, "laya-multilingual": "multilingual"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


H2H = _load("bench_h2h", ROOT / "experiments" / "bench_h2h.py")


def laya_suite(lang):
    """Laya's own sampling (research/eval/laya_eval.py build_suite): (keys per case, gold index, texts)."""
    from datasets import load_dataset
    d = load_dataset("mteb/amazon_massive_intent", lang, split="test")
    rows = [{"text": r["text"], "label_text": r["label_text"]} for r in d]
    labels = sorted({r["label_text"] for r in rows})
    rng = random.Random(SEED)
    out = []
    for row in rows[:PER_LANG]:
        pool = [x for x in labels if x != row["label_text"]]
        keys = [row["label_text"]] + rng.sample(list(pool), min(N_OPTS - 1, len(pool)))
        rng.shuffle(keys)
        out.append((keys, keys.index(row["label_text"]), row["text"]))
    return out


def cases_for(lang):
    """bench_h2h cases, asserted identical to Laya's sampling and option rendering."""
    cases = H2H.build_task(f"massive:{lang}", PER_LANG)
    ref = laya_suite(lang)
    assert len(cases) == len(ref) == PER_LANG, (lang, len(cases), len(ref))
    for c, (keys, gold, text) in zip(cases, ref):
        assert [k for k, _ in c["options"]] == keys and c["gold"] == gold and c["state"]["utterance"] == text, c["id"]
        assert [d for _, d in c["options"]] == [k.replace("_", " ").replace(".", ": ") for k in keys], c["id"]
    return cases


def rows_path(lang, system):
    return ROWS / f"{lang}_{system}.jsonl"


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ------------------------------------------------------------------ tez
def tez_record(server, c, lang):
    t = time.perf_counter()
    p, z, pn, calls = H2H.tez_decide(server, c)
    ms = (time.perf_counter() - t) * 1000
    return dict(id=c["id"], task=f"massive:{lang}", lang=lang, system="tez", gold=c["gold"], k=len(c["options"]),
                probabilities=[float(x) for x in p], logits=[float(x) for x in z], pred=int(np.argmax(p)),
                confidence=float(np.max(p)), correct=int(int(np.argmax(p)) == c["gold"]), ms=ms, prompt_n=pn, calls=calls)


def check_h2h(server, lang, cases, n_check):
    """Reuse results/h2h rows only if they are the same rows and a re-score reproduces them."""
    old = read_rows(H2H_ROWS / f"rows_massive_{lang}_tez.jsonl")
    info = {"lang": lang, "h2h_rows": len(old)}
    if len(old) != len(cases):
        info["reuse"] = False; info["why"] = "row count differs"; return None, info
    same = all(o["id"] == c["id"] and o["gold"] == c["gold"] and o["k"] == len(c["options"]) for o, c in zip(old, cases))
    if not same:
        info["reuse"] = False; info["why"] = "ids / gold / option counts differ"; return None, info
    diffs, agree = [], []
    for o, c in list(zip(old, cases))[:n_check]:
        r = tez_record(server, c, lang)
        diffs.append(float(np.max(np.abs(np.asarray(r["probabilities"]) - np.asarray(o["probabilities"])))))
        agree.append(r["pred"] == o["pred"])
    info.update(checked=len(agree), pred_agreement=float(np.mean(agree)), max_abs_prob_diff=float(max(diffs)),
                median_abs_prob_diff=float(np.median(diffs)))
    # The question is whether these are the same rows (same prompts). Same prompt => near-identical probabilities on
    # most rows; a different prompt (other options, order or wording) moves them by tenths. llama.cpp's numbers move
    # slightly with how much of the prompt was cached, which can flip a near-tie (full re-runs: ar 100/100 same answers,
    # hi and km one point apart), so the fingerprint is the median difference, and agreement is recorded, not required.
    ok = float(np.median(diffs)) < 1e-2
    info["reuse"] = bool(ok)
    if not ok:
        info["why"] = "re-score did not reproduce the stored answers"
        return None, info
    rows = [dict(o, lang=lang, system="tez", confidence=float(np.max(o["probabilities"])),
                 correct=int(o["pred"] == o["gold"]), source="results/h2h/rows_massive_%s_tez.jsonl" % lang) for o in old]
    return rows, info


def run_tez(server, langs, n_check, reuse):
    props = H2H.SESSION.get(f"{server}/props", timeout=30).json()
    print("llama-server model:", props.get("model_path"), props.get("build_info"), flush=True)
    checks = []
    for lang in langs:
        path = rows_path(lang, "tez")
        cases = cases_for(lang)
        if len([r for r in read_rows(path) if "error" not in r]) == len(cases):
            print(f"[tez] {lang}: already done", flush=True); continue
        if reuse and lang in H2H_LANGS:
            rows, info = check_h2h(server, lang, cases, n_check)
            checks.append(info)
            print(f"[tez] {lang}: h2h check {info}", flush=True)
            if rows is not None:
                path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                continue
        done = {r["id"] for r in read_rows(path) if "error" not in r}
        t0 = time.perf_counter()
        with open(path, "a", encoding="utf-8") as f:
            for c in cases:
                if c["id"] in done:
                    continue
                rec = None
                for attempt in range(3):
                    try:
                        rec = tez_record(server, c, lang); break
                    except Exception as exc:  # noqa: BLE001
                        err = f"{type(exc).__name__}: {str(exc)[:200]}"; time.sleep(1.0)
                if rec is None:
                    rec = dict(id=c["id"], lang=lang, system="tez", gold=c["gold"], error=err)
                f.write(json.dumps(rec) + "\n"); f.flush()
        rows = [r for r in read_rows(path) if "error" not in r]
        print(f"[tez] {lang}: acc={np.mean([r['correct'] for r in rows]):.3f} n={len(rows)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    if checks:
        ck = OUT / "massive51_h2h_reuse_check.json"
        prev = json.loads(ck.read_text(encoding="utf-8")) if ck.exists() else []
        prev = [p for p in prev if p["lang"] not in {c["lang"] for c in checks}] + checks
        ck.write_text(json.dumps(prev, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ laya
def run_laya(system, device, langs, threads):
    import torch
    import laya
    apps = _load("vs_laya_apps", ROOT / "experiments" / "vs_laya_apps.py")
    if threads:
        torch.set_num_threads(threads)
    agent = laya.load("convaiinnovations/laya", subfolder=LAYA[system], device=device)
    agent.model.eval()
    print(f"[{system}] device {agent.device} dtype {agent.dtype} threads {torch.get_num_threads()} "
          f"temps served {agent.temperature_by_options} raw {getattr(agent, 'temperature_by_options_raw', None)}", flush=True)
    for lang in langs:
        path = rows_path(lang, system)
        if len(read_rows(path)) == PER_LANG:
            print(f"[{system}] {lang}: already done", flush=True); continue
        cases = cases_for(lang)
        lcases = [dict(c, question=H2H.laya_question(c)) for c in cases]
        lgs, idx, secs, dropped, _ = apps.laya_score_cases(agent, lcases)
        recs = []
        for c, z, (qt, k) in zip(cases, lgs, idx):
            if z is None:
                recs.append(dict(id=c["id"], lang=lang, system=system, gold=c["gold"], error="dropped")); continue
            t_served, t_raw, bucket = apps.laya_temps(agent, qt, k)
            p = apps.softmax_t(z, t_served)
            p_raw = apps.softmax_t(z, t_raw)
            recs.append(dict(id=c["id"], lang=lang, system=system, gold=c["gold"], k=int(k), probabilities=[float(x) for x in p],
                             marker_logits=[float(x) for x in z], bucket=bucket, temperature=t_served, temperature_raw=t_raw,
                             confidence=float(np.max(p)), confidence_raw_temperature=float(np.max(p_raw)),
                             pred=int(np.argmax(p)), correct=int(int(np.argmax(p)) == c["gold"])))
        path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
        ok = [r for r in recs if "error" not in r]
        print(f"[{system}] {lang:6s} acc={np.mean([r['correct'] for r in ok]):.3f} conf={np.mean([r['confidence'] for r in ok]):.3f} "
              f"({secs:.1f}s, dropped {dropped})", flush=True)


# ------------------------------------------------------------------ aggregate
def lang_metrics(recs, conf_key="confidence"):
    g = np.array([r["gold"] for r in recs])
    pred = np.array([r["pred"] for r in recs])
    corr = (pred == g).astype(float)
    conf = np.array([r[conf_key] for r in recs])
    return {"n": len(recs), "accuracy": float(corr.mean()), "macro_f1": H2H.macro_f1(g, pred), "ece": H2H.ece15(conf, corr),
            "mean_confidence": float(conf.mean())}


def aggregate():
    from laya.router import Router
    systems, ece, detail = {}, {}, {}
    for s in ["tez", "laya", "laya-multilingual"]:
        per, pe = {}, {}
        for lang in LANGS51:
            recs = [r for r in read_rows(rows_path(lang, s)) if "error" not in r]
            if len(recs) != PER_LANG:
                continue
            m = lang_metrics(recs)
            per[lang] = round(m["accuracy"], 4)
            pe[lang] = round(m["ece"], 4)
            detail.setdefault(s, {})[lang] = {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
            if s != "tez":
                mr = lang_metrics(recs, "confidence_raw_temperature")
                detail[s][lang]["ece_raw_shipped_temperature"] = round(mr["ece"], 4)
                detail[s][lang]["mean_confidence_raw_shipped_temperature"] = round(mr["mean_confidence"], 4)
        systems[s] = per
        ece[s] = pe
    if systems.get("laya") and systems.get("laya-multilingual"):
        systems["laya-routed"] = {l: (systems["laya"][l] if l == "en" else systems["laya-multilingual"][l])
                                  for l in LANGS51 if l in systems["laya"] and l in systems["laya-multilingual"]}
        ece["laya-routed"] = {l: (ece["laya"][l] if l == "en" else ece["laya-multilingual"][l]) for l in systems["laya-routed"]}
    usable = {s: int(sum(1 for a in v.values() if a > USABLE)) for s, v in systems.items()}
    macro = {s: round(float(np.mean(list(v.values()))), 4) for s, v in systems.items() if v}

    # extras: the router's own per-utterance detection (no language hint)
    router, det = Router(), {}
    if systems.get("laya") and systems.get("laya-multilingual"):
        n_en_route = {}
        for lang in LANGS51:
            cases = H2H.build_task(f"massive:{lang}", PER_LANG)
            en = {r["id"]: r for r in read_rows(rows_path(lang, "laya"))}
            ml = {r["id"]: r for r in read_rows(rows_path(lang, "laya-multilingual"))}
            corr, n_en = [], 0
            for c in cases:
                pick = router.route(c["state"], {"intent": H2H.laya_question(c)}).model
                n_en += pick == "english"
                corr.append((en if pick == "english" else ml)[c["id"]]["correct"])
            det[lang] = round(float(np.mean(corr)), 4)
            n_en_route[lang] = n_en
    pub = json.loads(PUBLISHED.read_text(encoding="utf-8")) if PUBLISHED.exists() else None
    laya_pub, repro = {}, {}
    if pub:
        bm = pub["part_a"]["by_model"]
        laya_pub = {"laya": {l: bm["english"]["per_language"][l]["accuracy"] for l in LANGS51},
                    "laya-multilingual": {l: bm["multilingual"]["per_language"][l]["accuracy"] for l in LANGS51}}
        laya_pub["laya-routed"] = {l: (laya_pub["laya"][l] if l == "en" else laya_pub["laya-multilingual"][l]) for l in LANGS51}
        for s in ("laya", "laya-multilingual"):
            if systems.get(s):
                d = {l: round(systems[s][l] - laya_pub[s][l], 4) for l in systems[s]}
                repro[s] = {"languages_identical_accuracy": int(sum(1 for v in d.values() if abs(v) < 1e-9)),
                            "languages_compared": len(d), "max_abs_delta": max(abs(v) for v in d.values()),
                            "macro_accuracy_rerun": macro[s], "macro_accuracy_published": bm["english" if s == "laya" else "multilingual"]["macro_accuracy"],
                            "usable_rerun": usable[s], "usable_published": bm["english" if s == "laya" else "multilingual"]["languages_above_random"],
                            "per_language_delta": {l: v for l, v in d.items() if abs(v) > 1e-9}}
        laya_pub_usable = {s: int(sum(1 for a in v.values() if a > USABLE)) for s, v in laya_pub.items()}
    reuse_ck = OUT / "massive51_h2h_reuse_check.json"
    rerun = {}
    for p in sorted(ROWS.glob("*_tez_rerun_check.jsonl")):
        lang = p.name.split("_tez_rerun_check")[0]
        new, old = read_rows(p), read_rows(H2H_ROWS / f"rows_massive_{lang}_tez.jsonl")
        if [r["id"] for r in new] == [r["id"] for r in old]:
            d = [float(np.max(np.abs(np.asarray(a["probabilities"]) - np.asarray(b["probabilities"])))) for a, b in zip(new, old)]
            rerun[lang] = {"rows": len(new), "pred_agreement": float(np.mean([a["pred"] == b["pred"] for a, b in zip(new, old)])),
                           "accuracy_rerun": float(np.mean([a["pred"] == a["gold"] for a in new])),
                           "accuracy_stored": float(np.mean([b["pred"] == b["gold"] for b in old])),
                           "median_max_abs_prob_diff": float(np.median(d)), "max_abs_prob_diff": float(max(d)),
                           "file": str(p.relative_to(ROOT)).replace("\\", "/")}
    doc = {
        "meta": {
            "title": "MASSIVE intent, 51 languages, 20 options (Laya's protocol), same rows for every system",
            "date": datetime.date.today().isoformat(),
            "script": "experiments/vs_laya_massive51.py",
            "protocol": ("mteb/amazon_massive_intent test, first 100 rows per language, 20 options = gold + 19 sampled "
                         "(fresh random.Random(13) per language, then shuffled), question 'What is the user asking for in "
                         "`utterance`?'. Rows asserted identical to Laya's laya_eval.build_suite before scoring."),
            "usable_rule": "accuracy > 3 x random = 0.15 (Laya's languages_above_random)",
            "hardware": {"gpu": "NVIDIA GeForce RTX 5080 Laptop GPU 16 GB (WDDM)", "cpu": "Intel Core Ultra 9 275HX (24 cores)"},
            "tez": "Gemma 4 12B Q8_0, zero-shot letters (bench_h2h.tez_decide), llama-server b11100 on the RTX 5080 Laptop",
            "laya": "laya 0.3.5 package, convaiinnovations/laya root + multilingual checkpoints, Laya's batched harness, CPU fp32 (accuracy is temperature-free)",
            "laya_routed": "laya on 'en', laya-multilingual on every other language (laya/router.py with the language code given)",
            "tez_reuse_of_results_h2h": json.loads(reuse_ck.read_text(encoding="utf-8")) if reuse_ck.exists() else None,
            "tez_full_rerun_vs_results_h2h": rerun or None,
            "laya_published_source": "github.com/NandhaKishorM/laya research/results/cpu_51_language_sweep.json (laya 0.2.0, CPU); copy in results/vs_laya/laya_published/",
            "reproduction_note": ("laya (English) reproduces Laya's published per-language accuracy in all 51 languages (identical, "
                                  "macro 0.2269, 23 usable). laya-multilingual as released does not: 6/51 languages identical, "
                                  "max |delta| 0.16, macro 0.4008 vs 0.3661 published, usable 48 vs 45. The same released weights "
                                  "gave the same answers on CUDA bf16 in results/h2h (1,087 of 1,100 rows), and the typed-decisions "
                                  "and English checkpoints reproduce to the row in apps.json through the same code, so the gap sits "
                                  "in the multilingual checkpoint's environment (Laya's run loaded a local directory with laya 0.2.0); "
                                  "not isolated. The figure can show either set: 'systems' holds our reruns, 'laya_published' theirs."),
            "rows": "results/vs_laya/massive51_rows/<lang>_<system>.jsonl",
        },
        "langs": LANGS51,
        "random": RANDOM,
        "systems": systems,
        "usable_3x_random": usable,
        "macro_accuracy": macro,
        "laya_published": laya_pub,
        "laya_published_usable_3x_random": laya_pub_usable if pub else None,
        "reproduction": repro,
        "extras": {"ece": ece, "per_language_detail": detail,
                   "laya-routed-by-detection": det or None,
                   "laya-routed-by-detection_usable_3x_random": int(sum(1 for a in det.values() if a > USABLE)) if det else None,
                   "laya-routed-by-detection_note": "Router().route(state) per utterance with no language hint (default english when undecided)"},
    }
    (OUT / "massive51.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT / 'massive51.json'}")
    print("usable:", usable, "macro:", macro)
    if repro:
        print("reproduction:", {s: {k: v for k, v in r.items() if k != "per_language_delta"} for s, r in repro.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["tez", "laya", "laya-multilingual"])
    ap.add_argument("--langs", default="all")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--server", default="http://127.0.0.1:8091")
    ap.add_argument("--check", type=int, default=10, help="rows re-scored to validate reuse of results/h2h")
    ap.add_argument("--no-reuse", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    ROWS.mkdir(parents=True, exist_ok=True)
    langs = LANGS51 if a.langs == "all" else a.langs.split(",")
    if a.system == "tez":
        run_tez(a.server, langs, a.check, not a.no_reuse)
    elif a.system:
        run_laya(a.system, a.device, langs, a.threads)
    if a.aggregate:
        aggregate()


if __name__ == "__main__":
    main()
