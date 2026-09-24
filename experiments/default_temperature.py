"""A default letter temperature per question type, fitted on pooled labelled decisions (CPU only, saved rows only).

Tez applies T = 1 to the letter logits of a question that was never fitted (tez/engine.py, decide_question:
softmax(z, fq.letters.temperature if letters_ok else 1.0)). This script measures what one temperature per question
type (noul, choice, score) would do instead: fitted by NLL on the pooled labelled decisions of every task in
results/ read with the default model (Gemma 4 12B Q8_0, Ollama blob sha256-047dae1d...), and evaluated
leave-one-task-out. Nothing is re-run: no model is loaded, no server is contacted.

Readout, exactly as the engine does it (tez/backends.py letters_from_response, tez/engine.py letter_logits):
  z   the option letters' log-probabilities from llama-server's top-200 list; a letter absent from the list gets
      (lowest listed log-probability - 2). Every input below was read with n_probs = 200 and that same floor, so
      the saved logits already carry the engine's rule for the truncated mass. Files that keep only
      probabilities (voice, the JevBench runtime rows) hold softmax(z) at T = 1, so log p = z - logsumexp(z) and
      softmax(log p / T) = softmax(z / T) = p^(1/T) renormalised, exactly (no probability in them is 0 or 1).
  > 26 options: a chunked tournament; options that left it are -inf (probability 0 at every T) and the
      temperature acts on the final round only (Banking77: 4 finalists per row).
  p = softmax(z / T)  (tez.readout.softmax_rows).

Protocol
  unit      a task = one dataset or benchmark (MASSIVE's 51 languages and XNLI's 10 are one task each, JevBench's
            three tiers one task, each of Laya's seven application workflows its own task).
  pooled T  for each type, argmin over T in [0.05, 20] of the mean NLL of all decisions of that type (each decision
            weighs the same; the 120-point grid of tez.readout.T_GRID, then a golden-section refinement in log T).
  LOTO      for each type and each task with decisions of that type: T fitted on the other tasks' decisions of the
            type, measured on the held-out task's. A shipped value is the fit on all tasks.
  oracle    T fitted on the held-out task's own decisions (in-sample; a reference, not a deployable setting).
Settings compared: (a) T = 1 as shipped; (b) one pooled T per type; (b') as (b) with choice split into the buckets
2 / 3-5 / 6-10 / 11-26 options (a tournament goes by its finalists, the options its tempered softmax spans); (r) as (b)
with choice split only at 10 options (2-10 / 11-26), the rule the report recommends; (c) the per-task oracle. Variants
(task-balanced pools, one T for everything, runtime prompt layout only, ...) are each leave-one-task-out too.
Metrics (the engine's definitions, imported from tez/readout.py): accuracy (argmax), ECE-15 (tez.readout.ece: max
probability vs correctness, 15 equal-width bins (lo, hi]; the same code as bench_h2h.ece15 behind BENCHMARKS.md),
NLL (tez.readout.nll, p clipped at 1e-12), multi-class Brier (sum over options, as bench_h2h), and for score
questions the MAE of the expected level sum(i * p_i) against the gold level.

  python experiments/default_temperature.py            # prints the tables, writes results/calibration/*.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tez.readout import T_GRID, ece, fit_temperature_logits, nll, softmax_rows  # noqa: E402  (engine definitions)

RES = ROOT / "results"
OUT_DIR = RES / "calibration"
OUT_JSON = OUT_DIR / "default_temperature.json"
OUT_MD = OUT_DIR / "default_temperature.md"
TYPES = ("noul", "choice", "score")
MODEL_BLOB = "sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795"
CLIP = 1e-12
H2H_TASKS = ["typed_decisions", "ag_news", "emotion", "banking77", "sst5", "boolq", "prompt_injections"]
XNLI_LANGS = ["en", "de", "fr", "es", "ar", "hi", "th", "zh", "ru", "tr"]
MASSIVE_H2H_LANGS = ["en", "de", "fr", "es", "ja", "zh-CN", "ar", "hi", "th", "ko", "km"]
APPS = ["email_spam", "phishing", "guardrails_jailbreak", "moderation_toxicity", "rag_relevance", "support_triage",
        "model_routing_domain"]
BUCKETS = ("2", "3-5", "6-10", "11-26", ">26")
N_LABELS = (5, 10, 20, 50)
DRAWS = 100
SEED = 20260924

EXCLUDED = [
    ("results/INVALID_wrongmodel_*", "quarantined: runs that hit the wrong model"),
    ("results/*gemma3-4b*, *llama32-3b*, *qwen3*/qwen35* (authored144, perturb108, h2h_qwen35-9b, h2h_q4b_L*, "
     "speed/*q4b*), jevbench_tez_qwen35-9b-q8_short.json", "another model than the default Gemma 4 12B Q8_0"),
    ("results/*gemma4-12b-q4km*", "Q4_K_M quantisation, not the default Q8_0 weights"),
    ("results/h2h/rows_*_laya-*.jsonl, results/vs_laya/*/*laya*", "Laya's checkpoints, not Tez"),
    ("results/authored144_gemma4-12b-q8_0_run2/_run3.jsonl", "repeats of run 1 (bit-identical per BENCHMARKS.md)"),
    ("results/authored144_{rev,perm021,perm102,perm120,perm201}_gemma4-12b-q8_0.jsonl, perturb108_gemma4-12b-q8_0.jsonl",
     "the same 144 / 36 authored items with permuted options or perturbations: would count items several times"),
    ("results/authored144_{cf,cfempty,cfmask}_gemma4-12b-q8_0.jsonl", "content-free inputs: the gold does not apply"),
    ("results/authored144_gemma4-12b-q8_0_{digits,lower,sys}.jsonl", "answer-symbol / system-prompt variants the runtime does not use"),
    ("results/h2h/rows_massive_*_tez.jsonl", "identical to results/vs_laya/massive51_rows/<lang>_tez.jsonl (checked row by row); "
     "MASSIVE is read once, from massive51_rows"),
    ("results/vs_laya/massive51_rows/*_tez_rerun_check.jsonl", "re-runs of four languages kept only as a determinism check"),
    ("results/jevbench_tez_gemma4-12b-q8_0*.json", "the same 231 items through the benchmark harness (its own noul option wording); "
     "the runtime rows (tez serve) are used instead"),
    ("results/jevbench_tez_server_short.json", "first pass after a restart; the warm second pass (short_warm) is the one BENCHMARKS.md quotes"),
    ("results/jevbench_hard_think64_gemma4-12b-q8_0.jsonl, results/h2h/rows_typed_decisions_tez-think*.jsonl",
     "thinking-budget variants (not the one-pass readout)"),
    ("results/h2h/rows_typed_decisions_tez-fewshot4*.jsonl", "few-shot prefix: used only as a side check, not pooled"),
    ("results/voice_gemma4-12b-q8_0.jsonl, voicefast_{final_first,partial_last}_gemma4-12b-q8_0.jsonl",
     "the same 220 utterances with non-default prompt layouts"),
    ("results/*_stream.jsonl, voice_ddm_*, stream_*", "per-word prefixes of an utterance: the gold of a prefix is undefined"),
    ("results/residual_check_*, asr_bench_*, retrieval_narrow_*, option_elimination_*, cascade_*, perm_analysis_*, "
     "conformal_*, batch_calibration.json", "summaries or other readouts (no per-decision letter probabilities with gold)"),
    ("results/teacher_labels_gemma4-12b-q8_0.*", "typed-decisions train split (same task) and a duplicate of the test rows"),
    ("results/hidden_probe_*, probe_*, probe_cache_*", "probe readouts / other models' hidden states, not letters"),
    ("results/speed/*", "another session's timing runs; td_rows_prod_today_http.jsonl is used only as a reproducibility check"),
]


# ------------------------------------------------------------------------------------------ data
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


class Data:
    """One row per labelled decision: z (letter log-probabilities, -inf = left a tournament), gold index, metadata."""

    def __init__(self) -> None:
        self.z: list[np.ndarray] = []
        self.z_stored: list[np.ndarray] = []     # the logits exactly as saved (for reproducing bench_h2h's refit)
        self.cols: dict[str, list] = defaultdict(list)
        self.inputs: list[dict] = []

    def add(self, z, gold: int, *, task: str, subtask: str, qtype: str, source: str, prompt: str, rid: str,
            gold_score: float = float("nan"), ambiguous: bool = False, z_stored=None) -> None:
        z = np.asarray(z, float)
        if not 0 <= gold < len(z):
            raise ValueError(f"{source} {rid}: gold {gold} outside {len(z)} options")
        if qtype not in TYPES:
            raise ValueError(f"{source} {rid}: unknown type {qtype}")
        self.z.append(z)
        self.z_stored.append(np.asarray(z if z_stored is None else z_stored, float))
        for k, v in dict(task=task, subtask=subtask, qtype=qtype, source=source, prompt=prompt, rid=rid, gold=gold,
                         gold_score=gold_score, ambiguous=ambiguous).items():
            self.cols[k].append(v)

    def input(self, path: Path, **info) -> None:
        self.inputs.append({"path": rel(path), "sha256": sha256(path), **info})

    def finalize(self) -> "Data":
        for k, v in self.cols.items():
            setattr(self, k, np.asarray(v))
        self.k = np.array([len(z) for z in self.z])
        self.k_final = np.array([int(np.isfinite(z).sum()) for z in self.z])
        self.tournament = self.k > 26
        self.n = len(self.z)
        self.bucket = np.array([bucket_of(q, k) for q, k in zip(self.qtype, self.k)])
        self.bucket_final = np.array([bucket_of(q, k) for q, k in zip(self.qtype, self.k_final)])
        self.rule = np.array([rule_of(q, k) for q, k in zip(self.qtype, self.k_final)])
        if len({(t, r) for t, r in zip(self.task, self.rid)}) != self.n:
            raise ValueError("duplicate decision ids within a task")
        return self


def bucket_of(qtype: str, k: int) -> str:
    """The requested option-count buckets for choice (2, 3-5, 6-10, 11-26, > 26); other types are their own cell."""
    if qtype != "choice":
        return qtype
    if k <= 2:
        return "2"
    if k <= 5:
        return "3-5"
    if k <= 10:
        return "6-10"
    if k <= 26:
        return "11-26"
    return ">26"


def rule_of(qtype: str, k: int) -> str:
    """The two-level rule (r): choice with up to 10 options vs 11-26 (a tournament goes by its finalists, <= 20)."""
    if qtype != "choice":
        return qtype
    return "2-10" if k <= 10 else "11-26"


def floored_lower_bound(zs) -> int:
    """Rows where at least two letters share the exact lowest value: the engine's floor (lowest listed - 2) was used.
    A single floored letter cannot be told apart from a genuine minimum, so this is a lower bound."""
    n = 0
    for z in zs:
        f = np.asarray(z, float)
        f = f[np.isfinite(f)]
        n += int((f == f.min()).sum() >= 2)
    return n


def check_probs(z: np.ndarray, p, where: str) -> None:
    q = softmax_rows(z[None, :], 1.0)[0]
    d = float(np.abs(q - np.asarray(p, float)).max())
    if d > 1e-6:
        raise ValueError(f"{where}: stored probabilities differ from softmax(logits) by {d:.2e}")


def load(voice_ambiguous: bool = True) -> Data:
    D = Data()
    # --- Laya's public suite, head-to-head rows (experiments/bench_h2h.py, the runtime's prompt layout)
    summary = json.loads((RES / "h2h" / "summary.json").read_text(encoding="utf-8"))
    for task in H2H_TASKS + [f"xnli_{lang}" for lang in XNLI_LANGS]:
        path = RES / "h2h" / f"rows_{task}_tez.jsonl"
        name, sub = ("xnli", task.split("_", 1)[1]) if task.startswith("xnli_") else (task, task)
        model = summary[task.replace("xnli_", "xnli:")]["tez"]["model"]
        if MODEL_BLOB not in model:
            raise ValueError(f"{path}: model {model}")
        rows = read_jsonl(path)
        for r in rows:
            zs = np.asarray(r["logits"], float)
            p = np.asarray(r["probabilities"], float)
            z = np.where(p > 0, zs, -np.inf) if r["k"] > 26 else zs    # tournament: the engine puts -inf there
            check_probs(z, p, f"{path.name} {r['id']}")
            gs = float(r.get("extra", {}).get("gold_score", "nan")) if r["qtype"] == "score" else float("nan")
            D.add(z, int(r["gold"]), task=name, subtask=(r.get("workflow") or sub) if name == "typed_decisions" else sub,
                  qtype=r["qtype"], source=rel(path), prompt="engine", rid=r["id"], gold_score=gs, z_stored=zs)
        D.input(path, task=name, rows=len(rows), stored="logits + probabilities", prompt="engine layout (bench_h2h.tez_prompt)",
                model=model, rows_with_floored_letters_at_least=floored_lower_bound(r["logits"] for r in rows))
    # --- MASSIVE, 51 languages (experiments/vs_laya_massive51.py; the 11 h2h languages are byte-identical reuses)
    meta = json.loads((RES / "vs_laya" / "massive51.json").read_text(encoding="utf-8"))["meta"]
    for path in sorted((RES / "vs_laya" / "massive51_rows").glob("*_tez.jsonl")):
        if "rerun" in path.name:
            continue
        lang = path.name[: -len("_tez.jsonl")]
        rows = read_jsonl(path)
        for r in rows:
            z = np.asarray(r["logits"], float)
            check_probs(z, r["probabilities"], f"{path.name} {r['id']}")
            D.add(z, int(r["gold"]), task="massive", subtask=lang, qtype="choice", source=rel(path), prompt="engine",
                  rid=r["id"])
        D.input(path, task="massive", rows=len(rows), stored="logits + probabilities", prompt="engine layout",
                model=meta["tez"], rows_with_floored_letters_at_least=floored_lower_bound(r["logits"] for r in rows))
    # --- Laya's seven application workflows (experiments/vs_laya_apps.py)
    apps_meta = json.loads((RES / "vs_laya" / "apps.json").read_text(encoding="utf-8"))["meta"]["tez"]
    if MODEL_BLOB not in apps_meta["model_path"]:
        raise ValueError("apps: unexpected model")
    qtypes = {}
    for c in read_jsonl(RES / "vs_laya" / "apps_cases.jsonl"):
        qtypes.setdefault(c["task"], c["question"]["type"])
    for app in APPS:
        path = RES / "vs_laya" / "apps_rows" / f"{app}_tez.jsonl"
        rows = read_jsonl(path)
        for r in rows:
            z = np.asarray(r["logits"], float)
            check_probs(z, r["probabilities"], f"{path.name} {r['id']}")
            D.add(z, int(r["gold"]), task=app, subtask=app, qtype=qtypes[app], source=rel(path), prompt="engine", rid=r["id"])
        D.input(path, task=app, rows=len(rows), stored="logits + probabilities",
                prompt="engine layout, options rendered by tez.schema.Question.options", model=apps_meta["model_path"],
                rows_with_floored_letters_at_least=floored_lower_bound(r["logits"] for r in rows))
    D.input(RES / "vs_laya" / "apps_cases.jsonl", task="(question types of the seven apps)", rows=None, stored="cases")
    # --- JevBench public tiers through the runtime (tez serve -> llama-server; release/jevbench/run_public.py)
    for name in ("jevbench_tez_server_short_warm.json", "jevbench_tez_server_hard.json"):
        path = RES / name
        d = json.loads(path.read_text(encoding="utf-8"))
        m = d["manifest"]
        if not (m["endpoint"].startswith("http://127.0.0.1:8787") and m["healthz"]["model"] == "gemma-4-12b-q8_0"
                and all(s.startswith("tez-") for s in m["server_models"])):
            raise ValueError(f"{path}: not the local Tez runtime")
        for r in d["rows"]:
            if r["status"] != "ok":
                raise ValueError(f"{path}: row {r['id']} status {r['status']}")
            with np.errstate(divide="ignore"):
                z = np.log(np.asarray(r["probabilities"], float))
            D.add(z, int(r["gold"]), task="jevbench", subtask=r["tier"], qtype=r["qtype"], source=rel(path), prompt="engine",
                  rid=r["id"])
        D.input(path, task="jevbench", rows=len(d["rows"]), stored="probabilities only (runtime answers, T = 1)",
                prompt="the runtime itself", model=f"{m['server_models'][0]} via {m['endpoint']}",
                rows_with_floored_letters_at_least=None, floor_note="not recoverable from probabilities; the runtime floors "
                "missing letters at (lowest listed - 2) before the softmax, so the stored probabilities already include it")
    # --- SemIf authored144 (experiments/run_direct.py: SemIf's own prompt in the Gemma 4 template)
    path = RES / "authored144_gemma4-12b-q8_0.jsonl"
    man = json.loads((RES / "authored144_gemma4-12b-q8_0.manifest.json").read_text(encoding="utf-8"))
    if MODEL_BLOB not in man["model_path"] or man["n_probs"] != 200:
        raise ValueError("authored144: unexpected model or n_probs")
    rows = read_jsonl(path)
    for r in rows:
        prov = r.get("provenance") or {}
        if "typesafe" in json.dumps(prov).lower() or "jev" in json.dumps(prov).lower():
            raise ValueError(f"authored144 {r['id']}: label provenance mentions TypeSafe/Jev")
        z = np.asarray(r["option_logprobs"], float)
        check_probs(z, r["probabilities"], f"{path.name} {r['id']}")
        D.add(z, int(r["gold"]), task="semif", subtask=r["family"], qtype="choice", source=rel(path), prompt="semif",
              rid=r["id"])
    D.input(path, task="semif", rows=len(rows), stored="option logprobs + probabilities",
            prompt="SemIf direct-options-v1 in the Gemma 4 template", model=man["model_path"],
            rows_with_floored_letters=sum(1 for r in rows if r["missing_letters"]))
    # --- voice commands, transcript last, final wording (experiments/run_voice_fast.py; 16 options incl. none)
    path = RES / "voicefast_final_last_gemma4-12b-q8_0.jsonl"
    man = json.loads((RES / "voicefast_final_last_gemma4-12b-q8_0.manifest.json").read_text(encoding="utf-8"))
    if MODEL_BLOB not in man["model_path"] or man["n_probs"] != 200:
        raise ValueError("voice: unexpected model or n_probs")
    rows = read_jsonl(path)
    for r in rows:
        amb = bool(r.get("then") or r.get("alt"))
        if amb and not voice_ambiguous:
            continue
        with np.errstate(divide="ignore"):
            z = np.log(np.asarray(r["probabilities"], float))
        D.add(z, r["option_ids"].index(r["gold_intent"]), task="voice", subtask=r["style"], qtype="choice", source=rel(path),
              prompt="voice", rid=r["id"], ambiguous=amb)
    D.input(path, task="voice", rows=len(rows), stored="probabilities only (T = 1)",
            prompt="voice router prompt (actions first, transcript last)", model=man["model_path"],
            rows_with_floored_letters=sum(1 for r in rows if r["missing_letters"]))
    return D.finalize()


# ------------------------------------------------------------------------------------------ readout
def softmax_T(Z: np.ndarray, T) -> np.ndarray:
    """tez.readout.softmax_rows with a temperature per row (same arithmetic)."""
    T = np.asarray(T, float)
    A = np.asarray(Z, float) / (T.reshape(-1, 1) if T.ndim else float(T))
    finite = np.isfinite(A)
    m = np.max(np.where(finite, A, -np.inf), axis=1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    E = np.where(finite, np.exp(np.clip(np.where(finite, A, 0.0) - m, -745.0, 0.0)), 0.0)
    s = E.sum(axis=1, keepdims=True)
    return E / np.where(s > 0, s, 1.0)


def blocks(D: Data, idx: np.ndarray, stored: bool = False) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Rows grouped by option count: [(positions within idx, Z of shape (m, k), gold of shape (m,))]."""
    src = D.z_stored if stored else D.z
    out = []
    ks = D.k[idx]
    for k in np.unique(ks):
        pos = np.flatnonzero(ks == k)
        out.append((pos, np.stack([src[i] for i in idx[pos]]), D.gold[idx[pos]]))
    return out


def readout(D: Data, idx: np.ndarray, T, stored: bool = False) -> dict:
    """Per-decision outputs at temperature T (a scalar or one per row of idx)."""
    idx = np.asarray(idx)
    T = np.broadcast_to(np.asarray(T, float), (len(idx),))
    n = len(idx)
    conf, pred, nll_i, brier, ev = np.zeros(n), np.zeros(n, int), np.zeros(n), np.zeros(n), np.zeros(n)
    med = np.zeros(n, int)
    y = D.gold[idx]
    for pos, Z, yy in blocks(D, idx, stored):
        P = softmax_T(Z, T[pos])
        med[pos] = (np.cumsum(P, axis=1) >= 0.5).argmax(axis=1)
        conf[pos] = P.max(axis=1)
        pred[pos] = P.argmax(axis=1)
        nll_i[pos] = -np.log(np.clip(P[np.arange(len(pos)), yy], CLIP, 1.0))
        onehot = np.zeros_like(P)
        onehot[np.arange(len(pos)), yy] = 1.0
        brier[pos] = ((P - onehot) ** 2).sum(axis=1)
        ev[pos] = P @ np.arange(P.shape[1])
    return dict(conf=conf, pred=pred, correct=pred == y, nll=nll_i, brier=brier, ev=ev, median=med, y=y,
                gold_score=D.gold_score[idx])


def metrics(out: dict, sel=None) -> dict:
    s = slice(None) if sel is None else sel
    conf, correct = out["conf"][s], out["correct"][s]
    if len(conf) == 0:
        return {"n": 0}
    return {"n": int(len(conf)), "accuracy": float(correct.mean()), "ece": ece(conf, correct), "nll": float(out["nll"][s].mean()),
            "brier": float(out["brier"][s].mean()), "mean_confidence": float(conf.mean())}


def score_metrics(out: dict, sel=None) -> dict:
    """The expected level sum(i * p_i) against the gold level (and against a benchmark's gold expected score)."""
    s = slice(None) if sel is None else sel
    ev, y, gs = out["ev"][s], out["y"][s], out["gold_score"][s]
    err = ev - y
    res = {"mae_level": float(np.abs(err).mean()), "mse_level": float((err ** 2).mean()),
           "within_1_level": float((np.abs(err) <= 1.0).mean()),
           "mae_median_level": float(np.abs(out["median"][s] - y).mean()),     # the median of the distribution
           "mae_mode_level": float(np.abs(out["pred"][s] - y).mean())}         # the argmax (the same at every T)
    ok = np.isfinite(gs)
    if ok.any():
        e2 = ev[ok] - gs[ok]
        res.update(mae_gold_expected_score=float(np.abs(e2).mean()), mse_gold_expected_score=float((e2 ** 2).mean()),
                   n_gold_expected_score=int(ok.sum()))
    return res


def mean_nll(blks, T: float, w=None) -> float:
    tot, wsum = 0.0, 0.0
    for pos, Z, yy in blks:
        P = softmax_T(Z, T)
        li = -np.log(np.clip(P[np.arange(len(pos)), yy], CLIP, 1.0))
        if w is None:
            tot += li.sum()
            wsum += len(pos)
        else:
            tot += (li * w[pos]).sum()
            wsum += w[pos].sum()
    return tot / wsum


def fit_T(D: Data, idx: np.ndarray, weights=None) -> tuple[float, bool]:
    """argmin_T mean NLL on the rows idx (optionally weighted): the engine's 120-point grid, then golden section
    in log T between the grid neighbours of the best point. Returns (T, whether the grid minimum sat on an edge)."""
    idx = np.asarray(idx)
    blks = blocks(D, idx)
    w = None if weights is None else np.asarray(weights, float)
    f = lambda t: mean_nll(blks, t, w)  # noqa: E731
    vals = np.array([f(t) for t in T_GRID])
    i = int(np.argmin(vals))
    a, b = math.log(T_GRID[max(i - 1, 0)]), math.log(T_GRID[min(i + 1, len(T_GRID) - 1)])
    g = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(math.exp(c)), f(math.exp(d))
    while b - a > 1e-6:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(math.exp(d))
    t = math.exp((a + b) / 2.0)
    if f(t) > vals[i]:        # never worse than the grid point
        t = float(T_GRID[i])
    return float(t), i in (0, len(T_GRID) - 1)


def task_weights(D: Data, idx: np.ndarray) -> np.ndarray:
    """Each task weighs the same in a pool: 1 / (its number of rows in the pool)."""
    cnt = Counter(D.task[idx])
    return np.array([1.0 / cnt[t] for t in D.task[idx]])


# ------------------------------------------------------------------------------------------ analyses
def type_tasks(D: Data, qtype: str, mask=None) -> list[str]:
    m = D.qtype == qtype if mask is None else (D.qtype == qtype) & mask
    return sorted(set(D.task[m]))


def loto_fits(D: Data, key: np.ndarray, pool_mask: np.ndarray, balanced: bool = False) -> dict:
    """For each cell value c of `key` (a type, or a type / option-count bucket): T on all pooled rows of c, and T with
    each task left out. {c: {"all": T, "edge": bool, "n": int, "tasks": [...], "loto": {task: T}}}."""
    res = {}
    for c in sorted(set(key[pool_mask])):
        cm = (key == c) & pool_mask
        idx = np.flatnonzero(cm)
        t_all, edge = fit_T(D, idx, task_weights(D, idx) if balanced else None)
        tasks = sorted(set(D.task[idx]))
        loto = {}
        for t in tasks:
            j = np.flatnonzero(cm & (D.task != t))
            if len(j):
                loto[t] = fit_T(D, j, task_weights(D, j) if balanced else None)[0]
        res[str(c)] = {"all": t_all, "edge": edge, "n": int(len(idx)), "tasks": [str(t) for t in tasks], "loto": loto}
    return res


def held_out_T(D: Data, key: np.ndarray, pool_mask: np.ndarray | None = None, balanced: bool = False) -> tuple[np.ndarray, dict, dict]:
    """Per-row temperature fitted without the row's task: the row's `key` cell, else (the cell has no other task) its
    type. A row whose task is not in the pool gets the pool's all-task fit, which is already held out."""
    pool_mask = np.ones(D.n, bool) if pool_mask is None else pool_mask
    fits = loto_fits(D, key, pool_mask, balanced)
    by_type = fits if np.array_equal(key, D.qtype) else loto_fits(D, D.qtype, pool_mask, balanced)

    def pick(f, t):
        if f is None:
            return None
        if t not in f["tasks"]:
            return f["all"]
        return f["loto"].get(t)

    T = np.ones(D.n)
    for i in range(D.n):
        t = str(D.task[i])
        v = pick(fits.get(str(key[i])), t)
        if v is None:
            v = pick(by_type.get(str(D.qtype[i])), t)
        if v is None:
            raise ValueError(f"no held-out temperature for {t} / {key[i]}")
        T[i] = v
    return T, fits, by_type


def oracle_fits(D: Data) -> dict:
    out = {}
    for q in TYPES:
        out[q] = {}
        for t in type_tasks(D, q):
            idx = np.flatnonzero((D.qtype == q) & (D.task == t))
            out[q][str(t)] = fit_T(D, idx)[0]
    return out


def evaluate_settings(D: Data, settings: dict[str, np.ndarray]) -> dict:
    """Metrics per (task, type) cell, per task, per subtask, and aggregates, for each per-row temperature setting."""
    outs = {name: readout(D, np.arange(D.n), T) for name, T in settings.items()}
    tasks = sorted(set(D.task))
    cells, per_task, per_sub = [], {}, {}
    for t in tasks:
        per_task[str(t)] = {name: metrics(o, D.task == t) for name, o in outs.items()}
        for q in TYPES:
            m = (D.task == t) & (D.qtype == q)
            if not m.any():
                continue
            cell = {"task": str(t), "type": q, "n": int(m.sum()), "k": sorted(int(x) for x in set(D.k[m]))}
            for name, o in outs.items():
                cell[name] = metrics(o, m)
                if q == "score":
                    cell[name].update(score_metrics(o, m))
                u = np.unique(settings[name][m])
                cell[f"T_{name}"] = float(u[0]) if len(u) == 1 else [float(x) for x in u]
            cells.append(cell)
        for s in sorted(set(D.subtask[D.task == t])):
            m = (D.task == t) & (D.subtask == s)
            per_sub[f"{t}/{s}"] = {name: metrics(o, m) for name, o in outs.items()}
    agg = {}
    keys = ("accuracy", "ece", "nll", "brier")
    for name in outs:
        agg[name] = {
            "macro_tasks": {k: float(np.mean([per_task[str(t)][name][k] for t in tasks])) for k in keys},
            "macro_cells": {k: float(np.mean([c[name][k] for c in cells])) for k in keys},
            "micro_all": metrics(outs[name]),
            "micro_by_type": {q: metrics(outs[name], D.qtype == q) for q in TYPES},
            "macro_by_type": {q: {k: float(np.mean([c[name][k] for c in cells if c["type"] == q])) for k in keys} for q in TYPES},
        }
    return {"cells": cells, "per_task": per_task, "per_subtask": per_sub, "aggregate": agg, "_outs": outs}


def worse_than_raw(res: dict, name: str) -> dict:
    cells = res["cells"]
    hi = max(cells, key=lambda c: c[name]["ece"])

    def pick(k):
        return [{"task": c["task"], "type": c["type"], "n": c["n"], f"{k}_a": c["raw"][k], f"{k}_setting": c[name][k]}
                for c in sorted(cells, key=lambda c: c["raw"][k] - c[name][k]) if c[name][k] > c["raw"][k]]
    gain = max(cells, key=lambda c: c["raw"]["ece"] - c[name]["ece"])
    return {"highest_ece": {"task": hi["task"], "type": hi["type"], "ece_a": hi["raw"]["ece"], "ece_setting": hi[name]["ece"]},
            "largest_ece_gain": {"task": gain["task"], "type": gain["type"], "ece_a": gain["raw"]["ece"], "ece_setting": gain[name]["ece"]},
            "ece_worse": pick("ece"), "nll_worse": pick("nll"), "brier_worse": pick("brier")}


ANCHOR_ENTRIES = ([(t, None) for t in H2H_TASKS] + [("massive", lang) for lang in MASSIVE_H2H_LANGS]
                  + [("xnli", lang) for lang in XNLI_LANGS])


def _entry_mask(D: Data, t: str, sub: str | None) -> np.ndarray:
    return (D.task == t) if sub is None else (D.task == t) & (D.subtask == sub)


def repo_refit_28(D: Data) -> dict:
    """bench_h2h.summarise's 'ece_refit': per entry, one T (all the entry's rows, whatever their type) fitted on the
    even rows and applied to the odd ones and vice versa, on the logits as stored, grid of 120, first minimum."""
    out = {}
    for t, sub in ANCHOR_ENTRIES:
        idx = np.flatnonzero(_entry_mask(D, t, sub))
        conf2, corr2 = [], []
        for fold in (0, 1):
            fit = idx[np.arange(len(idx)) % 2 == fold]
            ev = idx[np.arange(len(idx)) % 2 != fold]
            best, bt = 1e9, 1.0
            fb = blocks(D, fit, stored=True)
            for tt in np.exp(np.linspace(np.log(0.05), np.log(20), 120)):
                v = mean_nll(fb, tt)
                if v < best:
                    best, bt = v, float(tt)
            o2 = readout(D, ev, bt, stored=True)
            conf2 += o2["conf"].tolist()
            corr2 += o2["correct"].tolist()
        out[t if sub is None else f"{t}:{sub}"] = ece(np.array(conf2), np.array(corr2))
    return out


def anchor_28(D: Data, outs: dict, refit: dict | None = None) -> dict:
    """BENCHMARKS.md's 'mean ECE-15 as shipped' (0.212) averages the 28 head-to-head entries (seven tasks, MASSIVE in 11
    languages, XNLI in 10); '... after one temperature per task (2-fold OOF)' (0.078) is bench_h2h's per-entry refit.
    Both are reproduced from the rows, next to the settings in `outs`."""
    rows = []
    for t, sub in ANCHOR_ENTRIES:
        m = _entry_mask(D, t, sub)
        name = t if sub is None else f"{t}:{sub}"
        row = {"entry": name, "n": int(m.sum())}
        for s, o in outs.items():
            row[s] = ece(o["conf"][m], o["correct"][m])
        if refit is not None:
            row["repo_2fold_refit"] = refit[name]
        rows.append(row)
    names = list(outs) + (["repo_2fold_refit"] if refit is not None else [])
    return {"entries": rows, "mean": {k: float(np.mean([r[k] for r in rows])) for k in names}}


def map_fit(Z: np.ndarray, y: np.ndarray, centre: float, sd: float, min_rows: int = 5) -> float:
    """tez.readout.fit_temperature_logits with the prior's median moved to `centre` (and sd `sd`). With too few rows
    it returns the centre (the engine returns 1.0 today)."""
    if len(y) < min_rows:
        return float(centre)
    best, bt = math.inf, float(centre)
    for t in T_GRID:
        v = nll(softmax_rows(Z, t), y) + (math.log(t) - math.log(centre)) ** 2 / (2.0 * sd ** 2 * len(y))
        if v < best - 1e-12:
            best, bt = v, float(t)
    return bt


def prior_centre_simulation(D: Data, T0_rows: np.ndarray, key: np.ndarray, sds: dict, draws: int = DRAWS) -> dict:
    """A question fitted from only n labels: tez.readout.fit_temperature_logits (MAP, log-normal prior with median 1 and
    sd 1 in log T, weight 1/n) against the same fit with the prior's median at the default (T0 = the cell's held-out
    default, fitted without the task, so nothing of the task leaks), and with a prior sd equal to the spread of
    per-task log temperatures of the type. Cells = (task, type, option-count bucket) with at least 100 decisions; n
    labelled rows are drawn at random from the cell and the cell's other rows measure the fit (NLL, ECE-15)."""
    rng = np.random.default_rng(SEED)
    cells = []
    for q in TYPES:
        for t in type_tasks(D, q):
            for b in sorted(set(key[(D.qtype == q) & (D.task == t)])):
                idx = np.flatnonzero((D.qtype == q) & (D.task == t) & (key == b))
                if len(idx) < 100:
                    continue
                t0s = np.unique(T0_rows[idx])
                if len(t0s) != 1:
                    raise ValueError(f"{t}/{q}/{b}: more than one default in a cell")
                t0 = float(t0s[0])
                K = int(D.k[idx].max())
                Z = np.full((len(idx), K), -np.inf)
                for r, i in enumerate(idx):
                    Z[r, : D.k[i]] = D.z[i]
                y = D.gold[idx]
                cell = {"task": str(t), "type": q, "bucket": str(b), "n_rows": int(len(idx)), "T0": t0, "by_n": {}}
                for n in N_LABELS:
                    acc = defaultdict(list)
                    for _ in range(draws):
                        perm = rng.permutation(len(idx))
                        fi, ev = perm[:n], perm[n:]
                        fits = {"T = 1, no labels (engine today)": 1.0, "default, no labels": t0,
                                "fit, prior median 1, sd 1 (engine today)": fit_temperature_logits(Z[fi], y[fi]),
                                "fit, prior median = default, sd 1": map_fit(Z[fi], y[fi], t0, 1.0),
                                "fit, prior median = default, sd = spread across tasks": map_fit(Z[fi], y[fi], t0, sds[q])}
                        for name, tt in fits.items():
                            P = softmax_rows(Z[ev], tt)
                            acc[name + ":nll"].append(nll(P, y[ev]))
                            acc[name + ":ece"].append(ece(P.max(1), P.argmax(1) == y[ev]))
                    cell["by_n"][str(n)] = {k: float(np.mean(v)) for k, v in acc.items()}
                cells.append(cell)
    names = [k[:-4] for k in cells[0]["by_n"][str(N_LABELS[0])] if k.endswith(":nll")]
    summary = {}
    for n in N_LABELS:
        summary[str(n)] = {name: {"nll": float(np.mean([c["by_n"][str(n)][name + ":nll"] for c in cells])),
                                  "ece": float(np.mean([c["by_n"][str(n)][name + ":ece"] for c in cells]))} for name in names}
    return {"cells": cells, "macro_over_cells": summary, "n_cells": len(cells), "draws": draws, "seed": SEED,
            "prior_sd_spread_by_type": sds}


def fewshot_side_check(T_by_bucket: dict) -> dict:
    """typed-decisions with 4 worked examples of the question in the prefix (not pooled): T = 1 against the (r) defaults
    held out from typed-decisions (the engine can put a schema's examples in the prefix too, in its own wording)."""
    path = RES / "h2h" / "rows_typed_decisions_tez-fewshot4-all.jsonl"
    F = Data()
    for r in read_jsonl(path):
        F.add(np.asarray(r["logits"], float), int(r["gold"]), task="typed_decisions_fewshot4", subtask=r["workflow"],
              qtype=r["qtype"], source=rel(path), prompt="engine + 4 shots", rid=r["id"])
    F.finalize()
    T = np.array([T_by_bucket[b] for b in F.rule])
    out = {"file": rel(path), "sha256": sha256(path), "n": int(F.n), "T": {b: T_by_bucket[b] for b in sorted(set(F.rule))}}
    for name, tt in (("raw", 1.0), ("default", T)):
        o = readout(F, np.arange(F.n), tt)
        out[name] = {"all": metrics(o), **{q: metrics(o, F.qtype == q) for q in TYPES}}
    return out


def tail_mass(D: Data, T_values: dict) -> dict:
    """Mean share of the tempered probability outside each decision's four highest-scoring letters (choice decisions
    whose tempered softmax spans more than 4 options), per task: why a many-option question needs a lower T."""
    out = {}
    for t in type_tasks(D, "choice"):
        idx = np.flatnonzero((D.qtype == "choice") & (D.task == t) & (D.k_final > 4))
        if not len(idx):
            continue
        row = {"n": int(len(idx)), "options": sorted(int(x) for x in set(D.k_final[idx]))}
        for name, T in T_values.items():
            vals = [1.0 - np.sort(softmax_T(Z, T), axis=1)[:, ::-1][:, :4].sum(axis=1) for _, Z, _ in blocks(D, idx)]
            row[name] = float(np.concatenate(vals).mean())
        out[str(t)] = row
    return out


def published_checks(D: Data) -> dict:
    """T = 1 metrics recomputed here against the numbers the repository already published for the same rows:
    results/h2h/summary.json (28 entries), results/vs_laya/apps.json (7 apps), the JevBench runtime summaries."""
    diffs = defaultdict(float)

    def cmp(tag, idx, pub, keys):
        mm = metrics(readout(D, idx, 1.0))
        for k, pk in keys.items():
            if pub.get(pk) is not None:
                diffs[tag] = max(diffs[tag], abs(mm[k] - float(pub[pk])))

    keys = {"accuracy": "accuracy", "ece": "ece", "nll": "nll", "brier": "brier"}
    summary = json.loads((RES / "h2h" / "summary.json").read_text(encoding="utf-8"))
    for t, sub in ANCHOR_ENTRIES:
        cmp("h2h/summary.json (28 entries)", np.flatnonzero(_entry_mask(D, t, sub)),
            summary[t if sub is None else f"{t}:{sub}"]["tez"], keys)
    for task in json.loads((RES / "vs_laya" / "apps.json").read_text(encoding="utf-8"))["tasks"]:
        cmp("vs_laya/apps.json (7 apps)", np.flatnonzero(D.task == task["id"]), task["systems"]["tez"], keys)
    for name, tiers in (("jevbench_tez_server_short_warm.json", ("original", "easy")), ("jevbench_tez_server_hard.json", ("hard",))):
        s = json.loads((RES / name).read_text(encoding="utf-8"))["summary"]
        for tier in tiers:
            cmp("JevBench runtime summaries (3 tiers)", np.flatnonzero((D.task == "jevbench") & (D.subtask == tier)), s[tier],
                {"accuracy": "accuracy", "ece": "ece15", "nll": "nll", "brier": "brier"})
    return {"max_abs_difference": dict(diffs)}


def scipy_check(D: Data, t_ours: float) -> dict | None:
    """The pooled noul temperature recomputed with scipy (log_softmax + bounded Brent), independently of fit_T."""
    try:
        from scipy.optimize import minimize_scalar
        from scipy.special import log_softmax
    except ImportError:
        return None
    idx = np.flatnonzero(D.qtype == "noul")
    Z, y = np.stack([D.z[i] for i in idx]), D.gold[idx]

    def f(u):
        return -float(np.mean(np.maximum(log_softmax(Z / math.exp(u), axis=1)[np.arange(len(y)), y], math.log(CLIP))))
    r = minimize_scalar(f, bounds=(math.log(T_GRID[0]), math.log(T_GRID[-1])), method="bounded", options={"xatol": 1e-9})
    t = math.exp(r.x)
    if abs(t - t_ours) > 1e-3 * t_ours:
        raise AssertionError(f"noul pooled T: scipy {t} vs fit_T {t_ours}")
    return {"noul_pooled_T_scipy": t, "noul_pooled_T_fit_T": t_ours}


def reproducibility_check(D: Data) -> dict | None:
    """typed-decisions re-read by another session through llama-server with the runtime's layout (results/speed,
    untracked): per-decision agreement with the h2h rows used here. Informational only."""
    path = RES / "speed" / "td_rows_prod_today_http.jsonl"
    if not path.exists():
        return None
    rows = {r["id"]: r for r in read_jsonl(path)}
    diffs, agree, n = [], 0, 0
    for i in np.flatnonzero(D.task == "typed_decisions"):
        rid = str(D.rid[i])
        r = rows.get(rid[3:] if rid.startswith("td-") else rid)      # that file drops the "td-" prefix
        if r is None or len(r["z"]) != D.k[i]:
            continue
        p1 = softmax_rows(D.z[i][None], 1.0)[0]
        p2 = softmax_rows(np.asarray(r["z"], float)[None], 1.0)[0]
        diffs.append(float(np.abs(p1 - p2).max()))
        agree += int(np.argmax(p1) == np.argmax(p2))
        n += 1
    if n == 0:
        return {"file": rel(path), "matched": 0}
    return {"file": rel(path), "sha256": sha256(path), "matched": n, "argmax_agreement": agree / n,
            "median_max_abs_dp": float(np.median(diffs)), "p95_max_abs_dp": float(np.percentile(diffs, 95))}


# ------------------------------------------------------------------------------------------ tables
SET = ("raw", "default_loto", "bucket_loto", "oracle")
SET_LABEL = "a / b / b′ / c"
SET5 = ("raw", "default_loto", "bucket_loto", "rule_loto", "oracle")
SET5_LABEL = "a / b / b′ / r / c"


def f3(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def f2(x) -> str:
    if isinstance(x, list):
        return "/".join(f"{v:.2f}" for v in x)
    return "–" if x is None else f"{x:.2f}"


def md_table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def joined(d: dict, key: str, settings=SET) -> str:
    return " / ".join(f3(d[s][key]) for s in settings)


def tables(R: dict) -> str:
    out = []
    n_tasks, n_cells = len(R["data"]["tasks"]), len(R["loto"]["cells"])
    # data
    rows = [[t, d["prompt"], *(str(d["types"].get(q, "")) for q in TYPES), str(d["n"]),
             ", ".join(f"{k}:{v}" for k, v in d["option_counts"].items())] for t, d in R["data"]["tasks"].items()]
    tot = R["data"]["by_type"]
    rows.append([f"**all ({n_tasks} tasks)**", "", *(f"**{tot[q]['n']}** ({len(tot[q]['tasks'])} tasks)" for q in TYPES),
                 f"**{R['data']['n']}**", ""])
    out.append("### Data: decisions per task and type (options = option count: decisions)\n\n"
               + md_table(["task", "prompt", "noul", "choice", "score", "decisions", "options"], rows, "llrrrrl"))
    # fitted temperatures
    F = R["fitted"]

    def frow(label, f, bal, orc=None):
        return [label, f"**{f2(f['all'])}**",
                f"{f2(f['loto_median'])} ({f2(f['loto_min'])}–{f2(f['loto_max'])})" if f["loto"] else "–", f2(bal["all"]),
                f"{f2(orc['median'])} ({f2(orc['min'])}–{f2(orc['max'])})" if orc else "", f2(orc["sd_log"]) if orc else "",
                str(len(f["tasks"])), str(f["n"])]

    rows = [frow(f"{q} (b, r)" if q != "choice" else "choice (b)", F["per_type"][q], F["per_type_task_balanced"][q], F["oracle"][q])
            for q in TYPES]
    rows += [frow(f"choice, {b} options (b′)", f, F["choice_by_option_count_task_balanced"][b])
             for b, f in F["choice_by_option_count"].items() if b not in TYPES]
    rows += [frow(f"choice, {b} options (r)", f, F["rule_two_level_task_balanced"][b])
             for b, f in F["rule_two_level"].items() if b not in TYPES]
    out.append("### Fitted temperatures (pooled NLL; LOTO = the same fit with each task left out in turn)\n\n" + md_table(
        ["cell", "T pooled on all tasks", "LOTO folds: median (min–max)", "T task-balanced", "per-task oracle: median (min–max)",
         "sd of log T across tasks", "tasks", "decisions"], rows))
    # LOTO per cell, the requested comparison
    rows = [[c["task"], c["type"], str(c["n"]), f3(c["raw"]["accuracy"]), joined(c, "ece"), joined(c, "nll"), joined(c, "brier"),
             f2(c["T_default_loto"]), f2(c["T_bucket_loto"]), f2(c["T_oracle"])] for c in R["loto"]["cells"]]
    out.append("### Leave-one-task-out, per task and type\n\n(a) T = 1; (b) one pooled T per type, fitted without the task; "
               "(b′) as (b) with choice split into the buckets 2 / 3–5 / 6–10 / 11–26 options (a tournament by its finalists); "
               "(c) per-task oracle.\n\n" + md_table(
                   ["task", "type", "n", "accuracy", f"ECE-15 {SET_LABEL}", f"NLL {SET_LABEL}", f"Brier {SET_LABEL}",
                    "T (b)", "T (b′)", "T (c)"], rows, "llrrrrrrrr"))
    # the recommended rule per cell
    rows = [[c["task"], c["type"], str(c["n"]), f"{f3(c['raw']['ece'])} → {f3(c['rule_loto']['ece'])}",
             f"{f3(c['raw']['nll'])} → {f3(c['rule_loto']['nll'])}", f"{f3(c['raw']['brier'])} → {f3(c['rule_loto']['brier'])}",
             f2(c["T_rule_loto"]), f3(c["oracle"]["ece"]), f2(c["T_oracle"])] for c in R["loto"]["cells"]]
    out.append("### The recommended rule (r), leave-one-task-out: noul, score, and choice with up to 10 / 11–26 options\n\n" + md_table(
        ["task", "type", "n", "ECE-15 a → r", "NLL a → r", "Brier a → r", "T (r)", "ECE-15 (c)", "T (c)"], rows, "llrrrrrrr"))
    # summary
    A = R["loto"]["aggregate"]

    def agg_row(label, key, sub=None):
        d = {s: (A[s][key][sub] if sub else A[s][key]) for s in SET5}
        acc = A["raw"]["micro_by_type"][sub]["accuracy"] if sub else A["raw"][key]["accuracy"]
        return [label, " / ".join(f3(d[s]["ece"]) for s in SET5), " / ".join(f3(d[s]["nll"]) for s in SET5),
                " / ".join(f3(d[s]["brier"]) for s in SET5), f3(acc)]

    rows = [agg_row(f"mean over the {n_tasks} tasks", "macro_tasks"), agg_row(f"mean over the {n_cells} task×type cells", "macro_cells")]
    rows += [agg_row(f"{q}: mean over its tasks", "macro_by_type", q) for q in TYPES]
    rows.append(agg_row("all decisions pooled (over- and under-confidence cancel in the bins)", "micro_all"))
    out.append(f"### Summary ({SET5_LABEL})\n\n" + md_table(["", f"ECE-15 {SET5_LABEL}", f"NLL {SET5_LABEL}", f"Brier {SET5_LABEL}",
                                                             "accuracy (every setting)"], rows))
    lines = []
    for name, lab in (("default_loto", "(b)"), ("bucket_loto", "(b′)"), ("rule_loto", "(r)")):
        W = R["loto"]["worse_than_raw"][name]
        h, g = W["highest_ece"], W["largest_ece_gain"]
        lines.append(f"- {lab} highest ECE: **{h['task']}** ({h['type']}) {f3(h['ece_a'])} → {f3(h['ece_setting'])}; "
                     f"largest gain: {g['task']} ({g['type']}) {f3(g['ece_a'])} → {f3(g['ece_setting'])}")
        for k, lab_k in (("ece", "ECE"), ("nll", "NLL"), ("brier", "Brier")):
            lst = ", ".join(f"{x['task']} ({x['type']}, n {x['n']}) {f3(x[k + '_a'])} → {f3(x[k + '_setting'])}" for x in W[f"{k}_worse"])
            lines.append(f"  - {lab_k} worse than T = 1: {lst or 'none'}")
    lines.append("- argmax changes against T = 1: " + ", ".join(f"{k} {v}" for k, v in R["loto"]["argmax_changes_vs_raw"].items()))
    out.append("\n".join(lines))
    # anchor
    M = R["anchor_28"]["mean"]
    out.append("### Against BENCHMARKS.md: mean ECE-15 over its 28 head-to-head entries (7 tasks, MASSIVE × 11, XNLI × 10)\n\n" + md_table(
        ["setting", "mean ECE-15"],
        [["(a) as shipped, T = 1 (BENCHMARKS.md: 0.212)", f3(M["raw"])],
         ["(b) held-out pooled default per type", f3(M["default_loto"])],
         ["(b′) as (b), choice by option-count bucket", f3(M["bucket_loto"])],
         ["(r) as (b), choice up to 10 / 11–26 options", f3(M["rule_loto"])],
         ["one T per entry, 2-fold out-of-fold, bench_h2h (BENCHMARKS.md: 0.078)", f3(M["repo_2fold_refit"])],
         ["(c) per-task oracle per type", f3(M["oracle"])]], "lr"))
    # score readout
    S3 = ("raw", "default_loto", "oracle")
    rows = []
    for c in R["score_readout"]["cells"]:
        rows.append([c["task"], str(c["n"]), joined(c, "mae_level", S3), joined(c, "mse_level", S3), joined(c, "within_1_level", S3),
                     " / ".join(f3(c[s].get("mae_gold_expected_score")) for s in S3),
                     joined(c, "mae_median_level", S3), f3(c["raw"]["mae_mode_level"])])
    a = R["score_readout"]["all_score_decisions"]
    rows.append(["all score decisions", str(R["data"]["by_type"]["score"]["n"]), joined(a, "mae_level", S3), joined(a, "mse_level", S3),
                 joined(a, "within_1_level", S3), "", joined(a, "mae_median_level", S3), f3(a["raw"]["mae_mode_level"])])
    out.append("### Score questions: the expected level sum(i·p_i) at T = 1 / (b = r) / (c)\n\n" + md_table(
        ["task", "n", "MAE vs gold level", "squared error vs gold level", "within 1 level",
         "MAE vs the benchmark's gold expected score", "MAE of the median level", "MAE of the argmax level (any T)"], rows))
    # variants
    rows = [[name, ", ".join(f"{k} {f2(v)}" for k, v in s["T_all"].items()), str(s["tasks"]), f3(s["macro_tasks"]["ece"]),
             f3(s["macro_tasks"]["nll"]), f3(s["macro_tasks"]["brier"]), f3(s.get("anchor_28_mean_ece"))]
            for name, s in R["sensitivity"].items()]
    out.append("### Variants, each leave-one-task-out (mean over the variant's tasks; T = 1 gives ECE "
               f"{f3(A['raw']['macro_tasks']['ece'])}, NLL {f3(A['raw']['macro_tasks']['nll'])}, Brier {f3(A['raw']['macro_tasks']['brier'])})\n\n"
               + md_table(["variant", "T fitted on all tasks", "tasks", "ECE-15", "NLL", "Brier", "28-entry mean ECE"], rows, "llrrrrr"))
    # prior centre
    S = R["prior_centre"]["macro_over_cells"]
    names = list(S[str(N_LABELS[0])])
    rows = [[name] + [f"{f3(S[str(n)][name]['nll'])} / {f3(S[str(n)][name]['ece'])}" for n in N_LABELS] for name in names]
    sd = R["prior_centre"]["prior_sd_spread_by_type"]
    out.append(f"### A question fitted from n labels: prior centred at 1 vs at the default (NLL / ECE-15 on the cell's other rows; "
               f"mean over {R['prior_centre']['n_cells']} task×cell combinations with ≥ 100 decisions, {R['prior_centre']['draws']} "
               f"draws each; default = (r), held out; spread sd = " + ", ".join(f"{q} {f2(v)}" for q, v in sd.items()) + ")\n\n"
               + md_table(["method"] + [f"n = {n}" for n in N_LABELS], rows))
    # few-shot side check
    fs = R["side_checks"]["typed_decisions_fewshot4"]
    rows = [[k, f"{f3(fs['raw'][k]['ece'])} → {f3(fs['default'][k]['ece'])}", f"{f3(fs['raw'][k]['nll'])} → {f3(fs['default'][k]['nll'])}",
             f"{f3(fs['raw'][k]['brier'])} → {f3(fs['default'][k]['brier'])}"] for k in ("all",) + TYPES]
    out.append("### Side check: typed-decisions with 4 worked examples in the prefix (not pooled), T = 1 → (r) held out from "
               "typed-decisions\n\n" + md_table(["", "ECE-15", "NLL", "Brier"], rows))
    # tail mass
    TM = R["tail_mass_outside_top4"]
    names = [k for k in next(iter(TM.values())) if k.startswith("T = ")]
    rows = [[t, ", ".join(str(k) for k in d["options"]), str(d["n"])] + [f3(d[k]) for k in names] for t, d in TM.items()]
    out.append("### Why many options need a lower T: mean probability outside each decision's four best letters\n\n"
               + md_table(["task", "options", "n"] + names, rows, "llr" + "r" * len(names)))
    # self-checks
    C = R["self_checks"]
    lines = ["### Self-checks", ""]
    lines += [f"- T = 1 metrics against {k}: largest absolute difference {v:.1e}" for k, v in C["published_T1_metrics"]["max_abs_difference"].items()]
    if C.get("scipy"):
        lines.append(f"- pooled noul T with scipy (log_softmax, bounded Brent): {C['scipy']['noul_pooled_T_scipy']:.4f}; "
                     f"this script: {C['scipy']['noul_pooled_T_fit_T']:.4f}")
    rc = R["side_checks"].get("typed_decisions_rerun_agreement")
    if rc and rc.get("matched"):
        lines.append(f"- typed-decisions re-read through llama-server by another session ({rc['file']}): {rc['matched']} decisions matched, "
                     f"argmax agreement {rc['argmax_agreement']:.3f}, median largest |Δp| {rc['median_max_abs_dp']:.1e}")
    out.append("\n".join(lines))
    return "\n\n".join(out)


# ------------------------------------------------------------------------------------------ main
def fitted_summary(f: dict) -> dict:
    out = {}
    for c, v in f.items():
        vals = list(v["loto"].values())
        out[c] = {"all": v["all"], "grid_edge": v["edge"], "n": v["n"], "tasks": v["tasks"], "loto": v["loto"],
                  "loto_median": float(np.median(vals)) if vals else None, "loto_min": float(min(vals)) if vals else None,
                  "loto_max": float(max(vals)) if vals else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-write", action="store_true", help="print the tables only")
    ap.add_argument("--draws", type=int, default=DRAWS, help="draws per cell in the few-labels simulation")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    D = load()
    # self-checks: the per-row softmax is the engine's, and the MAP fit with median 1 is the engine's
    sel = np.flatnonzero(D.task == "ag_news")[:50]
    Zt, yt = np.stack([D.z[i] for i in sel]), D.gold[sel]
    assert np.allclose(softmax_T(Zt, 1.7), softmax_rows(Zt, 1.7))
    assert map_fit(Zt, yt, 1.0, 1.0) == fit_temperature_logits(Zt, yt)

    data = {"n": int(D.n), "tasks": {}, "by_type": {}}
    for t in sorted(set(D.task), key=lambda t: (-int((D.task == t).sum()), t)):
        m = D.task == t
        data["tasks"][str(t)] = {"n": int(m.sum()), "types": {q: int((m & (D.qtype == q)).sum()) for q in TYPES if (m & (D.qtype == q)).any()},
                                 "prompt": str(sorted(set(D.prompt[m]))[0]), "subtasks": sorted(str(s) for s in set(D.subtask[m])),
                                 "option_counts": {str(k): int(v) for k, v in sorted(Counter(D.k[m]).items())}}
    for q in TYPES:
        data["by_type"][q] = {"n": int((D.qtype == q).sum()), "tasks": type_tasks(D, q)}
    data["choice_by_bucket"] = {b: {"n": int((D.bucket_final == b).sum()), "tasks": sorted(set(D.task[D.bucket_final == b]))}
                                for b in BUCKETS if (D.bucket_final == b).any()}
    data["voice_rows_with_alternative_answers"] = int(D.ambiguous.sum())
    data["banking77_gold_eliminated_in_tournament"] = int(sum(1 for i in np.flatnonzero(D.task == "banking77")
                                                             if not np.isfinite(D.z[i][D.gold[i]])))
    print(f"{D.n} decisions, {len(set(D.task))} tasks: " + ", ".join(f"{q} {data['by_type'][q]['n']}" for q in TYPES), flush=True)

    # (b) per type, (b') per type with choice by option count (a tournament by its finalists), (c) oracle
    T_b, fits, _ = held_out_T(D, D.qtype)
    T_bb, fits_bf, _ = held_out_T(D, D.bucket_final)
    T_r, fits_r, _ = held_out_T(D, D.rule)
    orc = oracle_fits(D)
    T_c = np.array([orc[q][t] for q, t in zip(D.qtype, D.task)])
    print("pooled T per type:", {q: round(fits[q]["all"], 3) for q in TYPES},
          "choice by option count:", {b: round(v["all"], 3) for b, v in fits_bf.items() if b not in TYPES},
          "two-level rule:", {b: round(v["all"], 3) for b, v in fits_r.items() if b not in TYPES}, flush=True)
    res = evaluate_settings(D, {"raw": np.ones(D.n), "default_loto": T_b, "bucket_loto": T_bb, "rule_loto": T_r, "oracle": T_c})
    base = res["_outs"]["raw"]["pred"]
    changes = {name: int((o["pred"] != base).sum()) for name, o in res["_outs"].items()}
    if any(changes.values()):
        raise AssertionError(f"a temperature changed an argmax: {changes}")
    refit28 = repo_refit_28(D)
    anchor = anchor_28(D, res["_outs"], refit28)

    oracle_summary = {}
    for q in TYPES:
        vals = np.array(list(orc[q].values()))
        oracle_summary[q] = {"per_task": orc[q], "median": float(np.median(vals)), "min": float(vals.min()), "max": float(vals.max()),
                             "sd_log": float(np.std(np.log(vals), ddof=1)), "geometric_mean": float(np.exp(np.mean(np.log(vals)))),
                             "at_grid_edge": [t for t, v in orc[q].items() if v >= T_GRID[-1] - 1e-9 or v <= T_GRID[0] + 1e-9]}

    # ---- variants, each fully leave-one-task-out and measured on every task of its data
    sens = {}

    def variant(name, Dx, T_rows, T_all, note=None):
        r = evaluate_settings(Dx, {"raw": np.ones(Dx.n), "v": T_rows})
        a = anchor_28(Dx, {"v": r["_outs"]["v"]})["mean"]["v"] if set(t for t, _ in ANCHOR_ENTRIES) <= set(Dx.task) else None
        sens[name] = {"T_all": T_all, "tasks": len(set(Dx.task)), "macro_tasks": r["aggregate"]["v"]["macro_tasks"],
                      "macro_by_type": r["aggregate"]["v"]["macro_by_type"], "micro_all": r["aggregate"]["v"]["micro_all"],
                      "anchor_28_mean_ece": a,
                      "cells": [{"task": c["task"], "type": c["type"], "n": c["n"], "T": c["T_v"],
                                 **{f"{k}_raw": c["raw"][k] for k in ("ece", "nll", "brier")},
                                 **{k: c["v"][k] for k in ("ece", "nll", "brier")}} for c in r["cells"]]}
        if note:
            sens[name]["note"] = note

    choice_T = lambda f: {b: v["all"] for b, v in f.items()}  # noqa: E731
    variant("(b) one pooled T per type", D, T_b, choice_T(fits))
    variant("(b′) per type, choice by option count, tournaments by finalists", D, T_bb, choice_T(fits_bf))
    variant("(r) per type, choice up to 10 options / 11-26 options", D, T_r, choice_T(fits_r))
    T_bal, fits_bal, _ = held_out_T(D, D.qtype, balanced=True)
    variant("(b) with task-balanced pools", D, T_bal, choice_T(fits_bal))
    T_bbal, fits_bbal, _ = held_out_T(D, D.bucket_final, balanced=True)
    variant("(b′) with task-balanced pools", D, T_bbal, choice_T(fits_bbal))
    T_rbal, fits_rbal, _ = held_out_T(D, D.rule, balanced=True)
    variant("(r) with task-balanced pools", D, T_rbal, choice_T(fits_rbal))
    allkey = np.array(["all"] * D.n)
    T_one, fits_one, _ = held_out_T(D, allkey)
    variant("one T for every type and option count", D, T_one, choice_T(fits_one))
    T_own, fits_own, _ = held_out_T(D, D.bucket)
    variant("(b′) with tournaments as their own bucket (> 26)", D, T_own, choice_T(fits_own),
            "Banking77 is the only > 26 task, so held out it falls back to the pooled choice T")
    eng = D.prompt == "engine"
    T_eng, fits_eng, _ = held_out_T(D, D.rule, eng)
    variant("(r) pooled on the runtime prompt layout only (no SemIf, no voice)", D, T_eng, choice_T(fits_eng),
            "held out from MASSIVE, the 11-26 cell has no other task and falls back to the pooled choice T")
    Dv = load(voice_ambiguous=False)
    T_v, fits_v, _ = held_out_T(Dv, Dv.rule)
    variant("(r) without the 33 voice rows that accept alternative answers", Dv, T_v, choice_T(fits_v))
    Dw = load()
    Dw.task = np.array([f"td:{s}" if t == "typed_decisions" else t for t, s in zip(Dw.task, Dw.subtask)])
    T_w, fits_w, _ = held_out_T(Dw, Dw.rule)
    variant("(r) with typed-decisions as its 4 workflows (held out one at a time)", Dw, T_w, choice_T(fits_w))

    # ---- score readout
    outs = res["_outs"]
    score = {"cells": [{k: c[k] for k in ("task", "n", "k", "raw", "default_loto", "oracle", "T_default_loto", "T_oracle")}
                       for c in res["cells"] if c["type"] == "score"],
             "all_score_decisions": {name: score_metrics(outs[name], D.qtype == "score") for name in ("raw", "default_loto", "oracle")},
             "note": "b' = b for score questions (only choice is split by option count). 'Score from untempered probabilities' = "
                     "the T = 1 column, with tempered probabilities and confidence alongside."}

    # ---- prior centre for per-question fits
    sds = {q: oracle_summary[q]["sd_log"] for q in TYPES}
    prior = prior_centre_simulation(D, T_r, D.rule, sds, args.draws)

    R = {
        "title": "A default letter temperature per question type (Gemma 4 12B Q8_0, zero-shot letters)",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "experiments/default_temperature.py",
        "command": "python experiments/default_temperature.py" + (f" --draws {args.draws}" if args.draws != DRAWS else ""),
        "python": platform.python_version(), "numpy": np.__version__,
        "model": {"weights": f"Ollama blob {MODEL_BLOB} (Gemma 4 12B Q8_0)", "readout": "letters, llama-server top-200 log-probs"},
        "readout": ("z = option-letter log-probabilities from llama-server's top-200 list, a missing letter floored at "
                    "(lowest listed - 2) (tez/backends.py letters_from_response; every input used n_probs 200 and this floor); "
                    "options that left a tournament = -inf (tez/engine.py letter_logits); p = softmax(z / T) "
                    "(tez/readout.py softmax_rows). Probability-only files: z = log p, which tempers identically."),
        "protocol": {
            "unit": "task = dataset/benchmark; MASSIVE (51 languages) and XNLI (10) one task each; JevBench's three tiers one task; "
                    "each of Laya's seven application workflows one task; typed-decisions one task (split into its workflows as a variant)",
            "types": "question type as served: noul, choice, score (SemIf and voice option sets are choice questions)",
            "option_count_buckets": "choice only: 2, 3-5, 6-10, 11-26 options; a tournament (> 26 options) goes by its number of "
                                    "finalists, the options the tempered softmax actually spans (a variant puts it in its own > 26 bucket)",
            "pooled_fit": "argmin over T in [0.05, 20] of the mean NLL (p clipped at 1e-12) of all decisions in the cell, each decision "
                          "weighing the same (task-balanced pools as a variant); tez.readout.T_GRID then golden-section refinement in log T",
            "leave_one_task_out": "for each cell and each task in it: T fitted on every other task's decisions of the cell, applied to "
                                  "the held-out task's; a cell with no other task falls back to the type's held-out T",
            "oracle": "T fitted on the task's own decisions of the type (in-sample; reference only)",
            "settings": {"a": "T = 1 (as shipped)", "b": "held-out pooled T per type",
                         "b'": "held-out pooled T per type, choice by option-count bucket (2, 3-5, 6-10, 11-26; tournament by finalists)",
                         "r": "held-out pooled T per type, choice split at 10 options (2-10, 11-26; tournament by finalists)",
                         "c": "per-task oracle per type"},
            "metrics": "accuracy; ECE-15 = tez.readout.ece (max probability vs correctness, 15 equal-width bins (lo, hi]); NLL = "
                       "tez.readout.nll; Brier = sum over options of (p - onehot)^2; score: |sum(i p_i) - gold level| (and squared)",
            "aggregates": "macro = unweighted mean over tasks (or task x type cells); micro = all decisions pooled",
        },
        "inputs": D.inputs,
        "excluded": [{"files": f, "reason": r} for f, r in EXCLUDED],
        "provenance_checks": {
            "jev_or_typesafe_outputs_used": False,
            "notes": [
                "Jev was never run in this repository; no input row is a Jev or TypeSafe answer.",
                "JevBench rows are Tez's answers from the local runtime (manifest endpoint http://127.0.0.1:8787/v1/systemone, "
                "server model tez-0.1.0 (gemma-4-12b-q8_0, letters), backend http://127.0.0.1:8091), checked in load(); "
                "their gold labels are JevBench's authored labels (original/easy: rubric reviewed before inference; hard: "
                "authored with a written rationale and cross-model review before any system run, per the items' provenance).",
                "SemIf authored144 labels are project-authored (provenance.source 'project-authored', checked per row); "
                "SemIf's 'typesafe' source (agreement with a released model reference) is not in data/ and not used.",
                "typed-decisions gold is the mean of three samples from an unnamed ~4B-class 'teacher endpoint' (dataset card, "
                "LocalLLaMA/typed-decisions); the card states the benchmark is not affiliated with TypeSafe and lists Jev "
                "only as a separately measured leaderboard row. The teacher's identity is not disclosed.",
            ],
        },
        "data": data,
        "fitted": {
            "per_type": fitted_summary(fits),
            "per_type_task_balanced": fitted_summary(fits_bal),
            "choice_by_option_count": fitted_summary(fits_bf),
            "rule_two_level": fitted_summary(fits_r),
            "rule_two_level_task_balanced": fitted_summary(fits_rbal),
            "choice_by_option_count_task_balanced": fitted_summary(fits_bbal),
            "choice_by_option_count_tournament_own_bucket": fitted_summary(fits_own),
            "one_T_every_type": fitted_summary(fits_one),
            "oracle": oracle_summary,
        },
        "loto": {"cells": res["cells"], "per_task": res["per_task"], "per_subtask": res["per_subtask"],
                 "aggregate": res["aggregate"], "argmax_changes_vs_raw": changes,
                 "worse_than_raw": {name: worse_than_raw(res, name) for name in ("default_loto", "bucket_loto", "rule_loto")}},
        "anchor_28": anchor,
        "score_readout": score,
        "prior_centre": prior,
        "sensitivity": sens,
        "self_checks": {"published_T1_metrics": published_checks(D), "scipy": scipy_check(D, fits["noul"]["all"])},
        "tail_mass_outside_top4": tail_mass(D, {"T = 1": 1.0, f"T = {fits_r['11-26']['all']:.2f} (11-26 default)": fits_r["11-26"]["all"],
                                                f"T = {fits_r['2-10']['all']:.2f} (2-10 default)": fits_r["2-10"]["all"]}),
        "side_checks": {"typed_decisions_fewshot4": fewshot_side_check({b: v["loto"]["typed_decisions"] for b, v in fits_r.items()
                                                                        if "typed_decisions" in v["loto"]}),
                        "typed_decisions_rerun_agreement": reproducibility_check(D)},
    }
    text = tables(R)
    print()
    print(text)
    if args.no_write:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(R, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"\nwrote {rel(OUT_JSON)}")
    if OUT_MD.exists():
        md = OUT_MD.read_text(encoding="utf-8")
        start, end = "<!-- tables:start -->", "<!-- tables:end -->"
        if start in md and end in md:
            md = md[: md.index(start) + len(start)] + "\n" + text + "\n" + md[md.index(end):]
            OUT_MD.write_text(md, encoding="utf-8")
            print(f"refreshed the tables in {rel(OUT_MD)}")


if __name__ == "__main__":
    main()
