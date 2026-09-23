"""Tez vs Laya on Laya's seven application workflows, same rows (results/vs_laya/apps.json).

Laya's definition: research/scripts/bench_apps.py (github.com/NandhaKishorM/laya), published numbers in
research/results/app_benchmark_results.json (laya 0.2.1, CPU, 400 cases per task, seed 13). The rows are
rebuilt exactly as bench_apps.build() builds them: same datasets, splits, filters and caps, the question
wording verbatim, and ONE random.Random(13) consumed in Laya's suite order (phishing shuffle, jailbreak
shuffle, toxicity shuffle, MS MARCO passage choice, routing shuffle). Email states use laya 0.2.1's
email_state (vendored below), the version that produced the published file. The built rows are frozen in
results/vs_laya/apps_cases.jsonl and every system reads that file, so all of them answer byte-identical rows.

Systems
  tez                    Gemma 4 12B Q8_0 in llama-server, zero-shot letter readout through bench_h2h.tez_decide
                         (same prompt layout as the runtime: instructions + options first, state last; each Laya
                         question rendered with tez.schema.Question.options(), so a noul question is
                         "no, <false criterion or: the statement does not hold>" / "yes, <...>").
  laya                   convaiinnovations/laya (repo root)            } Laya's own scoring path (bench_local.py
  laya-multilingual      convaiinnovations/laya, subfolder multilingual } score_cases: length-sorted batches,
  laya-typed-decisions   convaiinnovations/laya, subfolder typed-decisions } raw marker logits, softmax at the
                         checkpoint's bucket temperature), CPU fp32 as in Laya's published run.
  Per-decision latency for Laya is a separate pass (--time): one Agent.system_one call per case, batch of one,
  on --device (the GPU when it is free), so ms_p50 is comparable with Tez's per-decision time.

Usage (from the repo root)
  python experiments/vs_laya_apps.py --build                              # freeze the rows
  python experiments/vs_laya_apps.py --system tez                         # llama-server on :8091
  python experiments/vs_laya_apps.py --system laya --device cpu           # and laya-multilingual, laya-typed-decisions
  python experiments/vs_laya_apps.py --system laya --time --device cuda   # per-case latency pass
  python experiments/vs_laya_apps.py --aggregate                          # -> results/vs_laya/apps.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "vs_laya"
ROWS = OUT / "apps_rows"
CASES = OUT / "apps_cases.jsonl"
PUBLISHED = OUT / "laya_published" / "app_benchmark_results.json"
SEED = 13
N = 400


def _load_h2h():
    spec = importlib.util.spec_from_file_location("bench_h2h", ROOT / "experiments" / "bench_h2h.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


H2H = _load_h2h()

# id, label, Laya suite name, held out of Laya's training mix (bench_apps in_training=False)
TASKS = [
    ("email_spam", "Email spam", "app.email_spam", False),
    ("phishing", "Phishing", "app.phishing", False),
    ("guardrails_jailbreak", "LLM guardrails (jailbreak)", "app.guardrails_jailbreak", True),
    ("moderation_toxicity", "Moderation (toxicity)", "app.moderation_toxicity", True),
    ("rag_relevance", "RAG passage relevance", "app.rag_relevance", False),
    ("support_triage", "Support triage (10-way queue)", "app.support_triage", False),
    ("model_routing_domain", "Model routing (domain)", "app.model_routing_domain", True),
]
TASK_IDS = [t[0] for t in TASKS]
LAYA = {"laya": None, "laya-multilingual": "multilingual", "laya-typed-decisions": "typed-decisions"}
LAYA_PUBLISHED_NAME = {"laya": "english", "laya-multilingual": "multilingual", "laya-typed-decisions": "typed-decisions"}
SYSTEMS = ["tez"] + list(LAYA)

# ------------------------------------------------------------------ laya 0.2.1 email_state (Apache-2.0)
# Copied from the laya 0.2.1 wheel (laya/email.py, unchanged in the repo until 2026-09-21), so the email
# states are the ones Laya's published run fed its checkpoints. Later laya versions clean mail differently.
_QUOTE_HEADERS = [
    re.compile(r"^\s*On .{0,300}wrote:\s*$", re.I),
    re.compile(r"^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}", re.I),
    re.compile(r"^\s*_{8,}\s*$"),
    re.compile(r"^\s*From:\s.+$", re.I),
]
_SIGNATURE_MARKERS = [
    re.compile(r"^\s*--\s*$"),
    re.compile(r"^\s*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[\w ,!.]*$", re.I),
    re.compile(r"^\s*sent from my (iphone|android|mobile|ipad)", re.I),
]
_DISCLAIMER = re.compile(
    r"(confidential|intended (solely )?for the (use of the )?(named )?(addressee|recipient)|"
    r"if you (have )?received this (e-?mail|message) in error)",
    re.I,
)


def clean_email_body_021(body: str, max_chars: int = 3000) -> str:
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n")
    lines = []
    for line in text.split("\n"):
        if any(p.match(line) for p in _QUOTE_HEADERS) and lines:
            break
        if line.lstrip().startswith(">"):
            continue
        lines.append(line.rstrip())
    cut = len(lines)
    for i in range(max(1, min(int(len(lines) * 0.6), len(lines) - 8)), len(lines)):
        if len(lines[i].strip()) <= 40 and any(p.match(lines[i]) for p in _SIGNATURE_MARKERS):
            cut = i
            break
    lines = lines[:cut]
    paragraphs = [p for p in re.split(r"\n\s*\n", "\n".join(lines)) if not _DISCLAIMER.search(p)]
    text = re.sub(r"[ \t]+", " ", "\n\n".join(p.strip() for p in paragraphs if p.strip()))
    return text[:max_chars]


def email_state_021(subject: str, body: str) -> Dict:
    return {"subject": (subject or "").strip(), "body": clean_email_body_021(body)}


# ------------------------------------------------------------------ rows, exactly as bench_apps.build()
QUEUES = {"Technical Support": "technical problems, bugs, outages, integrations",
          "Product Support": "help using a product or feature",
          "Customer Service": "general account or service questions",
          "IT Support": "internal IT, devices, access, networks",
          "Billing and Payments": "invoices, charges, refunds, payment methods",
          "Returns and Exchanges": "returning or exchanging an item",
          "Service Outages and Maintenance": "downtime, outages, scheduled maintenance",
          "Sales and Pre-Sales": "pricing, quotes, buying",
          "Human Resources": "employment, payroll, leave, hiring",
          "General Inquiry": "anything else"}
DOM = {"code": "software engineering, programming, refactoring, architecture, debugging",
       "math_or_logic": "mathematics, logic puzzles, proofs, complex calculation",
       "writing": "creative writing, essays, emails, blog posts, copywriting",
       "factual_lookup": "facts, definitions, trivia, history",
       "data_analysis": "statistics, SQL, data manipulation, metrics",
       "chitchat": "casual conversation, greetings, small talk"}
Q_SPAM = {"type": "noul", "instructions": "Is this email unsolicited spam or bulk marketing?"}
Q_PHISH = {"type": "noul",
           "instructions": "Is this email a phishing or scam attempt to steal money, credentials, or personal data?",
           "criteria": {"true": "phishing, scam, or fraud", "false": "a legitimate email (even if promotional)"}}
Q_JAIL = {"type": "noul", "instructions": "Does `prompt` try to make an AI assistant ignore its rules, policies or system instructions?"}
Q_TOX = {"type": "noul", "instructions": "Is `post` toxic: rude, disrespectful or likely to make someone leave the discussion?"}
Q_RAG = {"type": "noul", "instructions": "Does `passage` help answer `query`?"}
Q_QUEUE = {"type": "choice", "instructions": "Which support queue should handle this ticket?", "criteria": dict(QUEUES)}
Q_DOMAIN = {"type": "choice", "instructions": "What domain does `request` belong to?", "criteria": dict(DOM)}


def _case(task, i, src, state, qid, q, gold):
    return {"id": f"{task}-{i:03d}", "task": task, "src": src, "state": state, "qid": qid, "question": q, "gold": int(gold)}


def build_cases():
    """The seven application suites in bench_apps.build() order, sharing one random.Random(13)."""
    from datasets import load_dataset
    rng = random.Random(SEED)
    out = {}

    # 1. support triage (queues) - no rng
    d = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    keys, cases = list(QUEUES), []
    for i, r in enumerate(d):
        if r.get("language") != "en" or r.get("queue") not in QUEUES or not r.get("body"):
            continue
        cases.append(_case("support_triage", len(cases), f"Tobi-Bueck/customer-support-tickets/train/{i}",
                           {"subject": r["subject"] or "", "body": r["body"].replace("\\n", "\n")[:3000]},
                           "queue", Q_QUEUE, keys.index(r["queue"])))
        if len(cases) >= N:
            break
    out["support_triage"] = cases

    # 2. email spam - no rng
    d = load_dataset("SetFit/enron_spam", split="test")
    out["email_spam"] = [_case("email_spam", i, f"SetFit/enron_spam/test/{i}",
                               email_state_021(r.get("subject") or "", (r.get("message") or "")[:3000]),
                               "is_spam", Q_SPAM, int(r["label"]))
                         for i, r in enumerate(list(d)[:N])]

    # 2b. phishing - rng.shuffle over the filtered first 6,000 rows
    d = load_dataset("zefang-liu/phishing-email-dataset", split="train")
    rows = [(i, r) for i, r in enumerate(list(d)[:6000])
            if (r.get("Email Text") or "").strip() and r.get("Email Type") in ("Safe Email", "Phishing Email")]
    rng.shuffle(rows)
    out["phishing"] = [_case("phishing", j, f"zefang-liu/phishing-email-dataset/train/{i}",
                             {"email": r["Email Text"][:3000]}, "is_phishing", Q_PHISH,
                             int(r["Email Type"] == "Phishing Email"))
                       for j, (i, r) in enumerate(rows[:N])]

    # 3 & 5. toxic-chat (held out): jailbreak then toxicity, each a shuffle
    d = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
    rows = [(i, r) for i, r in enumerate(d) if (r.get("user_input") or "").strip()]
    jb = [x for x in rows if int(x[1].get("jailbreaking", 0)) == 1][:N // 2]
    nj = [x for x in rows if int(x[1].get("jailbreaking", 0)) == 0][:N - len(jb)]
    mix = jb + nj
    rng.shuffle(mix)
    out["guardrails_jailbreak"] = [_case("guardrails_jailbreak", j, f"lmsys/toxic-chat/toxicchat0124/test/{i}",
                                         {"prompt": r["user_input"][:3000]}, "jailbreak", Q_JAIL, int(r["jailbreaking"]))
                                   for j, (i, r) in enumerate(mix)]
    tox = [x for x in rows if int(x[1].get("toxicity", 0)) == 1][:N // 2]
    ntox = [x for x in rows if int(x[1].get("toxicity", 0)) == 0][:N - len(tox)]
    mix2 = tox + ntox
    rng.shuffle(mix2)
    out["moderation_toxicity"] = [_case("moderation_toxicity", j, f"lmsys/toxic-chat/toxicchat0124/test/{i}",
                                        {"post": r["user_input"][:3000]}, "toxic", Q_TOX, int(r["toxicity"]))
                                  for j, (i, r) in enumerate(mix2)]

    # 4. RAG passage relevance - one rng.choice per kept query, alternating positive / negative
    d = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
    cases = []
    for i, r in enumerate(d):
        texts, sel = r["passages"]["passage_text"], r["passages"]["is_selected"]
        pos = [t for t, s in zip(texts, sel) if s == 1]
        neg = [t for t, s in zip(texts, sel) if s == 0]
        if not pos or not neg:
            continue
        take_pos = len(cases) % 2 == 0
        p = rng.choice(pos if take_pos else neg)
        cases.append(_case("rag_relevance", len(cases), f"microsoft/ms_marco/v1.1/validation/{i}",
                           {"query": r["query"], "passage": p}, "relevant", Q_RAG, 1 if take_pos else 0))
        if len(cases) >= N:
            break
    out["rag_relevance"] = cases

    # 6. model routing (held out): gsm8k / mbpp / ag_news -> domain, one shuffle, N//3 each (399 rows)
    keys, pool = list(DOM), []
    g = load_dataset("openai/gsm8k", "main", split="test")
    pool += [(r["question"], "math_or_logic", f"openai/gsm8k/main/test/{i}") for i, r in enumerate(list(g)[:N // 3])]
    m = load_dataset("google-research-datasets/mbpp", "full", split="test")
    pool += [(r["text"], "code", f"google-research-datasets/mbpp/full/test/{i}") for i, r in enumerate(list(m)[:N // 3])]
    t = load_dataset("fancyzhx/ag_news", split="test")
    pool += [(r["text"][:400], "factual_lookup", f"fancyzhx/ag_news/test/{i}") for i, r in enumerate(list(t)[:N // 3])]
    rng.shuffle(pool)
    out["model_routing_domain"] = [_case("model_routing_domain", j, src, {"request": text}, "domain", Q_DOMAIN,
                                         keys.index(dom)) for j, (text, dom, src) in enumerate(pool[:N])]
    return out


def write_cases():
    suites = build_cases()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(CASES, "w", encoding="utf-8") as f:
        for tid in TASK_IDS:
            for c in suites[tid]:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    for tid in TASK_IDS:
        g = [c["gold"] for c in suites[tid]]
        print(f"  {tid:22s} {len(g):4d} rows  gold balance {np.bincount(g).tolist()}  sha {task_sha(suites[tid])}")
    print(f"wrote {CASES}")


def read_cases():
    by = {t: [] for t in TASK_IDS}
    with open(CASES, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            by[c["task"]].append(c)
    return by


def task_sha(cases):
    h = hashlib.sha256()
    for c in cases:
        h.update(json.dumps([c["id"], c["state"], c["question"], c["gold"]], ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ rows files
def rows_path(task, system, suffix=""):
    return ROWS / f"{task}_{system}{suffix}.jsonl"


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ------------------------------------------------------------------ tez
def tez_options(qid, q):
    from tez.schema import Question
    return Question(id=qid, type=q["type"], instructions=q["instructions"], criteria=q.get("criteria")).options()


def run_tez(server, tasks):
    props = H2H.SESSION.get(f"{server}/props", timeout=30).json()
    print("llama-server model:", props.get("model_path"), "build", props.get("build_info"), flush=True)
    by = read_cases()
    for tid in tasks:
        path = rows_path(tid, "tez")
        done = {r["id"] for r in read_rows(path) if "error" not in r}
        todo = [c for c in by[tid] if c["id"] not in done]
        print(f"[tez] {tid}: {len(done)} done, {len(todo)} to go", flush=True)
        t_task = time.perf_counter()
        with open(path, "a", encoding="utf-8") as f:
            for c in todo:
                opts = tez_options(c["qid"], c["question"])
                tcase = {"qtype": c["question"]["type"], "instructions": c["question"]["instructions"],
                         "options": opts, "state": c["state"]}
                rec, err = None, None
                for attempt in range(3):
                    try:
                        t = time.perf_counter()
                        p, z, pn, calls = H2H.tez_decide(server, tcase)
                        ms = (time.perf_counter() - t) * 1000
                        rec = dict(id=c["id"], task=tid, system="tez", gold=c["gold"], k=len(opts),
                                   probabilities=[float(x) for x in p], logits=[float(x) for x in z],
                                   pred=int(np.argmax(p)), confidence=float(np.max(p)),
                                   correct=int(int(np.argmax(p)) == c["gold"]), ms=ms, prompt_n=pn, calls=calls)
                        break
                    except Exception as exc:  # noqa: BLE001
                        err = f"{type(exc).__name__}: {str(exc)[:200]}"
                        time.sleep(1.0)
                if rec is None:
                    rec = dict(id=c["id"], task=tid, system="tez", gold=c["gold"], error=err)
                    print(f"  ERR {c['id']}: {err}", flush=True)
                f.write(json.dumps(rec) + "\n")
                f.flush()
        rows = [r for r in read_rows(path) if "error" not in r]
        acc = np.mean([r["correct"] for r in rows]) if rows else float("nan")
        print(f"[tez] {tid}: n={len(rows)} acc={acc:.4f} ms_p50={np.median([r['ms'] for r in rows]):.1f} "
              f"({time.perf_counter() - t_task:.0f}s)", flush=True)


# ------------------------------------------------------------------ laya (bench_local.score_cases, verbatim logic)
def load_laya(system, device):
    import laya
    agent = laya.load("convaiinnovations/laya", subfolder=LAYA[system], device=device)
    agent.model.eval()
    return agent


def to_internal(qdef):
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    return {"t": t, "ins": ins if isinstance(ins, str) else json.dumps(ins), "crit": crit}


def laya_score_cases(agent, cases, max_tokens=8192, max_seqs=64):
    """research/scripts/bench_local.py score_cases: raw marker logits, length-sorted batches."""
    import torch
    from laya.common import QTYPES, build_sequence, collate_items, render_options
    max_len = agent.cfg.get("max_len", 512)
    hml = agent.cfg.get("head_max_len", 192)
    items, index, dropped = [], [], 0
    for c in cases:
        q = to_internal(c["question"])
        try:
            ids, mk = build_sequence(agent.tok, c["state"], q, max_len, hml)
        except Exception:  # noqa: BLE001
            items.append(None); index.append((QTYPES[q["t"]], 0)); dropped += 1; continue
        if len(mk) != len(render_options(q)):
            items.append(None); index.append((QTYPES[q["t"]], 0)); dropped += 1; continue
        items.append({"ids": ids, "markers": mk, "qtype": QTYPES[q["t"]]})
        index.append((QTYPES[q["t"]], len(mk)))
    order = sorted([i for i, it in enumerate(items) if it is not None], key=lambda i: len(items[i]["ids"]))
    out = [None] * len(items)
    t0, i = time.time(), 0
    with torch.no_grad():
        while i < len(order):
            j, L = i, 0
            while j < len(order) and j - i < max_seqs and max(L, len(items[order[j]]["ids"])) * (j - i + 1) <= max_tokens:
                L = max(L, len(items[order[j]]["ids"])); j += 1
            j = max(j, i + 1)
            sel = [items[order[t]] for t in range(i, j)]
            b = collate_items([sel], agent.tok.pad_token_id)
            lg, _ = agent.model(b["input_ids"].to(agent.device), b["attention_mask"].to(agent.device),
                                b["marker_pos"].to(agent.device), b["marker_mask"].to(agent.device),
                                b["qtype"].to(agent.device))
            lg = lg.float().cpu().numpy()
            for r in range(j - i):
                out[order[i + r]] = lg[r, :len(sel[r]["markers"])]
            i = j
    return out, index, time.time() - t0, dropped, [len(it["ids"]) if it else None for it in items]


def laya_temps(agent, qt, k):
    """(served temperature, raw shipped temperature) for a bucket; laya >= 0.3 clamps to [0.5, 5]."""
    from laya.common import temp_bucket
    b = temp_bucket(qt, k)
    served = float(agent.temperature_by_options.get(b, agent.temperature[qt]))
    raw_all = getattr(agent, "temperature_by_options_raw", agent.temperature_by_options)
    raw_default = getattr(agent, "temperature_raw", agent.temperature)
    raw = float(raw_all.get(b, raw_default[qt]))
    return served, max(1e-3, raw), b


def softmax_t(z, t):
    z = np.asarray(z, float) / max(1e-3, float(t))
    e = np.exp(z - z.max())
    return e / e.sum()


def run_laya(system, device, tasks, threads):
    import torch
    if threads:
        torch.set_num_threads(threads)
    agent = load_laya(system, device)
    print(f"[{system}] device {agent.device} dtype {agent.dtype} threads {torch.get_num_threads()} "
          f"max_len {agent.cfg.get('max_len')} head_max_len {agent.cfg.get('head_max_len')}", flush=True)
    print(f"[{system}] temperatures served {agent.temperature} {agent.temperature_by_options}", flush=True)
    by = read_cases()
    for tid in tasks:
        cases = by[tid]
        lgs, idx, secs, dropped, lens = laya_score_cases(agent, cases)
        recs = []
        for c, z, (qt, k), L in zip(cases, lgs, idx, lens):
            if z is None:
                recs.append(dict(id=c["id"], task=tid, system=system, gold=c["gold"], error="dropped (build_sequence)"))
                continue
            t_served, t_raw, bucket = laya_temps(agent, qt, k)
            p = softmax_t(z, t_served)
            p_raw = softmax_t(z, t_raw)
            recs.append(dict(id=c["id"], task=tid, system=system, gold=c["gold"], k=int(k),
                             probabilities=[float(x) for x in p], logits=[float(x) for x in np.log(np.clip(p, 1e-300, None))],
                             marker_logits=[float(x) for x in z], bucket=bucket, temperature=t_served, temperature_raw=t_raw,
                             probabilities_raw_temperature=[float(x) for x in p_raw] if t_raw != t_served else None,
                             pred=int(np.argmax(p)), confidence=float(np.max(p)), correct=int(int(np.argmax(p)) == c["gold"]),
                             n_tokens=L))
        path = rows_path(tid, system)
        path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
        meta = {"system": system, "task": tid, "device": str(agent.device), "dtype": str(agent.dtype),
                "threads": torch.get_num_threads(), "seconds": round(secs, 2), "ms_per_case_batched": round(1000 * secs / max(1, len(cases)), 1),
                "dropped": dropped, "n": len(cases), "max_len": agent.cfg.get("max_len"), "head_max_len": agent.cfg.get("head_max_len")}
        rows_path(tid, system, ".meta").with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        ok = [r for r in recs if "error" not in r]
        print(f"[{system}] {tid:22s} n={len(ok)} acc={np.mean([r['correct'] for r in ok]):.4f} "
              f"conf={np.mean([r['confidence'] for r in ok]):.4f} {meta['ms_per_case_batched']:.1f} ms/case batched "
              f"({secs:.0f}s, dropped {dropped})", flush=True)


def time_laya(system, device, tasks, threads):
    """Per-decision latency: one Agent.system_one call per case (batch of one), as a caller would use it."""
    import torch
    if threads:
        torch.set_num_threads(threads)
    agent = load_laya(system, device)
    cuda = agent.device.type == "cuda"
    print(f"[{system} timing] device {agent.device} dtype {agent.dtype}", flush=True)
    if cuda:
        free, total = torch.cuda.mem_get_info()
        print(f"  cuda free {free / 2**20:.0f} MiB of {total / 2**20:.0f} MiB after load", flush=True)
    by = read_cases()
    for _ in range(5):   # warm-up
        agent.system_one({"text": "hello there"}, {"q": {"type": "noul", "instructions": "Is `text` a greeting?"}})
    for tid in tasks:
        recs = []
        for c in by[tid]:
            q = {c["qid"]: c["question"]}
            if cuda:
                torch.cuda.synchronize()
            t = time.perf_counter()
            try:
                res = agent.system_one(c["state"], q)["answers"][c["qid"]]
            except Exception as exc:  # noqa: BLE001
                recs.append(dict(id=c["id"], error=f"{type(exc).__name__}: {str(exc)[:160]}")); continue
            if cuda:
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 1000
            if c["question"]["type"] == "choice":
                p = [res["probabilities"][k] for k in c["question"]["criteria"]]
            else:
                p = [1 - res["noul"], res["noul"]]
            recs.append(dict(id=c["id"], ms=ms, pred=int(np.argmax(p)), gold=c["gold"]))
        path = rows_path(tid, system, f"_timing_{agent.device.type}")
        path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
        ok = [r for r in recs if "error" not in r]
        print(f"[{system} timing] {tid:22s} n={len(ok)} ms_p50={np.median([r['ms'] for r in ok]):.1f} "
              f"ms_p95={np.percentile([r['ms'] for r in ok], 95):.1f}", flush=True)


# ------------------------------------------------------------------ metrics and aggregate
def metrics(recs):
    """Accuracy, macro-F1, ECE-15 (bench_h2h / Laya's binning), mean confidence, Brier, NLL, acc@50% coverage,
    and ECE after a 2-fold out-of-fold temperature (the refit used in BENCHMARKS.md)."""
    g = np.array([r["gold"] for r in recs])
    P = [np.asarray(r["probabilities"], float) for r in recs]
    pred = np.array([int(np.argmax(p)) for p in P])
    conf = np.array([float(np.max(p)) for p in P])
    corr = (pred == g).astype(float)
    out = dict(n=len(recs), accuracy=float(corr.mean()), macro_f1=H2H.macro_f1(g, pred), ece=H2H.ece15(conf, corr),
               mean_confidence=float(conf.mean()),
               brier=float(np.mean([((p - np.eye(len(p))[gg]) ** 2).sum() for p, gg in zip(P, g)])),
               nll=float(np.mean([-math.log(max(float(p[gg]), 1e-12)) for p, gg in zip(P, g)])),
               acc_at_50_coverage=float(corr[np.argsort(-conf, kind="stable")[: max(1, len(conf) // 2)]].mean()))
    rows = [(r["gold"], np.asarray(r["logits"], float)) for r in recs]
    conf2, corr2 = [], []
    for fold in (0, 1):
        fit = [rw for i, rw in enumerate(rows) if i % 2 == fold]
        ev = [rw for i, rw in enumerate(rows) if i % 2 != fold]
        T = H2H.fit_temperature(fit)
        for gg, z in ev:
            p = H2H.softmax_t(z, T)
            conf2.append(float(p.max())); corr2.append(float(np.argmax(p) == gg))
    out["ece_refit"] = H2H.ece15(conf2, corr2)
    return out


def r4(x):
    return None if x is None else (round(float(x), 4) if isinstance(x, (float, np.floating)) else x)


def aggregate():
    import datetime
    pub = json.loads(PUBLISHED.read_text(encoding="utf-8")) if PUBLISHED.exists() else None
    by = read_cases()
    tasks_out, notes = [], []
    for tid, label, suite, held in TASKS:
        cases = by[tid]
        entry = {"id": tid, "label": label, "n": len(cases), "held_out": held, "laya_suite": suite,
                 "rows_sha16": task_sha(cases), "systems": {}, "laya_published": {}, "reproduction": {}}
        for sysname in SYSTEMS:
            recs_all = read_rows(rows_path(tid, sysname))
            recs = [r for r in recs_all if "error" not in r]
            if not recs:
                continue
            m = metrics(recs)
            m["dropped"] = len(cases) - len(recs)
            if sysname == "tez":
                m["ms_p50"] = float(np.median([r["ms"] for r in recs]))
                m["ms_p95"] = float(np.percentile([r["ms"] for r in recs], 95))
                m["ms_source"] = "per decision, llama-server /completion round trip (prompt cache on), RTX 5080 Laptop"
            else:
                meta_p = rows_path(tid, sysname, ".meta").with_suffix(".json")
                meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
                m["cpu_batched_ms_per_case_not_a_speed_result"] = meta.get("ms_per_case_batched")
                m["scored_on"] = f"{meta.get('device')} {meta.get('dtype')}, {meta.get('threads')} threads, Laya's batched harness"
                timing = None
                for dev in ("cuda", "cpu"):
                    tr = [r for r in read_rows(rows_path(tid, sysname, f"_timing_{dev}")) if "error" not in r]
                    if tr:
                        timing = (dev, tr)
                        break
                if timing:
                    dev, tr = timing
                    m["ms_p50"] = float(np.median([r["ms"] for r in tr]))
                    m["ms_p95"] = float(np.percentile([r["ms"] for r in tr], 95))
                    m["ms_source"] = (f"per decision, one Agent.system_one call per case on {dev}"
                                      + (" (RTX 5080 Laptop, bf16 autocast)" if dev == "cuda" else " (fp32)"))
                    pred_cpu = {r["id"]: r["pred"] for r in recs}
                    agree = [pred_cpu[r["id"]] == r["pred"] for r in tr if r["id"] in pred_cpu]
                    m["timing_pass_pred_agreement"] = float(np.mean(agree)) if agree else None
                else:
                    m["ms_p50"] = None
                raw = [r for r in recs if r.get("probabilities_raw_temperature")]
                if raw:
                    m2 = metrics([dict(r, probabilities=r["probabilities_raw_temperature"]) for r in recs if r.get("probabilities_raw_temperature")] +
                                 [r for r in recs if not r.get("probabilities_raw_temperature")])
                    m["ece_raw_shipped_temperature"] = m2["ece"]
                    m["mean_confidence_raw_shipped_temperature"] = m2["mean_confidence"]
            entry["systems"][sysname] = {k: r4(v) for k, v in m.items()}
        if pub:
            for sysname in LAYA:
                pm = pub["suites"].get(suite, {}).get(LAYA_PUBLISHED_NAME[sysname])
                if not pm:
                    continue
                entry["laya_published"][sysname] = pm["accuracy"]
                mine = entry["systems"].get(sysname)
                if mine:
                    entry["reproduction"][sysname] = {
                        "published": {k: pm.get(k) for k in ("n", "accuracy", "macro_f1", "ece", "mean_confidence", "ms_per_case")},
                        "rerun": {k: mine.get(k) for k in ("n", "accuracy", "macro_f1", "ece", "mean_confidence")},
                        "delta_accuracy": r4(mine["accuracy"] - pm["accuracy"]),
                        "delta_correct_rows": int(round((mine["accuracy"] - pm["accuracy"]) * mine["n"])),
                    }
        tasks_out.append(entry)

    # how close the Laya reruns are to Laya's published file
    repro_summary = {}
    for sysname in LAYA:
        cells = [e["reproduction"][sysname] for e in tasks_out if sysname in e["reproduction"]]
        if not cells:
            continue
        same = [all(c["published"].get(k) is not None and abs(c["rerun"][k] - c["published"][k]) < 6e-5
                    for k in ("accuracy", "macro_f1", "ece", "mean_confidence")) for c in cells]
        same_acc = [abs(c["delta_accuracy"]) < 1e-9 for c in cells]
        repro_summary[sysname] = {"tasks": len(cells), "identical_accuracy": int(sum(same_acc)),
                                  "identical_accuracy_f1_ece_confidence": int(sum(same)),
                                  "max_abs_delta_accuracy": max(abs(c["delta_accuracy"]) for c in cells)}
    for sysname, r in repro_summary.items():
        if r["identical_accuracy_f1_ece_confidence"] == r["tasks"]:
            notes.append(f"{sysname}: all {r['tasks']} tasks reproduce Laya's published accuracy, macro-F1, ECE and mean "
                         f"confidence to 4 decimals (same rows, same scoring path).")
        else:
            notes.append(f"{sysname}: does NOT reproduce Laya's published numbers: accuracy identical on "
                         f"{r['identical_accuracy']}/{r['tasks']} tasks, max |delta accuracy| {r['max_abs_delta_accuracy']:.4f}, "
                         f"confidences differ on every task; see tasks[].reproduction.")
    if repro_summary.get("laya-multilingual", {}).get("identical_accuracy_f1_ece_confidence", 1) == 0:
        notes.append("laya-multilingual: the released weights (sha256 9d628fd9..., identical in the convaiinnovations/laya "
                     "bundle and the standalone laya-multilingual repo) give the same answers here on CPU fp32 as our "
                     "earlier CUDA bf16 runs (1,087 of 1,100 MASSIVE rows), and the English and typed-decisions checkpoints reproduce to the row through "
                     "the same code, so the gap is in the multilingual checkpoint's environment, not the rows. Laya's "
                     "published run loaded a local directory (~/laya_models/laya-multilingual) with laya 0.2.x; candidates "
                     "are different local weights or a transformers-version difference for mmBERT (its encoder config is in "
                     "the transformers 5 format, sliding-window layers). Not isolated. The same shift shows on MASSIVE "
                     "(massive51.json).")
    notes.append("Laya ms_p50: per-case Agent.system_one on the RTX 5080 Laptop GPU (bf16 autocast), llama-server idle, "
                 "CPU jobs suspended; tez ms_p50: per-decision /completion round trip with the constant prefix cached.")

    props = {}
    try:
        props = H2H.SESSION.get("http://127.0.0.1:8091/props", timeout=10).json()
    except Exception:  # noqa: BLE001
        pass
    try:
        import laya
        laya_version = laya.__version__
    except Exception:  # noqa: BLE001
        laya_version = None
    meta = {
        "title": "Tez vs Laya on Laya's seven application workflows (same rows)",
        "date": datetime.date.today().isoformat(),
        "script": "experiments/vs_laya_apps.py",
        "rows": "results/vs_laya/apps_cases.jsonl (frozen inputs); per-row predictions in results/vs_laya/apps_rows/",
        "protocol": ("Rows rebuilt exactly as Laya's research/scripts/bench_apps.py: same datasets/splits/filters, "
                     "400 cases per task (model routing 399 = 3 x 133, as in Laya's run), one random.Random(13) "
                     "consumed in Laya's suite order, question wording verbatim, email states via laya 0.2.1's "
                     "email_state (the version behind Laya's published file)."),
        "tez": {"model": "Gemma 4 12B Q8_0 (GGUF), frozen, zero-shot letter readout", "server": "llama.cpp llama-server",
                "model_path": props.get("model_path"), "llama_cpp_build": props.get("build_info"),
                "server_flags": "-ngl 99 -c 4096 -b 512 -np 1 --swa-full --embeddings --pooling last",
                "readout": ("experiments/bench_h2h.py tez_decide: one /completion call, top-200 log-probs, softmax over "
                            "the option letters, temperature 1 (as shipped); noul options rendered like the runtime "
                            "(tez.schema.Question.options)")},
        "laya": {"package": f"laya {laya_version} (pip)", "checkpoints": "convaiinnovations/laya (root, multilingual, typed-decisions subfolders; weights bundled 2026-09-19, unchanged since)",
                 "scoring": ("Laya's own harness (bench_local.score_cases): raw marker logits, softmax at the checkpoint's "
                             "bucket temperature as served by the package (clamped to [0.5, 5] since laya 0.3; no app bucket "
                             "is affected unless noted per system), CPU fp32 like Laya's published run"),
                 "latency": "ms_p50 from a separate pass: one Agent.system_one call per case (see ms_source per system)"},
        "hardware": {"gpu": "NVIDIA GeForce RTX 5080 Laptop GPU 16 GB (WDDM)", "cpu": "Intel Core Ultra 9 275HX (24 cores)",
                     "os": platform.platform()},
        "metrics": ("accuracy; macro-F1 over the classes present; ECE with 15 equal-width confidence bins on the max "
                    "probability; mean confidence = mean max probability; ms_p50 = median wall-clock per decision; "
                    "ece_refit = ECE after a 2-fold out-of-fold temperature (BENCHMARKS.md)"),
        "laya_published_source": "github.com/NandhaKishorM/laya research/results/app_benchmark_results.json (laya 0.2.1, CPU, 2026-09-19); copy in results/vs_laya/laya_published/",
        "reproduction_summary": repro_summary,
        "notes": notes,
    }
    doc = {"meta": meta, "tasks": tasks_out}
    (OUT / "apps.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT / 'apps.json'}")
    for e in tasks_out:
        line = "  ".join(f"{s}={e['systems'][s]['accuracy']:.3f}" for s in SYSTEMS if s in e["systems"])
        pubs = "  ".join(f"{s}={v:.3f}" for s, v in e["laya_published"].items())
        print(f"  {e['id']:22s} {line}   | published {pubs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="rebuild and freeze the rows (apps_cases.jsonl)")
    ap.add_argument("--system", choices=SYSTEMS)
    ap.add_argument("--time", action="store_true", help="Laya per-case latency pass instead of scoring")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--server", default="http://127.0.0.1:8091")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    ROWS.mkdir(parents=True, exist_ok=True)
    tasks = TASK_IDS if a.tasks == "all" else a.tasks.split(",")
    if a.build:
        write_cases()
    if a.system == "tez":
        run_tez(a.server, tasks)
    elif a.system and a.time:
        time_laya(a.system, a.device, tasks, a.threads)
    elif a.system:
        run_laya(a.system, a.device, tasks, a.threads)
    if a.aggregate:
        aggregate()


if __name__ == "__main__":
    main()
