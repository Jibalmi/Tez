"""Pilot day 2a: Gemma 4 12B writes the question schemas, and the gold sets' question wording.

    python experiments/small_model/schemas.py cluster                  # language-homogeneous clusters of the pool (CPU)
    python experiments/small_model/schemas.py write --server URL       # the 12B writes 10 schemas per cluster (GPU)
    python experiments/small_model/schemas.py assign                   # validate, dedupe, filter, give 5 per state

Schemas (plan §5): clusters of about 46 states grouped by language and embedding; for each, the 12B sees five of its
states and writes 10 schemas: 4 choice (2-20 options), 3 yes/no, 3 score (3-7 levels). In FineWeb-2 clusters 7 of the
10 are written in the state's language (the plan's 25 % of questions in the state's language overall); everything
else in English. Large choices (27-255 options) come only from gold sets in the plan; the pilot's gold sets have at
most 20 options, so the pilot has none. The prompt shows only the JSON shape with placeholders: every word of every
schema is the 12B's (no text written by the agent running the pilot becomes training data).

Filters: tez.schema.parse_question validity; duplicates by MiniLM cosine >= 0.9 on the question text and by option
set; a choice schema whose option labels overlap an evaluation question's by 50 % or more is dropped (plan §5).
Output: runs/pilot/schemas.jsonl, runs/pilot/assign.jsonl (state id -> 5 questions), results/pilot/schemas_manifest.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from common import RESULTS, RUNS, THERMAL, dump, read_jsonl, thermal_guard, write_jsonl

SEED = 13
PER_CLUSTER = 46
N_SCHEMAS = 10
LANG_NAME = {"de": "German", "es": "Spanish", "fr": "French", "ja": "Japanese", "zh": "Chinese (Simplified)",
             "ar": "Arabic", "hi": "Hindi", "th": "Thai", "ko": "Korean"}
TMPL = ("<|turn>user\n", "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")

SCHEMA_JSON = {"type": "array", "minItems": N_SCHEMAS, "maxItems": N_SCHEMAS, "items": {"anyOf": [
    {"type": "object", "properties": {"type": {"const": "choice"}, "instructions": {"type": "string"},
                                      "criteria": {"type": "object", "additionalProperties": {"type": "string"}}},
     "required": ["type", "instructions", "criteria"], "additionalProperties": False},
    {"type": "object", "properties": {"type": {"const": "noul"}, "instructions": {"type": "string"}},
     "required": ["type", "instructions"], "additionalProperties": False},
    {"type": "object", "properties": {"type": {"const": "score"}, "instructions": {"type": "string"},
                                      "criteria": {"type": "array", "items": {"type": "string"}}},
     "required": ["type", "instructions", "criteria"], "additionalProperties": False}]}}

SHAPE = ('[\n  {"type": "choice", "instructions": "<question>", "criteria": {"<label_1>": "<what this option means>", '
         '"<label_2>": "<what this option means>", ...}},\n  {"type": "noul", "instructions": "<a yes/no question>"},\n'
         '  {"type": "score", "instructions": "<question>", "criteria": ["<lowest level>", "<next level>", ..., '
         '"<highest level>"]}\n]')


def schema_prompt(examples: list[str], lang: str | None) -> str:
    ex = "\n\n".join(f"Input {i + 1}:\n{e}" for i, e in enumerate(examples))
    if lang:
        lang_rule = (f"Write questions 1 to 3 in English and questions 4 to 10 in {LANG_NAME[lang]} (instructions, "
                     f"labels and descriptions in {LANG_NAME[lang]}).")
    else:
        lang_rule = "Write every question in English."
    return (
        "You design questions for an automated decision engine. The engine reads one input and answers a question "
        "about it by picking one option. Below are example inputs from one collection of similar inputs.\n\n"
        f"{ex}\n\n"
        f"Write {N_SCHEMAS} different questions that someone automating work on inputs like these would need "
        "answered: for example routing, triage, moderation, compliance, quality control, prioritisation, relevance, "
        "intent, topic, audience, tone, risk, completeness or next action. Each question must be answerable from a "
        "single input of this collection, and must make sense for most inputs of the collection, not only for the "
        "examples.\n"
        "Types, in this order: questions 1-4 are \"choice\" (between 2 and 20 mutually exclusive options that cover "
        "the usual cases; vary the number of options: one question with 2 or 3 options, one with 8 or more); "
        "questions 5-7 are \"noul\" (a yes/no question); questions 8-10 are \"score\" (an ordered scale of 3 to 7 "
        "levels, from lowest to highest).\n"
        "Option labels are short identifiers (snake_case); each description is a short phrase. "
        f"{lang_rule}\n"
        f"Answer with a JSON array only, in this shape:\n{SHAPE}")


def gold_prompt(dataset: str, examples: list[dict]) -> str:
    ex = "\n\n".join(f"Input {i + 1}: {json.dumps(e['state'], ensure_ascii=False)}\nOptions: "
                     f"{json.dumps(list(e['gold_q']['criteria']), ensure_ascii=False)}\nCorrect option: "
                     f"{e['gold_q']['label']}" for i, e in enumerate(examples))
    return ("You design questions for an automated decision engine. Below are inputs from one dataset, the options "
            "offered for each and the correct option.\n\n"
            f"{ex}\n\n"
            "Write the single question (one sentence, in English) that the engine should be asked so that the correct "
            "option is the answer, for any input of this dataset. Refer to the input's field by name in backquotes. "
            'Answer with JSON only: {"instructions": "<question>"}')


def generate(server: str, prompt: str, schema: dict, n_predict: int, temperature: float, seed: int) -> tuple[str, dict]:
    body = {"prompt": TMPL[0] + prompt + TMPL[1], "n_predict": n_predict, "temperature": temperature, "top_p": 0.95,
            "seed": seed, "cache_prompt": False}
    if schema is not None:
        body["json_schema"] = schema
    r = requests.post(f"{server}/completion", json=body, timeout=1800)
    r.raise_for_status()
    d = r.json()
    return d["content"], d.get("timings", {})


# ------------------------------------------------------------------------------------------------ steps
def cluster() -> None:
    from sklearn.cluster import KMeans
    pool = read_jsonl(RUNS / "pool.jsonl")
    emb = np.load(RUNS / "pool_emb.npy")
    groups = collections.defaultdict(list)
    for i, r in enumerate(pool):
        if r.get("gold_q"):
            continue
        g = r["lang"] if r["source"] == "fineweb2" else ("en" if r["lang"] in ("en", "eng") else "aya_multi")
        groups[g].append(i)
    clusters = []
    for g, idx in sorted(groups.items()):
        k = max(1, round(len(idx) / PER_CLUSTER))
        lab = KMeans(n_clusters=k, random_state=SEED, n_init=4).fit_predict(emb[idx]) if k > 1 else np.zeros(len(idx), int)
        for c in range(k):
            members = [idx[j] for j in range(len(idx)) if lab[j] == c]
            if members:
                clusters.append({"cid": f"{g}-{c:02d}", "group": g, "lang": g if g in LANG_NAME else None,
                                 "members": [pool[m]["id"] for m in members]})
    # gold rows join the nearest cluster centroid of their language group (English group for English gold rows)
    pos = {r["id"]: i for i, r in enumerate(pool)}
    cents = {c["cid"]: emb[[pos[m] for m in c["members"]]].mean(0) for c in clusters}
    by_id = {c["cid"]: c for c in clusters}
    for i, r in enumerate(pool):
        if not r.get("gold_q"):
            continue
        g = r["lang"] if r["lang"] in LANG_NAME else "en"
        cand = [cid for cid in cents if by_id[cid]["group"] == g] or [cid for cid in cents if by_id[cid]["group"] == "en"]
        best = max(cand, key=lambda cid: float(emb[i] @ cents[cid]) / float(np.linalg.norm(cents[cid])))
        by_id[best].setdefault("gold_members", []).append(r["id"])
    write_jsonl(RUNS / "clusters.jsonl", clusters)
    print(f"{len(clusters)} clusters:", collections.Counter(c["group"] for c in clusters))


def write(server: str, parallel: int, limit: int = 0, minutes: float = 0) -> None:
    pool = {r["id"]: r for r in read_jsonl(RUNS / "pool.jsonl")}
    clusters = read_jsonl(RUNS / "clusters.jsonl")
    out_path = RUNS / "schemas_raw.jsonl"
    done = {r["cid"] for r in read_jsonl(out_path) if r["items"]} if out_path.exists() else set()   # failures retried
    rng = random.Random(SEED)
    jobs = []
    for c in clusters:
        ex = rng.sample(c["members"], min(5, len(c["members"])))
        texts = []
        for sid in ex:
            st = pool[sid]["state"]
            t = json.dumps(st, ensure_ascii=False) if not isinstance(st, str) else st
            texts.append(t[:500] + (" ..." if len(t) > 500 else ""))
        if c["cid"] not in done:
            jobs.append((c, texts))
    if limit:
        jobs = jobs[:limit]
    gold_jobs = []
    gold_done = {r["dataset"] for r in read_jsonl(RUNS / "gold_questions.jsonl")} if (RUNS / "gold_questions.jsonl").exists() else set()
    by_ds = collections.defaultdict(list)
    for r in pool.values():
        if r.get("gold_q"):
            by_ds[r["gold_q"]["dataset"]].append(r)
    for ds, rows in by_ds.items():
        if ds not in gold_done and not limit:
            gold_jobs.append((ds, [r for r in rows if r["lang"] == "en"][:3] or rows[:3]))
    print(f"{len(jobs)} clusters and {len(gold_jobs)} gold datasets to write", flush=True)
    t0 = time.time()
    tok = [0]

    deadline = t0 + minutes * 60 if minutes else None

    def run_cluster(job):
        c, texts = job
        if deadline and time.time() > deadline:
            return None                                  # left for the next segment (resumable)
        thermal_guard(hot=84, cool=79, force=True)
        seed = int(c["cid"].encode().hex()[-6:], 16)
        for attempt in range(3):
            try:
                sch = None if attempt < 2 else SCHEMA_JSON      # grammar only on the last try: grammar sampling is
                                                                # CPU-bound and ran at 7 tok/s on 4 slots on 2026-09-25
                content, tm = generate(server, schema_prompt(texts, c["lang"]), sch, 2500, 0.7, seed + attempt)
                items = json.loads(content[content.index("["): content.rindex("]") + 1])
                if not isinstance(items, list) or len(items) < 5:
                    raise ValueError(f"{len(items) if isinstance(items, list) else type(items).__name__} items")
                return {"cid": c["cid"], "lang": c["lang"], "items": items, "timings": tm, "attempt": attempt}
            except Exception as exc:  # noqa: BLE001
                err = str(exc)[:200]
        return {"cid": c["cid"], "lang": c["lang"], "items": [], "error": err}

    with ThreadPoolExecutor(parallel) as ex, open(out_path, "a", encoding="utf-8") as f:
        for res in ex.map(run_cluster, jobs):
            if res is None:
                continue
            tok[0] += res.get("timings", {}).get("predicted_n", 0)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            el = time.time() - t0
            print(f"  {res['cid']}: {len(res['items'])} items; {tok[0]} tokens in {el:.0f} s "
                  f"({tok[0] / max(el, 1e-9):.1f} tok/s aggregate)", flush=True)
    gold_schema = {"type": "object", "properties": {"instructions": {"type": "string"}}, "required": ["instructions"],
                   "additionalProperties": False}
    with open(RUNS / "gold_questions.jsonl", "a", encoding="utf-8") as f:
        for ds, rows in gold_jobs:
            try:
                content, tm = generate(server, gold_prompt(ds, rows), None, 200, 0.0, SEED)
                json.loads(content[content.index("{"): content.rindex("}") + 1])["instructions"]
            except Exception:  # noqa: BLE001
                content, tm = generate(server, gold_prompt(ds, rows), gold_schema, 200, 0.0, SEED)
            instr = json.loads(content[content.index("{"): content.rindex("}") + 1])["instructions"]
            f.write(json.dumps({"dataset": ds, "instructions": instr, "timings": tm}, ensure_ascii=False) + "\n")
            print(f"  gold {ds}: {instr}", flush=True)
    dump(RESULTS / ("schema_generation_bulk.json" if not limit else f"schema_generation_timing_p{parallel}.json"), {"tokens": tok[0], "seconds": round(time.time() - t0, 1),
                                                     "tok_per_s_aggregate": round(tok[0] / max(time.time() - t0, 1e-9), 1),
                                                     "parallel_slots": parallel, "clusters": len(jobs),
                                                     "thermal": dict(THERMAL)})


def eval_option_sets() -> list[set[str]]:
    """Option label sets of every evaluation question (choice questions), normalised.

    TEZ_PILOT_EVAL_OPTION_SETS may name a JSON file written by `schemas.py option-sets` on a machine whose Hub cache
    already holds the evaluation datasets (newer huggingface_hub releases reject their legacy repository ids)."""
    import os
    cached = os.environ.get("TEZ_PILOT_EVAL_OPTION_SETS")
    if cached:
        with open(cached, encoding="utf-8") as f:
            return [set(x) for x in json.load(f)]
    import bench_h2h
    from datasets import load_dataset
    norm = lambda s: re.sub(r"[\s_\-:.]+", " ", str(s).casefold()).strip()  # noqa: E731
    sets = []
    for split in ("train", "test"):
        for r in load_dataset("LocalLLaMA/typed-decisions", "all", split=split):
            for q in json.loads(r["questions"]).values():
                if q["type"] == "choice":
                    sets.append({norm(k) for k in q["criteria"]})
    for r in read_jsonl(__import__("common").ROOT / "results" / "vs_laya" / "apps_cases.jsonl"):
        if r["question"]["type"] == "choice":
            sets.append({norm(k) for k in r["question"]["criteria"]})
    for task in ("ag_news", "emotion", "banking77"):
        c = bench_h2h.build_task(task, 1)[0]
        sets.append({norm(k) for k, _ in c["options"]})
    massive = load_dataset("mteb/amazon_massive_intent", "en", split="test")
    sets.append({norm(x) for x in set(massive["label_text"])})       # the whole intent schema (20-option subsets are inside it)
    root = __import__("common").ROOT
    for name in ("authored144", "perturbations108"):
        for r in read_jsonl(root / "data" / "semif" / f"{name}.jsonl"):
            sets.append({norm(o["id"]) for o in r["options"]})
    for tier in ("original", "easy", "hard"):
        for r in read_jsonl(root / "data" / "jevbench" / f"{tier}.jsonl"):
            sets.append({norm(x) for x in r["labels"]})
    voice = json.loads((root / "data" / "voice" / "intents.json").read_text(encoding="utf-8"))["intents"]
    sets.append({norm(x["id"]) for x in voice})
    return sets


def assign(device: str = "cuda") -> None:
    from eval_hashes import embed_texts
    from tez.schema import parse_question
    rng = random.Random(SEED)
    raw = read_jsonl(RUNS / "schemas_raw.jsonl")
    stats = collections.Counter()
    schemas = []
    for res in raw:
        for j, it in enumerate(res["items"]):
            stats["written"] += 1
            try:
                q = parse_question(f"{res['cid']}-{j}", it)
            except Exception:  # noqa: BLE001
                stats["invalid"] += 1
                continue
            if q.type == "choice" and not 2 <= len(q.criteria) <= 20:
                stats["invalid_option_count"] += 1
                continue
            if q.type == "score" and not 3 <= len(q.criteria) <= 7:
                stats["invalid_level_count"] += 1
                continue
            schemas.append({"sid": q.id, "cid": res["cid"], "lang": res["lang"] if j >= 3 else None, "q": q.to_wire()})
    # option-overlap with evaluation questions (choice schemas)
    evsets = eval_option_sets()
    norm = lambda s: re.sub(r"[\s_\-:.]+", " ", str(s).casefold()).strip()  # noqa: E731
    keep = []
    for s in schemas:
        if s["q"]["type"] == "choice":
            labs = {norm(k) for k in s["q"]["criteria"]}
            ov = max(len(labs & e) / len(labs) for e in evsets)
            if ov >= 0.5:
                stats["dropped_eval_option_overlap"] += 1
                continue
        keep.append(s)
    schemas = keep
    # duplicates: same option set, or MiniLM cosine >= 0.9 on the question text (instructions + options)
    seen_sets, keep = set(), []
    for s in schemas:
        key = (s["q"]["type"], tuple(sorted(s["q"].get("criteria") or []) if s["q"]["type"] == "choice" else ()))
        if s["q"]["type"] == "choice" and key in seen_sets:
            stats["dropped_same_option_set"] += 1
            continue
        seen_sets.add(key)
        keep.append(s)
    schemas = keep
    def plain(q: dict) -> str:
        """The question as text: type, instructions, option labels. The first version embedded the schema's JSON,
        whose shared scaffolding (type / instructions / criteria keys, quotes) inflated similarity: 473 of 1,250
        schemas kept at 0.9, against 843 on plain text."""
        crit = q.get("criteria")
        opts = list(crit) if q["type"] == "choice" else ([str(c) for c in crit] if q["type"] == "score" else [])
        return f"({q['type']}) {q['instructions']} | " + "; ".join(o.replace("_", " ") for o in opts)

    texts = [plain(s["q"]) for s in schemas]
    E = embed_texts(texts, device).astype(np.float32)
    keep_idx = []
    for i in range(len(schemas)):
        if keep_idx and float((E[keep_idx] @ E[i]).max()) >= 0.9:
            stats["dropped_cosine_ge_0.9"] += 1
            continue
        keep_idx.append(i)
    schemas = [schemas[i] for i in keep_idx]
    write_jsonl(RUNS / "schemas.jsonl", schemas)

    # 5 questions per state from its cluster; gold rows: the gold question + 4 from their cluster
    pool = {r["id"]: r for r in read_jsonl(RUNS / "pool.jsonl")}
    gold_q = {r["dataset"]: r["instructions"] for r in read_jsonl(RUNS / "gold_questions.jsonl")}
    by_c = collections.defaultdict(list)
    for s in schemas:
        by_c[s["cid"]].append(s)
    assign_rows = []
    for c in read_jsonl(RUNS / "clusters.jsonl"):
        avail = by_c.get(c["cid"], [])
        if len(avail) < 5:        # a cluster whose schemas mostly failed borrows from the others of its group
            extra = [s for s in schemas if s["cid"].rsplit("-", 1)[0] == c["group"] and s["cid"] != c["cid"]]
            avail = avail + rng.sample(extra, min(len(extra), 10 - len(avail)))
        for sid in c["members"]:
            qs = rng.sample(avail, 5)
            assign_rows.append({"id": sid, "questions": [{"qid": s["sid"], **s["q"]} for s in qs]})
        for sid in c.get("gold_members", []):
            g = pool[sid]["gold_q"]
            gq = {"qid": f"gold-{g['dataset']}", "type": "choice", "instructions": gold_q[g["dataset"]],
                  "criteria": g["criteria"], "gold": g["label"]}
            qs = rng.sample(avail, 4)
            assign_rows.append({"id": sid, "questions": [gq] + [{"qid": s["sid"], **s["q"]} for s in qs]})
    write_jsonl(RUNS / "assign.jsonl", assign_rows)
    types = collections.Counter(q["type"] for r in assign_rows for q in r["questions"])
    nopt = [len(q["criteria"]) for r in assign_rows for q in r["questions"] if q["type"] == "choice"]
    reuse = collections.Counter(q["qid"] for r in assign_rows for q in r["questions"])
    local = sum(1 for r in assign_rows for q in r["questions"] if q["qid"].startswith(tuple(LANG_NAME)) and
                int(q["qid"].rsplit("-", 1)[1]) >= 3)
    man = {"schemas_written": stats["written"], "filters": dict(stats), "schemas_kept": len(schemas),
           "gold_questions": gold_q, "states": len(assign_rows),
           "decisions": sum(len(r["questions"]) for r in assign_rows), "types": dict(types),
           "choice_options": {"min": min(nopt), "median": float(np.median(nopt)), "max": max(nopt)},
           "schema_reuse": {"median_states_per_schema": float(np.median(list(reuse.values()))),
                            "max": max(reuse.values())},
           "questions_in_state_language": local}
    dump(RESULTS / "schemas_manifest.json", man)
    print(json.dumps(man, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["cluster", "write", "assign", "option-sets"])
    ap.add_argument("--out", help="option-sets: the JSON file to write")
    ap.add_argument("--server", default="http://127.0.0.1:8851")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--device", default="cuda", help="embedding device for assign (cosine dedupe)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N clusters (the single-stream rate check)")
    ap.add_argument("--minutes", type=float, default=0, help="stop starting clusters after this long (resumable)")
    args = ap.parse_args()
    if args.step == "cluster":
        cluster()
    elif args.step == "write":
        write(args.server, args.parallel, args.limit, args.minutes)
    elif args.step == "option-sets":
        sets = eval_option_sets()
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump([sorted(x) for x in sets], f, ensure_ascii=False)
        print(f"{len(sets)} evaluation option sets -> {args.out}")
    else:
        assign(args.device)


if __name__ == "__main__":
    main()
