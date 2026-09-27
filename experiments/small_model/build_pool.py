"""Pilot day 1: sample the state pool, filter it against the evaluation suites, cluster it for schema writing.

    python experiments/small_model/build_pool.py sample     # raw states from the ledger's training sources
    python experiments/small_model/build_pool.py filter     # leakage guard + domain filter + clusters -> pool.jsonl

Sample (plan §8, week-1 day 1): 3,000 FineWeb (en), 2,000 FineWeb-2 in de, es, fr, ja, zh, ar, hi, th, ko (km held out),
500 Aya, 500 gold rows (CLINC with 20 sampled options, CommonsenseQA, Global-MMLU). Each source is oversampled by
about a quarter so the filters can drop rows and the counts still hold; the filter keeps the first N per source.
Web passages are cut at a word or sentence boundary to a length drawn log-uniformly between about 40 and 512 tokens,
so the pool covers short utterances as well as documents.

Filter (plan §6, leakage guard step 2, and the S-clean domain filter):
  1. drop on an exact hash match (state_key) or a normalised-text hash match with any evaluation state
  2. drop at MinHash Jaccard >= 0.8 (character 5-grams)
  3. drop at MiniLM cosine >= 0.95 with any evaluation state
  4. domain filter: drop a state nearer (cosine) to an evaluation workflow's centroid than to every training cluster
     centroid (k-means on the pool)
The manifest records how many rows each step dropped and the hash of the final pool.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import time

import numpy as np

from common import (DATA, HELD_OUT_LANGS, RESULTS, RUNS, dump, minhash, norm_text, read_jsonl, sha, state_hashes,
                    write_jsonl)

SEED = 13
RAW = RUNS / "pool_raw.jsonl"
POOL = RUNS / "pool.jsonl"
FW2 = {"de": "deu_Latn", "es": "spa_Latn", "fr": "fra_Latn", "ja": "jpn_Jpan", "zh": "cmn_Hani", "ar": "arb_Arab",
       "hi": "hin_Deva", "th": "tha_Thai", "ko": "kor_Hang"}
# characters per token, roughly, for the Gemma tokenizer (only used to aim the cut length)
CPT = {"en": 4.2, "de": 3.6, "es": 3.8, "fr": 3.8, "ja": 1.3, "zh": 1.2, "ar": 3.0, "hi": 2.6, "th": 2.4, "ko": 1.6}
TARGET = {"fineweb": 3150, "fineweb2": 2100, "aya": 525, "clinc": 175, "csqa": 175, "gmmlu": 175}   # 6,300 states: 6,000 train + 5 % val
OVER = 1.25
# A later, larger sample (--scale, --seed, --exclude-pool, --id-prefix) draws fresh rows disjoint from an earlier pool,
# under its own TEZ_PILOT_RUNS directory; the defaults reproduce the original pilot sample exactly.
SCALE = 1.0
PREFIX = ""
EXCLUDE_IDS: set[str] = set()
EXCLUDE_REFS: set[str] = set()
EXCLUDE_TEXT: set[str] = set()   # sha of normalised text of the earlier pool's states
AYA_HELD_OUT = {"Amharic", "Welsh", "Icelandic", "Javanese", "Khmer", "Mongolian", "Burmese", "Swahili"}
GMMLU_LANGS = ["en", "de", "es", "fr", "ja", "zh", "ar", "hi", "ko"]


def cut(text: str, lang: str, rng: random.Random) -> str:
    """Cut to a log-uniform target of 40-512 tokens at a boundary (sentence end if one is near, else whitespace)."""
    tokens = math.exp(rng.uniform(math.log(40), math.log(512)))
    limit = int(tokens * CPT.get(lang, 3.5))
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[:limit]
    m = max(head.rfind(p) for p in (". ", "。", "! ", "? ", "\n", "।", "؟ "))
    if m > limit * 0.6:
        return head[: m + 1].strip()
    sp = head.rfind(" ")
    return head[:sp].strip() if sp > limit * 0.6 else head.strip()


def stream(repo: str, folder: str, n: int, rng_seed: int, groups: int = 6):
    """n random rows from `groups` random parquet row groups (1,000 rows each) spread over the folder's shards, read by
    HTTP range requests (datasets' streaming resolver stalls on fineweb-2's thousands of files behind the proxy)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem
    rng = random.Random(rng_seed)
    files = sorted(f.path for f in HfApi().list_repo_tree(repo, path_in_repo=folder, repo_type="dataset")
                   if f.path.endswith(".parquet"))
    fs = HfFileSystem()
    rows: list[dict] = []
    for g in range(groups):
        path = rng.choice(files)
        p = pq.ParquetFile(fs.open(f"datasets/{repo}/{path}", block_size=1 << 20))
        rg = rng.randrange(p.metadata.num_row_groups)
        tb = p.read_row_group(rg, columns=["text", "id"]).to_pylist()
        for r in tb:
            r["ref"] = f"{path}#rg{rg}"
        rows.extend(tb)
    rng.shuffle(rows)
    yield from rows[:n]


def sample() -> None:
    rng = random.Random(SEED)
    rows: list[dict] = []
    t0 = time.time()

    n = int(TARGET["fineweb"] * OVER)
    for i, r in enumerate(stream("HuggingFaceFW/fineweb", "sample/10BT", n + len(EXCLUDE_REFS), SEED,
                                 groups=math.ceil(8 * SCALE))):
        if r["id"] in EXCLUDE_REFS:
            continue
        rows.append({"id": f"{PREFIX}fw-{i:05d}", "source": "fineweb", "lang": "en", "state": {"text": cut(r["text"], "en", rng)},
                     "src_ref": r["id"], "shard": r["ref"]})
    print(f"fineweb {sum(r['source'] == 'fineweb' for r in rows)} ({time.time() - t0:.0f} s)", flush=True)

    per = int(TARGET["fineweb2"] * OVER / len(FW2)) + 1
    for lang, cfg in FW2.items():
        for i, r in enumerate(stream("HuggingFaceFW/fineweb-2", f"data/{cfg}/train", per, SEED,
                                     groups=math.ceil(3 * SCALE))):
            if r["id"] in EXCLUDE_REFS:
                continue
            rows.append({"id": f"{PREFIX}fw2-{lang}-{i:04d}", "source": "fineweb2", "lang": lang,
                         "state": {"text": cut(r["text"], lang, rng)}, "src_ref": r["id"], "shard": r["ref"]})
        print(f"fineweb2 {lang} ({time.time() - t0:.0f} s)", flush=True)

    from datasets import load_dataset
    aya = load_dataset("CohereLabs/aya_dataset", split="train")
    idx = [i for i in range(len(aya)) if aya[i]["language"] not in AYA_HELD_OUT]
    rng.shuffle(idx)
    k = 0
    for i in idx:
        r = aya[i]
        if f"aya-{i}" in EXCLUDE_IDS or len(r["inputs"]) + len(r["targets"]) > 2400:
            continue
        rows.append({"id": f"{PREFIX}aya-{i}", "source": "aya", "lang": r["language_code"], "lang_name": r["language"],
                     "state": {"prompt": r["inputs"], "response": r["targets"]}})
        k += 1
        if k >= int(TARGET["aya"] * OVER):
            break
    print(f"aya {k} ({time.time() - t0:.0f} s)", flush=True)

    # gold rows: the question wording is written by the 12B later (schemas.py); here only the state, options and gold
    clinc = load_dataset("clinc/clinc_oos", "plus", split="train")
    names = clinc.features["intent"].names
    idx = list(range(len(clinc)))
    rng.shuffle(idx)
    for i in idx[: int(TARGET["clinc"] * OVER)]:
        r = clinc[i]
        gold = names[r["intent"]]
        if gold == "oos" or f"clinc-{i}" in EXCLUDE_IDS:
            continue
        keys = [gold] + rng.sample([x for x in names if x not in (gold, "oos")], 19)
        rng.shuffle(keys)
        rows.append({"id": f"{PREFIX}clinc-{i}", "source": "clinc", "lang": "en", "state": {"text": r["text"]},
                     "gold_q": {"dataset": "clinc", "type": "choice",
                                "criteria": {k: k.replace("_", " ") for k in keys}, "label": gold}})
    csqa = load_dataset("tau/commonsense_qa", split="train")
    idx = list(range(len(csqa)))
    rng.shuffle(idx)
    for i in idx[: int(TARGET["csqa"] * OVER)]:
        r = csqa[i]
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        if r["answerKey"] not in labels or len(set(texts)) != len(texts) or f"csqa-{i}" in EXCLUDE_IDS:
            continue
        rows.append({"id": f"{PREFIX}csqa-{i}", "source": "csqa", "lang": "en", "state": {"question": r["question"]},
                     "gold_q": {"dataset": "csqa", "type": "choice", "criteria": {t: None for t in texts},
                                "label": texts[labels.index(r["answerKey"])]}})
    per = int(TARGET["gmmlu"] * OVER / len(GMMLU_LANGS)) + 1
    for lang in GMMLU_LANGS:
        d = load_dataset("CohereLabs/Global-MMLU", lang, split="test")
        idx = list(range(len(d)))
        rng.shuffle(idx)
        for i in idx[:per]:
            r = d[i]
            opts = [r["option_a"], r["option_b"], r["option_c"], r["option_d"]]
            if len(set(opts)) != 4 or r["answer"] not in "ABCD" or f"gmmlu-{lang}-{i}" in EXCLUDE_IDS:
                continue
            rows.append({"id": f"{PREFIX}gmmlu-{lang}-{i}", "source": "gmmlu", "lang": lang, "state": {"question": r["question"]},
                         "gold_q": {"dataset": "gmmlu", "type": "choice", "criteria": {o: None for o in opts},
                                    "label": opts["ABCD".index(r["answer"])], "subject": r.get("subject")}})
    print(f"gold done ({time.time() - t0:.0f} s)", flush=True)
    write_jsonl(RAW, rows)
    print(collections.Counter(r["source"] for r in rows))


def filt(k_clusters: int = 64, device: str = "cuda") -> None:
    from datasketch import MinHashLSH

    rows = read_jsonl(RAW)
    from eval_hashes import load_hashes
    eval_h = load_hashes()
    ex_set, nm_set = set(), set()
    for v in eval_h["sources"].values():
        ex_set.update(v["exact"])
        nm_set.update(v["norm"])
    dropped = collections.Counter()
    drops: list[dict] = []

    # 0. empty / duplicate within the pool
    keep, seen = [], set(EXCLUDE_TEXT)
    for r in rows:
        t = norm_text(r["state"])
        if len(t) < 8 or sha(t) in seen:
            dropped["0_empty_or_duplicate_in_pool"] += 1
            continue
        seen.add(sha(t))
        keep.append(r)
    rows = keep

    # 1. exact / normalised hash
    keep = []
    for r in rows:
        ex, nh = state_hashes(r["state"])
        if ex in ex_set or nh in nm_set:
            dropped["1_hash_match"] += 1
            drops.append({"id": r["id"], "step": "hash"})
            continue
        keep.append(r)
    rows = keep

    # 2. MinHash Jaccard >= 0.8
    ev = np.load(RUNS / "eval_minhash.npz")
    lsh = MinHashLSH(threshold=0.8, num_perm=ev["sigs"].shape[1])
    from datasketch import MinHash
    from common import MINHASH_SEED
    for i, s in enumerate(ev["sigs"]):
        lsh.insert(f"e{i}", MinHash(num_perm=len(s), seed=MINHASH_SEED, hashvalues=s, scheme="affine32"))
    keep = []
    for r in rows:
        m = minhash(norm_text(r["state"]))
        hits = lsh.query(m)
        best = 0.0
        for h in hits:
            s = ev["sigs"][int(h[1:])]
            best = max(best, float(np.mean(s == m.hashvalues.astype(np.uint32))))
        if best >= 0.8:
            dropped["2_minhash_ge_0.8"] += 1
            drops.append({"id": r["id"], "step": "minhash", "jaccard": round(best, 3)})
            continue
        keep.append(r)
    rows = keep

    # 3. MiniLM cosine >= 0.95 ; 4. domain filter
    from eval_hashes import embed_texts
    emb = embed_texts([norm_text(r["state"]) for r in rows], device).astype(np.float32)
    ee = np.load(RUNS / "eval_emb.npz")
    E, esrc = ee["emb"].astype(np.float32), ee["src"]
    best = np.full(len(rows), -1.0, dtype=np.float32)
    for s in range(0, len(E), 20000):
        best = np.maximum(best, (emb @ E[s:s + 20000].T).max(axis=1))
    mask = best < 0.95
    for r, b in zip(rows, best):
        if b >= 0.95:
            drops.append({"id": r["id"], "step": "cosine", "cos": round(float(b), 3)})
    dropped["3_cosine_ge_0.95"] += int((~mask).sum())
    rows = [r for r, k in zip(rows, mask) if k]
    emb = emb[mask]

    # evaluation workflow centroids: one per evaluation source family (task or workflow; MASSIVE and XNLI per family)
    fam = np.array([s.split(":")[0] + ":" + s.split(":")[-1] if s.startswith(("typed_decisions", "laya_apps", "h2h"))
                    else s.split(":")[0] for s in esrc])
    cents, names = [], []
    for f in sorted(set(fam)):
        c = E[fam == f].mean(axis=0)
        cents.append(c / np.linalg.norm(c))
        names.append(f)
    C_eval = np.stack(cents)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k_clusters, random_state=SEED, n_init=4).fit(emb)
    C_tr = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    d_eval, d_tr = emb @ C_eval.T, emb @ C_tr.T
    near_eval = d_eval.max(axis=1) > d_tr.max(axis=1)
    by_fam = collections.Counter()
    for r, ne, j, de in zip(rows, near_eval, d_eval.argmax(axis=1), d_eval.max(axis=1)):
        if ne:
            by_fam[names[j]] += 1
            drops.append({"id": r["id"], "step": "domain", "nearest_eval": names[j], "cos": round(float(de), 3)})
    dropped["4_domain_nearer_eval_centroid"] += int(near_eval.sum())
    rows = [r for r, k in zip(rows, ~near_eval) if k]
    emb = emb[~near_eval]

    # keep N per source (targets), round-robin over the source's languages in sample order, so every language keeps
    # its share (the first version kept the first N in sample order and cut fineweb-2's last language, ko)
    by_src = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, r in enumerate(rows):
        by_src[r["source"]][r["lang"] if r["source"] in ("fineweb2", "gmmlu") else "all"].append(i)
    chosen = set()
    per = collections.Counter()
    for src, langs in by_src.items():
        queues = [list(v) for v in langs.values()]
        while per[src] < TARGET[src] and any(queues):
            for q in queues:
                if q and per[src] < TARGET[src]:
                    chosen.add(q.pop(0))
                    per[src] += 1
    dropped["5_over_target"] += len(rows) - len(chosen)
    idx = sorted(chosen)
    rows, emb = [rows[i] for i in idx], emb[idx]
    np.save(RUNS / "pool_emb.npy", emb)
    write_jsonl(POOL, rows)
    pool_hash = sha("".join(sha(json.dumps(r["state"], ensure_ascii=False, sort_keys=True), 64) for r in rows), 64)
    write_jsonl(RUNS / "pool_drops.jsonl", drops)
    manifest = {"raw_rows": len(read_jsonl(RAW)), "dropped": dict(sorted(dropped.items())),
                "domain_drops_by_nearest_eval_family": dict(by_fam.most_common()),
                "kept": dict(per), "kept_total": len(rows), "pool_sha256": pool_hash,
                "langs": dict(collections.Counter(r["lang"] for r in rows).most_common()),
                "held_out_langs_present": sorted({r["lang"] for r in rows} & set(HELD_OUT_LANGS)),
                "k_clusters_domain_filter": k_clusters}
    dump(RESULTS / "pool_manifest.json", manifest)
    print(json.dumps(manifest, indent=1))


def main() -> None:
    global SCALE, PREFIX, SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sample", "filter"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every source's target")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--exclude-pool", help="an earlier pool.jsonl whose rows, sources and texts this sample must avoid")
    ap.add_argument("--id-prefix", default="", help="prefix for new state ids, keeping them disjoint from earlier pools")
    args = ap.parse_args()
    SCALE, PREFIX, SEED = args.scale, args.id_prefix, args.seed
    for k in TARGET:
        TARGET[k] = round(TARGET[k] * SCALE)
    if args.exclude_pool:
        for r in read_jsonl(args.exclude_pool):
            EXCLUDE_IDS.add(r["id"])
            if r.get("src_ref"):
                EXCLUDE_REFS.add(r["src_ref"])
            EXCLUDE_TEXT.add(sha(norm_text(r["state"])))
    sample() if args.step == "sample" else filt(device=args.device)


if __name__ == "__main__":
    main()
