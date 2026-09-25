"""Leakage guard, step 1 (plan §6): hash every evaluation state before the first teacher call.

    python experiments/small_model/eval_hashes.py hash                 # CPU: hashes + MinHash + the distinct texts
    python experiments/small_model/eval_hashes.py embed --device cuda  # MiniLM embeddings of the distinct texts

Writes
  data/small_model/eval_hashes.json   committed: per source, the SHA-256 (64-bit prefix) of tez.schema.state_key and of
                                      the normalised text of every evaluation state, plus the MinHash parameters and the
                                      SHA-256 of the signature file
  runs/pilot/eval_texts.jsonl         gitignored: the distinct normalised evaluation texts with their source
  runs/pilot/eval_minhash.npz         gitignored: 5-gram MinHash signatures of those texts (64 permutations, seed 13)
  runs/pilot/eval_emb.npz             gitignored: MiniLM embeddings (cosine guard, domain centroids)
(The embeddings run on the GPU: this shared machine's CPU embeds about 20 texts a second under load, which made the
first all-CPU run take hours for 205,776 texts.)
"""
from __future__ import annotations

import argparse
import collections
import json
import time

import numpy as np

from common import DATA, MINHASH_PERM, MINHASH_SEED, RUNS, dump, file_sha256, minhash, norm_text, read_jsonl, state_hashes

EMBEDDER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def hash_phase() -> None:
    from common import eval_states
    t0 = time.time()
    by_source: dict[str, dict[str, list[str]]] = collections.defaultdict(lambda: {"exact": [], "norm": []})
    texts, srcs = [], []
    seen_norm: set[str] = set()
    for src, st in eval_states():
        ex, nh = state_hashes(st)
        by_source[src]["exact"].append(ex)
        by_source[src]["norm"].append(nh)
        if nh not in seen_norm:
            seen_norm.add(nh)
            texts.append(norm_text(st))
            srcs.append(src)
    print(f"{sum(len(v['exact']) for v in by_source.values())} evaluation states, {len(texts)} distinct normalised "
          f"({time.time() - t0:.0f} s)", flush=True)
    with open(RUNS / "eval_texts.jsonl", "w", encoding="utf-8") as f:
        for t, s in zip(texts, srcs):
            f.write(json.dumps({"src": s, "text": t}, ensure_ascii=False) + "\n")
    mh = RUNS / "eval_minhash.npz"
    reuse = False
    if mh.exists():
        old = np.load(mh)
        reuse = old["sigs"].shape[0] == len(texts) and list(old["src"]) == srcs
    if not reuse:
        sigs = np.zeros((len(texts), MINHASH_PERM), dtype=np.uint32)
        for i, t in enumerate(texts):
            sigs[i] = minhash(t).hashvalues.astype(np.uint32)
        np.savez_compressed(mh, sigs=sigs, src=np.array(srcs))
    print(f"minhash {'reused' if reuse else 'computed'} ({time.time() - t0:.0f} s)", flush=True)
    write_hashes({
        "title": "Evaluation states for the small-model pilot's leakage guard (plan §6, step 1)",
        "hash": "first 16 hex characters (64 bits) of SHA-256; exact = tez.schema.state_key(state), "
                "norm = experiments/small_model/common.py norm_text(state) (leaves joined, NFKC, casefold, whitespace)",
        "minhash": {"shingles": "character 5-grams of the normalised text", "num_perm": MINHASH_PERM, "seed": MINHASH_SEED,
                    "file": "runs/pilot/eval_minhash.npz (gitignored; rebuilt by this script)",
                    "sha256": file_sha256(mh), "rows": len(texts)},
        "embedder": EMBEDDER,
        "counts": {k: len(v["exact"]) for k, v in sorted(by_source.items())},
        "sources": {k: v for k, v in sorted(by_source.items())},
    })
    print(f"done in {time.time() - t0:.0f} s")


def write_hashes(obj: dict) -> None:
    """data/small_model/eval_hashes.json.gz (the hashes, gzipped: 360,666 states) and eval_hashes_manifest.json
    (everything but the hash lists, plus the SHA-256 of the .gz)."""
    import gzip
    gz = DATA / "eval_hashes.json.gz"
    with gzip.open(gz, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    man = {k: v for k, v in obj.items() if k != "sources"}
    man["file"] = "data/small_model/eval_hashes.json.gz"
    man["sha256"] = file_sha256(gz)
    dump(DATA / "eval_hashes_manifest.json", man)


def load_hashes() -> dict:
    import gzip
    with gzip.open(DATA / "eval_hashes.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


def embed_texts(texts: list[str], device: str, batch: int = 256) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMBEDDER, device=device)
    if device == "cuda":
        m.half()
    e = m.encode([t[:2000] for t in texts], batch_size=batch, normalize_embeddings=True, show_progress_bar=False,
                 convert_to_numpy=True).astype(np.float16)
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return e


def embed_phase(device: str) -> None:
    t0 = time.time()
    rows = read_jsonl(RUNS / "eval_texts.jsonl")
    emb = embed_texts([r["text"] for r in rows], device)
    np.savez_compressed(RUNS / "eval_emb.npz", emb=emb, src=np.array([r["src"] for r in rows]))
    print(f"embedded {len(rows)} texts on {device} in {time.time() - t0:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["hash", "embed"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    hash_phase() if args.phase == "hash" else embed_phase(args.device)


if __name__ == "__main__":
    main()
