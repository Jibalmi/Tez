"""Leakage guard, step 1 (plan §6): hash every evaluation state before the first teacher call.

    python experiments/small_model/eval_hashes.py            # hashes + MinHash + embeddings

Writes
  data/small_model/eval_hashes.json   committed: per source, the SHA-256 (64-bit prefix) of tez.schema.state_key and of
                                      the normalised text of every evaluation state, plus the MinHash parameters and the
                                      SHA-256 of the signature file
  runs/pilot/eval_minhash.npz         gitignored: 5-gram MinHash signatures of every normalised evaluation state
                                      (deterministic from the states; 64 permutations, seed 13)
  runs/pilot/eval_emb.npz             gitignored: MiniLM embeddings (cosine guard, domain centroids) with each row's source
"""
from __future__ import annotations

import argparse
import collections
import time

import numpy as np

from common import DATA, MINHASH_PERM, MINHASH_SEED, RUNS, dump, file_sha256, minhash, norm_text, state_hashes

EMBEDDER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--threads", type=int, default=6, help="CPU threads (the machine is shared)")
    args = ap.parse_args()
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

    sigs = np.zeros((len(texts), MINHASH_PERM), dtype=np.uint32)
    for i, t in enumerate(texts):
        sigs[i] = minhash(t).hashvalues.astype(np.uint32)
        if i % 50000 == 0:
            print(f"  minhash {i}/{len(texts)} ({time.time() - t0:.0f} s)", flush=True)
    np.savez_compressed(RUNS / "eval_minhash.npz", sigs=sigs, src=np.array(srcs))

    if not args.no_embed:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(args.threads)
        dev = "cuda" if torch.cuda.is_available() and args.threads <= 0 else "cpu"
        m = SentenceTransformer(EMBEDDER, device=dev)
        emb = m.encode([t[:2000] for t in texts], batch_size=128, normalize_embeddings=True, show_progress_bar=False,
                       convert_to_numpy=True).astype(np.float16)
        np.savez_compressed(RUNS / "eval_emb.npz", emb=emb, src=np.array(srcs))
        del m

    dump(DATA / "eval_hashes.json", {
        "title": "Evaluation states for the small-model pilot's leakage guard (plan §6, step 1)",
        "hash": "first 16 hex characters (64 bits) of SHA-256; exact = tez.schema.state_key(state), "
                "norm = experiments/small_model/common.py norm_text(state) (leaves joined, NFKC, casefold, whitespace)",
        "minhash": {"shingles": "character 5-grams of the normalised text", "num_perm": MINHASH_PERM, "seed": MINHASH_SEED,
                    "file": "runs/pilot/eval_minhash.npz (gitignored; rebuilt by this script)",
                    "sha256": file_sha256(RUNS / "eval_minhash.npz"), "rows": len(texts)},
        "embedder": EMBEDDER,
        "counts": {k: len(v["exact"]) for k, v in sorted(by_source.items())},
        "sources": {k: v for k, v in sorted(by_source.items())},
    })
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
