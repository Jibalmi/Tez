"""Retrieval-narrowed decisions for wide label spaces (Banking77, 77 intents).

The chunked tournament (bench_h2h.tez_decide) spends 5 forward passes per decision when k > 26.
Here a cheap embedder shortlists the top-k labels by cosine similarity to the input, and the frozen
LLM makes ONE letter decision among the shortlist. Measured on the same 400 Banking77 rows as the
head-to-head: shortlist recall@k, retrieval-only accuracy (argmax cosine), Tez-on-shortlist accuracy,
calls per decision, ms per decision. Embedders: MiniLM (sentence-transformers/all-MiniLM-L6-v2,
Apache-2.0, 22M params) or the serving model itself via llama-server /embedding (last-token pooling).

  py experiments/retrieval_narrow.py --embedder minilm --ks 5,10,15,20 --out results/retrieval_narrow_banking77_minilm.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("bench_h2h", ROOT / "experiments" / "bench_h2h.py")
h2h = importlib.util.module_from_spec(_spec); sys.modules["bench_h2h"] = h2h; _spec.loader.exec_module(h2h)  # type: ignore[union-attr]

SERVER = os.environ.get("TEZ_SERVER", "http://127.0.0.1:8091")


_MINILM = None


def embed_minilm(texts):
    global _MINILM
    from sentence_transformers import SentenceTransformer
    if _MINILM is None:   # load once, outside the timed region
        _MINILM = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return np.asarray(_MINILM.encode(texts, normalize_embeddings=True, batch_size=64))


def embed_server(texts):
    out = []
    for t in texts:
        d = h2h.SESSION.post(f"{SERVER}/embedding", json={"content": t}, timeout=600).json()
        v = np.asarray(d[0]["embedding"][0] if isinstance(d, list) else d["embedding"], dtype=float)
        out.append(v / (np.linalg.norm(v) + 1e-9))
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="banking77")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--embedder", default="minilm", choices=["minilm", "server"])
    ap.add_argument("--ks", default="5,10,15,20")
    ap.add_argument("--with-none", action="store_true", help="append 'none of these' and fall back to the full tournament when it wins")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cases = h2h.build_task(args.task, args.n)
    ref_path = ROOT / "results" / "h2h" / f"rows_{args.task}_tez.jsonl"
    ref = {}
    if ref_path.exists():
        for l in ref_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(l); ref[r["id"]] = r
        same = sum(c["id"] in ref for c in cases)
        print(f"reference rows matched: {same}/{len(cases)}", flush=True)
    options = cases[0]["options"]
    label_texts = [d for _, d in options]
    inputs = [c["state"]["text"] if "text" in c["state"] else json.dumps(c["state"], ensure_ascii=False) for c in cases]

    emb = embed_minilm if args.embedder == "minilm" else embed_server
    E_lab = emb(label_texts)                      # labels: embedded once per schema (model load happens here)
    t0 = time.perf_counter(); E_in = emb(inputs)  # inputs: the per-decision cost
    t_embed = (time.perf_counter() - t0) / len(cases) * 1000
    sims = E_in @ E_lab.T
    order = np.argsort(-sims, axis=1)
    gold = np.array([c["gold"] for c in cases])
    res = dict(task=args.task, n=len(cases), embedder=args.embedder, ms_embed_per_input=t_embed,
               retrieval_only_acc=float(np.mean(order[:, 0] == gold)),
               recall_at={str(k): float(np.mean([gold[i] in order[i, :k] for i in range(len(cases))])) for k in range(1, 27)},
               tournament_ref_acc=float(np.mean([int(np.argmax(ref[c["id"]]["probabilities"])) == c["gold"] for c in cases if c["id"] in ref])) if ref else None,
               tournament_ref_ms=float(np.median([ref[c["id"]]["ms"] for c in cases if c["id"] in ref])) if ref else None,
               runs={})
    print(f"retrieval-only acc {res['retrieval_only_acc']:.3f}; recall@5 {res['recall_at']['5']:.3f} @10 {res['recall_at']['10']:.3f} @20 {res['recall_at']['20']:.3f}; tournament ref {res['tournament_ref_acc']}", flush=True)

    for k in [int(x) for x in args.ks.split(",")]:
        rows, ms_all, calls_all = [], [], []
        for i, c in enumerate(cases):
            short = [int(j) for j in order[i, :k]]
            opts = [options[j] for j in short]
            t1 = time.perf_counter()
            if args.with_none:
                p, z, pn = h2h.tez_score_letters(SERVER, h2h.tez_prompt(c, opts + [("none", "none of the options above fits")]), k + 1); calls = 1
                if int(np.argmax(p)) == k:
                    pf, _, _, nc = h2h.tez_decide(SERVER, c); calls += nc; pred = int(np.argmax(pf)); conf = float(pf.max())
                else:
                    pred = short[int(np.argmax(p[:-1]))]; conf = float(p[:-1].max())
            else:
                p, z, pn = h2h.tez_score_letters(SERVER, h2h.tez_prompt(c, opts), k); calls = 1
                pred = short[int(np.argmax(p))]; conf = float(p.max())
            ms = (time.perf_counter() - t1) * 1000 + t_embed
            rows.append(dict(id=c["id"], gold=int(c["gold"]), pred=pred, conf=conf, in_shortlist=bool(c["gold"] in short), calls=calls, ms=ms))
            ms_all.append(ms); calls_all.append(calls)
            if (i + 1) % 100 == 0:
                print(f"  k={k} {i + 1}/{len(cases)} acc so far {np.mean([r['pred'] == r['gold'] for r in rows]):.3f}", flush=True)
        acc = float(np.mean([r["pred"] == r["gold"] for r in rows]))
        acc_in = float(np.mean([r["pred"] == r["gold"] for r in rows if r["in_shortlist"]])) if any(r["in_shortlist"] for r in rows) else None
        res["runs"][f"k{k}{'_none' if args.with_none else ''}"] = dict(k=k, with_none=args.with_none, acc=acc, acc_given_in_shortlist=acc_in,
                                                                        recall=res["recall_at"][str(k)], ms_p50=float(np.median(ms_all)),
                                                                        calls_mean=float(np.mean(calls_all)), rows=rows)
        print(f"k={k}: acc {acc:.3f} (recall {res['recall_at'][str(k)]:.3f}, acc|in {acc_in}), ms p50 {np.median(ms_all):.0f}, calls {np.mean(calls_all):.2f}", flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
