"""Hidden-state probe through llama-server's embedding endpoint, so it works on the 12B Q8 GGUF that
does not fit in-process. Start the server with `--embeddings --pooling last` (any -c that fits the
prompts); `/embedding` then returns the final-layer hidden state of the LAST token for a causal
model. Fit one logistic-regression probe per question schema on the typed-decisions train split,
evaluate on the test split, and compare with the letter-logit readout of the same model.

  py experiments/hidden_probe_server.py --out results/hidden_probe_server_gemma4-12b-q8_0.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hp", ROOT / "experiments" / "hidden_probe.py")
hp = importlib.util.module_from_spec(spec); spec.loader.exec_module(hp)  # type: ignore[union-attr]
TAIL = "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
SESSION = requests.Session()


def embed(server, prompt):
    d = SESSION.post(f"{server}/embedding", json={"content": prompt}, timeout=600).json()
    e = d[0]["embedding"] if isinstance(d, list) else d["embedding"]
    if isinstance(e[0], list):
        e = e[0]
    return np.asarray(e, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8091")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-train", type=int, default=0)
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    train = hp.load_split("train"); test = hp.load_split("test")
    if args.limit_train:
        train = train[: args.limit_train]
    t0 = time.perf_counter()
    cache = Path(args.out).with_suffix(".emb")

    def embed_all(cases, tag):
        ck = Path(str(cache) + f".{tag}.npy")
        rows = list(np.load(ck)) if ck.exists() else []
        for i in range(len(rows), len(cases)):
            rows.append(embed(args.server, f"<|turn>user\n{hp.prompt_text(cases[i])}{TAIL}"))
            if (i + 1) % 250 == 0 or i + 1 == len(cases):
                np.save(ck, np.stack(rows)); print(f"{tag}: {i+1}/{len(cases)} embedded, {time.perf_counter()-t0:.0f}s", flush=True)
        return np.stack(rows)

    Xtr = embed_all(train, "train"); Xte = embed_all(test, "test")
    print(f"embeddings done in {time.perf_counter()-t0:.0f}s, dim {Xtr.shape[1]}", flush=True)
    gtr = np.array([c["gold"] for c in train]); gte = np.array([c["gold"] for c in test])
    acc_lr = acc_nc = 0; nll = 0.0; n = 0
    per_q = {}
    for qk in sorted({c["qkey"] for c in test}):
        itr = [i for i, c in enumerate(train) if c["qkey"] == qk]; ite = [i for i, c in enumerate(test) if c["qkey"] == qk]
        if not itr or not ite:
            continue
        ytr, yte = gtr[itr], gte[ite]
        if len(set(ytr.tolist())) < 2:
            p = np.full(len(ite), int(ytr[0])); acc_lr += (p == yte).sum(); acc_nc += (p == yte).sum(); n += len(ite); continue
        clf = LogisticRegression(max_iter=3000, C=0.5).fit(Xtr[itr], ytr)
        P = clf.predict_proba(Xte[ite]); cls = list(clf.classes_)
        pl = np.array([cls[int(np.argmax(r))] for r in P]); ok = (pl == yte).sum(); acc_lr += ok
        nll += sum(-math.log(max(P[j][cls.index(int(y))] if int(y) in cls else 1e-12, 1e-12)) for j, y in enumerate(yte))
        cents = {y: Xtr[itr][ytr == y].mean(0) for y in set(ytr.tolist())}
        pn = np.array([min(cents, key=lambda y: np.linalg.norm(Xte[i] - cents[y])) for i in ite]); acc_nc += (pn == yte).sum()
        per_q["/".join(qk)] = dict(n=len(ite), logreg_acc=float(ok / len(ite)))
        n += len(ite)
    res = dict(n_train=len(train), n_test=n, dim=int(Xtr.shape[1]), logreg_acc=float(acc_lr / n), logreg_nll=float(nll / n), centroid_acc=float(acc_nc / n), per_question=per_q)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "per_question"}, indent=1))


if __name__ == "__main__":
    main()
