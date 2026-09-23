"""Depth-pruned Qwen3.5-4B GGUFs served by llama.cpp: speed and accuracy of the two readouts.

Each GGUF keeps the first N blocks plus the output norm and head (experiments/gguf_truncate.py), so
  * the served letter logits are the logit lens at layer N (zero-shot readout), and
  * llama-server `--embeddings --pooling last` returns the output-normed layer-N state of the last
    token, which a per-question logistic probe reads (trained on the train split, as everywhere).
Measured per cut on the typed-decisions test split: letter accuracy, probe accuracy, ms per request.

  py experiments/pruned_server_bench.py --cuts L20:tools/models/Qwen3.5-4B-Q8_0-L20.gguf,L32:tools/models/Qwen3.5-4B-Q8_0.gguf --out results/pruned_server_qwen35-4b.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["TEZ_TEMPLATE"] = "qwen3"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


h2h = _load("bench_h2h", "bench_h2h.py")
hp = _load("hp", "hidden_probe.py")
EXE = r"C:\temp\llamacpp\llama-server.exe"
S = requests.Session()


def wrap(c):
    return f"<|im_start|>user\n{hp.prompt_text(c)}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def start(path, port, log):
    p = subprocess.Popen([EXE, "-m", str(ROOT / path), "-ngl", os.environ.get("TEZ_NGL", "99"), "-c", "4096", "-b", "512", "--port", str(port), "--host", "127.0.0.1",
                          "-np", "1", "--no-webui", "--embeddings", "--pooling", "last"], cwd=r"C:\temp\llamacpp",
                         stdout=open(log, "w"), stderr=subprocess.STDOUT)
    for _ in range(240):
        try:
            if S.get(f"http://127.0.0.1:{port}/health", timeout=2).json().get("status") == "ok":
                return p
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    p.kill(); raise RuntimeError("server did not start")


def embed(server, prompt):
    d = S.post(f"{server}/embedding", json={"content": prompt, "embd_normalize": -1}, timeout=600).json()
    if isinstance(d, list):
        d = d[0]
    e = d["embedding"]
    return np.asarray(e[-1] if isinstance(e[0], list) else e, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", required=True)
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    train, test = hp.load_split("train"), hp.load_split("test")
    gtr = np.array([c["gold"] for c in train]); gte = np.array([c["gold"] for c in test])
    qkeys = sorted(set(c["qkey"] for c in train))
    itr = {q: np.array([i for i, c in enumerate(train) if c["qkey"] == q]) for q in qkeys}
    ite = {q: np.array([i for i, c in enumerate(test) if c["qkey"] == q]) for q in qkeys}
    res = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    server = f"http://127.0.0.1:{args.port}"
    for item in args.cuts.split(","):
        name, path = item.split(":", 1)
        if name in res:
            continue
        log = Path(os.environ.get("TEMP", ".")) / f"pruned_{name}.log"
        proc = start(path, args.port, log)
        try:
            # zero-shot letters on the test split (prefix caching on, as Tez runs)
            ok, ms_l = 0, []
            for c in test:
                t0 = time.perf_counter(); p, z, pn = h2h.tez_score_letters(server, wrap(c), len(c["options"])); ms_l.append((time.perf_counter() - t0) * 1000)
                ok += int(int(np.argmax(p)) == c["gold"])
            letters = ok / len(test)
            # served embeddings for train + test, probe per question
            feats = {}
            ms_e = []
            for split, cases in (("train", train), ("test", test)):
                X = []
                for c in cases:
                    t0 = time.perf_counter(); X.append(embed(server, wrap(c))); ms_e.append((time.perf_counter() - t0) * 1000)
                feats[split] = np.stack(X)
            np.savez(ROOT / "results" / f"probe_cache_served_{name}.npz", **feats)
        finally:
            proc.kill(); time.sleep(3)
        Xtr, Xte = feats["train"], feats["test"]
        best = None
        for C in (0.05, 0.5, 5.0):   # choose C by 3-fold CV on the TRAIN split only
            sc = np.mean([cross_val_score(LogisticRegression(max_iter=3000, C=C), Xtr[itr[q]], gtr[itr[q]], cv=3).mean() for q in qkeys if len(np.unique(gtr[itr[q]])) > 1])
            if best is None or sc > best[0]:
                best = (sc, C)
        okp = 0
        for q in qkeys:
            y = gtr[itr[q]]
            if len(np.unique(y)) < 2:
                okp += int((gte[ite[q]] == y[0]).sum()); continue
            okp += int((LogisticRegression(max_iter=3000, C=best[1]).fit(Xtr[itr[q]], y).predict(Xte[ite[q]]) == gte[ite[q]]).sum())
        res[name] = dict(gguf=path, letters_acc=letters, probe_acc=okp / len(test), probe_C=best[1],
                         ms_letters_p50=float(np.median(ms_l)), ms_embed_p50=float(np.median(ms_e)), dim=int(Xtr.shape[1]))
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[name].items()}, flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
