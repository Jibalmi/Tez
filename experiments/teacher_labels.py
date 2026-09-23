"""Zero-shot letter readout of the serving model (Gemma 4 12B Q8 via llama-server) on the typed-decisions
TRAIN and TEST splits, in the same row order as the hidden-state cache (hidden_probe.load_split).

The train-split answers are free pseudo-labels: they let a probe be trained with no human labels
(weak-to-strong) and give the online cold-start loop a stronger zero-shot starting point.
Resumable: rows already in the .jsonl are skipped.

  py experiments/teacher_labels.py --out results/teacher_labels_gemma4-12b-q8_0.npz
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


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


h2h = _load("bench_h2h", "bench_h2h.py")
hp = _load("hp", "hidden_probe.py")
SERVER = os.environ.get("TEZ_SERVER", "http://127.0.0.1:8091")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); log = out.with_suffix(".jsonl")
    done = {}
    if log.exists():
        for l in log.read_text(encoding="utf-8").splitlines():
            r = json.loads(l); done[(r["split"], r["i"])] = r
    f = log.open("a", encoding="utf-8")
    res = {}
    for split in ("train", "test"):
        cases = hp.load_split(split)
        P = np.zeros((len(cases), 26), dtype=np.float32); ms = []
        t0 = time.perf_counter()
        for i, c in enumerate(cases):
            if (split, i) in done:
                p = np.array(done[(split, i)]["p"]); P[i, :len(p)] = p; continue
            k = len(c["options"])
            t1 = time.perf_counter()
            p, z, pn = h2h.tez_score_letters(SERVER, h2h.tez_prompt(c, c["options"]), k)
            ms.append((time.perf_counter() - t1) * 1000)
            P[i, :k] = p
            f.write(json.dumps({"split": split, "i": i, "id": c["id"], "p": [float(x) for x in p]}) + "\n"); f.flush()
            if (i + 1) % 500 == 0:
                acc = np.mean([int(np.argmax(P[j, :len(cases[j]["options"])])) == cases[j]["gold"] for j in range(i + 1)])
                print(f"{split} {i + 1}/{len(cases)} acc so far {acc:.3f} {time.perf_counter() - t0:.0f}s", flush=True)
        g = np.array([c["gold"] for c in cases]); pred = np.array([int(np.argmax(P[j, :len(cases[j]['options'])])) for j in range(len(cases))])
        res[split] = dict(n=len(cases), acc=float((pred == g).mean()), ms_p50=float(np.median(ms)) if ms else None)
        np.save(out.with_name(out.stem + f".{split}.npy"), P)
        print(split, res[split], flush=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", out.with_suffix(".json"))


if __name__ == "__main__":
    main()
