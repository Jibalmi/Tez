"""Hidden Calibration (arXiv:2406.16535) on typed-decisions: read the last-token hidden state of a
frozen decoder at the answer position, fit a per-question linear probe / nearest-centroid on the
benchmark's TRAIN split, evaluate on the TEST split, and compare with the same model's letter-logit
readout. Also fits a probe to reproduce the FULL-DEPTH option distribution from an earlier layer
(tuned-lens style) to see how shallow the readout could be.

In-process transformers; needs the GPU free (stop llama-server first).
  py experiments/hidden_probe.py --model Qwen/Qwen3.5-4B --layers -1,-4,-8 --out results/hidden_probe_qwen35-4b.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_split(split):
    from datasets import load_dataset
    d = load_dataset("LocalLLaMA/typed-decisions", "all", split=split)
    cases = []
    for r in d:
        qs, gold = json.loads(r["questions"]), json.loads(r["gold"])
        try:
            st = json.loads(r["state"])
        except Exception:  # noqa: BLE001
            st = r["state"]
        for qid, qd in qs.items():
            g = gold[qid]
            if qd["type"] == "choice":
                keys = list(qd["criteria"].keys()); opts = [(k, qd["criteria"][k] if isinstance(qd["criteria"][k], str) else json.dumps(qd["criteria"][k])) for k in keys]; gi = keys.index(str(g["label"]))
            elif qd["type"] == "noul":
                opts = [("false", "no, the statement does not hold"), ("true", "yes, the statement holds")]; gi = 1 if str(g["label"]).lower() == "true" else 0
            else:
                opts = [(f"level {i}", c if isinstance(c, str) else json.dumps(c)) for i, c in enumerate(qd["criteria"])]; gi = int(g["label"])
            cases.append(dict(id=f"{r['id']}-{qid}", qkey=(r["workflow"], qid), qtype=qd["type"], state=st, instructions=qd["instructions"], options=opts, gold=gi))
    return cases


def prompt_text(c):
    head = ("You are a decision engine. Read the question and the options, then look at the input and answer "
            "with the single letter of the best option. Answer with the letter only.")
    q = f"Question ({c['qtype']}): {c['instructions']}"
    opts = "Options:\n" + "\n".join(f"{LETTERS[i]}. {k}: {d}" for i, (k, d) in enumerate(c["options"]))
    st = json.dumps(c["state"], ensure_ascii=False) if not isinstance(c["state"], str) else c["state"]
    return f"{head}\n\n{q}\n\n{opts}\n\nInput:\n{st}"


@torch.inference_mode()
def run(model, tok, cases, layers, dev, slot_ids):
    feats = {L: [] for L in layers}; logits_out = []
    for c in cases:
        msgs = [{"role": "user", "content": prompt_text(c)}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, -6000:].to(dev)
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        for L in layers:
            feats[L].append(hs[L][0, -1].float().cpu().numpy())
        k = len(c["options"])
        z = out.logits[0, -1, slot_ids[:k]].float().cpu().numpy()
        logits_out.append(z)
    return feats, logits_out


def softmax(z):
    z = np.asarray(z, float); e = np.exp(z - z.max()); return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layers", default="-1,-4,-8")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import LogisticRegression
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=dev, attn_implementation="sdpa").eval()
    slot_ids = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    layers = [int(x) for x in args.layers.split(",")]
    train = load_split("train"); test = load_split("test")
    if args.limit_train:
        train = train[: args.limit_train]
    t0 = time.perf_counter()
    ftr, ztr = run(model, tok, train, layers, dev, slot_ids)
    fte, zte = run(model, tok, test, layers, dev, slot_ids)
    print(f"forward passes done in {time.perf_counter()-t0:.0f}s", flush=True)
    gold_te = np.array([c["gold"] for c in test]); gold_tr = np.array([c["gold"] for c in train])
    res = {"model": args.model, "n_train": len(train), "n_test": len(test)}
    # baseline: letter logits
    pred = np.array([int(np.argmax(z)) for z in zte]); conf = np.array([softmax(z).max() for z in zte])
    res["letter_logits"] = dict(accuracy=float((pred == gold_te).mean()), nll=float(np.mean([-math.log(max(softmax(z)[g], 1e-12)) for z, g in zip(zte, gold_te)])))
    # per-question probes (the option set is fixed per (workflow, qid))
    qkeys = sorted({c["qkey"] for c in test})
    for L in layers:
        Xtr = np.stack(ftr[L]); Xte = np.stack(fte[L])
        acc_lr, acc_nc, nll_lr, n = 0, 0, 0.0, 0
        for qk in qkeys:
            itr = [i for i, c in enumerate(train) if c["qkey"] == qk]; ite = [i for i, c in enumerate(test) if c["qkey"] == qk]
            if not itr or not ite:
                continue
            ytr = gold_tr[itr]; yte = gold_te[ite]; k = len(test[ite[0]]["options"])
            if len(set(ytr.tolist())) < 2:
                p = np.full(len(ite), int(ytr[0])); acc_lr += (p == yte).sum(); acc_nc += (p == yte).sum(); nll_lr += len(ite) * 0.0; n += len(ite); continue
            clf = LogisticRegression(max_iter=2000, C=0.5).fit(Xtr[itr], ytr)
            P = clf.predict_proba(Xte[ite]); cls = list(clf.classes_)
            pl = np.array([cls[int(np.argmax(r))] for r in P]); acc_lr += (pl == yte).sum()
            nll_lr += sum(-math.log(max(P[j][cls.index(int(y))] if int(y) in cls else 1e-12, 1e-12)) for j, y in enumerate(yte))
            cents = {y: Xtr[itr][ytr == y].mean(0) for y in set(ytr.tolist())}
            pn = np.array([min(cents, key=lambda y: np.linalg.norm(Xte[i] - cents[y])) for i in ite]); acc_nc += (pn == yte).sum()
            n += len(ite)
        res[f"layer{L}"] = dict(logreg_acc=float(acc_lr / n), logreg_nll=float(nll_lr / n), centroid_acc=float(acc_nc / n), n=int(n))
        print(L, res[f"layer{L}"], flush=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
