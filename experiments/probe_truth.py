"""A universal "is this answer right?" probe, tested on workflows it has never seen.

For every typed-decisions decision and every option j, the frozen Qwen3.5-4B reads the standard
prompt and is then made to answer with option j's letter (teacher forcing, one extra token on the
cached prefix). The hidden state AT that answer token is kept. One logistic probe is trained across
questions to say whether the forced answer is the gold one; a decision is the option with the
highest P(correct). Evaluated leave-one-workflow-out, so the held-out workflow's questions, options
and wording are never seen in training. Variants: raw states, and states centred across the
options of the same decision (removes the shared context, keeps what differs between answers).

  py experiments/probe_truth.py --out results/probe_truth_qwen35-4b.json
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("hp", ROOT / "experiments" / "hidden_probe.py")
hp = importlib.util.module_from_spec(_s); _s.loader.exec_module(hp)  # type: ignore[union-attr]
LAYERS = [12, 16, 20, 24, 26, 28, 30, 32]


def chat(tok, content):
    msgs = [{"role": "user", "content": content}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def extract(model, tok, cases, letter_ids, tag):
    feats = np.zeros((len(cases), 5, len(LAYERS), model.config.get_text_config().hidden_size), dtype=np.float16)
    t0 = time.perf_counter()
    for i, c in enumerate(cases):
        ids = tok(chat(tok, hp.prompt_text(c)), return_tensors="pt", add_special_tokens=False).input_ids[:, -6000:].to("cuda")
        o = model(input_ids=ids, use_cache=True)
        for j in range(len(c["options"])):
            pkv = copy.deepcopy(o.past_key_values)
            oj = model(input_ids=torch.tensor([[letter_ids[j]]], device="cuda"), past_key_values=pkv, use_cache=True, output_hidden_states=True)
            feats[i, j] = torch.stack([oj.hidden_states[L][0, -1] for L in LAYERS]).to(torch.float16).cpu().numpy()
        if (i + 1) % 500 == 0:
            print(f"{tag} {i + 1}/{len(cases)} {time.perf_counter() - t0:.0f}s", flush=True)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tag", default="qwen35-4b")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    train, test = hp.load_split("train"), hp.load_split("test")
    cache = ROOT / "results" / f"probe_cache_truth_{args.tag}.npz"
    if cache.exists():
        F = dict(np.load(cache))
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
        letter_ids = [tok.encode(L, add_special_tokens=False)[0] for L in hp.LETTERS]
        F = {"train": extract(model, tok, train, letter_ids, "train"), "test": extract(model, tok, test, letter_ids, "test")}
        np.savez(cache, **F)
        del model; torch.cuda.empty_cache()
    wf_tr = np.array([c["qkey"][0] for c in train]); wf_te = np.array([c["qkey"][0] for c in test])
    ktr = np.array([len(c["options"]) for c in train]); kte = np.array([len(c["options"]) for c in test])
    gtr = np.array([c["gold"] for c in train]); gte = np.array([c["gold"] for c in test])
    wfs = sorted(set(wf_tr))

    def flat(H, k, g, rows, li, centre):
        X, y, owner = [], [], []
        for i in rows:
            h = H[i, :k[i], li].astype(np.float32)
            if centre:
                h = h - h.mean(0, keepdims=True)
            X.append(h); y += [int(j == g[i]) for j in range(k[i])]; owner += [i] * k[i]
        return np.concatenate(X), np.array(y), np.array(owner)

    def decide(clf, H, k, g, rows, li, centre):
        ok = 0
        for i in rows:
            h = H[i, :k[i], li].astype(np.float32)
            if centre:
                h = h - h.mean(0, keepdims=True)
            ok += int(np.argmax(clf.decision_function(h)) == g[i])
        return ok / len(rows)

    res = {"layers": LAYERS, "lowo": {}, "in_distribution": {}}
    for li, L in enumerate(LAYERS):
        for centre in (False, True):
            key = f"L{L}{'_centred' if centre else ''}"
            X, y, _ = flat(F["train"], ktr, gtr, np.arange(len(train)), li, centre)
            clf = LogisticRegression(max_iter=3000, C=0.05).fit(X, y)
            res["in_distribution"][key] = decide(clf, F["test"], kte, gte, np.arange(len(test)), li, centre)
            R = {}
            for w in wfs:
                X, y, _ = flat(F["train"], ktr, gtr, np.where(wf_tr != w)[0], li, centre)
                clf = LogisticRegression(max_iter=3000, C=0.05).fit(X, y)
                R[w] = decide(clf, F["test"], kte, gte, np.where(wf_te == w)[0], li, centre)
            R["mean"] = float(np.mean([R[w] for w in wfs])); res["lowo"][key] = R
            print(f"{key:14s} in-distribution {res['in_distribution'][key]:.3f} | unseen workflow mean {R['mean']:.3f} | " + " ".join(f"{w[:10]}:{R[w]:.2f}" for w in wfs), flush=True)
            Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
