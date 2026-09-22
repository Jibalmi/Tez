"""Hidden-state probes on the public classification tasks (Laya's benchmarks), same rows as bench_h2h.

For each task: take up to --n-train rows from the dataset's TRAIN split (seeded), the SAME 400 test
rows bench_h2h used (seed 13), forward through the frozen model once with output_hidden_states,
keep the last-token state at a few layers, fit ONE logistic-regression probe per task, evaluate.
Compare with the frozen model's own letter readout on the same test rows.

  py experiments/hidden_probe_tasks.py --model Qwen/Qwen3.5-4B --tag qwen35-4b --tasks banking77,emotion,sst5,massive:en,xnli:en,boolq --layers 18,22,26,-1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bh", ROOT / "experiments" / "bench_h2h.py")
bh = importlib.util.module_from_spec(spec); spec.loader.exec_module(bh)  # type: ignore[union-attr]
LETTERS = bh.LETTERS


def train_cases(task, n, seed=7):
    """Build train rows with the same option sets / phrasing as bench_h2h.build_task (test)."""
    from datasets import load_dataset
    rng = random.Random(seed)
    if task == "banking77":
        d = load_dataset("banking77", split="train"); names = d.features["label"].names
        idx = rng.sample(range(len(d)), n)
        return [dict(id=f"tr-{i}", task=task, lang="en", state={"text": d[i]["text"]}, qtype="choice", instructions="Which banking intent is `text`?", options=[(k, bh.humanise(k)) for k in names], gold=d[i]["label"]) for i in idx]
    if task == "emotion":
        d = load_dataset("dair-ai/emotion", "split", split="train"); names = d.features["label"].names
        idx = rng.sample(range(len(d)), n)
        return [dict(id=f"tr-{i}", task=task, lang="en", state={"text": d[i]["text"]}, qtype="choice", instructions="What emotion does the writer of `text` express?", options=[(k, k) for k in names], gold=d[i]["label"]) for i in idx]
    if task == "sst5":
        d = load_dataset("SetFit/sst5", split="train"); levels = ["very negative", "negative", "neutral", "positive", "very positive"]
        idx = rng.sample(range(len(d)), n)
        return [dict(id=f"tr-{i}", task=task, lang="en", state={"text": d[i]["text"]}, qtype="score", instructions="How positive is the sentiment of `text`?", options=[(f"level {j}", lv) for j, lv in enumerate(levels)], gold=int(d[i]["label"])) for i in idx]
    if task == "boolq":
        d = load_dataset("boolq", split="train")
        idx = rng.sample(range(len(d)), n)
        return [dict(id=f"tr-{i}", task=task, lang="en", state={"passage": d[i]["passage"], "question": d[i]["question"]}, qtype="noul", instructions="Based on `passage`, is the answer to `question` yes?", options=[("false", "no, the answer is no"), ("true", "yes, the answer is yes")], gold=1 if d[i]["answer"] else 0) for i in idx]
    if task.startswith("xnli:"):
        lang = task.split(":")[1]; d = load_dataset("xnli", lang, split="train")
        idx = rng.sample(range(len(d)), n)
        return [dict(id=f"tr-{i}", task=task, lang=lang, state={"premise": d[i]["premise"], "hypothesis": d[i]["hypothesis"]}, qtype="choice", instructions="What is the relation of `hypothesis` to `premise`?",
                     options=[("entailment", "the hypothesis follows from the premise"), ("neutral", "the hypothesis may or may not be true given the premise"), ("contradiction", "the hypothesis contradicts the premise")], gold=d[i]["label"]) for i in idx]
    if task.startswith("massive:"):
        # MASSIVE test rows carry 20 sampled options each; for a task-level probe use the FULL 60-intent label space on both splits
        lang = task.split(":")[1]; d = load_dataset("mteb/amazon_massive_intent", lang, split="train"); labels = sorted(set(d["label_text"]))
        idx = rng.sample(range(len(d)), n)
        return [dict(id=f"tr-{i}", task=task, lang=lang, state={"utterance": d[i]["text"]}, qtype="choice", instructions="What is the user asking for in `utterance`?", options=[(k, bh.humanise(k)) for k in labels], gold=labels.index(d[i]["label_text"])) for i in idx], labels
    raise ValueError(task)


@torch.inference_mode()
def feats(model, tok, cases, layers, slot_ids):
    H = {L: [] for L in layers}; Z = []
    for c in cases:
        if len(c["options"]) <= 26:
            content = bh.tez_prompt(c, c["options"]).split("<|turn>user\n", 1)[1].rsplit(bh.TAIL, 1)[0]
        else:   # wider than the alphabet (Banking77, MASSIVE full space): number the options; the probe reads the state, not a symbol
            head = ("You are a decision engine. Read the question and the options, then look at the input and answer "
                    "with the number of the best option. Answer with the number only.")
            opts = "Options:\n" + "\n".join(f"{i + 1}. {k}: {d}" if d and d != k else f"{i + 1}. {k}" for i, (k, d) in enumerate(c["options"]))
            st = json.dumps(c["state"], ensure_ascii=False)
            content = f"{head}\n\nQuestion ({c['qtype']}): {c['instructions']}\n\n{opts}\n\nInput:\n{st}"
        msgs = [{"role": "user", "content": content}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, -4096:].to("cuda")
        o = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        for L in layers:
            H[L].append(o.hidden_states[L][0, -1].float().cpu().numpy())
        k = min(len(c["options"]), 26); Z.append(o.logits[0, -1, slot_ids[:k]].float().cpu().numpy())
    return {L: np.stack(v) for L, v in H.items()}, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tag", default="qwen35-4b")
    ap.add_argument("--tasks", default="banking77,emotion,sst5,massive:en,xnli:en,boolq")
    ap.add_argument("--layers", default="18,22,26,-1")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import LogisticRegression
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    slot_ids = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    layers = [int(x) for x in args.layers.split(",")]
    out_path = Path(args.out or f"results/hidden_probe_tasks_{args.tag}.json")
    res = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for task in args.tasks.split(","):
        if task in res:
            continue
        t0 = time.perf_counter()
        tr = train_cases(task, args.n_train)
        labels = None
        if isinstance(tr, tuple):
            tr, labels = tr
        te = bh.build_task(task, 400 if ":" not in task else 100)
        if labels is not None:   # massive: re-express the test rows over the full label space
            for c in te:
                gold_key = c["options"][c["gold"]][0]; c["options"] = [(k, bh.humanise(k)) for k in labels]; c["gold"] = labels.index(gold_key)
        Htr, Ztr = feats(model, tok, tr, layers, slot_ids); Hte, Zte = feats(model, tok, te, layers, slot_ids)
        gtr = np.array([c["gold"] for c in tr]); gte = np.array([c["gold"] for c in te])
        r = {"n_train": len(tr), "n_test": len(te), "n_options": len(te[0]["options"]), "seconds": time.perf_counter() - t0}
        # letter readout of the same model on the test rows (only meaningful when options <= 26)
        if len(te[0]["options"]) <= 26:
            pred = np.array([int(np.argmax(z)) for z in Zte]); r["letter_acc"] = float((pred == gte).mean())
        for L in layers:
            clf = LogisticRegression(max_iter=4000, C=0.5).fit(Htr[L], gtr)
            P = clf.predict_proba(Hte[L]); cls = list(clf.classes_); pl = np.array([cls[int(np.argmax(p))] for p in P])
            acc = float((pl == gte).mean()); conf = P.max(1); corr = (pl == gte).astype(float)
            r[f"probe_layer{L}"] = dict(acc=acc, ece=float(bh.ece15(conf, corr)), macro_f1=float(bh.macro_f1(gte, pl)))
        res[task] = r
        out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(task, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)}, {k: round(v["acc"], 3) for k, v in r.items() if isinstance(v, dict)}, flush=True)


if __name__ == "__main__":
    main()
