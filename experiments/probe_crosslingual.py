"""Zero-shot cross-lingual transfer of a hidden-state probe (MASSIVE intent, 11 languages).

Train ONE logistic probe on English inputs only (MASSIVE en train, 2,000 rows, all 60 intents),
then apply it unchanged to the head-to-head test rows of 11 languages (first 100 test rows each,
Laya's protocol). Instructions and the intent list stay in English; only the utterance changes
language, exactly as in bench_h2h. Reported per layer and language:
  * 60-way accuracy over the full intent list;
  * accuracy restricted to each row's 20 candidate intents (directly comparable with the
    zero-shot 12B letter readout and Laya's checkpoints on the same rows).

  py experiments/probe_crosslingual.py --model Qwen/Qwen3.5-4B --tag qwen35-4b --out results/probe_crosslingual_qwen35-4b.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_s = importlib.util.spec_from_file_location("bench_h2h", ROOT / "experiments" / "bench_h2h.py")
bh = importlib.util.module_from_spec(_s); sys.modules["bench_h2h"] = bh; _s.loader.exec_module(bh)  # type: ignore[union-attr]

HEAD = ("You are a decision engine. Read the question and the options, then look at the input and answer "
        "with the number of the best option. Answer with the number only.")
INSTR = "What is the user asking for in `utterance`?"


def prompt(labels, utterance):
    opts = "Options:\n" + "\n".join(f"{i + 1}. {k}: {bh.humanise(k)}" for i, k in enumerate(labels))
    st = json.dumps({"utterance": utterance}, ensure_ascii=False)
    return f"{HEAD}\n\nQuestion (choice): {INSTR}\n\n{opts}\n\nInput:\n{st}"


@torch.inference_mode()
def extract(model, tok, texts, tag):
    H = None; t0 = time.perf_counter()
    for i, t in enumerate(texts):
        msgs = [{"role": "user", "content": t}]
        try:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(s, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        o = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        hs = torch.stack([h[0, -1] for h in o.hidden_states]).to(torch.float16).cpu().numpy()
        if H is None:
            H = np.zeros((len(texts),) + hs.shape, dtype=np.float16)
        H[i] = hs
        if (i + 1) % 250 == 0:
            print(f"{tag} {i + 1}/{len(texts)} {time.perf_counter() - t0:.0f}s", flush=True)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tag", default="qwen35-4b")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression
    d = load_dataset("mteb/amazon_massive_intent", "en", split="train")
    labels = sorted(set(d["label_text"]))
    rng = random.Random(7); idx = rng.sample(range(len(d)), args.n_train)
    tr_text = [d[i]["text"] for i in idx]; tr_y = np.array([labels.index(d[i]["label_text"]) for i in idx])
    test = {}
    for lang in bh.MASSIVE_LANGS:
        cases = bh.build_task(f"massive:{lang}", 100)
        test[lang] = dict(text=[c["state"]["utterance"] for c in cases],
                          y=np.array([labels.index(c["options"][c["gold"]][0]) for c in cases]),
                          cand=[[labels.index(k) for k, _ in c["options"]] for c in cases])
    cache = ROOT / "results" / f"probe_cache_crosslingual_{args.tag}.npz"
    if cache.exists():
        F = dict(np.load(cache))
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
        F = {"H_train": extract(model, tok, [prompt(labels, t) for t in tr_text], "train-en")}
        for lang in bh.MASSIVE_LANGS:
            F[f"H_{lang}"] = extract(model, tok, [prompt(labels, t) for t in test[lang]["text"]], lang)
        np.savez(cache, **F)
        del model; torch.cuda.empty_cache()
    nL = F["H_train"].shape[1]
    res = {"model": args.model, "n_train": args.n_train, "labels": len(labels), "layers": {}}
    for L in list(range(12, nL, 2)) + [nL - 1]:
        clf = LogisticRegression(max_iter=4000, C=0.5).fit(F["H_train"][:, L].astype(np.float32), tr_y)
        cls = list(clf.classes_); R = {}
        for lang in bh.MASSIVE_LANGS:
            P = clf.predict_proba(F[f"H_{lang}"][:, L].astype(np.float32))
            full = np.zeros((len(P), len(labels))); full[:, cls] = P
            y = test[lang]["y"]
            a60 = float((full.argmax(1) == y).mean())
            a20 = float(np.mean([c[int(np.argmax(full[j, c]))] == y[j] for j, c in enumerate(test[lang]["cand"])]))
            R[lang] = dict(acc60=a60, acc20=a20)
        R["macro_acc20"] = float(np.mean([R[l]["acc20"] for l in bh.MASSIVE_LANGS])); R["macro_acc60"] = float(np.mean([R[l]["acc60"] for l in bh.MASSIVE_LANGS]))
        res["layers"][str(L)] = R
        print(f"L{L}: macro 20-way {R['macro_acc20']:.3f} 60-way {R['macro_acc60']:.3f} | " + " ".join(f"{l}:{R[l]['acc20']:.2f}" for l in bh.MASSIVE_LANGS), flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
