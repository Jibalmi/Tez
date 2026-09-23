"""Select-and-copy attention readout (typed-decisions, frozen Qwen3.5-4B).

Some attention heads, at the answer position, look straight at the option that will be chosen
(Tulchinskii et al. 2024, arXiv:2410.02343; Wiegreffe et al. 2024, arXiv:2407.15018). Reading the
decision from those heads' attention weights, instead of from the letter logits, needs no training
beyond choosing the heads. Qwen3.5 has softmax attention only in every fourth layer (3, 7, ..., 31);
those are the layers read here.

For every decision: attention of the last prompt token to each option's letter token and to the last
token of each option line, per head. Heads are ranked by train-split accuracy (argmax over options);
the readout sums the top-h heads. Also leave-one-workflow-out: heads chosen on three workflows,
tested on the fourth, i.e. a readout for a workflow it has never seen.

  py experiments/attn_readout.py --out results/attn_readout_qwen35-4b.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("hp", ROOT / "experiments" / "hidden_probe.py")
hp = importlib.util.module_from_spec(_s); _s.loader.exec_module(hp)  # type: ignore[union-attr]


def chat(tok, content):
    msgs = [{"role": "user", "content": content}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def extract(model, tok, cases, tag):
    out_letter, out_end = [], []
    t0 = time.perf_counter()
    for i, c in enumerate(cases):
        s = chat(tok, hp.prompt_text(c))
        enc = tok(s, return_offsets_mapping=True, add_special_tokens=False)
        offs = enc["offset_mapping"]; ids = torch.tensor([enc["input_ids"]], device=model.device)
        opt_start = s.index("Options:\n") + len("Options:\n")
        pos_letter, pos_end = [], []
        cur = opt_start
        for j, (k, d) in enumerate(c["options"]):
            line = f"{hp.LETTERS[j]}. {k}: {d}"
            a = s.index(line, cur); b = a + len(line); cur = b
            pos_letter.append(min(t for t, (x, y) in enumerate(offs) if y > a))          # token holding the letter
            pos_end.append(max(t for t, (x, y) in enumerate(offs) if x < b))              # last token of the line
        o = model(input_ids=ids, output_attentions=True, use_cache=False)
        att = [a_ for a_ in o.attentions if a_ is not None]                              # full-attention layers only
        A = torch.stack([a_[0, :, -1, :] for a_ in att]).float()                          # [n_layers, heads, T]
        out_letter.append(A[:, :, pos_letter].cpu().numpy()); out_end.append(A[:, :, pos_end].cpu().numpy())
        if (i + 1) % 500 == 0:
            print(f"{tag} {i + 1}/{len(cases)} {time.perf_counter() - t0:.0f}s", flush=True)
    return out_letter, out_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N cases per split")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=args.device, attn_implementation="eager").eval()
    train, test = hp.load_split("train"), hp.load_split("test")
    if args.limit:
        train, test = train[: args.limit], test[: args.limit]
    Ltr, Etr = extract(model, tok, train, "train"); Lte, Ete = extract(model, tok, test, "test")
    del model; torch.cuda.empty_cache()
    nlay, nh = Ltr[0].shape[:2]
    gtr = np.array([c["gold"] for c in train]); gte = np.array([c["gold"] for c in test])
    wtr = np.array([c["qkey"][0] for c in train]); wte = np.array([c["qkey"][0] for c in test])

    def head_acc(S, g, rows):
        acc = np.zeros((nlay, nh))
        for i in rows:
            acc += (np.argmax(S[i], -1) == g[i])
        return acc / max(1, len(rows))

    def readout(S, g, rows, heads):
        ok = 0
        for i in rows:
            sc = sum(S[i][l, h] for l, h in heads); ok += int(np.argmax(sc) == g[i])
        return ok / max(1, len(rows))

    res = {"layers_with_attention": int(nlay), "heads_per_layer": int(nh), "readouts": {}}
    for name, (Str, Ste) in (("attention to the option letter", (Ltr, Lte)), ("attention to the end of the option line", (Etr, Ete))):
        acc = head_acc(Str, gtr, range(len(train))); order = np.dstack(np.unravel_index(np.argsort(-acc.ravel()), acc.shape))[0]
        R = {"best_single_head_train": float(acc.max()), "top_heads": [[int(l), int(h), float(acc[l, h])] for l, h in order[:10]]}
        for h in (1, 3, 5, 10, 20):
            R[f"top{h}_test"] = readout(Ste, gte, range(len(test)), [tuple(x) for x in order[:h]])
        lowo = {}
        for w in sorted(set(wtr)):
            a_w = head_acc(Str, gtr, np.where(wtr != w)[0]); o_w = np.dstack(np.unravel_index(np.argsort(-a_w.ravel()), a_w.shape))[0]
            lowo[w] = {f"top{h}": readout(Ste, gte, np.where(wte == w)[0], [tuple(x) for x in o_w[:h]]) for h in (1, 3, 5, 10)}
        R["unseen_workflow"] = lowo
        R["unseen_workflow_mean"] = {f"top{h}": float(np.mean([lowo[w][f"top{h}"] for w in lowo])) for h in (1, 3, 5, 10)}
        res["readouts"][name] = R
        print(name, {k: v for k, v in R.items() if k.startswith("top") and k.endswith("test")}, "| unseen workflow", R["unseen_workflow_mean"], flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
