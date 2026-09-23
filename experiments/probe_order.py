"""Decide-then-bind, measured across depth: does a probe survive reordering the options?

Probes are trained on the standard prompts (options in their usual order) and predict the chosen
OPTION (its identity, not its letter). At test time the options are presented in reverse order, so
every letter points at a different option. If a layer encodes the decision as content ("approve"),
the probe still finds the right option; if it encodes the bound symbol ("letter B"), it fails.
Wong et al. (arXiv:2601.03914) predict content first, symbol late; this measures where the switch is.

Cost: one pass over the 2,000 test prompts with reversed options (all layers cached); the probes and
the standard-order features come from results/probe_cache_<tag>.npz.

  py experiments/probe_order.py --out results/probe_order_qwen35-4b.json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tag", default="qwen35-4b")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    train, test = hp.load_split("train"), hp.load_split("test")
    cache = ROOT / "results" / f"probe_cache_reversed_{args.tag}.npz"
    if cache.exists():
        R = dict(np.load(cache))
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
        slot = [tok.encode(L, add_special_tokens=False)[0] for L in hp.LETTERS]
        H = None; Z = np.full((len(test), 26), -1e4, dtype=np.float32); t0 = time.perf_counter()
        with torch.inference_mode():
            for i, c in enumerate(test):
                rc = dict(c); rc["options"] = list(reversed(c["options"]))
                ids = tok(chat(tok, hp.prompt_text(rc)), return_tensors="pt", add_special_tokens=False).input_ids[:, -6000:].to("cuda")
                o = model(input_ids=ids, output_hidden_states=True, use_cache=False)
                hs = torch.stack([h[0, -1] for h in o.hidden_states]).to(torch.float16).cpu().numpy()
                if H is None:
                    H = np.zeros((len(test),) + hs.shape, dtype=np.float16)
                H[i] = hs; k = len(c["options"]); Z[i, :k] = o.logits[0, -1, slot[:k]].float().cpu().numpy()
                if (i + 1) % 500 == 0:
                    print(f"reversed {i + 1}/{len(test)} {time.perf_counter() - t0:.0f}s", flush=True)
        R = {"H_test_rev": H, "Z_test_rev": Z}
        np.savez(cache, **R)
        del model; torch.cuda.empty_cache()
    B = np.load(ROOT / "results" / f"probe_cache_{args.tag}.npz")
    gtr = np.array([c["gold"] for c in train]); gte = np.array([c["gold"] for c in test]); k = np.array([len(c["options"]) for c in test])
    qk = sorted(set(c["qkey"] for c in train))
    itr = {q: np.array([i for i, c in enumerate(train) if c["qkey"] == q]) for q in qk}
    ite = {q: np.array([i for i, c in enumerate(test) if c["qkey"] == q]) for q in qk}
    # letters on reversed prompts: letter j now points at original option k-1-j
    zr = R["Z_test_rev"]; pred_rev_letters = np.array([k[i] - 1 - int(np.argmax(zr[i, :k[i]])) for i in range(len(test))])
    zo = B["Z_test"]; pred_letters = np.array([int(np.argmax(zo[i, :k[i]])) for i in range(len(test))])
    res = {"letters": {"standard_order_acc": float((pred_letters == gte).mean()), "reversed_order_acc": float((pred_rev_letters == gte).mean()),
                       "same_option_chosen": float((pred_letters == pred_rev_letters).mean())}, "probe": {}}
    print("letters", res["letters"], flush=True)
    nL = B["H_train"].shape[1]
    for L in list(range(8, nL, 2)) + [nL - 1]:
        Xtr = B["H_train"][:, L].astype(np.float32); Xs = B["H_test"][:, L].astype(np.float32); Xr = R["H_test_rev"][:, L].astype(np.float32)
        ps = np.zeros(len(test), int); pr = np.zeros(len(test), int); pr_as_letter = np.zeros(len(test), int)
        for q in qk:
            y = gtr[itr[q]]
            if len(np.unique(y)) < 2:
                ps[ite[q]] = pr[ite[q]] = y[0]; pr_as_letter[ite[q]] = y[0]; continue
            clf = LogisticRegression(max_iter=3000, C=0.5).fit(Xtr[itr[q]], y)
            ps[ite[q]] = clf.predict(Xs[ite[q]]); pr[ite[q]] = clf.predict(Xr[ite[q]])
        # if the layer carried the LETTER, the reversed prompt would map to the mirrored option
        pr_as_letter = np.array([k[i] - 1 - pr[i] for i in range(len(test))])
        res["probe"][str(L)] = dict(standard=float((ps == gte).mean()), reversed_read_as_content=float((pr == gte).mean()),
                                    reversed_read_as_letter=float((pr_as_letter == gte).mean()))
        r = res["probe"][str(L)]
        print(f"L{L:2d}: standard order {r['standard']:.3f} | reversed, probe unchanged {r['reversed_read_as_content']:.3f} | (if it read letters: {r['reversed_read_as_letter']:.3f})", flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
