"""Hidden-state probe, phase 2: cache every layer's last-token state once, then sweep offline.

  1. Forward the typed-decisions train (6,000) and test (2,000) decisions through the frozen model
     ONCE, saving the last-token hidden state of EVERY layer (float16) to results/probe_cache_<tag>.npz
     plus the letter logits.
  2. Offline, per question schema: logistic-regression probe at every layer -> accuracy / NLL curve;
     data-efficiency curve at the best layer (train rows per question: 25, 50, 100, all);
     ensemble of probe probabilities with the letter-logit readout; conformal act/escalate on the probe.

  py experiments/hidden_probe_sweep.py --model Qwen/Qwen3.5-4B --tag qwen35-4b --out results/hidden_probe_sweep_qwen35-4b.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hp", ROOT / "experiments" / "hidden_probe.py")
hp = importlib.util.module_from_spec(spec); spec.loader.exec_module(hp)  # type: ignore[union-attr]


def softmax(z):
    z = np.asarray(z, float); e = np.exp(z - z.max()); return e / e.sum()


def extract(model_id, train, test, cache):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    slot_ids = [tok.encode(L, add_special_tokens=False)[0] for L in hp.LETTERS]
    out = {}
    for tag, cases in (("train", train), ("test", test)):
        H, Z = None, []
        t0 = time.perf_counter()
        with torch.inference_mode():
            for i, c in enumerate(cases):
                msgs = [{"role": "user", "content": hp.prompt_text(c)}]
                try:
                    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                except TypeError:
                    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, -6000:].to("cuda")
                o = model(input_ids=ids, output_hidden_states=True, use_cache=False)
                hs = torch.stack([h[0, -1] for h in o.hidden_states]).to(torch.float16).cpu().numpy()  # [L+1, d]
                if H is None:
                    H = np.zeros((len(cases),) + hs.shape, dtype=np.float16)
                H[i] = hs
                k = len(c["options"]); z = np.full(26, -1e4, dtype=np.float32); z[:k] = o.logits[0, -1, slot_ids[:k]].float().cpu().numpy(); Z.append(z)
                if (i + 1) % 500 == 0:
                    print(f"{tag} {i+1}/{len(cases)} {time.perf_counter()-t0:.0f}s", flush=True)
        out[f"H_{tag}"] = H; out[f"Z_{tag}"] = np.stack(Z)
    np.savez(cache, **out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tag", default="qwen35-4b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--offline", action="store_true", help="use the cached features only")
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    train = hp.load_split("train"); test = hp.load_split("test")
    cache = ROOT / "results" / f"probe_cache_{args.tag}.npz"
    if cache.exists():
        d = np.load(cache); F = {k: d[k] for k in d.files}
    else:
        if args.offline:
            raise SystemExit("no cache")
        F = extract(args.model, train, test, cache)
    Htr, Hte, Ztr, Zte = F["H_train"].astype(np.float32), F["H_test"].astype(np.float32), F["Z_train"], F["Z_test"]
    gtr = np.array([c["gold"] for c in train]); gte = np.array([c["gold"] for c in test])
    qk_tr = [c["qkey"] for c in train]; qk_te = [c["qkey"] for c in test]
    qkeys = sorted(set(qk_te))
    idx_tr = {q: [i for i, k in enumerate(qk_tr) if k == q] for q in qkeys}; idx_te = {q: [i for i, k in enumerate(qk_te) if k == q] for q in qkeys}
    nL = Htr.shape[1]

    def fit_eval(layer, n_per_q=None, seed=0, return_probs=False):
        rng = np.random.default_rng(seed)
        acc = 0; nll = 0.0; n = 0; probs = {}
        for q in qkeys:
            itr = idx_tr[q]; ite = idx_te[q]
            if n_per_q and len(itr) > n_per_q:
                itr = list(rng.choice(itr, n_per_q, replace=False))
            ytr, yte = gtr[itr], gte[ite]; k = len(test[ite[0]]["options"])
            if len(set(ytr.tolist())) < 2:
                p = np.full(len(ite), int(ytr[0])); acc += (p == yte).sum(); n += len(ite)
                for j, i in enumerate(ite):
                    pr = np.zeros(k); pr[int(ytr[0])] = 1.0; probs[i] = pr
                continue
            clf = LogisticRegression(max_iter=3000, C=0.5).fit(Htr[itr, layer], ytr)
            P = clf.predict_proba(Hte[ite, layer]); cls = list(clf.classes_)
            for j, i in enumerate(ite):
                pr = np.full(k, 1e-6);
                for ci, cl in enumerate(cls):
                    pr[int(cl)] = P[j, ci]
                pr /= pr.sum(); probs[i] = pr
                acc += int(np.argmax(pr) == yte[j]); nll += -math.log(max(pr[yte[j]], 1e-12)); n += 1
        r = dict(acc=acc / n, nll=nll / n, n=n)
        return (r, probs) if return_probs else r

    res = {"model": args.model, "n_layers": int(nL), "letter_logits": {}}
    # letter baseline
    ok = 0; nl = 0.0
    for i, c in enumerate(test):
        k = len(c["options"]); p = softmax(Zte[i, :k]); ok += int(np.argmax(p) == gte[i]); nl += -math.log(max(p[gte[i]], 1e-12))
    res["letter_logits"] = dict(acc=ok / len(test), nll=nl / len(test))
    # layer sweep (every 2nd layer + last few)
    layers = sorted(set(list(range(0, nL, 2)) + [nL - 1, nL - 2, nL - 3]))
    res["layer_sweep"] = {}
    best = (0, None)
    for L in layers:
        r = fit_eval(L); res["layer_sweep"][str(L)] = r; print(f"layer {L:2d}/{nL-1}: acc {r['acc']:.3f} nll {r['nll']:.3f}", flush=True)
        if r["acc"] > best[0]:
            best = (r["acc"], L)
    res["best_layer"] = best[1]
    # data efficiency at the best layer
    res["data_efficiency"] = {}
    for npq in (10, 25, 50, 100, 200):
        rs = [fit_eval(best[1], npq, seed)["acc"] for seed in range(3)]
        res["data_efficiency"][str(npq)] = dict(mean=float(np.mean(rs)), std=float(np.std(rs))); print(f"{npq} rows/question: {np.mean(rs):.3f} ± {np.std(rs):.3f}", flush=True)
    # ensemble with letter logits at the best layer
    r, probs = fit_eval(best[1], return_probs=True)
    for w in (0.0, 0.25, 0.5):
        ok = 0
        for i, c in enumerate(test):
            k = len(c["options"]); pl = softmax(Zte[i, :k]); pe = (1 - w) * probs[i] + w * pl; ok += int(np.argmax(pe) == gte[i])
        res[f"ensemble_w_letter={w}"] = ok / len(test)
    # conformal on the probe (split by case id, Mondrian by qtype), alpha 0.1
    case_of = [c["id"].rsplit("-", 1)[0] for c in test]
    cases_u = sorted(set(case_of)); rng = np.random.default_rng(13); rng.shuffle(cases_u); half = set(cases_u[: len(cases_u) // 2])
    A = [i for i in range(len(test)) if case_of[i] in half]; B = [i for i in range(len(test)) if case_of[i] not in half]
    cov = act = act_ok = 0; nn = 0
    for cal, ev in ((A, B), (B, A)):
        qs = {}
        for qt in {test[i]["qtype"] for i in cal}:
            s = np.sort([1 - probs[i][gte[i]] for i in cal if test[i]["qtype"] == qt]); m = len(s); kk = min(m - 1, int(math.ceil((m + 1) * 0.9)) - 1); qs[qt] = float(s[kk])
        for i in ev:
            S = [j for j in range(len(probs[i])) if 1 - probs[i][j] <= qs[test[i]["qtype"]]]
            cov += gte[i] in S; nn += 1
            if len(S) == 1:
                act += 1; act_ok += S[0] == gte[i]
    res["conformal_probe_alpha0.1"] = dict(coverage=cov / nn, act_rate=act / nn, acc_when_acting=act_ok / max(1, act))
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "layer_sweep"}, indent=1))


if __name__ == "__main__":
    main()
