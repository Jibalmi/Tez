"""Many decisions from ONE forward pass (typed-decisions, frozen Qwen3.5-4B).

Today every question is its own prompt (options first, state last), so a row with five questions
costs five passes over the state. Two single-pass layouts are compared with that baseline:

  A "state first, all questions":  Input: <state>  then  Question 1 ... Options ... Answer 1:
                                    Question 2 ... Answer 2:  ...  in one sequence. Each decision is
                                    read at its own "Answer n:" marker (hidden state -> per-question
                                    probe; and the letter logits at the marker, zero-shot).
  B "state only":                  one pass over the state alone; ONE hidden vector per row, read by
                                    a separate probe for each question.

Features (all layers) are cached aligned with hidden_probe.load_split order, so the probes are
directly comparable with the per-question-prompt cache (0.793 at layer 26).

  py experiments/probe_multiq.py --out results/probe_multiq_qwen35-4b.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("hp", ROOT / "experiments" / "hidden_probe.py")
hp = importlib.util.module_from_spec(_s); _s.loader.exec_module(hp)  # type: ignore[union-attr]
HEAD_A = ("You are a decision engine. Read the input, then answer every question below with the single letter "
          "of its best option.")


def rows_of(split):
    """Group the decisions of hp.load_split(split) by row, keeping the dataset's question order."""
    cases = hp.load_split(split)
    rows = {}
    for c in cases:
        qid = str(c["qkey"][1]); rid = c["id"][: -(len(qid) + 1)]   # ids are f"{row_id}-{qid}"
        rows.setdefault(rid, dict(state=c["state"], qs=[]))["qs"].append(c)
    return cases, rows


def state_text(st):
    return json.dumps(st, ensure_ascii=False) if not isinstance(st, str) else st


def layout_a(row):
    parts = [HEAD_A, "", "Input:", state_text(row["state"]), ""]
    for n, c in enumerate(row["qs"], 1):
        parts.append(f"Question {n} ({c['qtype']}): {c['instructions']}")
        parts.append("Options:")
        parts += [f"{hp.LETTERS[i]}. {k}: {d}" for i, (k, d) in enumerate(c["options"])]
        parts.append(f"Answer {n}:")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def chat(tok, content):
    msgs = [{"role": "user", "content": content}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def extract(model, tok, split, sp_ids, nsp_ids):
    cases, rows = rows_of(split)
    pos_of = {c["id"]: i for i, c in enumerate(cases)}
    HA = ZA = HB = None; row_of = np.zeros(len(cases), dtype=np.int32); ntok = dict(a=0, b=0)
    t0 = time.perf_counter()
    for r_i, (rid, row) in enumerate(rows.items()):
        # ---- layout A: all questions in one sequence, read at each "Answer n:" marker
        s = chat(tok, layout_a(row))
        enc = tok(s, return_offsets_mapping=True, add_special_tokens=False)
        offs = enc["offset_mapping"]; ids = torch.tensor([enc["input_ids"]], device="cuda")
        ntok["a"] += ids.shape[1]
        marks = []
        for n in range(1, len(row["qs"]) + 1):
            e = s.rindex(f"Answer {n}:") + len(f"Answer {n}:")
            t = max(j for j, (a, b) in enumerate(offs) if a < e)
            marks.append(t)
        o = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        hs = torch.stack(o.hidden_states)[:, 0]                      # [L+1, T, d]
        feats = hs[:, marks].permute(1, 0, 2).to(torch.float16).cpu().numpy()   # [n_q, L+1, d]
        lg = o.logits[0, marks].float()
        z = torch.logsumexp(torch.stack([lg[:, sp_ids], lg[:, nsp_ids]]), 0).cpu().numpy()   # " A" or "A"
        if HA is None:
            HA = np.zeros((len(cases),) + feats.shape[1:], dtype=np.float16); ZA = np.full((len(cases), 26), -1e4, dtype=np.float32)
        for j, c in enumerate(row["qs"]):
            i = pos_of[c["id"]]; HA[i] = feats[j]; k = len(c["options"]); ZA[i, :k] = z[j, :k]
        # ---- layout B: the state alone, one vector per row
        sb = chat(tok, "Input:\n" + state_text(row["state"]))
        idb = tok(sb, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        ntok["b"] += idb.shape[1]
        ob = model(input_ids=idb, output_hidden_states=True, use_cache=False)
        hb = torch.stack([h[0, -1] for h in ob.hidden_states]).to(torch.float16).cpu().numpy()
        if HB is None:
            HB = np.zeros((len(rows),) + hb.shape, dtype=np.float16)
        HB[r_i] = hb
        for c in row["qs"]:
            row_of[pos_of[c["id"]]] = r_i
        if (r_i + 1) % 200 == 0:
            print(f"{split} rows {r_i + 1}/{len(rows)} {time.perf_counter() - t0:.0f}s", flush=True)
    return dict(HA=HA, ZA=ZA, HB=HB, row_of=row_of, ntok_a=ntok["a"], ntok_b=ntok["b"], n_rows=len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tag", default="qwen35-4b")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    cache = ROOT / "results" / f"probe_cache_multiq_{args.tag}.npz"
    if cache.exists():
        F = dict(np.load(cache))
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
        sp_ids = [tok.encode(" " + L, add_special_tokens=False)[-1] for L in hp.LETTERS]
        nsp_ids = [tok.encode(L, add_special_tokens=False)[0] for L in hp.LETTERS]
        # the per-question baseline: tokens of the standard prompts, for the cost comparison
        F = {}
        for split in ("train", "test"):
            r = extract(model, tok, split, sp_ids, nsp_ids)
            for k, v in r.items():
                F[f"{k}_{split}"] = v
        cases_te = hp.load_split("test")
        F["ntok_perq_test"] = sum(len(tok(chat(tok, hp.prompt_text(c)), add_special_tokens=False).input_ids) for c in cases_te)
        np.savez(cache, **F)
        del model; torch.cuda.empty_cache()
    train, test = hp.load_split("train"), hp.load_split("test")
    gtr = np.array([c["gold"] for c in train]); gte = np.array([c["gold"] for c in test]); kte = np.array([len(c["options"]) for c in test])
    qkeys = sorted(set(c["qkey"] for c in train))
    itr = {q: np.array([i for i, c in enumerate(train) if c["qkey"] == q]) for q in qkeys}
    ite = {q: np.array([i for i, c in enumerate(test) if c["qkey"] == q]) for q in qkeys}
    base = np.load(ROOT / "results" / f"probe_cache_{args.tag}.npz")
    Hq_tr, Hq_te = base["H_train"], base["H_test"]

    def probe(Xtr, Xte):
        ok = 0
        for q in qkeys:
            y = gtr[itr[q]]
            if len(np.unique(y)) < 2:
                ok += int((gte[ite[q]] == y[0]).sum()); continue
            clf = LogisticRegression(max_iter=3000, C=0.5).fit(Xtr[itr[q]], y)
            ok += int((clf.predict(Xte[ite[q]]) == gte[ite[q]]).sum())
        return ok / len(test)

    nL = F["HA_train"].shape[1]
    ZA = F["ZA_test"]
    res = {"model": args.model, "rows_test": int(F["n_rows_test"]), "decisions_test": len(test),
           "tokens_test": {"per_question_prompts": int(F["ntok_perq_test"]), "A_all_questions_one_pass": int(F["ntok_a_test"]), "B_state_only": int(F["ntok_b_test"])},
           "passes_test": {"per_question_prompts": len(test), "A_all_questions_one_pass": int(F["n_rows_test"]), "B_state_only": int(F["n_rows_test"])},
           "zero_shot_letters": {"per_question_prompt": float(np.mean([int(np.argmax(base["Z_test"][i, :kte[i]])) == gte[i] for i in range(len(test))])),
                                 "A_at_markers": float(np.mean([int(np.argmax(ZA[i, :kte[i]])) == gte[i] for i in range(len(test))]))},
           "probe": {}}
    print(json.dumps({k: res[k] for k in ("tokens_test", "passes_test", "zero_shot_letters")}), flush=True)
    for L in (12, 16, 18, 20, 22, 24, 26, 28, nL - 1):
        HBtr = F["HB_train"][F["row_of_train"], L].astype(np.float32); HBte = F["HB_test"][F["row_of_test"], L].astype(np.float32)
        r = dict(per_question_prompt=probe(Hq_tr[:, L].astype(np.float32), Hq_te[:, L].astype(np.float32)),
                 A_all_questions_one_pass=probe(F["HA_train"][:, L].astype(np.float32), F["HA_test"][:, L].astype(np.float32)),
                 B_state_only=probe(HBtr, HBte))
        res["probe"][str(L)] = r
        print(f"L{L}: per-question prompt {r['per_question_prompt']:.3f} | A one pass {r['A_all_questions_one_pass']:.3f} | B state only {r['B_state_only']:.3f}", flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
