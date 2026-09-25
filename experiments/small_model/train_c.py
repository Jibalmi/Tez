"""Pilot day 3, option C (the control arm): Qwen3.5-0.8B with LoRA r = 16, trained on the teacher's letters.

    python experiments/small_model/train_c.py --arm c-qwen08-30k

Question-first letters, so it drops into the existing readout: each decision is Tez's own prompt
(tez.prompt.build_prompt, template qwen3, layout question_first) and the student's distribution is the softmax over
the option letters' logits at the answer position. Loss: KL(teacher || student) over the options (+ 0.5 x CE on gold
rows), with the option order reshuffled every epoch (target permuted to match). One epoch by default (the 12B's
targets are the same as arm A's).

Writes runs/pilot/models/<arm>/ (LoRA adapter, temperatures) and results/pilot/train_<arm>.json.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import RESULTS, RUNS, dump
from dataset import load_examples, summary
from train_a import fit_type_temps

BASE = "Qwen/Qwen3.5-0.8B"
LETTERS = [chr(65 + i) for i in range(26)]


def load_base(dtype=torch.bfloat16, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype).to(device)
    letter_ids = []
    for L in LETTERS:
        t = tok(L, add_special_tokens=False)["input_ids"]
        assert len(t) == 1, (L, t)
        letter_ids.append(t[0])
    return tok, model, torch.tensor(letter_ids, device=device)


def prompt_for(q_type, instructions, options, state):
    from tez.prompt import QUESTION_FIRST, build_prompt
    from tez.schema import Question
    crit = None
    if q_type == "choice":
        crit = {k: d for k, d in options}
    q = Question("q", q_type, instructions, crit)
    return build_prompt(q, state, "qwen3", options, None, QUESTION_FIRST)


def letter_logp(model, tok, letter_ids, prompts, ks, device, max_len=1536):
    enc = tok(prompts, add_special_tokens=False, padding=True, truncation=True, max_length=max_len,
              return_tensors="pt", padding_side="left")
    out = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device),
                logits_to_keep=1)
    last = out.logits[:, -1, :].float()
    K = max(ks)
    z = last[:, letter_ids[:K]]
    mask = torch.arange(K, device=device)[None, :] < torch.tensor(ks, device=device)[:, None]
    z = torch.where(mask, z, torch.full_like(z, -1e4))
    return F.log_softmax(z, -1), mask


def items(exs, rng):
    out = []
    for e in exs:
        for d in e.decisions:
            perm = list(range(len(d.options)))
            rng.shuffle(perm)
            opts = [d.options[i] for i in perm]
            out.append({"prompt": prompt_for(d.type, d.instructions, opts, e.state), "k": len(opts),
                        "target": d.target[perm], "gold": perm.index(d.gold) if d.gold is not None else None,
                        "type": d.type})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="c-qwen08-30k")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--minutes", type=float, default=100)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device("cuda")
    from peft import LoraConfig, get_peft_model
    tok, model, letter_ids = load_base(device=dev)
    tok.padding_side = "left"
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear",
                     exclude_modules=["lm_head"], task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    train, _, val = load_examples()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    n_items = sum(len(e.decisions) for e in train)
    total = (n_items // args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.05 * total))) *
                                              max(0.0, (total - s) / max(1, total)))
    t0 = time.time()
    step, tokens, stopped, log = 0, 0, False, []
    model.train()
    for ep in range(args.epochs):
        its = items(train, rng)
        its.sort(key=lambda x: len(x["prompt"]))
        bs = [its[i:i + args.batch] for i in range(0, len(its), args.batch)]
        rng.shuffle(bs)
        for b in bs:
            logp, mask = letter_logp(model, tok, letter_ids, [x["prompt"] for x in b], [x["k"] for x in b], dev)
            K = logp.shape[1]
            tgt = torch.tensor(np.array([np.pad(x["target"], (0, K - x["k"])) for x in b]), dtype=torch.float32, device=dev)
            loss = (tgt * (torch.log(tgt.clamp_min(1e-12)) - logp) * mask).sum(-1).mean()
            gold = torch.tensor([-100 if x["gold"] is None else x["gold"] for x in b], device=dev)
            if (gold >= 0).any():
                loss = loss + 0.5 * F.nll_loss(logp, gold, ignore_index=-100) * (gold >= 0).float().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            step += 1
            tokens += int(sum(len(tok(x["prompt"], add_special_tokens=False)["input_ids"]) for x in b)) if step % 20 == 0 else 0
            if step % 20 == 0:
                el = time.time() - t0
                print(f"ep {ep} step {step}/{total} loss {loss.item():.4f} {el:.0f} s "
                      f"(~{tokens * 20 / el:.0f} tok/s est)", flush=True)
            if time.time() - t0 > args.minutes * 60:
                stopped = True
                break
        if stopped:
            break
    train_s = time.time() - t0
    # validation: agreement with the teacher + per-type temperatures
    model.eval()
    recs, agree = [], []
    with torch.no_grad():
        vi = items(val, random.Random(0))
        for i in range(0, len(vi), 32):
            b = vi[i:i + 32]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logp, _ = letter_logp(model, tok, letter_ids, [x["prompt"] for x in b], [x["k"] for x in b], dev)
            lp = logp.float().cpu().numpy()
            for x, row in zip(b, lp):
                y = int(np.argmax(x["target"]))
                agree.append(int(np.argmax(row[: x["k"]]) == y))
                recs.append((x["type"], row[: x["k"]], y))
    temps = fit_type_temps(recs)
    out = RUNS / "models" / args.arm
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")
    info = {"arm": args.arm, "backbone": BASE, "lora": {"r": 16, "alpha": 32, "targets": "all-linear"},
            "epochs": args.epochs, "steps": step, "total_steps_planned": total, "stopped_early": stopped,
            "train_seconds": round(train_s, 1), "decisions_per_s": round(step * args.batch / train_s, 2),
            "est_train_tok_per_s": round(tokens * 20 / train_s, 1), "lr": args.lr, "batch": args.batch,
            "val_agreement_with_teacher": round(float(np.mean(agree)), 4), "temperatures": temps,
            "train_summary": summary(train), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
    (out / "student.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    dump(RESULTS / f"train_{args.arm}.json", info)
    print(json.dumps({k: v for k, v in info.items() if k != "train_summary"}, indent=1))


if __name__ == "__main__":
    main()
