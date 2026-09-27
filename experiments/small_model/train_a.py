"""Pilot day 3, option A: distil an mmBERT student from the teacher's reads.

    python experiments/small_model/train_a.py --arm a-base-30k --backbone jhu-clsp/mmBERT-base --data 30k
    python experiments/small_model/train_a.py --arm a-base-10k --backbone jhu-clsp/mmBERT-base --data 10k
    python experiments/small_model/train_a.py --arm a-small-30k --backbone jhu-clsp/mmBERT-small --data 30k

Loss (plan §4): KL(teacher || student) over each question's options + 0.5 x cross-entropy on gold labels (CLINC,
CommonsenseQA, Global-MMLU rows) + dev-0.4b's ordinal term (squared CDF distance) for score questions. Every epoch
reshuffles question order and option order (the target is permuted to match) and repacks each state's questions:
half the states as one row with all 5, the rest split into random groups of 1-4. Full fine-tune, bf16 autocast,
sequence length 2,048. After training, one temperature per question type is fitted on the teacher-labelled
validation split (NLL against the teacher's argmax).

Writes runs/pilot/models/<arm>/ (encoder, head, tokenizer, temperatures) and results/pilot/train_<arm>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import RESULTS, RUNS, THERMAL, dump, thermal_guard, thermal_guard_or_checkpoint
from dataset import load_examples, summary
from student import Encoder, Q, Student, collate, lr_lambda, n_params, ordinal_loss


def make_rows(exs, enc: Encoder, rng: random.Random, pack_all: float = 0.5):
    rows = []
    for e in exs:
        items = []
        for d in e.decisions:
            perm = list(range(len(d.options)))
            rng.shuffle(perm)
            opts = [d.options[i] for i in perm]
            tgt = d.target[perm]
            gold = perm.index(d.gold) if d.gold is not None else None
            items.append((Q(d.type, d.instructions, opts), tgt, gold, d.type, np.argsort(perm).tolist()))
        rng.shuffle(items)
        if rng.random() < pack_all or len(items) == 1:
            groups = [items]
        else:
            groups, rest = [], items[:]
            while rest:
                n = rng.randint(1, min(4, len(rest)))
                groups.append(rest[:n])
                rest = rest[n:]
        st_text = None
        for g in groups:
            row = enc.encode(e.state, [x[0] for x in g], st_text)
            row.meta = {"targets": [x[1] for x in g], "gold": [x[2] for x in g], "types": [x[3] for x in g],
                        "canonical_order": [x[4] for x in g]}
            rows.append(row)
    return rows


def batches(rows, budget: int, rng: random.Random):
    """Length-bucketed batches of at most `budget` padded tokens (random bucket order)."""
    rows = rows[:]
    rng.shuffle(rows)
    out = []
    for s in range(0, len(rows), 256):
        chunk = sorted(rows[s:s + 256], key=lambda r: len(r.ids))
        cur, cur_max = [], 0
        for r in chunk:
            m = max(cur_max, len(r.ids))
            if cur and m * (len(cur) + 1) > budget:
                out.append(cur)
                cur, m = [], len(r.ids)
            cur.append(r)
            cur_max = m
        if cur:
            out.append(cur)
    rng.shuffle(out)
    return out


def targets_tensor(rows, K, device):
    tg, gold, isscore, mask = [], [], [], []
    for r in rows:
        for t, g, ty in zip(r.meta["targets"], r.meta["gold"], r.meta["types"]):
            v = np.zeros(K, np.float32)
            v[: len(t)] = t
            m = np.zeros(K, np.float32)
            m[: len(t)] = 1
            tg.append(v)
            mask.append(m)
            gold.append(-100 if g is None else g)
            isscore.append(ty == "score")
    f = lambda a, dt=torch.float32: torch.tensor(np.array(a), dtype=dt, device=device)  # noqa: E731
    return f(tg), f(gold, torch.long), f(isscore, torch.bool), f(mask)


def ordered_score_loss(logp, target, mask, rows):
    """Restore semantic level order before comparing cumulative score probabilities."""
    K = logp.shape[1]
    orders = [list(order) + list(range(len(order), K))
              for row in rows for order in row.meta["canonical_order"]]
    index = torch.tensor(orders, dtype=torch.long, device=logp.device)
    return ordinal_loss(logp.gather(1, index), target.gather(1, index), mask.gather(1, index))


def step_loss(model, rows, device, ce_w=0.5, ord_w=1.0):
    batch = collate(rows, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logp = model(batch)
    K = logp.shape[1]
    tgt, gold, isscore, mask = targets_tensor(rows, K, device)
    kl = (tgt * (torch.log(tgt.clamp_min(1e-12)) - logp) * mask).sum(-1)
    loss = kl.mean()
    if (gold >= 0).any():
        ce = F.nll_loss(logp, gold, ignore_index=-100, reduction="sum") / max(1, int((gold >= 0).sum()))
        loss = loss + ce_w * ce * (gold >= 0).float().mean()
    if isscore.any():
        loss = loss + ord_w * (ordered_score_loss(logp, tgt, mask, rows) * isscore.float()).mean()
    return loss, logp, tgt, mask


@torch.no_grad()
def evaluate_val(model, val, enc, device, budget, guard=thermal_guard):
    model.eval()
    rng = random.Random(0)
    rows = make_rows(val, enc, rng, pack_all=1.0)
    agree, kls, recs = [], [], []
    for b in batches(rows, budget, rng):
        if device.type == "cuda":
            guard()
        _, logp, tgt, mask = step_loss(model, b, device)
        lp = logp.float().cpu().numpy()
        t = tgt.cpu().numpy()
        types = [ty for r in b for ty in r.meta["types"]]
        for i in range(len(t)):
            k = int(mask[i].sum())
            agree.append(int(np.argmax(lp[i, :k]) == np.argmax(t[i, :k])))
            kls.append(float((t[i, :k] * (np.log(np.clip(t[i, :k], 1e-12, None)) - lp[i, :k])).sum()))
            recs.append((types[i], lp[i, :k], int(np.argmax(t[i, :k]))))
    model.train()
    return float(np.mean(agree)), float(np.mean(kls)), recs


def fit_type_temps(recs) -> dict:
    """One temperature per question type (choice split at 10 options), NLL against the teacher's argmax."""
    out = {}
    groups = {}
    for ty, lp, y in recs:
        key = "choice_many" if ty == "choice" and len(lp) > 10 else ty
        groups.setdefault(key, []).append((lp, y))
    for key, rows in groups.items():
        best, bt = 1e9, 1.0
        for t in np.exp(np.linspace(np.log(0.2), np.log(5), 60)):
            nll = 0.0
            for lp, y in rows:
                z = lp / t
                z = z - z.max()
                nll -= z[y] - math.log(np.exp(z).sum())
            if nll < best:
                best, bt = nll, float(t)
        out[key] = round(bt, 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--data", choices=["10k", "30k"], default="30k")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--budget", type=int, default=16384, help="padded tokens per batch")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--select-best-val", action="store_true", help="save the epoch with minimum teacher-validation KL")
    ap.add_argument("--minutes", type=float, default=50, help="segment length: checkpoint and exit 3 past this")
    ap.add_argument("--device", default="cuda", help="cuda (the pilot) or cpu (smoke tests)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.backbone)
    enc = Encoder(tok, args.max_len)
    train_all, train_10k, val = load_examples()
    train = train_all if args.data == "30k" else train_10k
    print(json.dumps({"train": summary(train), "val": summary(val)}), flush=True)
    out = RUNS / "models" / args.arm
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt.pt"
    best_path = out / "best_val.pt"

    model = Student(args.backbone).to(dev)
    model.train()
    enc_params = [p for n, p in model.named_parameters() if n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr},
                             {"params": model.head.parameters(), "lr": args.head_lr}], weight_decay=0.01,
                            fused=dev.type == "cuda")

    def epoch_batches(ep: int):
        """Deterministic per epoch, so a resumed segment skips exactly the batches already trained."""
        r = random.Random(args.seed * 1000 + ep)
        return batches(make_rows(train, enc, r), args.budget, r)

    total = len(epoch_batches(0)) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(max(1, int(0.06 * total)), total))
    st = {"ep": 0, "b": 0, "step": 0, "tokens": 0, "log": [], "train_s": 0.0, "segments": 0}
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        st = ck["state"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        if dev.type == "cuda" and "cuda_rng" in ck:
            torch.cuda.set_rng_state_all([state.cpu() for state in ck["cuda_rng"]])
        print(f"resumed at epoch {st['ep']} batch {st['b']} (step {st['step']})", flush=True)
    st["segments"] += 1
    t0 = time.time()
    prior_seconds, prior_paused = st["train_s"], st.get("paused_s", 0.0)

    def checkpoint():
        st["train_s"] = prior_seconds + time.time() - t0
        st["paused_s"] = prior_paused + THERMAL["paused_s"]
        st["max_gpu_c"] = max(st.get("max_gpu_c", 0), THERMAL["max_c"])
        tmp = ckpt_path.with_suffix(".pt.tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "state": st,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if dev.type == "cuda" else []}, tmp)
        tmp.replace(ckpt_path)

    def guard():
        thermal_guard_or_checkpoint(checkpoint)

    stopped = False
    while st["ep"] < args.epochs:
        bs = epoch_batches(st["ep"])
        for bi in range(st["b"], len(bs)):
            if dev.type == "cuda":
                guard()
            loss, *_ = step_loss(model, bs[bi], dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; refusing to publish a student")
            st["tokens"] += sum(len(r.ids) for r in bs[bi])
            st["step"] += 1
            st["b"] = bi + 1
            if st["step"] % 250 == 0:
                checkpoint()
            if st["step"] % 50 == 0:
                el = prior_seconds + time.time() - t0
                print(f"ep {st['ep']} step {st['step']}/{total} loss {loss.item():.4f} "
                      f"{st['tokens'] / el:.0f} tok/s {el:.0f} s", flush=True)
            if time.time() - t0 > args.minutes * 60 and st["b"] < len(bs):
                stopped = True
                break
        if stopped:
            break
        agree, kl, _ = evaluate_val(model, val, enc, dev, args.budget, guard=guard)
        st["log"].append({"epoch": st["ep"], "step": st["step"], "val_agreement": round(agree, 4),
                          "val_kl": round(kl, 4), "seconds": round(prior_seconds + time.time() - t0, 1)})
        print(f"== epoch {st['ep']}: val agreement {agree:.4f} KL {kl:.4f}", flush=True)
        if args.select_best_val and kl < st.get("best_val_kl", float("inf")):
            st["best_val_kl"], st["best_val_epoch"] = kl, st["ep"] + 1
            temporary = best_path.with_suffix(".pt.tmp")
            torch.save(model.state_dict(), temporary)
            temporary.replace(best_path)
        st["ep"] += 1
        st["b"] = 0
    st["train_s"] = prior_seconds + time.time() - t0
    st["paused_s"] = prior_paused + THERMAL["paused_s"]
    st["max_gpu_c"] = max(st.get("max_gpu_c", 0), THERMAL["max_c"])
    if stopped:
        checkpoint()
        print(f"segment ended at epoch {st['ep']} batch {st['b']}: checkpoint saved", flush=True)
        return 3
    if args.select_best_val:
        model.load_state_dict(torch.load(best_path, map_location=dev, weights_only=True))
    agree, kl, recs = evaluate_val(model, val, enc, dev, args.budget, guard=guard)
    temps = fit_type_temps(recs)
    model.enc.save_pretrained(out / "encoder")
    tok.save_pretrained(out / "encoder")
    torch.save(model.head.state_dict(), out / "head.pt")
    info = {"arm": args.arm, "backbone": args.backbone, "data": args.data, "params": n_params(model),
            "epochs_run": len(st["log"]), "steps": st["step"], "segments": st["segments"],
            "train_seconds": round(st["train_s"], 1), "train_tokens": st["tokens"],
            "train_tok_per_s": round(st["tokens"] / st["train_s"], 1),
            "train_tok_per_s_excluding_thermal_pauses": round(st["tokens"] / max(1e-9, st["train_s"] - st["paused_s"]), 1),
            "thermal_paused_s": round(st["paused_s"], 1), "max_gpu_c": st["max_gpu_c"], "lr": args.lr,
            "head_lr": args.head_lr, "budget_tokens": args.budget, "max_len": args.max_len, "log": st["log"],
            "score_loss_order": "canonical_levels", "seed": args.seed,
            "selected_by": "minimum validation KL" if args.select_best_val else "final epoch",
            "selected_epoch": st.get("best_val_epoch", st["ep"]),
            "val_agreement_with_teacher": round(agree, 4), "val_kl": round(kl, 4), "temperatures": temps,
            "train_summary": summary(train), "val_summary": summary(val),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev.type == "cuda" else None}
    (out / "student.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    dump(RESULTS / f"train_{args.arm}.json", info)
    print(json.dumps({k: v for k, v in info.items() if k not in ("log", "train_summary", "val_summary")}, indent=1))
    if ckpt_path.exists():
        ckpt_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
