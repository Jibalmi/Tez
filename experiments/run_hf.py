"""In-process symbol-logit readout with Hugging Face transformers -- SemIf's own method.

Mirrors data/semif/direct.py: tokenizer.apply_chat_template(..., add_generation_prompt=True,
enable_thinking=False), every answer letter must be one exact round-trip token, the answer
boundary must not change tokenization, one forward pass, last-position logits restricted to
the declared slots, softmax. No decoding.

Modes
  fresh      one full forward pass per decision (what the accuracy tables use)
  prefix     share the KV cache of the common prefix (system + evidence) across the decisions
             that use the same state, evaluating only the suffix (criterion + options + tail)
  batchperm  score all 6 orderings of a 3-option row in ONE batched forward pass
Timing uses CUDA synchronisation around the forward only, so numbers are compute, not HTTP.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("semif_core", ROOT / "data" / "semif" / "core.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)  # type: ignore[union-attr]
LETTERS = core.LETTERS


def slot_ids(tok, count):
    out = []
    for L in LETTERS[:count]:
        enc = tok.encode(L, add_special_tokens=False)
        if len(enc) != 1 or tok.decode(enc) != L:
            raise ValueError(f"slot {L!r} is not one exact round-trip token: {enc}")
        out.append(enc[0])
    if len(out) != len(set(out)):
        raise ValueError("slot tokens collide")
    return out


def render(tok, row, template_kwargs):
    return tok.apply_chat_template(core.direct_messages(row), tokenize=False, add_generation_prompt=True, **template_kwargs)


def encode(tok, row, template_kwargs, max_tokens=4096):
    prompt = render(tok, row, template_kwargs)
    ids = tok.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > max_tokens:
        raise ValueError(f"{row['id']}: {len(ids)} tokens")
    slots = slot_ids(tok, len(row["options"]))
    for L, t in zip(LETTERS, slots):
        if tok.encode(prompt + L, add_special_tokens=False) != ids + [t]:
            raise ValueError(f"{row['id']}: answer boundary changes tokenization for {L}")
    return prompt, ids, slots


@torch.inference_mode()
def last_logits(model, ids_batch, attn=None, past=None):
    kw = dict(use_cache=past is not None or False, return_dict=True)
    if past is not None:
        kw["past_key_values"] = past
        kw["use_cache"] = True
    try:
        out = model(input_ids=ids_batch, attention_mask=attn, logits_to_keep=1, **kw)
    except TypeError:
        out = model(input_ids=ids_batch, attention_mask=attn, **kw)
    return out.logits[:, -1, :].float(), (out.past_key_values if past is not None else None)


def record(row, slots, logits_row, timing_ms, prompt_n, prompt, mode, extra=None):
    sel = logits_row[slots].tolist()
    full = torch.log_softmax(logits_row, dim=-1)
    letter_mass = float(torch.exp(full[slots]).sum())
    top = int(torch.argmax(logits_row))
    return {
        "id": row["id"], "family": row.get("family"), "group_id": row.get("group_id"), "split": row.get("split"),
        "provenance": row.get("provenance"), "gold": row.get("label"), "target_distribution": row.get("target_distribution"),
        "option_ids": [o["id"] for o in row["options"]],
        "probabilities": core.softmax(sel), "option_logprobs": sel,
        "missing_letters": [], "top1_token_id": top, "top1_is_slot": top in slots, "letter_mass": letter_mass,
        "prompt_n": prompt_n, "forward_ms": timing_ms, "prompt_ms": timing_ms, "http_ms": None,
        "prompt_sha256": core.digest(prompt), "mode": mode, **(extra or {}),
        "readout": "transformers last-position logits restricted to declared answer slots",
        "probability_status": "conditional option score; uncalibrated as decision confidence",
    }


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="fresh", choices=["fresh", "prefix", "batchperm"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-thinking-kwarg", action="store_true", help="template has no enable_thinking parameter")
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="cuda", attn_implementation="sdpa")
    model.eval()
    load_s = time.perf_counter() - t0
    dev = next(model.parameters()).device
    tk = {} if args.no_thinking_kwarg else {"enable_thinking": False}

    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    for r in rows:
        core.validate_row(r)

    # warm-up (CUDA graphs/kernels)
    for r in rows[: args.warmup]:
        _, ids, _ = encode(tok, r, tk)
        last_logits(model, torch.tensor([ids], device=dev))
    sync()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    wall0 = time.perf_counter()

    if args.mode == "fresh":
        for r in rows:
            prompt, ids, slots = encode(tok, r, tk)
            x = torch.tensor([ids], device=dev)
            sync(); t = time.perf_counter()
            lg, _ = last_logits(model, x)
            sync(); ms = (time.perf_counter() - t) * 1000
            recs.append(record(r, slots, lg[0], ms, len(ids), prompt, "fresh"))

    elif args.mode == "prefix":
        # The shared prefix is everything up to (and including) the evidence value: SemIf's
        # payload is {"evidence":..., "criterion":..., "options":...}, so we split the rendered
        # prompt at the '"criterion"' key and trim to a clean token boundary.
        from transformers import DynamicCache
        by_state = {}
        for r in rows:
            by_state.setdefault(json.dumps(r["state"], ensure_ascii=False, sort_keys=True), []).append(r)
        for state_key, group in by_state.items():
            prompt0, ids0, _ = encode(tok, group[0], tk)
            cut = prompt0.find('"criterion"')
            pre = prompt0[:cut]
            pre_ids = tok.encode(pre, add_special_tokens=False)
            while pre_ids and ids0[: len(pre_ids)] != pre_ids:
                pre = pre[:-1]
                pre_ids = tok.encode(pre, add_special_tokens=False)
            sync(); t = time.perf_counter()
            cache = DynamicCache()
            _, cache = last_logits(model, torch.tensor([pre_ids], device=dev), past=cache)
            sync(); pre_ms = (time.perf_counter() - t) * 1000
            for r in group:
                prompt, ids, slots = encode(tok, r, tk)
                assert ids[: len(pre_ids)] == pre_ids, r["id"]
                suf = ids[len(pre_ids):]
                c2 = copy.deepcopy(cache)
                sync(); t = time.perf_counter()
                lg, _ = last_logits(model, torch.tensor([suf], device=dev), past=c2)
                sync(); ms = (time.perf_counter() - t) * 1000
                recs.append(record(r, slots, lg[0], ms, len(suf), prompt, "prefix",
                                   {"prefix_n": len(pre_ids), "prefix_ms": pre_ms, "state_group_size": len(group)}))

    elif args.mode == "batchperm":
        for r in rows:
            n = len(r["options"])
            perms = list(itertools.permutations(range(n)))
            prompts, idss, slotss, rows_p = [], [], [], []
            for p in perms:
                rp = dict(r, options=[r["options"][j] for j in p], label=list(p).index(r["label"]))
                prompt, ids, slots = encode(tok, rp, tk)
                prompts.append(prompt); idss.append(ids); slotss.append(slots); rows_p.append(rp)
            L = max(len(x) for x in idss)
            pad = tok.pad_token_id
            x = torch.tensor([[pad] * (L - len(ids)) + ids for ids in idss], device=dev)
            attn = torch.tensor([[0] * (L - len(ids)) + [1] * len(ids) for ids in idss], device=dev)
            sync(); t = time.perf_counter()
            lg, _ = last_logits(model, x, attn=attn)
            sync(); ms = (time.perf_counter() - t) * 1000
            # average by option id, report in original order
            acc = {o["id"]: 0.0 for o in r["options"]}
            for k, (rp, slots) in enumerate(zip(rows_p, slotss)):
                pr = core.softmax(lg[k][slots].tolist())
                for o, pv in zip(rp["options"], pr):
                    acc[o["id"]] += pv / len(perms)
            base = record(r, slotss[0], lg[0], ms, L, prompts[0], "batchperm", {"n_perms": len(perms)})
            base["probabilities"] = [acc[o["id"]] for o in r["options"]]
            base["option_logprobs"] = [float(torch.log(torch.tensor(v) + 1e-12)) for v in base["probabilities"]]
            recs.append(base)

    wall = time.perf_counter() - wall0
    with out.open("w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    ms = sorted(r["forward_ms"] for r in recs)
    manifest = {
        "tag": args.tag, "model": args.model, "dtype": args.dtype, "mode": args.mode, "rows": len(recs),
        "data": args.data, "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "load_seconds": load_s, "wall_seconds": wall, "decisions_per_second": len(recs) / wall,
        "forward_ms_p50": statistics.median(ms), "forward_ms_p95": ms[int(0.95 * (len(ms) - 1))],
        "mean_prompt_tokens": sum(r["prompt_n"] for r in recs) / len(recs),
        "mean_letter_mass": sum(r["letter_mass"] for r in recs) / len(recs),
        "top1_not_slot_rows": sum(1 for r in recs if not r["top1_is_slot"]),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in manifest.items() if k in
                      ("mode", "rows", "decisions_per_second", "forward_ms_p50", "forward_ms_p95", "mean_prompt_tokens",
                       "mean_letter_mass", "top1_not_slot_rows", "peak_vram_gb", "load_seconds")}))


if __name__ == "__main__":
    main()
