"""What does reading the decision at 60% depth save? Truncate the frozen model to its first N layers
and time a forward pass on typed-decisions prompts (CUDA-synchronised), full depth vs N in
{16, 20, 24, 28}. The probe results say layer 20-28 of 32 already carry the decision.

  py experiments/early_exit_bench.py --model Qwen/Qwen3.5-4B --out results/early_exit_qwen35-4b.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hp", ROOT / "experiments" / "hidden_probe.py")
hp = importlib.util.module_from_spec(spec); spec.loader.exec_module(hp)  # type: ignore[union-attr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--depths", default="16,20,24,28")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    layers_all = model.model.layers
    full = len(layers_all)
    test = hp.load_split("test")[: args.n]
    prompts = []
    for c in test:
        msgs = [{"role": "user", "content": hp.prompt_text(c)}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, -4096:].to("cuda"))
    res = {"model": args.model, "full_layers": full, "n_prompts": len(prompts), "mean_tokens": statistics.mean(p.shape[1] for p in prompts)}
    import torch.nn as nn
    for depth in [full] + [int(x) for x in args.depths.split(",")]:
        model.model.layers = nn.ModuleList(list(layers_all)[:depth])
        with torch.inference_mode():
            for p in prompts[:5]:
                model.model(input_ids=p, use_cache=False)
            torch.cuda.synchronize(); ms = []
            for p in prompts:
                t = time.perf_counter(); model.model(input_ids=p, use_cache=False); torch.cuda.synchronize(); ms.append((time.perf_counter() - t) * 1000)
        res[f"depth_{depth}"] = dict(ms_p50=statistics.median(ms), ms_p95=sorted(ms)[int(0.95 * (len(ms) - 1))], speedup=None)
        res[f"depth_{depth}"]["speedup"] = res[f"depth_{full}"]["ms_p50"] / res[f"depth_{depth}"]["ms_p50"]
        print(depth, res[f"depth_{depth}"], flush=True)
    model.model.layers = layers_all
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
