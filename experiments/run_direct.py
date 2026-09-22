"""Direct symbol-logit readout via llama-server, byte-compatible with SemIf's prompt.

For each decision row {id, state, question, options[{id, description}], label, ...}:
  1. Build SemIf's exact system+user messages (imported from SemIf's own core.py).
  2. Render them into the Gemma 4 chat template with thinking disabled. Gemma has no
     separate system turn, so the system text is folded into the user turn, which is
     what HF's Gemma template does too.
  3. One forward pass (n_predict=1, temperature=0) and read top_logprobs at the first
     answer position; keep only the declared answer-slot letters and softmax over them.
     Because logprobs are already log-softmax over the full vocabulary, the restricted
     softmax is identical to SemIf's softmax over raw slot logits.

Nothing is decoded. Output is one JSONL row per decision plus a manifest.
"""
from __future__ import annotations

import argparse
import os
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SEMIF_CORE = ROOT / "data" / "semif" / "core.py"


def load_semif_core():
    spec = importlib.util.spec_from_file_location("semif_core", SEMIF_CORE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


core = load_semif_core()
if os.environ.get("TEZ_SYMBOLS"):
    # verbalizer ablation: replace the A..P answer symbols (e.g. "123456789" or "abcdefghij");
    # SemIf's instruction says "uppercase letter" -- reword it so the prompt stays consistent
    core.LETTERS = os.environ["TEZ_SYMBOLS"]
    core.DIRECT_SYSTEM = core.DIRECT_SYSTEM.replace("uppercase letter", os.environ.get("TEZ_SYMBOL_WORD", "label"))
LETTERS = core.LETTERS


def _split(messages: list[dict]) -> tuple[str, str]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
    return system, user


def render_prompt(messages: list[dict], template: str) -> str:
    """Render SemIf's system+user messages into a model family's chat template with the
    generation prompt appended, so the next token is the answer letter."""
    system, user = _split(messages)
    folded = f"{system}\n\n{user}" if system else user
    if template == "gemma4":
        # Thinking disabled: the official template emits an EMPTY thought channel and the
        # answer token is read at the position immediately after <channel|>.
        # System text is folded into the user turn (Gemma 3 convention). Gemma 4's own
        # template has a native system block -- see "gemma4sys" for that rendering.
        return f"<|turn>user\n{folded}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    if template == "gemma4sys":
        # Native Gemma 4 system turn, as apply_chat_template would render it.
        return (f"<|turn>system\n{system}<turn|>\n<|turn>user\n{user}<turn|>\n"
                f"<|turn>model\n<|channel>thought\n<channel|>")
    if template == "gemma3":
        # Gemma 3 has no system role and no thinking; system text folds into the user turn.
        return f"<start_of_turn>user\n{folded}<end_of_turn>\n<start_of_turn>model\n"
    if template == "qwen3":
        # Qwen3 / Qwen3.5 chat format with thinking disabled: empty <think> block, letter at the next position.
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    if template == "llama3":
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    raise ValueError(f"unknown template {template!r}")


def gemma4_prompt(messages: list[dict]) -> str:  # kept for probe_gemma.py compatibility
    return render_prompt(messages, "gemma4")


def permute_row(row: dict, perm: list[int]) -> dict:
    """Reorder options so new_options[i] = options[perm[i]]; remap gold and target."""
    n = len(row["options"])
    if sorted(perm) != list(range(n)):
        raise ValueError(f"perm {perm} is not a permutation of {n} options")
    out = dict(row)
    out["options"] = [row["options"][j] for j in perm]
    if isinstance(row.get("label"), int):
        out["label"] = perm.index(row["label"])
    if isinstance(row.get("target_distribution"), list):
        out["target_distribution"] = [row["target_distribution"][j] for j in perm]
    return out


def reverse_row(row: dict) -> dict:
    """Reverse option order (option_reversal perturbation)."""
    return permute_row(row, list(range(len(row["options"])))[::-1])


def content_free_row(row: dict, text: str = "N/A") -> dict:
    """Zhao et al. contextual calibration: same question/options, content-free state.
    Zhao et al. average over several placeholders ("N/A", "[MASK]", empty); pass --cf-text."""
    out = dict(row)
    out["state"] = text if text else " "
    return out


def score_row(server: str, row: dict, n_probs: int, cache: bool, template: str = "gemma4") -> dict:
    messages = core.direct_messages(row)
    prompt = render_prompt(messages, template)
    if os.environ.get("TEZ_BACKEND", "llamacpp") == "ollama":
        # fallback backend (top_logprobs capped at 20): same readout, letters floored if missing
        import importlib.util as _iu
        _s = _iu.spec_from_file_location("backend", ROOT / "experiments" / "backend.py"); _b = _iu.module_from_spec(_s); _s.loader.exec_module(_b)  # type: ignore[union-attr]
        t0 = time.perf_counter()
        p, z, pn = _b.score_letters(prompt, len(row["options"]), n_probs, "ollama")
        option_ids = [o["id"] for o in row["options"]]
        return {"id": row["id"], "family": row.get("family"), "group_id": row.get("group_id"), "split": row.get("split"),
                "provenance": row.get("provenance"), "gold": row.get("label"), "target_distribution": row.get("target_distribution"),
                "option_ids": option_ids, "probabilities": [float(x) for x in p], "option_logprobs": [float(x) for x in z],
                "missing_letters": [], "top1_token": None, "top1_logprob": None, "top1_is_slot": True,
                "prompt_n": pn, "prompt_ms": None, "http_ms": (time.perf_counter() - t0) * 1000, "prompt_sha256": core.digest(prompt),
                "readout": "ollama /api/generate top_logprobs(20) at first answer position, restricted to declared answer slots",
                "probability_status": "conditional option score; uncalibrated as decision confidence"}
    body = {
        "prompt": prompt,
        "n_predict": 1,
        "n_probs": n_probs,
        "temperature": 0,
        "cache_prompt": cache,
        "samplers": [],
    }
    t0 = time.perf_counter()
    for attempt in range(2):
        try:
            resp = requests.post(f"{server}/completion", json=body, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == 1:
                raise
            time.sleep(1.0)
            last = exc
    http_ms = (time.perf_counter() - t0) * 1000.0

    tops = data["completion_probabilities"][0]["top_logprobs"]
    lp_by_tok = {}
    for t in tops:
        lp_by_tok.setdefault(t["token"], t["logprob"])  # first (highest) wins
    floor = min(t["logprob"] for t in tops) - 2.0

    option_ids = [o["id"] for o in row["options"]]
    letters = LETTERS[: len(option_ids)]
    logprobs, missing = [], []
    for L in letters:
        if L in lp_by_tok:
            logprobs.append(float(lp_by_tok[L]))
        else:
            logprobs.append(floor)
            missing.append(L)
    probs = core.softmax(logprobs)

    top1 = tops[0]
    timings = data.get("timings", {})
    return {
        "id": row["id"],
        "family": row.get("family"),
        "group_id": row.get("group_id"),
        "split": row.get("split"),
        "provenance": row.get("provenance"),
        "gold": row.get("label"),
        "target_distribution": row.get("target_distribution"),
        "option_ids": option_ids,
        "probabilities": probs,
        "option_logprobs": logprobs,
        "missing_letters": missing,
        "top1_token": top1["token"],
        "top1_logprob": top1["logprob"],
        "top1_is_slot": top1["token"] in letters,
        "prompt_n": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "http_ms": http_ms,
        "prompt_sha256": core.digest(prompt),
        "readout": "llama-server top_logprobs at first answer position, restricted to declared answer slots",
        "probability_status": "conditional option score; uncalibrated as decision confidence",
    }


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8091")
    ap.add_argument("--n-probs", type=int, default=200)
    ap.add_argument("--cache", action="store_true", help="cache_prompt=true (default: fresh scoring)")
    ap.add_argument("--reverse", action="store_true", help="reverse option order on the fly")
    ap.add_argument("--perm", default="", help="explicit option permutation, e.g. 2,0,1 (new[i]=old[perm[i]])")
    ap.add_argument("--content-free", action="store_true", help="replace state with a placeholder (contextual calibration prior)")
    ap.add_argument("--cf-text", default="N/A", help="placeholder for --content-free: N/A | [MASK] | '' (Zhao et al. average all three)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--template", default="gemma4", choices=["gemma4", "gemma4sys", "gemma3", "llama3", "qwen3"])
    args = ap.parse_args()

    data_path = Path(args.data)
    rows = [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    for r in rows:
        core.validate_row(r)

    if os.environ.get("TEZ_BACKEND", "llamacpp") == "ollama":
        props = {"model_path": "ollama:" + os.environ.get("TEZ_OLLAMA_MODEL", "gemma4:12b-it-q8_0"), "model_alias": "ollama"}
    else:
        props = requests.get(f"{args.server}/props", timeout=30).json()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    n_ok = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            r = row
            if args.perm:
                r = permute_row(r, [int(x) for x in args.perm.split(",")])
            elif args.reverse:
                r = reverse_row(r)
            if args.content_free:
                r = content_free_row(r, args.cf_text)
            rec = score_row(args.server, r, args.n_probs, args.cache, args.template)
            rec["variant"] = {"reverse": args.reverse, "perm": args.perm, "content_free": args.content_free,
                              "cf_text": args.cf_text if args.content_free else None, "template": args.template}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
            if i % 24 == 0 or i == len(rows):
                el = time.perf_counter() - started
                print(f"  {i}/{len(rows)}  {el:.0f}s  ({i/el:.2f} dec/s)", flush=True)
    elapsed = time.perf_counter() - started

    recs = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()]
    pms = [r["prompt_ms"] for r in recs if r.get("prompt_ms") is not None]
    manifest = {
        "tag": args.tag,
        "data": str(data_path),
        "data_sha256": sha256_file(data_path),
        "rows": len(recs),
        "server": args.server,
        "model_path": props.get("model_path"),
        "model_alias": props.get("model_alias"),
        "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
        "n_probs": args.n_probs,
        "cache_prompt": args.cache,
        "template": args.template,
        "reverse": args.reverse,
        "perm": args.perm,
        "content_free": args.content_free,
        "cf_text": args.cf_text if args.content_free else None,
        "prompt_version": "semif direct-options-v1 rendered into gemma4 template (thinking off)",
        "wall_seconds": elapsed,
        "decisions_per_second": len(recs) / elapsed if elapsed else None,
        "prompt_ms_median": statistics.median(pms) if pms else None,
        "prompt_ms_p95": sorted(pms)[int(0.95 * (len(pms) - 1))] if pms else None,
        "missing_letter_rows": sum(1 for r in recs if r["missing_letters"]),
        "top1_not_slot_rows": sum(1 for r in recs if not r["top1_is_slot"]),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("rows", "wall_seconds", "decisions_per_second", "prompt_ms_median", "missing_letter_rows", "top1_not_slot_rows")}))


if __name__ == "__main__":
    main()
