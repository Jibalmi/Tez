"""Is llama-server's prompt cache safe with the hybrid Qwen3.5 GGUFs (llama.cpp b11100)?

The runtime switches caching off for Qwen3.5 because partial prefix reuse crashed b11100 (tez/backends.py). This
checks, on a fresh server per scenario (a crash must not contaminate the next one), which reuse patterns survive and
whether reuse actually saves work (`timings.prompt_n` / `cache_n`):

  identical        the same prompt twice
  extend_n0        prompt P (n_predict 0, only evaluated), then P + " more words"   (exact-prefix extension)
  extend_n1        the same with n_predict 1 (a sampled token follows P in the slot)
  voice_letters    the voice loop's per-word prompts: prefix + '"open' + tail, then prefix + '"open notes' + tail
                   (the new prompt shares the prefix but not the old tail: partial reuse, needs a rollback)
  embed_extend     /embedding of a stem, then of the stem + one word (exact extension through /embedding)
  diverge          two typed-decisions prompts that share only the instruction prefix (partial reuse)

Reproduce (GPU free of other models; uses port 8095):
  python experiments/speed_qwen_cache_check.py --model tools/models/Qwen3.5-4B-Q8_0-L24.gguf --tag L24
  python experiments/speed_qwen_cache_check.py --model tools/models/Qwen3.5-4B-Q8_0.gguf --tag L32
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import speed_common as sc
import speed_voice as sv


def post(url, path, body):
    t = time.perf_counter()
    try:
        r = sc.S.post(f"{url}{path}", json=body, timeout=120)
        ms = (time.perf_counter() - t) * 1000
        d = r.json()
        if isinstance(d, list):
            d = d[0]
        tm = d.get("timings", {}) if isinstance(d, dict) else {}
        return dict(ok=r.status_code == 200, status=r.status_code, ms=round(ms, 1), prompt_n=tm.get("prompt_n"),
                    cache_n=tm.get("cache_n"), prompt_ms=tm.get("prompt_ms"),
                    error=(d.get("error") if isinstance(d, dict) else None))
    except Exception as exc:  # noqa: BLE001
        return dict(ok=False, error=f"{exc.__class__.__name__}: {str(exc)[:200]}", ms=round((time.perf_counter() - t) * 1000, 1))


def alive(url):
    try:
        return sc.S.get(f"{url}/health", timeout=5).json().get("status") == "ok"
    except Exception:  # noqa: BLE001
        return False


def scenarios():
    tdc = sc.typed_decisions("test")[:2]
    stem = sv.prompt_stem("open", "qwen3")
    comp = lambda p, n=1: {"prompt": p, "n_predict": n, "n_probs": 20, "temperature": 0, "samplers": [], "cache_prompt": True}  # noqa: E731
    return {
        "identical": [("/completion", comp(sv.prompt_letters("open notes", "qwen3"))),
                      ("/completion", comp(sv.prompt_letters("open notes", "qwen3")))],
        "extend_n0": [("/completion", comp(stem, 0)), ("/completion", comp(stem + " notes and type hello", 0)),
                      ("/completion", comp(stem + " notes and type hello world", 1))],
        "extend_n1": [("/completion", comp(stem, 1)), ("/completion", comp(stem + " notes and type hello", 1))],
        "voice_letters": [("/completion", comp(sv.prompt_letters("open", "qwen3"))),
                          ("/completion", comp(sv.prompt_letters("open notes", "qwen3"))),
                          ("/completion", comp(sv.prompt_letters("open notes and", "qwen3")))],
        "embed_extend": [("/embedding", {"content": stem, "embd_normalize": -1}),
                         ("/embedding", {"content": stem + " notes", "embd_normalize": -1}),
                         ("/embedding", {"content": stem + " notes and type", "embd_normalize": -1})],
        "diverge": [("/completion", comp(sc.prompt_today(tdc[0], "qwen3"))), ("/completion", comp(sc.prompt_today(tdc[1], "qwen3")))],
        # two user turns: the server checkpoints the recurrent state at user-message starts, so a second prompt that
        # diverges inside the last user turn restores a checkpoint (the partial-reuse path most likely to crash)
        "multiturn_checkpoint": [("/completion", comp(multiturn(tdc[0]["instructions"], sc.state_text(tdc[0]["state"])))),
                                 ("/completion", comp(multiturn(tdc[0]["instructions"], sc.state_text(tdc[1]["state"])))),
                                 ("/completion", comp(multiturn(tdc[1]["instructions"], sc.state_text(tdc[1]["state"]))))],
    }


def multiturn(question: str, state: str) -> str:
    pre, post = sc.TEMPLATES["qwen3"]
    turn1 = (f"{pre}You are a decision engine. Answer each question with one letter.<|im_end|>\n<|im_start|>assistant\n"
             "<think>\n\n</think>\n\nUnderstood.<|im_end|>\n")
    return f"{turn1}{pre}Question: {question}\n\nInput:\n{state}{post}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--extra", default="", help="extra llama-server flags, space separated")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    res = {"meta": {"script": "experiments/speed_qwen_cache_check.py", "model": args.model, "extra": args.extra,
                    "date": time.strftime("%Y-%m-%dT%H:%M:%S")}, "scenarios": {}}
    for name, steps in scenarios().items():
        log = Path(os.environ.get("TEMP", ".")) / f"qwen_cache_{args.tag}_{name}.log"
        p = sc.start_server(sc.ROOT / args.model if not os.path.isabs(args.model) else args.model, args.port,
                            extra=args.extra.split() if args.extra else [], log=log)
        try:
            out = []
            for path, body in steps:
                r = post(url, path, body)
                r["path"] = path
                r["server_alive_after"] = alive(url)
                out.append(r)
                if not r["server_alive_after"]:
                    break
            time.sleep(0.5)
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
            crashed = not all(o["server_alive_after"] for o in out)
            res["scenarios"][name] = {"steps": out, "crashed": crashed, "pr": sc.props(url).get("build_info") if not crashed else None,
                                      "log_tail": tail if crashed else [l for l in tail if "cache" in l.lower() or "checkpoint" in l.lower()][-6:]}
            print(name, "CRASH" if crashed else "ok", [(o.get("prompt_n"), o.get("cache_n"), o.get("ms")) for o in out], flush=True)
        finally:
            sc.stop_server(p)
    sc.dump(f"qwen_cache_check_{args.tag}.json", res)


if __name__ == "__main__":
    main()
