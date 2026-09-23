"""Where does a one-pass decision's latency go? Same 100 typed-decisions prompts, one server at a time, caching off:
  /completion with n_predict 1 and n_probs 0 / 20 / 200 (the letter readout needs the letters' log-probs),
  /embedding (the probe readout: last-token state).
If n_probs dominates, the probe readout is not only more accurate but several times faster.

  py experiments/latency_breakdown.py --out results/latency_breakdown.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("hp", ROOT / "experiments" / "hidden_probe.py")
hp = importlib.util.module_from_spec(_s); _s.loader.exec_module(hp)  # type: ignore[union-attr]
EXE = r"C:\temp\llamacpp\llama-server.exe"
GEMMA = "C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795"
S = requests.Session()


def wrap(c, tmpl):
    u = hp.prompt_text(c)
    if tmpl == "qwen3":
        return f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return f"<|turn>user\n{u}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"


def start(path, port, extra, log):
    p = subprocess.Popen([EXE, "-m", path, "-ngl", "99", "-c", "4096", "-b", "512", "--port", str(port), "--host", "127.0.0.1", "-np", "1", "--no-webui",
                          "--embeddings", "--pooling", "last"] + extra, cwd=r"C:\temp\llamacpp", stdout=open(log, "w"), stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            if S.get(f"http://127.0.0.1:{port}/health", timeout=2).json().get("status") == "ok":
                return p
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    p.kill(); raise RuntimeError("server did not start")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma-separated model names to run (default: all)")
    args = ap.parse_args()
    cases = hp.load_split("test")[: args.n]
    models = [("Qwen3.5-4B, 24 of 32 blocks", str(ROOT / "tools/models/Qwen3.5-4B-Q8_0-L24.gguf"), "qwen3", []),
              ("Qwen3.5-4B, all 32 blocks", str(ROOT / "tools/models/Qwen3.5-4B-Q8_0.gguf"), "qwen3", []),
              ("Gemma 4 12B Q8", GEMMA, "gemma4", ["--swa-full"])]
    models += [("Qwen3.5-4B, 20 of 32 blocks", str(ROOT / "tools/models/Qwen3.5-4B-Q8_0-L20.gguf"), "qwen3", []),
               ("Qwen3.5-4B, 29 of 32 blocks", str(ROOT / "tools/models/Qwen3.5-4B-Q8_0-L29.gguf"), "qwen3", [])]
    if args.only:
        keep = [x.strip() for x in args.only.split(";")]; models = [m for m in models if m[0] in keep]
    res = json.loads(Path(args.out).read_text(encoding="utf-8")) if Path(args.out).exists() else {}
    url = f"http://127.0.0.1:{args.port}"
    for name, path, tmpl, extra in models:
        proc = start(path, args.port, extra, Path(os.environ.get("TEMP", ".")) / "latency_breakdown.log")
        try:
            R = {}
            for label, fn in (
                ("completion, n_probs 0", lambda pr: S.post(f"{url}/completion", json={"prompt": pr, "n_predict": 1, "n_probs": 0, "temperature": 0, "cache_prompt": False, "samplers": []}, timeout=300)),
                ("completion, n_probs 20", lambda pr: S.post(f"{url}/completion", json={"prompt": pr, "n_predict": 1, "n_probs": 20, "temperature": 0, "cache_prompt": False, "samplers": []}, timeout=300)),
                ("completion, n_probs 200", lambda pr: S.post(f"{url}/completion", json={"prompt": pr, "n_predict": 1, "n_probs": 200, "temperature": 0, "cache_prompt": False, "samplers": []}, timeout=300)),
                ("embedding (probe readout)", lambda pr: S.post(f"{url}/embedding", json={"content": pr, "embd_normalize": -1}, timeout=300)),
            ):
                for c in cases[:3]:
                    fn(wrap(c, tmpl))
                ms = []
                for c in cases:
                    t0 = time.perf_counter(); fn(wrap(c, tmpl)); ms.append((time.perf_counter() - t0) * 1000)
                R[label] = dict(p50=float(np.median(ms)), p90=float(np.percentile(ms, 90)))
                print(f"{name:32s} {label:28s} p50 {R[label]['p50']:.1f} ms  p90 {R[label]['p90']:.1f}", flush=True)
            res[name] = R
        finally:
            proc.kill(); time.sleep(3)
        Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
