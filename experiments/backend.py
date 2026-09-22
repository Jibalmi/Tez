"""Scoring backends for the one-pass letter readout.

  llamacpp  llama-server /completion (n_probs up to 200, --swa-full prefix reuse)     -- preferred
  ollama    /api/generate raw prompt with logprobs (top_logprobs capped at 20)         -- fallback when
            the llama-server binary is unavailable (Avast quarantines it on this machine)

Both return (probs over the first k letters, logits, prompt_tokens_evaluated). Letters missing
from the returned top-k are floored 2 nats below the lowest returned logprob, as everywhere else.
"""
from __future__ import annotations

import os
import string

import numpy as np
import requests

LETTERS = string.ascii_uppercase
SESSION = requests.Session()
BACKEND = os.environ.get("TEZ_BACKEND", "llamacpp")
LLAMACPP = os.environ.get("TEZ_SERVER", "http://127.0.0.1:8091")
OLLAMA = os.environ.get("TEZ_OLLAMA", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("TEZ_OLLAMA_MODEL", "gemma4:12b-it-q8_0")


def _from_tops(tops, k):
    lp = {}
    for t in tops:
        lp.setdefault(t["token"], t["logprob"])
    floor = min(t["logprob"] for t in tops) - 2.0
    z = np.array([lp.get(LETTERS[i], floor) for i in range(k)], dtype=float)
    p = np.exp(z - z.max()); p /= p.sum()
    missing = [LETTERS[i] for i in range(k) if LETTERS[i] not in lp]
    return p, z, missing


def score_letters(prompt: str, k: int, n_probs: int = 200, backend: str | None = None):
    backend = backend or BACKEND
    if backend == "llamacpp":
        body = {"prompt": prompt, "n_predict": 1, "n_probs": n_probs, "temperature": 0, "cache_prompt": True, "samplers": []}
        d = SESSION.post(f"{LLAMACPP}/completion", json=body, timeout=300).json()
        p, z, _ = _from_tops(d["completion_probabilities"][0]["top_logprobs"], k)
        return p, z, d.get("timings", {}).get("prompt_n")
    if backend == "ollama":
        body = {"model": OLLAMA_MODEL, "prompt": prompt, "raw": True, "stream": False, "keep_alive": "60m",
                "options": {"num_predict": 1, "temperature": 0, "num_ctx": int(os.environ.get("TEZ_OLLAMA_CTX", "8192"))}, "logprobs": True, "top_logprobs": 20}
        d = SESSION.post(f"{OLLAMA}/api/generate", json=body, timeout=600).json()
        p, z, _ = _from_tops(d["logprobs"][0]["top_logprobs"], k)
        return p, z, d.get("prompt_eval_count")
    raise ValueError(backend)


def generate(prompt: str, n_predict: int, stop: list[str], backend: str | None = None):
    """Greedy generation (for the thinking-budget experiment). Returns (text, n_tokens)."""
    backend = backend or BACKEND
    if backend == "llamacpp":
        d = SESSION.post(f"{LLAMACPP}/completion", json={"prompt": prompt, "n_predict": n_predict, "temperature": 0, "cache_prompt": True, "stop": stop}, timeout=600).json()
        return d["content"], d.get("tokens_predicted", 0)
    d = SESSION.post(f"{OLLAMA}/api/generate", json={"model": OLLAMA_MODEL, "prompt": prompt, "raw": True, "stream": False, "keep_alive": "60m",
                                                    "options": {"num_predict": n_predict, "temperature": 0, "stop": stop}}, timeout=600).json()
    return d["response"], d.get("eval_count", 0)
