"""Probe: can we read full-vocabulary last-position logits from gemma4:12b via llama.cpp,
and does symbol-slot scoring behave (no thinking token, letters are single tokens)?"""
from __future__ import annotations

import json
import math
import os
import sys
import time

GGUF = os.path.expanduser(
    r"~\.ollama\models\blobs\sha256-1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606"
)
LETTERS = "ABCDEFGHIJKLMNOP"


def register_cuda_dlls() -> None:
    """The cu124 llama-cpp-python wheel links cudart64_12/cublas64_12, which live in the
    nvidia-* pip packages and are not on PATH by default on Windows."""
    import glob
    import site

    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def gemma4_prompt(user_text: str) -> str:
    """Gemma 4 generation prompt with thinking disabled: the template ends with an EMPTY
    thought channel, and the answer token is read at the position right after <channel|>."""
    return f"<|turn>user\n{user_text}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"


def main() -> None:
    register_cuda_dlls()
    from llama_cpp import Llama

    t0 = time.perf_counter()
    llm = Llama(model_path=GGUF, n_gpu_layers=-1, n_ctx=4096, logits_all=False, verbose=False)
    print(f"load {time.perf_counter() - t0:.1f}s")
    md = llm.metadata
    tmpl = md.get("tokenizer.chat_template", "")
    print("chat_template present:", bool(tmpl), "| len", len(tmpl))
    print("arch:", md.get("general.architecture"), "| name:", md.get("general.name"))
    print("--- template head ---")
    print(tmpl[:900])
    print("--- template tail ---")
    print(tmpl[-700:])

    # single-token round trip for answer slots (SemIf's _slot_ids check)
    print("\n--- slot tokens ---")
    slot_ids = {}
    for L in LETTERS[:6]:
        ids = llm.tokenize(L.encode(), add_bos=False, special=False)
        back = llm.detokenize(ids).decode()
        ids_sp = llm.tokenize((" " + L).encode(), add_bos=False, special=False)
        print(f"{L!r}: ids={ids} roundtrip={back!r}   ' {L}': ids={ids_sp}")
        slot_ids[L] = ids[0] if len(ids) == 1 else None

    # build a Gemma-style prompt manually (avoid template thinking defaults)
    user = (
        "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
        "Respond with only its uppercase letter, with no explanation or reasoning.\n\n"
        + json.dumps(
            {
                "evidence": "The basket holds a car key, an apple, a hammer, a cloud photo, a pear and a brick.",
                "criterion": "Which listed item is a fruit?",
                "options": [
                    {"letter": "A", "description": "car"},
                    {"letter": "B", "description": "apple"},
                    {"letter": "C", "description": "hammer"},
                    {"letter": "D", "description": "cloud"},
                    {"letter": "E", "description": "pear"},
                    {"letter": "F", "description": "brick"},
                ],
            }
        )
    )
    prompt = gemma4_prompt(user)
    toks = llm.tokenize(prompt.encode(), add_bos=True, special=True)
    print("prompt tail tokens:", [llm.detokenize([t]).decode(errors="replace") for t in toks[-6:]])
    print(f"\nprompt tokens: {len(toks)}")

    llm.reset()
    t1 = time.perf_counter()
    llm.eval(toks)
    dt = time.perf_counter() - t1
    logits = llm.scores[llm.n_tokens - 1]  # full vocab, last position
    import numpy as np

    logits = np.asarray(logits, dtype=np.float64)
    print(f"eval {dt*1000:.0f} ms  | vocab {logits.shape[0]}")
    top = np.argsort(-logits)[:12]
    print("top-12 next tokens:")
    for i in top:
        print(f"   {llm.detokenize([int(i)]).decode(errors='replace')!r:16s} logit={logits[i]:.2f}")
    letters = list("ABCDEF")
    sel = np.array([logits[slot_ids[L]] for L in letters])
    p = np.exp(sel - sel.max())
    p /= p.sum()
    print("slot softmax:", {L: round(float(x), 4) for L, x in zip(letters, p)})
    print("slot logits :", {L: round(float(x), 2) for L, x in zip(letters, sel)})


if __name__ == "__main__":
    main()
