"""Shared pieces of the small-model pilot (docs/plans/tez-small-model.md, week 1).

Paths, the state normalisation used by the leakage guard, hashing, MinHash shingles, and the evaluation suites'
states (every split, every row the repo has results for). Large artefacts live in runs/pilot/ (gitignored); code,
small result files and the licence ledger are committed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

THREADS = int(os.environ.get("RUN_HEAVY_THREADS", "6"))   # the machine's cap for heavy jobs (C:/temp/run_heavy.py)
if "torch" in sys.modules:           # never import torch here: the in-process teacher loads llama.cpp's libomp, and
    sys.modules["torch"].set_num_threads(THREADS)   # torch's libiomp5md first makes it abort (OMP error 15)

RUNS = ROOT / "runs" / "pilot"                   # gitignored: weights, caches, datasets
DATA = ROOT / "data" / "small_model"             # committed: eval hashes (small), licence ledger copy
RESULTS = ROOT / "results" / "pilot"             # committed: metrics, manifests
for _d in (RUNS, DATA, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

GGUF_12B = "C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795"
LLAMA_LIB = "C:/temp/llamacpp"
LOCK_OWNER = "projects-0c-pilot"

# MASSIVE's 51 locales in the mteb/amazon_massive_intent configs; the plan's 8 held-out languages (pilot: km)
HELD_OUT_LANGS = ["am", "cy", "is", "jv", "km", "mn", "my", "sw"]
MASSIVE11 = ["en", "de", "fr", "es", "ja", "zh-CN", "ar", "hi", "th", "ko", "km"]


# ------------------------------------------------------------------------------------------ normalisation and hashes
def leaves(value: Any) -> Iterator[str]:
    """String and number leaves of a state, in document order (keys are not content)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, bool) or value is None:
        return
    elif isinstance(value, (int, float)):
        yield str(value)
    elif isinstance(value, dict):
        for v in value.values():
            yield from leaves(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from leaves(v)


def norm_text(state: Any) -> str:
    """The normalised content of a state: its leaves joined, NFKC, case-folded, whitespace collapsed. Two states
    that carry the same text under different field names (MASSIVE's {"utterance": t} and a pool row's {"text": t})
    normalise to the same string."""
    s = " ".join(leaves(state))
    s = unicodedata.normalize("NFKC", s).casefold()
    return re.sub(r"\s+", " ", s).strip()


def sha(text: str, n: int = 16) -> str:
    """SHA-256 hex, truncated to n characters (64 bits by default: collision odds are negligible at this scale)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def state_hashes(state: Any) -> tuple[str, str]:
    """(exact, normalised): the SHA-256 of tez.schema.state_key(state) and of norm_text(state)."""
    from tez.schema import state_key
    return sha(state_key(state)), sha(norm_text(state))


def shingles(text: str, n: int = 5) -> set[str]:
    """Character 5-grams of normalised text (character, not word, grams: ja, zh and th have no spaces)."""
    t = text if len(text) >= n else text.ljust(n)
    return {t[i:i + n] for i in range(len(t) - n + 1)}


MINHASH_PERM = 64
MINHASH_SEED = 13


def minhash(text: str):
    from datasketch import MinHash
    m = MinHash(num_perm=MINHASH_PERM, seed=MINHASH_SEED, scheme="affine32")   # datasketch 2.0's default, uint32
    for s in shingles(text):
        m.update(s.encode("utf-8"))
    return m


def file_sha256(path: str | os.PathLike, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dump(path: str | os.PathLike, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------------------------------ evaluation states
def eval_states() -> Iterator[tuple[str, Any]]:
    """(source, state) for every evaluation state the plan lists: every split of typed-decisions, every task under
    results/h2h/ (rebuilt with bench_h2h.build_task, which is how those rows were made), Laya's seven application
    workflows, all 51 MASSIVE test sets (full, not only the 100 rows scored), XNLI's test sets, SemIf, voice and the
    JevBench tiers. The evaluation-only NC sets (toxic-chat, customer-support-tickets) enter through Laya's workflows."""
    from datasets import get_dataset_config_names, load_dataset

    for split in ("train", "test"):
        for r in load_dataset("LocalLLaMA/typed-decisions", "all", split=split):
            try:
                st = json.loads(r["state"])
            except Exception:  # noqa: BLE001
                st = r["state"]
            yield f"typed_decisions:{split}:{r['workflow']}", st

    import bench_h2h
    for task in ("ag_news", "emotion", "banking77", "sst5", "boolq", "prompt_injections"):
        for c in bench_h2h.build_task(task, 400):
            yield f"h2h:{task}", c["state"]
    for lang in bench_h2h.XNLI_LANGS:
        for r in load_dataset("xnli", lang, split="test"):
            yield f"xnli:{lang}", {"premise": r["premise"], "hypothesis": r["hypothesis"]}

    for r in read_jsonl(ROOT / "results" / "vs_laya" / "apps_cases.jsonl"):
        yield f"laya_apps:{r['task']}", r["state"]

    for lang in get_dataset_config_names("mteb/amazon_massive_intent"):
        for r in load_dataset("mteb/amazon_massive_intent", lang, split="test"):
            yield f"massive:{lang}", {"utterance": r["text"]}

    for name in ("authored144", "perturbations108"):
        for r in read_jsonl(ROOT / "data" / "semif" / f"{name}.jsonl"):
            yield f"semif:{name}", r["state"]
    for r in read_jsonl(ROOT / "data" / "voice" / "commands.jsonl"):
        yield "voice", r["text"]
    for tier in ("original", "easy", "hard"):
        for r in read_jsonl(ROOT / "data" / "jevbench" / f"{tier}.jsonl"):
            yield f"jevbench:{tier}", r.get("state", r.get("input", r))


# ------------------------------------------------------------------------------------------ thermal governor
# This laptop's GPU climbs from 70 C to 89 C in about 5 minutes of sustained decode (2026-09-26 13:10-13:15), and
# C:/temp/run_heavy.py kills any job whose GPU stays above 88 C for 15 s. Every GPU loop of the pilot calls
# thermal_guard(): every `every` seconds it reads the temperature, and at `hot` or above it pauses until `cool`.
# The first setting (80 / 72) left the writer paused for its full 10 minutes at 16:10: the idle GPU in a chassis
# heated by other sessions' CPU jobs cools only about 1 C per 20 s from 80 C, and idles at 82-84 C at 20:30.
THERMAL = {"checks": 0, "pauses": 0, "paused_s": 0.0, "max_c": 0}
_THERMAL_LAST = [0.0]


def gpu_temp() -> int | None:
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        return int(out.strip())
    except Exception:  # noqa: BLE001
        return None


def thermal_guard(hot: int = 87, cool: int = 85, every: float = 3.0, force: bool = False, max_s: float = 60) -> float:
    """Pause (seconds returned) while the GPU is at `hot` C or above, until it is back to `cool` (at most max_s).
    87 / 85 C, checked every 3 s, at most 60 s: the NVIDIA driver's own target is 87 C (software slowdown) and
    run_heavy kills only after 15 s above 88 C. With other sessions holding the CPU near 60 % the idle GPU sits at
    82-84 C (2026-09-26 20:30); at 86 / 82 / 180 s the teacher spent 388 of its first 506 s paused."""
    import time as _t
    now = _t.time()
    if not force and now - _THERMAL_LAST[0] < every:
        return 0.0
    _THERMAL_LAST[0] = now
    t = gpu_temp()
    THERMAL["checks"] += 1
    if t is None:
        return 0.0
    THERMAL["max_c"] = max(THERMAL["max_c"], t)
    if t < hot:
        return 0.0
    t0 = _t.time()
    while True:
        _t.sleep(3)
        t = gpu_temp()
        if t is None or t <= cool or _t.time() - t0 > max_s:
            break
    dt = _t.time() - t0
    THERMAL["pauses"] += 1
    THERMAL["paused_s"] += dt
    _THERMAL_LAST[0] = _t.time()
    return dt
