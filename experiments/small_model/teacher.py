"""Pilot day 2b: Gemma 4 12B reads every assigned decision, in two option orders, in process.

    python experiments/small_model/teacher.py --minutes 100        # resumable; stops before the deadline

Each state's 5 questions x 2 orders (original, reversed) = 10 state-first prompts go into ONE InprocBackend.read_many
call: the state is evaluated once and the 10 suffixes in one batched decode (plan §5b's rate: 289 ms per state for
10 readings on ticket-sized states). Stored per question: the letter log-probabilities (full-vocabulary normaliser,
T = 1) in both orders, the option keys in the original order, and whether __none__ was added (20 % of choice
questions, plan §4). The student's target is built later (dataset.py): the average of the two orders' distributions,
each at the §5c default temperature of its question type.

Checkpoints: runs/pilot/teacher/part-NNNN.jsonl every 25 states, published atomically; a rerun skips saved states.
Cooling timeouts save the current buffer and exit 3 so the driver releases the lock and re-queues.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from common import GGUF_12B, LLAMA_LIB, RESULTS, RUNS, THERMAL, ThermalCooldownTimeout, dump, read_jsonl, thermal_guard

OUT = RUNS / "teacher"
NONE_SHARE = 0.2
CHECKPOINT_STATES = 25


def save_part(part: int, rows: list[dict]) -> None:
    """Only complete, flushed shards are visible to a resumed teacher."""
    path = OUT / f"part-{part:04d}.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def shared_gpu_mb() -> int:
    """This process's shared (spilled) GPU memory in MB, from the Windows GPU counters (C:/temp/gpu_lock.py snapshot)."""
    import os
    import subprocess
    import sys as _sys
    try:
        out = subprocess.run([_sys.executable, "C:/temp/gpu_lock.py", "snapshot"], capture_output=True, text=True,
                             timeout=60).stdout
        procs = json.loads(out)["processes"]
        return sum(p["mb"] for p in procs if p["pid"] == os.getpid() and p["kind"] == "shared")
    except Exception:  # noqa: BLE001
        return -1


def add_none(state_id: str, qid: str) -> bool:
    """Deterministic 20 % draw of the choice questions that get __none__."""
    h = int(hashlib.sha256(f"{state_id}|{qid}".encode()).hexdigest()[:8], 16)
    return (h / 0xFFFFFFFF) < NONE_SHARE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=100, help="stop starting new states after this long")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-ctx", type=int, default=2048,
                    help="KV cells (swa_full). 4,096 spilled out of VRAM beside 1.9 GB of desktop VRAM on 2026-09-26")
    args = ap.parse_args()
    from tez.inproc import InprocBackend
    from tez.prompt import STATE_FIRST, build_prompt
    from tez.schema import parse_question

    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    for p in sorted(OUT.glob("part-*.jsonl")):
        done.update(r["id"] for r in read_jsonl(p))
    pool = {r["id"]: r for r in read_jsonl(RUNS / "pool.jsonl")}
    todo = [a for a in read_jsonl(RUNS / "assign.jsonl") if a["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} states already read, {len(todo)} to go", flush=True)
    if not todo:
        return 0

    import os
    import psutil
    from tez.inproc import find_library, load_api
    api = load_api(find_library(LLAMA_LIB))
    default_params = api.llama_model_default_params

    def no_mmap_params():
        """LLAMA_LOAD_MODE_NONE (0), lazy mode off (0): the weights go to the GPU and no 12.7 GB mapped copy of the
        GGUF stays resident in RAM (run_heavy killed a segment at 14:37 when free RAM fell below 4 GB)."""
        mp = default_params()
        mp.load_mode = 0
        mp.lazy_mode = 0
        return mp

    api.llama_model_default_params = no_mmap_params
    be = InprocBackend(GGUF_12B, template="gemma4", lib=LLAMA_LIB, n_ctx=args.n_ctx, n_seq_max=16,
                       n_threads=int(os.environ.get("RUN_HEAVY_THREADS", "6")), api=api)
    t_load = time.time()
    be.load()
    load_s = time.time() - t_load
    mem = {"rss_gb_after_load": round(psutil.Process().memory_info().rss / 2 ** 30, 2),
           "free_ram_gb_after_load": round(psutil.virtual_memory().available / 2 ** 30, 1)}
    print(f"loaded in {load_s:.0f} s; {mem}", flush=True)
    part = max((int(p.stem.split("-")[1]) for p in OUT.glob("part-*.jsonl")), default=-1) + 1
    buf, t0, ms, n_read = [], time.time(), [], 0
    deadline = t0 + args.minutes * 60
    stopped_for = None
    try:
        for a in todo:
            if time.time() > deadline:
                print("deadline reached", flush=True)
                break
            thermal_guard(force=True, max_s=min(60.0, max(0.0, deadline - time.time())))
            if time.time() >= deadline:
                print("deadline reached during cooling", flush=True)
                break
            state = pool[a["id"]]["state"]
            prompts, ks, meta = [], [], []
            for qd in a["questions"]:
                q = parse_question(qd["qid"], {k: qd[k] for k in ("type", "instructions", "criteria") if k in qd})
                ab = q.type == "choice" and add_none(a["id"], qd["qid"]) and len(q.criteria) < 26
                opts = q.options(ab)
                for order in (opts, opts[::-1]):
                    prompts.append(build_prompt(q, state, "gemma4", order, None, STATE_FIRST))
                    ks.append(len(order))
                meta.append({"qid": qd["qid"], "type": q.type, "keys": q.keys(ab), "abstain": ab})
            t = time.perf_counter()
            res = be.read_many(prompts, ks)
            dt_ms = (time.perf_counter() - t) * 1000
            ms.append(dt_ms)
            qs_out = []
            for j, m in enumerate(meta):
                zo = res[2 * j][0].logits
                zr = res[2 * j + 1][0].logits
                qs_out.append({**m, "z_orig": [round(float(x), 4) for x in zo], "z_rev": [round(float(x), 4) for x in zr]})
            buf.append({"id": a["id"], "questions": qs_out, "ms": round(dt_ms, 1),
                        "prompt_tokens": sum(r[0].tokens for r in res), "batch_prompt_n": res[0][0].timings["batch_prompt_n"]})
            n_read += 1
            if n_read == 3:
                # llama.cpp keeps the token-embedding table (262,144 x 3,840 at Q8_0, about 1.07 GB) in pinned host
                # memory, which Windows counts as shared GPU memory: ~1,130 MB is the design, not a spill.
                spill = shared_gpu_mb()
                if spill > 1700:
                    raise SystemExit(f"spilled into shared GPU memory ({spill} MB, 1,130 expected): lower --n-ctx")
                print(f"  shared GPU memory {spill} MB (1,130 expected: the host-side token embeddings)", flush=True)
            if n_read % 50 == 0:
                el = time.time() - t0
                print(f"  {n_read} states in {el:.0f} s: p50 {np.median(ms):.0f} ms/state, thermal {THERMAL}", flush=True)
            if len(buf) >= CHECKPOINT_STATES:
                save_part(part, buf)
                part += 1
                buf = []
            if n_read % 200 == 0:
                el = time.time() - t0
                print(f"  {n_read} states in {el:.0f} s: p50 {np.median(ms):.0f} ms/state, {n_read / el:.2f} states/s",
                      flush=True)
    except ThermalCooldownTimeout as exc:
        stopped_for = str(exc)
        print(f"cooling stop: {exc}; checkpointing and releasing the lock", flush=True)
    finally:
        try:
            if buf:
                save_part(part, buf)
        finally:
            be.close()
    el = time.time() - t0
    stats_path = RESULTS / "teacher_timing.json"
    prev = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {"runs": []}
    prev["runs"].append({"states": n_read, "seconds": round(el, 1), "load_s": round(load_s, 1),
                         "ms_per_state_p50": round(float(np.median(ms)), 1) if ms else None,
                         "ms_per_state_p95": round(float(np.percentile(ms, 95)), 1) if ms else None,
                         "readings_per_state": 10, "n_ctx": args.n_ctx, **mem, "load_mode": "none (no mmap)",
                         "backend": "tez.inproc InprocBackend, llama.cpp b11100, state_first, read_many",
                         "stopped_for": stopped_for, "checkpoint_states": CHECKPOINT_STATES,
                         "thermal": dict(THERMAL)})
    dump(stats_path, prev)
    print(f"read {n_read} states in {el:.0f} s", flush=True)
    return 3 if stopped_for else 0


if __name__ == "__main__":
    raise SystemExit(main())
