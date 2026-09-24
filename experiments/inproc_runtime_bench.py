"""Many questions about one state through the runtime itself: a running `tez serve`, with the in-process backend or a
llama-server behind it (results/speed/inproc_runtime_*.json, results/speed/http_runtime_*.json).

The protocol and pacing are experiments/speed_multiq.py's: Laya's protocol (the payout ticket, a NEW ticket id every
call, so the state is never a cache hit), distinct questions alternating a 3-option choice and a noul
(speed_common.laya_questions(n, unique=True)), 1 / 5 / 10 / 50 questions per call, warm-up calls then timed calls; after
every call the GPU idles max(--gap, --duty x the call's time), and each size starts once the GPU is at or below --cool
degrees (the laptop GPU idles at 82-85 C and throttles). The difference: every call is a POST /v1/systemone to tez serve
with the server's default layout (auto: state first for two or more questions), so the numbers include the runtime.

  laya   latency per call (p50 / p95), per question, the server's own latency_ms and backend time (Server-Timing), the
         layout it reports (X-Tez-Layout), nvidia-smi telemetry per size, and gpu_lock snapshots before and after
  td     the typed-decisions test split (400 rows x 5 questions = 2,000 decisions), one call per row, zero-shot; accuracy,
         and per decision agreement with another run's rows (--compare, e.g. the HTTP state-first rows of the speed study)
         with McNemar's test and the gap between the letter logits (recovered from the probabilities and the reported
         temperature, relative to the top letter)

Reproduce (GPU lock held; the production llama-server on :8091 for the HTTP arm, stopped for the in-process arm):
  python -m tez serve --backend http://127.0.0.1:8091 --template gemma4 --port 8788
  python experiments/inproc_runtime_bench.py laya --tez http://127.0.0.1:8788 --kind http --llama-port 8091 --tag http_runtime
  python -m tez serve --backend inproc:<gguf> --template gemma4 --llama-lib C:/temp/llamacpp --port 8789
  python experiments/inproc_runtime_bench.py laya --tez http://127.0.0.1:8789 --kind inproc --tag inproc_runtime
  python experiments/inproc_runtime_bench.py td --tez http://127.0.0.1:8789 --kind inproc --tag inproc_runtime \
      --compare td_rows_prod_statefirst_http.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speed_common as sc  # noqa: E402


def listening_pid(port: int) -> int | None:
    """The process listening on a local port (Windows)."""
    ps = f"(Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess"
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60).stdout.split()
    return int(out[0]) if out and out[0].isdigit() else None


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(sc.ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=30).stdout.strip() or None
    except OSError:
        return None


def server_timing(header: str | None) -> dict:
    out = {}
    for part in (header or "").split(","):
        m = re.match(r"\s*([a-z]+);dur=([0-9.]+)", part)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def post(sess, url: str, body: dict) -> tuple[float, dict, dict]:
    t = time.perf_counter()
    r = sess.post(f"{url}/v1/systemone", json=body, timeout=900)
    wall = (time.perf_counter() - t) * 1000.0
    if r.status_code != 200:
        raise RuntimeError(f"tez serve answered {r.status_code}: {r.text[:300]}")
    j = r.json()
    st = server_timing(r.headers.get("server-timing"))
    return wall, j, {"layout": r.headers.get("x-tez-layout"), "tez_ms": j["tez"]["latency_ms"], "backend_ms": st.get("backend"),
                     "tokens": j["usage"]["input_tokens"]}


class CpuMonitor:
    """Total CPU load (Windows typeperf, one sample a second) during a measurement: llama.cpp's CPU threads meet at
    barriers even with every layer on the GPU, so another process saturating the CPU slows every call."""

    def __enter__(self):
        import threading
        self.samples: list[float] = []
        self.p = subprocess.Popen(["typeperf", r"\Processor(_Total)\% Processor Time", "-si", "1"], stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True)

        def read():
            for line in self.p.stdout:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    try:
                        self.samples.append(float(parts[1].strip('"')))
                    except ValueError:
                        pass
        threading.Thread(target=read, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.p.terminate()

    def summary(self) -> dict:
        a = np.asarray(self.samples, float)
        if not len(a):
            return {"samples": 0}
        return {"samples": int(len(a)), "cpu_pct_p50": round(float(np.median(a)), 1),
                "cpu_pct_p90": round(float(np.percentile(a, 90)), 1), "cpu_pct_max": round(float(a.max()), 1)}


BUSY_PS = ("$a = Get-Process | Select-Object Id,ProcessName,CPU; Start-Sleep -Seconds 1; "
           "$b = Get-Process | Select-Object Id,ProcessName,CPU; "
           "$b | ForEach-Object { $x = $_; $o = $a | Where-Object { $_.Id -eq $x.Id }; "
           "if ($o -and $x.CPU) { '{0}|{1}|{2:N2}' -f $x.Id,$x.ProcessName,($x.CPU - $o.CPU) } }")


def busy_processes(top: int = 5) -> list[dict]:
    """The processes using the most CPU over one second (pid, name, cores used), for the record."""
    try:
        rows = subprocess.run(["powershell", "-NoProfile", "-Command", BUSY_PS], capture_output=True, text=True,
                              timeout=120).stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = []
    for r in rows:
        pid, name, cores = (r.split("|") + ["", "", ""])[:3]
        try:
            out.append({"pid": int(pid), "name": name, "cores": float(cores.replace(",", ""))})
        except ValueError:
            continue
    return sorted(out, key=lambda x: -x["cores"])[:top]


class Guard:
    """gpu_lock snapshots (and Ollama's loaded models) before and after, judged with speed_common's contamination rule;
    our own processes are tez serve and, for the HTTP arm, the llama-server behind it."""

    def __init__(self, own: set[int]):
        self.own = own

    def __enter__(self):
        self.before = sc.lock_snapshot()
        return self

    def __exit__(self, *exc):
        self.after = sc.lock_snapshot()

    def result(self) -> dict:
        rb, ra = sc.contamination(self.before, self.own), sc.contamination(self.after, self.own)
        return {"own_pids": sorted(self.own), "before": self.before, "after": self.after, "contaminated": bool(rb or ra),
                "reasons_before": rb, "reasons_after": ra}


def meta_of(a: argparse.Namespace) -> dict:
    s = sc.session()
    health = s.get(f"{a.tez}/healthz", timeout=60).json()
    meta = {"script": "experiments/inproc_runtime_bench.py", "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": vars(a),
            "git_head": git_head(), "python": platform.python_version(), "tez_healthz": health}
    if a.llama_port:
        pr = sc.props(f"http://127.0.0.1:{a.llama_port}")
        meta.update(llama_model_path=pr.get("model_path"), llama_build=pr.get("build_info"), total_slots=pr.get("total_slots"))
    return meta


def own_pids(a: argparse.Namespace) -> set[int]:
    pids = {listening_pid(sc.port_of(a.tez))}
    if a.llama_port:
        pids.add(listening_pid(a.llama_port))
    return {p for p in pids if p}


# ---------------------------------------------------------------------------------------------- Laya's protocol
def mode_laya(a: argparse.Namespace) -> dict:
    sess = sc.session()
    out: dict = {"sizes": {}, "gpu": {}, "telemetry": {}}
    counter = a.seed_offset
    for n in sc.NS:
        qs = sc.laya_questions(n, unique=True)
        rec: dict[str, list] = {"walls": [], "tez_ms": [], "backend_ms": [], "tokens": [], "layouts": []}
        reps = a.reps if n < 50 else a.reps50
        out["gpu"][str(n)] = {"before": sc.wait_cool(a.cool, 240) if a.cool else sc.gpu_now()}
        out.setdefault("cpu", {})[str(n)] = {"busy_before": busy_processes()}
        with sc.GpuMonitor() as mon, CpuMonitor() as cpu:
            for i in range(a.warmup + reps):
                counter += 1
                wall, j, info = post(sess, a.tez, {"model": "tez-latest", "state": sc.fresh_state(counter), "questions": qs})
                if len(j["answers"]) != n:
                    raise RuntimeError(f"{len(j['answers'])} answers for {n} questions")
                if a.gap or a.duty:
                    time.sleep(max(a.gap, a.duty * wall / 1000.0))
                if i >= a.warmup:
                    rec["walls"].append(wall)
                    for k in ("tez_ms", "backend_ms", "tokens"):
                        rec[k].append(info[k])
                    rec["layouts"].append(info["layout"])
        g = mon.summary()
        out["gpu"][str(n)]["during"] = g
        out["cpu"][str(n)]["during"] = cpu.summary()
        out["telemetry"][str(n)] = [list(r) for r in mon.rows]      # temp C, SM MHz, power W, util %, throttle flags
        st = sc.per_call_stats(rec["walls"], n)
        st["tez_reported_ms_p50"] = round(float(np.median(rec["tez_ms"])), 2)
        bms = [x for x in rec["backend_ms"] if x is not None]
        st["backend_ms_p50"] = round(float(np.median(bms)), 2) if bms else None
        st["input_tokens_per_call"] = int(np.median(rec["tokens"]))
        st["layouts"] = sorted(set(rec["layouts"]))
        st["samples_ms"] = [round(w, 2) for w in rec["walls"]]
        out["sizes"][str(n)] = st
        print(f"{a.tag:18s} {n:2d}q  p50 {st['p50']:8.1f}  p95 {st['p95']:8.1f}  per q {st['ms_per_question_p50']:6.2f}  "
              f"tez {st['tez_reported_ms_p50']:8.1f}  backend {st['backend_ms_p50']}  layout {st['layouts']}  "
              f"sm {g.get('sm_mhz_busy_p50')} MHz {g.get('temp_c_p50')} C throttled {g.get('thermal_throttle_frac')}  "
              f"cpu {out['cpu'][str(n)]['during'].get('cpu_pct_p50')}%",
              flush=True)
    return out


# ---------------------------------------------------------------------------------------------- typed-decisions
def wire_question(c: dict) -> dict:
    """A typed-decisions case as a wire question whose runtime prompt is the one the speed study used (checked equal)."""
    if c["qtype"] == "choice":
        return {"type": "choice", "instructions": c["instructions"], "criteria": {k: d for k, d in c["options"]}}
    if c["qtype"] == "noul":
        return {"type": "noul", "instructions": c["instructions"]}
    return {"type": "score", "instructions": c["instructions"], "criteria": [d for _, d in c["options"]]}


def probs_of(answer: dict) -> np.ndarray:
    if answer["type"] == "noul":
        return np.array([1.0 - answer["noul"], answer["noul"]])
    return np.array(list(answer["probabilities"].values()), dtype=float)


def mode_td(a: argparse.Namespace) -> dict:
    sess = sc.session()
    cases = sc.typed_decisions("test")
    rows = sc.rows_of(cases)
    if a.limit_rows:
        rows = dict(list(rows.items())[: a.limit_rows])
    recs, walls, tez_ms = [], [], []
    t0 = time.perf_counter()
    with sc.GpuMonitor() as mon, CpuMonitor() as cpu:
        for r_i, row in enumerate(rows.values()):
            qs = {str(c["qkey"][1]): wire_question(c) for c in row}
            wall, j, info = post(sess, a.tez, {"model": "tez-latest", "state": row[0]["state"], "questions": qs})
            walls.append(wall)
            tez_ms.append(info["tez_ms"])
            for c in row:
                qid = str(c["qkey"][1])
                p = probs_of(j["answers"][qid])
                t = float((j["tez"]["questions"].get(qid) or {}).get("temperature") or 1.0)
                recs.append({"id": c["id"], "gold": c["gold"], "qtype": c["qtype"], "k": len(c["options"]),
                             "pred": int(np.argmax(p)), "p": [float(x) for x in p], "temperature": t, "layout": info["layout"]})
            if (r_i + 1) % 100 == 0:
                acc = np.mean([r["pred"] == r["gold"] for r in recs])
                print(f"  {r_i + 1}/{len(rows)} rows  acc {acc:.4f}  {time.perf_counter() - t0:.0f}s", flush=True)
    preds, golds = [r["pred"] for r in recs], [r["gold"] for r in recs]
    kept = [c for c in cases if c["id"] in {r["id"] for r in recs}]
    out = {"n_decisions": len(recs), "n_rows": len(rows), "accuracy": round(sc.accuracy(preds, golds), 4),
           "by_type": sc.by_type(kept, preds), "per_row_call_ms": sc.stats(walls), "per_row_tez_ms": sc.stats(tez_ms),
           "layouts": sorted({r["layout"] for r in recs}), "wall_s": round(time.perf_counter() - t0, 1), "gpu": mon.summary(),
           "cpu": cpu.summary()}
    rows_file = sc.OUT / f"{a.tag}_td_rows.jsonl"
    rows_file.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    out["rows_file"] = str(rows_file.relative_to(sc.ROOT)).replace("\\", "/")
    for comp in [x for x in a.compare.split(",") if x]:
        other = {}
        with open(sc.OUT / comp, encoding="utf-8") as fh:
            for line in fh:
                o = json.loads(line)
                other[o["id"]] = o
        pairs = [(r, other[r["id"]]) for r in recs if r["id"] in other]
        gaps = []
        for r, o in pairs:
            if "z" not in o:
                continue
            ours = r["temperature"] * np.log(np.clip(np.asarray(r["p"]), 1e-300, None))
            theirs = np.asarray(o["z"], dtype=float)
            ours, theirs = ours - ours.max(), theirs - theirs.max()
            keep = (theirs > -10) & (ours > -10)        # letters llama-server listed well inside its top n_probs
            gaps.append(float(np.max(np.abs(ours[keep] - theirs[keep]))) if keep.any() else 0.0)
        out[f"vs_{comp}"] = {"n": len(pairs), "agreement": round(float(np.mean([r["pred"] == o["pred"] for r, o in pairs])), 4),
                             "mcnemar": sc.mcnemar([r["pred"] == r["gold"] for r, _ in pairs],
                                                   [o["pred"] == o["gold"] for _, o in pairs]),
                             "other_accuracy": round(float(np.mean([o["pred"] == o["gold"] for _, o in pairs])), 4),
                             "letter_logit_gap_nats": {"p50": round(float(np.median(gaps)), 4) if gaps else None,
                                                       "p95": round(float(np.percentile(gaps, 95)), 4) if gaps else None,
                                                       "note": "max over letters (both sides above -10 relative to the top "
                                                               "letter) of |our logit - theirs|, logits relative to the top "
                                                               "letter; ours recovered as temperature x log p"}}
    print(json.dumps({k: v for k, v in out.items() if k not in ("gpu",)}, indent=1), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["laya", "td"])
    ap.add_argument("--tez", required=True, help="the running tez serve, e.g. http://127.0.0.1:8789")
    ap.add_argument("--kind", choices=["inproc", "http"], required=True, help="the backend behind tez serve")
    ap.add_argument("--llama-port", type=int, default=None, help="http arm: the llama-server's port (its pid is ours)")
    ap.add_argument("--serve-cmd", default="", help="the tez serve command line, recorded with the results")
    ap.add_argument("--gap", type=float, default=0.3)
    ap.add_argument("--duty", type=float, default=1.0)
    ap.add_argument("--cool", type=float, default=85.0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--reps50", type=int, default=4)
    ap.add_argument("--seed-offset", type=int, default=700_000)
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--compare", default="", help="td: comma-separated rows files in results/speed/ to compare with")
    ap.add_argument("--tag", required=True, help="output: results/speed/<tag>_<mode>.json")
    a = ap.parse_args()
    meta = meta_of(a)
    with Guard(own_pids(a)) as guard:
        res = mode_laya(a) if a.mode == "laya" else mode_td(a)
    g = guard.result()
    print("gpu_guard contaminated:", g["contaminated"], g["reasons_before"] + g["reasons_after"], flush=True)
    path = sc.dump(f"{a.tag}_{a.mode}.json", {"meta": meta, **res, "gpu_guard": g})
    print("wrote", path, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    main()
