"""Run timed GPU measurements only when the laptop is quiet, then put everything back.

Other sessions share this laptop's CPU and GPU; timings taken while they run are skewed (the runtime timings of
2026-09-24 ran with the CPU at 80-100 %). This waits until, over one window:

  - total CPU load averages below --cpu-max,
  - the cooperative GPU lock (C:/temp/gpu_lock.py) is free,
  - Ollama has no model loaded,
  - no process other than the production llama-server holds more than 300 MB of GPU memory,

then takes the lock and runs a plan of steps, recording the CPU load and a GPU snapshot around each step. Whatever
happens, it stops the servers it started, restores the production llama-server and releases the lock.

  python experiments/run_when_idle.py --plan runtime             # the idle rerun of the runtime timings
  python experiments/run_when_idle.py --plan runtime --dry-run   # print the plan and the idle checks only

Launched detached (Start-Process) so it survives the session that started it; progress goes to
results/speed/idle_runs/<timestamp>/run.log.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
LOCK = [sys.executable, "C:/temp/gpu_lock.py"]
OWNER = "projects-37-idle-timing"
PROD_GGUF = "C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795"
LLAMA_LIB = "C:/temp/llamacpp"
SERVERS_PS1 = str(ROOT / "experiments" / "speed_servers.ps1")


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


class Log:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"{now()}  {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def run(cmd: list[str], log: Log, timeout: float | None = None) -> int:
    log("$ " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    for stream in (p.stdout, p.stderr):
        for line in (stream or "").strip().splitlines()[-40:]:
            log("  " + line)
    log(f"  exit {p.returncode}")
    return p.returncode


def run_detached(cmd: list[str], log: Log, timeout: float = 180) -> int:
    """For commands that start a long-lived process (speed_servers.ps1 start-prod): no output pipes, because the
    server inherits them and a captured run would wait for it to exit."""
    log("$ " + " ".join(cmd))
    code = subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          timeout=timeout).returncode
    log(f"  exit {code}")
    return code


def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def wait_http(url: str, seconds: float, log: Log) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if http_ok(url):
            return True
        time.sleep(2)
    log(f"  {url} did not come up within {seconds:.0f} s")
    return False


# GPU memory held by these is the desktop itself, as gpu_lock.py's contamination rule has it
GPU_ALLOW = ("dwm", "code", "chrome", "msedge", "winword", "explorer")


def gpu_holders() -> list[dict]:
    out = subprocess.run([*LOCK, "snapshot"], capture_output=True, text=True).stdout
    try:
        snap = json.loads(out)
    except json.JSONDecodeError:
        return []
    return [p for p in snap.get("processes", []) if p.get("kind", "").lower().startswith("dedicated")]


def prod_pid() -> int | None:
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if (p.info["name"] or "").lower().startswith("llama-server") and "--port 8091" in " ".join(p.info["cmdline"] or []):
            return p.info["pid"]
    return None


def idle_reasons(cpu_max: float, window: float) -> list[str]:
    """Why the machine is not idle right now (empty list = idle). Samples the CPU over `window` seconds."""
    reasons = []
    status = json.loads(subprocess.run([*LOCK, "status"], capture_output=True, text=True).stdout or "{}")
    if status.get("lock") and not status.get("stale"):
        reasons.append(f"GPU lock held by {status['lock'].get('owner')} ({status['lock'].get('purpose')})")
    ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    if len(ps) > 1:
        reasons.append("Ollama has a model loaded: " + ps[1].split()[0])
    keep = prod_pid()
    for h in gpu_holders():
        name = str(h.get("name", "")).lower()
        if h.get("mb", 0) > 300 and h.get("pid") != keep and "llama-server" not in name and name not in GPU_ALLOW:
            reasons.append(f"{h.get('name')} (pid {h.get('pid')}) holds {h.get('mb')} MB of GPU memory")
    samples = [psutil.cpu_percent(interval=5) for _ in range(max(1, int(window / 5)))]
    avg = sum(samples) / len(samples)
    if avg >= cpu_max:
        procs = [p for p in psutil.process_iter(["name"]) if p.pid not in (0, 4)]   # not the idle process or System
        for p in procs:
            try:
                p.cpu_percent(None)
            except psutil.Error:
                pass
        time.sleep(2)
        busy = []
        for p in procs:
            try:
                busy.append((p.cpu_percent(None), p.info["name"]))
            except psutil.Error:
                pass
        top = ", ".join(f"{n} {c:.0f} %" for c, n in sorted(busy, reverse=True)[:3])
        reasons.append(f"CPU averaged {avg:.0f} % over {window:.0f} s (limit {cpu_max:.0f} %; busiest: {top})")
    return reasons


# ------------------------------------------------------------------------------------------------ plans
def plan_runtime(out: Path, log: Log) -> None:
    """The runtime timings on an idle machine: tez serve over HTTP, then tez serve in process."""
    bench = [sys.executable, "experiments/inproc_runtime_bench.py", "laya"]
    # 1. HTTP: tez serve in front of the production llama-server
    serve = subprocess.Popen([sys.executable, "-m", "tez", "serve", "--backend", "http://127.0.0.1:8091",
                              "--template", "gemma4", "--port", "8788"], cwd=ROOT,
                             stdout=(out / "serve_http.log").open("w"), stderr=subprocess.STDOUT)
    try:
        if wait_http("http://127.0.0.1:8788/healthz", 120, log):
            log(f"CPU before HTTP arm: {psutil.cpu_percent(interval=5):.0f} %")
            run([*bench, "--tez", "http://127.0.0.1:8788", "--kind", "http", "--llama-port", "8091",
                 "--tag", "http_runtime_idle"], log, timeout=3 * 3600)
    finally:
        serve.terminate()
        serve.wait(timeout=30)
    # 2. In process: the 12B cannot be loaded twice on 16 GB, so the production server stops for this arm
    run(["powershell", "-NoProfile", "-File", SERVERS_PS1, "stop-prod"], log)
    serve = subprocess.Popen([sys.executable, "-m", "tez", "serve", "--backend", f"inproc:{PROD_GGUF}",
                              "--template", "gemma4", "--llama-lib", LLAMA_LIB, "--port", "8789"], cwd=ROOT,
                             stdout=(out / "serve_inproc.log").open("w"), stderr=subprocess.STDOUT)
    try:
        if wait_http("http://127.0.0.1:8789/healthz", 300, log):
            log(f"CPU before in-process arm: {psutil.cpu_percent(interval=5):.0f} %")
            run([*bench, "--tez", "http://127.0.0.1:8789", "--kind", "inproc", "--tag", "inproc_runtime_idle"],
                log, timeout=3 * 3600)
    finally:
        serve.terminate()
        serve.wait(timeout=60)


PLANS = {"runtime": (plan_runtime, 240)}   # name: (function, lock minutes)


def restore_prod(log: Log) -> None:
    if prod_pid() is None:
        run_detached(["powershell", "-NoProfile", "-File", SERVERS_PS1, "start-prod"], log)
    ok = wait_http("http://127.0.0.1:8091/health", 300, log)
    log(f"production llama-server on :8091 {'ok' if ok else 'NOT healthy: check it'} (pid {prod_pid()})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", choices=sorted(PLANS), default="runtime")
    ap.add_argument("--cpu-max", type=float, default=25.0, help="idle means CPU below this (percent), averaged")
    ap.add_argument("--window", type=float, default=180.0, help="seconds of CPU sampling per idle check")
    ap.add_argument("--poll", type=float, default=300.0, help="seconds between idle checks")
    ap.add_argument("--max-wait-hours", type=float, default=36.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "results" / "speed" / "idle_runs" / f"{stamp}-{args.plan}"
    log = Log(out / "run.log")
    fn, minutes = PLANS[args.plan]
    log(f"plan {args.plan}: {fn.__doc__}  (pid {psutil.Process().pid})")

    deadline = time.time() + args.max_wait_hours * 3600
    while True:
        reasons = idle_reasons(args.cpu_max, args.window)
        if not reasons:
            break
        log("not idle: " + "; ".join(reasons))
        if args.dry_run:
            return 0
        if time.time() > deadline:
            log(f"gave up after {args.max_wait_hours} h without an idle window")
            return 2
        time.sleep(args.poll)
    log("idle")
    if args.dry_run:
        return 0

    if run([*LOCK, "acquire", "--owner", OWNER, "--purpose", f"idle rerun: {args.plan}", "--minutes", str(minutes),
            "--pid", str(psutil.Process().pid)], log) != 0:
        log("could not take the GPU lock")
        return 3
    try:
        (out / "gpu_before.json").write_text(subprocess.run([*LOCK, "snapshot"], capture_output=True, text=True).stdout)
        fn(out, log)
        (out / "gpu_after.json").write_text(subprocess.run([*LOCK, "snapshot"], capture_output=True, text=True).stdout)
    except Exception as exc:  # noqa: BLE001 - always restore below
        log(f"step failed: {exc!r}")
    finally:
        restore_prod(log)
        run([*LOCK, "release", "--owner", OWNER], log)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
