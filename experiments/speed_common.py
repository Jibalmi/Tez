"""Shared helpers for the speed_* experiments (results in results/speed/).

Not a script: imported by speed_overhead.py, speed_multiq.py, speed_probe_multiq.py, speed_voice.py and
speed_qwen_cache_check.py. It holds
  * the model paths and the exact production llama-server command (restored at the end of every session),
  * start/stop helpers for extra llama-server configurations on a side port,
  * latency statistics (p50 / p95 / mean),
  * the typed-decisions loader (experiments/hidden_probe.py, the split every Tez number uses),
  * the two prompt layouts compared for many questions on one state:
      today        instructions + question + options first, state LAST (tez/prompt.py, bench_h2h.py)
      statefirst   instructions + state FIRST, then the question and its options (the state is a shared prefix)
  * Laya's latency protocol (experiments/vs_laya_speed.py: payout ticket, alternating choice / noul questions),
  * an HTTP letter scorer for llama-server /completion that also returns the server's own timings.
"""
from __future__ import annotations

import importlib.util
import json
import os
import string
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "speed"
OUT.mkdir(parents=True, exist_ok=True)

LLAMA_DIR = r"C:\temp\llamacpp"
EXE = os.path.join(LLAMA_DIR, "llama-server.exe")
GEMMA = "C:/Users/migue/.ollama/models/blobs/sha256-047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795"
QWEN = {
    "q4b": str(ROOT / "tools" / "models" / "Qwen3.5-4B-Q8_0.gguf"),
    "q4b_L20": str(ROOT / "tools" / "models" / "Qwen3.5-4B-Q8_0-L20.gguf"),
    "q4b_L24": str(ROOT / "tools" / "models" / "Qwen3.5-4B-Q8_0-L24.gguf"),
    "q4b_L29": str(ROOT / "tools" / "models" / "Qwen3.5-4B-Q8_0-L29.gguf"),
}
# The production server (task brief): restored with exactly these arguments when a session ends.
PROD_ARGS = ["-m", GEMMA, "-ngl", "99", "-c", "4096", "-b", "512", "--port", "8091", "--host", "127.0.0.1", "-np", "1",
             "--no-webui", "--swa-full", "--embeddings", "--pooling", "last"]
PROD_URL = "http://127.0.0.1:8091"
TEZ_URL = "http://127.0.0.1:8787"

LETTERS = string.ascii_uppercase
TEMPLATES = {
    "gemma4": ("<|turn>user\n", "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
    "qwen3": ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
}
HEAD_TODAY = ("You are a decision engine. Read the question and the options, then look at the input and answer "
              "with the single letter of the best option. Answer with the letter only.")
HEAD_STATEFIRST = ("You are a decision engine. Read the input, then the question and its options, and answer "
                   "with the single letter of the best option. Answer with the letter only.")


# ---------------------------------------------------------------------------------------------- http
def session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False          # never reach a local server through the corporate proxy
    return s


S = session()


def wait_health(url: str, timeout_s: float = 300.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if S.get(f"{url}/health", timeout=2).json().get("status") == "ok":
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    return False


def start_server(model: str, port: int, extra: Sequence[str] = (), ctx: int = 4096, batch: int = 512, np_slots: int = 1,
                 log: str | Path | None = None, embeddings: bool = True, ngl: str = "99") -> subprocess.Popen:
    """Start a llama-server on a side port (never the production port unless asked) and wait until healthy."""
    args = [EXE, "-m", model, "-ngl", ngl, "-c", str(ctx), "-b", str(batch), "--port", str(port), "--host", "127.0.0.1",
            "-np", str(np_slots), "--no-webui"] + (["--embeddings", "--pooling", "last"] if embeddings else []) + list(extra)
    log = Path(log) if log else Path(os.environ.get("TEMP", ".")) / f"speed_server_{port}.log"
    p = subprocess.Popen(args, cwd=LLAMA_DIR, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    if not wait_health(f"http://127.0.0.1:{port}"):
        p.kill()
        raise RuntimeError(f"llama-server did not start: {' '.join(args)} (log {log})")
    return p


def stop_server(p: subprocess.Popen | None) -> None:
    if p is None:
        return
    p.kill()
    try:
        p.wait(timeout=30)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(3)          # let the driver release the VRAM


def props(url: str) -> dict:
    return S.get(f"{url}/props", timeout=30).json()


# ---------------------------------------------------------------------------------------------- GPU telemetry
_GPU_Q = "temperature.gpu,clocks.sm,power.draw,utilization.gpu,clocks_throttle_reasons.active"
_THERMAL = 0x20 | 0x40 | 0x08 | 0x80      # SW thermal, HW thermal, HW slowdown, HW power brake


def gpu_now() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={_GPU_Q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        t, sm, pw, ut, th = [x.strip() for x in out.split(",")]
        return {"temp_c": float(t), "sm_mhz": float(sm), "power_w": float(pw), "util": float(ut), "throttle": th}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


class GpuMonitor:
    """Samples nvidia-smi every `ms` during a measurement: temperature, SM clock, power, thermal-throttle flags."""

    def __init__(self, ms: int = 500):
        self.ms = ms
        self.rows: list[tuple] = []

    def __enter__(self):
        import threading
        self.p = subprocess.Popen(["nvidia-smi", f"--query-gpu={_GPU_Q}", "--format=csv,noheader,nounits", f"-lms={self.ms}"],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

        def read():
            for line in self.p.stdout:
                try:
                    t, sm, pw, ut, th = [x.strip() for x in line.split(",")]
                    self.rows.append((float(t), float(sm), float(pw), float(ut), int(th, 16)))
                except Exception:  # noqa: BLE001
                    pass
        self.th = threading.Thread(target=read, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *exc):
        self.p.terminate()
        try:
            self.p.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.p.kill()

    def summary(self) -> dict:
        if not self.rows:
            return {"samples": 0}
        a = np.asarray([r[:4] for r in self.rows], float)
        busy = a[:, 3] >= 20
        th = np.asarray([r[4] & _THERMAL != 0 for r in self.rows])
        b = a[busy] if busy.any() else a
        return {"samples": int(len(a)), "temp_c_p50": float(np.median(a[:, 0])), "temp_c_max": float(a[:, 0].max()),
                "sm_mhz_busy_p50": float(np.median(b[:, 1])), "sm_mhz_busy_p10": float(np.percentile(b[:, 1], 10)),
                "power_w_busy_p50": float(np.median(b[:, 2])), "busy_frac": round(float(busy.mean()), 3),
                "thermal_throttle_frac": round(float(th.mean()), 3)}


LOCK_TOOL = r"C:\temp\gpu_lock.py"
ALLOWED_VRAM = {"dwm", "Code", "chrome", "WINWORD"}   # desktop processes tolerated next to a measurement


def lock_snapshot() -> dict:
    """`python C:/temp/gpu_lock.py snapshot` (per-process dedicated / shared VRAM) plus Ollama's loaded models."""
    try:
        out = subprocess.run([sys.executable, LOCK_TOOL, "snapshot"], capture_output=True, text=True, timeout=180).stdout
        snap = json.loads(out)
    except Exception as exc:  # noqa: BLE001
        snap = {"error": str(exc), "processes": []}
    try:
        snap["ollama_ps"] = [m.get("name") for m in S.get("http://127.0.0.1:11434/api/ps", timeout=5).json().get("models", [])]
    except Exception as exc:  # noqa: BLE001
        snap["ollama_ps"] = f"unreachable ({exc.__class__.__name__})"
    return snap


def port_of(url: str) -> int | None:
    try:
        return int(url.rstrip("/").rsplit(":", 1)[1].split("/")[0])
    except (IndexError, ValueError):
        return None


def server_pid(port: int) -> int | None:
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | Where-Object { $_.CommandLine -match "
          f"'--port {port}( |$)' }} | ForEach-Object {{ $_.ProcessId }}")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60).stdout.split()
    return int(out[0]) if out else None


def contamination(snap: dict, own: set[int]) -> list[str]:
    """The lead's rule: contaminated if any process other than ours (dwm, Code, chrome, WINWORD tolerated) holds more
    than 300 MB of VRAM, if our process has more than 200 MB in shared (system) memory, or if Ollama has a model loaded."""
    reasons = []
    for p in snap.get("processes", []):
        if p["pid"] in own:
            if p["kind"] == "shared" and p["mb"] > 200:
                reasons.append(f"our process {p['name']} ({p['pid']}) has {p['mb']} MB in shared system memory (VRAM spill)")
            continue
        if p["kind"] == "dedicated" and p["mb"] > 300 and p["name"] not in ALLOWED_VRAM:
            reasons.append(f"{p['name']} (pid {p['pid']}) holds {p['mb']} MB of VRAM")
    ol = snap.get("ollama_ps")
    if isinstance(ol, list) and ol:
        reasons.append(f"ollama has {ol} loaded")
    if snap.get("error"):
        reasons.append(f"snapshot failed: {snap['error']}")
    return reasons


class Guard:
    """Snapshot before and after a timed run; `result()` goes into the run's JSON as gpu_guard."""

    def __init__(self, port: int | None = None, inproc: bool = False):
        self.port, self.inproc = port, inproc

    def __enter__(self):
        self.own = {os.getpid()} if self.inproc else set()
        if self.port:
            pid = server_pid(self.port)
            if pid:
                self.own.add(pid)
        self.before = lock_snapshot()
        return self

    def __exit__(self, *exc):
        self.after = lock_snapshot()

    def result(self) -> dict:
        rb, ra = contamination(self.before, self.own), contamination(self.after, self.own)
        return {"own_pids": sorted(self.own), "before": self.before, "after": self.after,
                "contaminated": bool(rb or ra), "reasons_before": rb, "reasons_after": ra}


def wait_cool(target_c: float = 80.0, timeout_s: float = 240.0) -> dict:
    """Wait (at most timeout_s) until the GPU is at or below target_c; returns the reading it stopped at."""
    t0 = time.time()
    g = gpu_now()
    while time.time() - t0 < timeout_s and g.get("temp_c", 0) > target_c:
        time.sleep(5)
        g = gpu_now()
    g["waited_s"] = round(time.time() - t0, 1)
    return g


# ---------------------------------------------------------------------------------------------- stats
def stats(ms: Sequence[float]) -> dict:
    a = np.asarray([x for x in ms if x is not None], float)
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "p50": round(float(np.percentile(a, 50)), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "mean": round(float(a.mean()), 2), "min": round(float(a.min()), 2), "max": round(float(a.max()), 2)}


def per_call_stats(call_ms: Sequence[float], n_questions: int) -> dict:
    st = stats(call_ms)
    st["ms_per_question_p50"] = round(st["p50"] / n_questions, 2)
    st["ms_per_question_p95"] = round(st["p95"] / n_questions, 2)
    return st


def dump(name: str, obj: Any) -> Path:
    p = OUT / name
    p.write_text(json.dumps(obj, indent=1, default=_json_default), encoding="utf-8")
    return p


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------------------------- data
def _load_module(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


_HP = None


def hp():
    global _HP
    if _HP is None:
        _HP = _load_module("hp", "hidden_probe.py")
    return _HP


def typed_decisions(split: str = "test") -> list[dict]:
    """hidden_probe.load_split: one case per decision, {id, qkey, qtype, state, instructions, options, gold}."""
    return hp().load_split(split)


def rows_of(cases: list[dict]) -> dict[str, list[dict]]:
    """Group decisions by typed-decisions row (ids are f"{row_id}-{qid}"), keeping the dataset's question order."""
    rows: dict[str, list[dict]] = {}
    for c in cases:
        qid = str(c["qkey"][1])
        rows.setdefault(c["id"][: -(len(qid) + 1)], []).append(c)
    return rows


def state_text(st: Any) -> str:
    return st if isinstance(st, str) else json.dumps(st, ensure_ascii=False)


def option_lines(options: Sequence[tuple[str, Any]]) -> str:
    return "\n".join(f"{LETTERS[i]}. {k}: {d}" if d and d != k else f"{LETTERS[i]}. {k}" for i, (k, d) in enumerate(options))


def user_today(qtype: str, instructions: str, options, state: Any) -> str:
    return (f"{HEAD_TODAY}\n\nQuestion ({qtype}): {instructions}\n\nOptions:\n{option_lines(options)}"
            f"\n\nInput:\n{state_text(state)}")


def statefirst_prefix(state: Any, template: str) -> str:
    """The shared part of every state-first prompt of one state (template head + instructions + state)."""
    pre, _ = TEMPLATES[template]
    return f"{pre}{HEAD_STATEFIRST}\n\nInput:\n{state_text(state)}\n\n"


def statefirst_suffix(qtype: str, instructions: str, options, template: str) -> str:
    _, post = TEMPLATES[template]
    return f"Question ({qtype}): {instructions}\n\nOptions:\n{option_lines(options)}{post}"


def prompt_today(case: dict, template: str) -> str:
    pre, post = TEMPLATES[template]
    return f"{pre}{user_today(case['qtype'], case['instructions'], case['options'], case['state'])}{post}"


def prompt_statefirst(case: dict, template: str) -> str:
    return statefirst_prefix(case["state"], template) + statefirst_suffix(case["qtype"], case["instructions"], case["options"], template)


# ---------------------------------------------------------------------------------------------- Laya's protocol
LAT_STATE = {"ticket": {"subject": "Payout failing",
             "messages": [{"from": "customer",
                           "text": "Hi, my Stripe payouts have failed for 3 days and I am losing sales. Please help ASAP. " * 6}]}}
Q_NOUL = {"type": "noul", "instructions": "Does `ticket.messages[0].text` express urgency?"}
Q_CHOICE = {"type": "choice", "instructions": "Which team should handle this?",
            "criteria": {"billing": "payments", "technical": "bugs and integrations", "sales": "pricing"}}
NS = [1, 5, 10, 50]


def laya_questions(n: int, unique: bool = False) -> dict:
    """Laya's protocol repeats two questions verbatim. unique=True makes every question of a call distinct (its
    number appended to the instructions), as real calls are: then no prompt can be served from an earlier identical one."""
    out = {}
    for i in range(n):
        q = dict(Q_NOUL if i % 2 else Q_CHOICE)
        if unique:
            q["instructions"] = f"{q['instructions']} (question {i + 1})"
        out["q%d" % i] = q
    return out


def fresh_state(i: int) -> dict:
    """vs_laya_speed.fresh_state: a new ticket id at the start of the subject and the message on every call."""
    return {"ticket": {"subject": f"Ticket {i}: payout failing",
                       "messages": [{"from": "customer",
                                     "text": f"Ticket {i}. " + "Hi, my Stripe payouts have failed for 3 days and I am losing sales. Please help ASAP. " * 6}]}}


def wire_to_case(qid: str, q: dict, state: Any) -> dict:
    """A wire-format question as a case dict with the runtime's options (tez/schema.py Question.options)."""
    from tez.schema import parse_question
    Q = parse_question(qid, q)
    return dict(id=qid, qtype=Q.type, instructions=Q.instructions_text, options=Q.options(), state=state)


# ---------------------------------------------------------------------------------------------- letters over HTTP
def letters_from_tops(tops: list[dict], k: int) -> np.ndarray:
    lp: dict[str, float] = {}
    for t in tops:
        lp.setdefault(t["token"], float(t["logprob"]))
    floor = min(float(t["logprob"]) for t in tops) - 2.0
    return np.array([lp.get(LETTERS[i], floor) for i in range(k)], dtype=float)


def completion_body(prompt: Any, n_probs: int = 200, cache: bool = True, **extra) -> dict:
    b = {"prompt": prompt, "n_predict": 1, "n_probs": n_probs, "temperature": 0, "samplers": [], "cache_prompt": cache}
    b.update(extra)
    return b


def score_http(url: str, prompt: str, k: int, n_probs: int = 200, cache: bool = True, sess: requests.Session | None = None,
               **extra) -> dict:
    """One /completion: letter log-probs + the server's timings + the client round trip."""
    s = sess or S
    t0 = time.perf_counter()
    r = s.post(f"{url}/completion", json=completion_body(prompt, n_probs, cache, **extra), timeout=600)
    http_ms = (time.perf_counter() - t0) * 1000
    data = r.json()
    if "completion_probabilities" not in data:     # greedy first token was end-of-sequence: suppress it and re-read
        data = s.post(f"{url}/completion", json=completion_body(prompt, n_probs, cache, ignore_eos=True, **extra), timeout=600).json()
        http_ms = (time.perf_counter() - t0) * 1000
    z = letters_from_tops(data["completion_probabilities"][0]["top_logprobs"], k)
    tm = data.get("timings", {}) or {}
    return dict(z=z, http_ms=http_ms, prompt_n=tm.get("prompt_n"), prompt_ms=tm.get("prompt_ms"),
                cache_n=tm.get("cache_n"), predicted_ms=tm.get("predicted_ms"), n_ctx_used=data.get("tokens_evaluated"))


def softmax(z) -> np.ndarray:
    z = np.asarray(z, float)
    e = np.exp(z - z.max())
    return e / e.sum()


def accuracy(preds: Sequence[int], golds: Sequence[int]) -> float:
    return float(np.mean(np.asarray(preds) == np.asarray(golds)))


def by_type(cases: list[dict], preds: Sequence[int]) -> dict:
    out = {}
    for t in ("choice", "noul", "score"):
        idx = [i for i, c in enumerate(cases) if c["qtype"] == t]
        if idx:
            out[t] = round(float(np.mean([preds[i] == cases[i]["gold"] for i in idx])), 4)
    return out


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    """Exact two-sided McNemar test on paired correctness."""
    from math import comb
    a = np.asarray(a_correct, bool)
    b = np.asarray(b_correct, bool)
    n01 = int((a & ~b).sum())
    n10 = int((~a & b).sum())
    n = n01 + n10
    if n == 0:
        return {"a_only": n01, "b_only": n10, "p": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": n01, "b_only": n10, "p": round(p, 5)}
