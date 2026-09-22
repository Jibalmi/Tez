"""End-to-end voice -> decision -> action loop with per-stage timing.

    mic/wav audio --(VAD+chunks)--> faster-whisper partial transcripts --> one-pass decision
    (llama-server, transcript-last prompt, KV prefix cached) --> commit policy --> action

Three input modes so every stage can be timed in isolation:
  --text "open notes and type hello"   words arrive at --wpm; no ASR; times the decision loop
  --wav file.wav                        audio is fed in real time in --chunk-ms pieces through ASR
  --mic                                 live microphone (sounddevice), 16 kHz mono

Commit policy (from results/stream_analysis_*.json): commit at the first partial whose
argmax is NOT 'none' with p >= --tau, stable for --stable consecutive partials. A confident
'none' on a partial means "keep listening", never "do nothing".

Actions are DRY RUN unless --execute.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
# ASR partials can contain non-cp1252 characters; line-buffer because CTranslate2+torch abort at
# interpreter teardown on Windows and block-buffered output would be lost.
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m  # dataclasses (3.14) resolve cls.__module__ through sys.modules
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


vf = _load("run_voice_fast")
va = _load("voice_actions")
sp = _load("stream_policy")
_load("asr_bench").add_nvidia_dlls()  # CUDA 12 cuBLAS/cuDNN for CTranslate2


class Decider:
    """run_voice_fast.score with the transcript-last prompt + the class-aware commit/refine policy
    of stream_policy.py, applied incrementally: OPEN actions fire at pmax>=tau on one partial,
    MEDIA/PLAY need `stable` consecutive partials, type_text/close_app wait for end of utterance,
    a confident 'none' on a partial never fires."""

    def __init__(self, server, variant="final", tau=0.9, stable=2, n_probs=200):
        self.server, self.variant, self.tau, self.stable, self.n_probs = server, variant, tau, stable, n_probs
        self.intents = json.loads((ROOT / "data" / "voice" / "intents.json").read_text(encoding="utf-8"))["intents"]
        self.ids = [i["id"] for i in self.intents]
        self.reset()
        vf.score(server, vf.build_prompt("warm up", self.intents, variant, "last"), len(self.ids), n_probs, True)

    def reset(self):
        self.history, self.run, self.done, self.cursor = [], [], [], 0

    def step(self, transcript: str) -> dict:
        """transcript = everything heard so far. After an action has fired, only the words AFTER
        the ones that triggered it are scored (the residual), so "open notes and type X" becomes
        two decisions: open_notes on "open notes", then type_text on "and type X"."""
        words = transcript.split()
        residual = " ".join(words[self.cursor:]) if self.cursor < len(words) else transcript
        t0 = time.perf_counter()
        prior = self.done[-1] if (self.cursor and self.done) else None
        r = vf.score(self.server, vf.build_prompt(residual, self.intents, self.variant, "last", prior), len(self.ids), self.n_probs, True)
        p = r["probabilities"]
        i = max(range(len(p)), key=lambda j: p[j])
        a = self.ids[i]
        d = dict(transcript=transcript, scored=residual, pred=a, pmax=p[i], prompt_n=r["prompt_n"], prompt_ms=r["prompt_ms"],
                 decide_ms=(time.perf_counter() - t0) * 1000, commit=None)
        self.history.append(d)
        self.run = self.run + [a] if (self.run and self.run[-1] == a) else [a]
        if a == "none" or p[i] < self.tau or a in sp.DEFERRED:
            return d
        if a in sp.SLOT_PLAY:
            anc = sp.REFINES[a]
            if anc not in self.done and len(self.run) >= self.stable:  # "play X" is platform-ambiguous; ASR partials flicker
                d["commit"] = anc      # open the app now; the search itself waits for the full query
                self.done.append(anc)
            return d
        need = 1 if a in sp.OPEN else self.stable
        if len(self.run) >= need and not (self.done and self.done[-1] == a):
            d["commit"] = a
            self.done.append(a)
            if a in sp.OPEN or a in sp.MEDIA or a in sp.PLAY:
                self.cursor = len(words)   # subsequent words are a new decision
                self.run = []
        return d

    def finish(self) -> tuple[str, str] | None:
        """End of utterance: the deferred / final action on the residual, if it differs from what
        already ran. Returns (action, text_to_extract_slots_from)."""
        if not self.history:
            return None
        last = self.history[-1]
        a = last["pred"]
        if a != "none" and (not self.done or self.done[-1] != a):
            self.done.append(a)
            return a, last["scored"]
        return None


def run_text(dec, text, wpm, execute):
    words = text.split()
    gap = 60.0 / wpm
    t_start = time.perf_counter()
    log = []
    dec.reset()
    for k in range(1, len(words) + 1):
        target = t_start + (k - 1) * gap
        while time.perf_counter() < target:
            time.sleep(0.001)
        t_arr = time.perf_counter()
        d = dec.step(" ".join(words[:k]))
        t_dec = time.perf_counter()
        row = dict(word=k, arrival_ms=(t_arr - t_start) * 1000, decided_ms=(t_dec - t_start) * 1000, **d)
        if d["commit"]:
            a = va.run_action(d["commit"], d["scored"], execute=execute)
            row["action"] = a.steps
            row["dispatched_ms"] = (time.perf_counter() - t_start) * 1000
        log.append(row)
        print(f"  w{k:<2d} +{row['arrival_ms']:6.0f}ms  {d['scored']!r:40s} -> {d['pred']:16s} p={d['pmax']:.2f} "
              f"tok={d['prompt_n']:<3d} {d['decide_ms']:5.1f}ms {'ACT ' + str(row.get('action')) if 'action' in row else ''}")
    fin = dec.finish()
    if fin:
        a = va.run_action(fin[0], fin[1], execute=execute)
        print(f"  end of utterance -> {fin[0]}: {a.steps}")
    else:
        print(f"  end of utterance -> nothing further (ran: {dec.done or 'nothing'})")
    return log


class Asr:
    def __init__(self, model="base.en", device="cuda", compute="float16"):
        from faster_whisper import WhisperModel
        self.m = WhisperModel(model, device=device, compute_type=compute)
        self.m.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1, language="en")  # warm

    def partial(self, audio: np.ndarray) -> tuple[str, float]:
        t0 = time.perf_counter()
        segs, _ = self.m.transcribe(audio, beam_size=1, language="en", vad_filter=False, condition_on_previous_text=False,
                                    without_timestamps=True)
        text = " ".join(s.text.strip() for s in segs).strip()
        return text, (time.perf_counter() - t0) * 1000


def run_stream(dec, asr, chunks, chunk_ms, execute, silence_ms=600, rms_gate=0.01):
    """chunks: iterator of float32 arrays at 16 kHz arriving in real time."""
    buf = np.zeros(0, dtype=np.float32)
    t_start = time.perf_counter()
    last_voice = None
    norm = lambda s: " ".join(s.lower().replace(",", "").replace(".", "").replace("?", "").split())
    last_text = ""
    dec.reset()
    for ch in chunks:
        buf = np.concatenate([buf, ch])
        now = (time.perf_counter() - t_start) * 1000
        if np.sqrt(np.mean(ch ** 2)) > rms_gate:
            last_voice = now
        if last_voice is None:
            continue
        text, asr_ms = asr.partial(buf)
        text = norm(text)
        if text and text != last_text:
            last_text = text
            d = dec.step(text)
            print(f"  +{now:6.0f}ms asr {asr_ms:5.0f}ms {d['scored']!r:40s} -> {d['pred']:16s} p={d['pmax']:.2f} tok={d['prompt_n']} {d['decide_ms']:5.1f}ms")
            if d["commit"]:
                a = va.run_action(d["commit"], d["scored"], execute=execute)
                print(f"     ACT @ +{(time.perf_counter()-t_start)*1000:.0f}ms: {a.steps}")
        if now - last_voice > silence_ms and len(buf) > 16000 * 0.5:
            fin = dec.finish()
            if fin:
                a = va.run_action(fin[0], fin[1], execute=execute)
                print(f"     end-of-speech @ +{now:.0f}ms -> {fin[0]}: {a.steps}")
            else:
                print(f"     end-of-speech @ +{now:.0f}ms -> nothing further (ran: {dec.done or 'nothing'})")
            buf = np.zeros(0, dtype=np.float32); last_voice = None; last_text = ""; dec.reset()


def wav_chunks(path, chunk_ms):
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import math
        idx = np.arange(0, len(audio), sr / 16000.0)
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    n = int(16000 * chunk_ms / 1000)
    t0 = time.perf_counter()
    for i in range(0, len(audio) + n, n):
        target = t0 + (i / 16000.0)
        while time.perf_counter() < target:
            time.sleep(0.001)
        yield audio[i:i + n] if i < len(audio) else np.zeros(n, dtype=np.float32)
        if i >= len(audio) + 16000:
            break


def mic_chunks(chunk_ms):
    import sounddevice as sd
    q = queue.Queue()
    n = int(16000 * chunk_ms / 1000)

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())
    with sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=n, callback=cb):
        while True:
            yield q.get()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8091")
    ap.add_argument("--text")
    ap.add_argument("--wav")
    ap.add_argument("--mic", action="store_true")
    ap.add_argument("--wpm", type=float, default=150)
    ap.add_argument("--chunk-ms", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--stable", type=int, default=2, help="consecutive partials required for MEDIA/PLAY actions (OPEN needs 1)")
    ap.add_argument("--variant", default="final")
    ap.add_argument("--asr", default="base.en")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    dec = Decider(args.server, args.variant, args.tau, args.stable)
    if args.text:
        log = run_text(dec, args.text, args.wpm, args.execute)
        if args.out:
            Path(args.out).write_text(json.dumps(log, indent=2), encoding="utf-8")
        return
    asr = Asr(args.asr)
    if args.wav:
        run_stream(dec, asr, wav_chunks(args.wav, args.chunk_ms), args.chunk_ms, args.execute)
    elif args.mic:
        print("listening... (ctrl-c to stop)")
        run_stream(dec, asr, mic_chunks(args.chunk_ms), args.chunk_ms, args.execute)
    else:
        ap.error("one of --text / --wav / --mic")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)  # skip interpreter teardown: CTranslate2 + torch abort there on Windows
