"""Text-to-speech candidate benchmark for the Theioma voice loop.

The plan, the metrics' definitions and the pass/fail gates are in docs/plans/voice-tts-plan.md; the fixed utterance
set is data/voice/tts_utterances.json. Run it from the harness's own environment (.venv-tts, see
experiments/tts_requirements.txt), one command per engine, so a scheduler can call it:

  .venv-tts\\Scripts\\python experiments\\tts_bench.py list
  .venv-tts\\Scripts\\python experiments\\tts_bench.py run --engine kokoro-onnx --variant fp32 --device cuda \\
      --tag idle-2026-10-01 --reps 10 --cold-runs 5 --idle-gaps 3,15 --lock acquire --score
  .venv-tts\\Scripts\\python experiments\\tts_bench.py score results/tts/kokoro-onnx-fp32-cuda_idle-2026-10-01.json
  .venv-tts\\Scripts\\python experiments\\tts_bench.py playback --tag idle-2026-10-01
  .venv-tts\\Scripts\\python experiments\\tts_bench.py suite --name control,gpu,cpu --tag r1-A --print

Exit codes: 0 ok, 2 a dependency or model file is missing, 3 the GPU lock is held by someone else (--lock check),
4 the engine failed (the partial rows are still written).

A call is one utterance through one engine. Per call it records, on the monotonic clock (time.perf_counter_ns):
  ttfa_ms     from the call to the first audio chunk the engine hands back (the call includes text chunking and
              grapheme-to-phoneme conversion; it excludes playback, which `playback` measures separately)
  total_ms    from the call to the last chunk; audio_ms is the audio produced; rtf = total_ms / audio_ms
  gaps_ms     time between consecutive chunks; stall_ms is how long a player that started at the first chunk and
              plays at 1x would have waited for audio (0 means the stream never underran)
  cpu_cores   process CPU time over wall time during the call (1.0 = one core busy)
Phases: `cold` (first call after the model loads, in this process; --cold-runs N adds N fresh processes), `first`
(each utterance once, in order: new text and new lengths, which is what dynamic replies look like), `repeat` (reps
1..N-1, interleaved), `idle` (a short utterance after sleeping G seconds, to catch GPU clock ramp-up), `barge` (the
long passage, paced at 1x with a bounded lookahead, cancelled D ms after its first audio: stop_ms is how long the
engine keeps computing after the cancel, next_ttfa_ms how soon the next utterance starts).
GPU: NVML is sampled every 50 ms (device memory, SM clock, utilisation, power); nvidia-smi and
`python C:/temp/gpu_lock.py snapshot` are recorded before and after the run, and this process's dedicated VRAM is
read from the Windows GPU performance counters after load and at the end.
Output: results/tts/<engine-id>_<tag>.json (manifest, per-phase summary, per-call rows). Audio of the `first` phase
goes to .cache/tts/<run>/ for scoring (not committed); --keep-wavs copies a few samples to results/tts/samples/.
`score` adds intelligibility (WER of faster-whisper on the audio, CPU by default) and a naturalness proxy
(UTMOS22-strong, via SpeechMOS pinned to v1.2.0) to an existing result file.

A run is marked timing_valid=false, a functional check rather than a measurement, when it was a --smoke run; when the
system CPU (> --cpu-idle-max) or the GPU (> 10 %) was busy before the load; when this process spilled into shared GPU
memory (> --spill-max-mb); or when it evicted other processes' VRAM (its dedicated memory exceeds the device total's
growth by > --evict-max-mb: WDDM oversubscribes silently).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata as md
import json
import os
import platform
import queue
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UTTS = ROOT / "data" / "voice" / "tts_utterances.json"
OUT_DIR = ROOT / "results" / "tts"
CACHE = ROOT / ".cache" / "tts"
MODELS = ROOT / "tools" / "tts"
GPU_LOCK = Path(os.environ.get("GPU_LOCK_PY", "C:/temp/gpu_lock.py"))
HARNESS_VERSION = 1

EXIT_DEPS, EXIT_LOCKED, EXIT_ENGINE = 2, 3, 4
WARMUP_TEXTS = ["This is a warm-up sentence.", "Another short one, to settle the kernels.", "Ready."]


def now_ns() -> int:
    return time.perf_counter_ns()


def ms(ns: int) -> float:
    return round(ns / 1e6, 3)


# ---------------------------------------------------------------------------------------------------------------------
# text chunking (applied the same way to every engine that does not stream text in by itself)
# ---------------------------------------------------------------------------------------------------------------------

_SENT = re.compile(r"(?<=[.!?])\s+")
_CLAUSE = re.compile(r"(?<=[,;:])\s+")


def split_text(text: str, policy: str) -> list[str]:
    """none: the whole text. sentence: split after . ! ?. clause: also after , ; : (pieces under 3 words are merged
    into the next one, so a lone "Okay," does not become its own call). first-clause: the first clause alone, then
    whole sentences (a small first piece for time to first audio, larger later pieces for prosody)."""
    text = " ".join(text.split())
    if policy in ("none", "native"):
        return [text]
    sents = [s for s in _SENT.split(text) if s]
    if policy == "sentence":
        return sents
    clauses = [c for s in sents for c in _CLAUSE.split(s) if c]
    merged: list[str] = []
    carry = ""
    for c in clauses:
        cur = f"{carry} {c}".strip() if carry else c
        if len(cur.split()) < 3 and c is not clauses[-1]:
            carry = cur
            continue
        merged.append(cur)
        carry = ""
    if carry:
        merged.append(carry)
    if policy == "clause":
        return merged
    if policy == "first-clause":
        first = merged[0]
        rest = text[len(first):].strip()
        return [first] + ([s for s in _SENT.split(rest) if s] if rest else [])
    raise ValueError(f"unknown chunking policy {policy!r}")


# ---------------------------------------------------------------------------------------------------------------------
# engines: one interface, lazy imports
# ---------------------------------------------------------------------------------------------------------------------

class MissingDependency(RuntimeError):
    pass


def need(module: str, pip: str):
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # noqa: PERF203
        raise MissingDependency(f"{module} is not importable ({exc}); install it into .venv-tts: {pip}") from exc


class Engine:
    """Adapter contract. load() does the heavy work once; synth() yields float32 mono chunks for one piece of text
    and should check `cancel` whenever the engine can stop; extras collects per-call side measurements (g2p_ms)."""

    name = "base"
    sample_rate = 24000
    native_streaming = False  # synth() yields several chunks for one piece on its own
    default_chunking = "clause"
    languages: frozenset[str] = frozenset({"en"})

    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.extras: dict[str, float] = {}

    def engine_id(self) -> str:
        return self.name

    def load(self) -> dict:
        return {}

    def synth(self, text: str, lang: str, cancel: threading.Event) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def model_files(self) -> list[Path]:
        return []

    def packages(self) -> list[str]:
        return []

    def gpu_memory(self) -> dict:
        return {}

    def close(self) -> None:
        pass


def ort_memcpy_probe(model: Path, conv_search: str) -> dict:
    """Count the Memcpy nodes onnxruntime inserts when it partitions the graph for CUDA (each one is a host<->device
    copy on every call). onnxruntime logs the count as a warning from native code, which Python cannot capture in
    its own process, so a short-lived child creates the same session and its stderr is read. Run before the real
    session exists, so the two never hold VRAM at once."""
    code = ("import onnxruntime as ort; getattr(ort, 'preload_dlls', lambda: None)(); "
            "so = ort.SessionOptions(); so.log_severity_level = 2; "
            f"ort.InferenceSession(r'{model}', so, providers=[('CUDAExecutionProvider', "
            f"{{'device_id': 0, 'cudnn_conv_algo_search': '{conv_search}'}}), 'CPUExecutionProvider'])")
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=600)
    except Exception as exc:
        return {"memcpy_nodes": None, "memcpy_probe_error": repr(exc)}
    # onnxruntime's Windows logger writes UTF-16 (wide) text, sometimes interleaved with narrow text
    log = p.stderr.replace(b"\x00", b"").decode("utf-8", "replace")
    log = re.sub(r"\x1b\[[0-9;]*m", "", log)
    m = re.search(r"(\d+) Memcpy nodes are added", log)
    return {"memcpy_nodes": int(m.group(1)) if m else 0, "memcpy_probe_rc": p.returncode,
            "memcpy_probe_log": log[m.start():m.start() + 120] if m else log.strip()[-240:]}


def ort_session(model: Path, device: str, conv_search: str, threads: int) -> tuple[Any, dict]:
    ort = need("onnxruntime", 'pip install "onnxruntime-gpu[cuda,cudnn]==1.30.0"')
    info: dict[str, Any] = {"onnxruntime": ort.__version__, "available_providers": ort.get_available_providers()}
    if device == "cuda" and hasattr(ort, "preload_dlls"):
        with contextlib.suppress(Exception):
            ort.preload_dlls()  # onnxruntime-gpu ships no CUDA DLLs; the nvidia-* wheels' bin/ are not on PATH
    so = ort.SessionOptions()
    so.log_severity_level = 2
    if threads:
        so.intra_op_num_threads = threads
    if device == "cuda":
        providers: list = [("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": conv_search}),
                           "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    if device == "cuda":
        info.update(ort_memcpy_probe(model, conv_search))
    sess = ort.InferenceSession(str(model), so, providers=providers)
    info.update(providers_requested=[p if isinstance(p, str) else p[0] for p in providers],
                providers_active=sess.get_providers(), cudnn_conv_algo_search=conv_search if device == "cuda" else None,
                intra_op_threads=threads or "default")
    if device == "cuda" and sess.get_providers()[0] != "CUDAExecutionProvider":
        info["warning"] = "CUDA was requested but the session runs on " + sess.get_providers()[0]
    return sess, info


class SineEngine(Engine):
    """Harness self-test: a tone whose synthesis time is simulated (--sine-rtf), in chunks of 200 ms."""

    name = "sine"
    native_streaming = True

    def synth(self, text, lang, cancel):
        dur = 0.06 * len(text.split()) + 0.2
        n = int(dur * self.sample_rate)
        t = np.arange(n) / self.sample_rate
        audio = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        step = int(0.2 * self.sample_rate)
        for i in range(0, n, step):
            if cancel.is_set():
                return
            time.sleep(self.a.sine_rtf * step / self.sample_rate)
            yield audio[i:i + step]


class KokoroOnnx(Engine):
    """Kokoro-82M v1.0 through kokoro-onnx 0.6.1 on onnxruntime (the runtime Theioma's sidecar uses).
    --variant fp32|fp16|int8 picks the export from the kokoro-onnx model-files-v1.0 release (Theioma runs fp16).
    --g2p misaki feeds English through misaki (lexicon first, espeak-ng for unknown words), as hexgrad's PyTorch
    package does, instead of kokoro-onnx's plain espeak-ng phonemisation."""

    name = "kokoro-onnx"
    languages = frozenset({"en", "es", "fr", "it", "pt-br", "hi", "ja", "zh"})
    LANG = {"en": "en-us", "es": "es", "fr": "fr-fr", "it": "it", "pt-br": "pt-br", "hi": "hi", "ja": "ja", "zh": "cmn"}
    FILES = {"fp32": "kokoro-v1.0.onnx", "fp16": "kokoro-v1.0.fp16.onnx", "int8": "kokoro-v1.0.int8.onnx"}

    def engine_id(self):
        g2p = "-misaki" if self.a.g2p == "misaki" else ""
        return f"kokoro-onnx-{self.a.variant or 'fp32'}{g2p}-{self.a.device}"

    def model_files(self):
        d = MODELS / "kokoro-onnx"
        return [d / self.FILES[self.a.variant or "fp32"], d / "voices-v1.0.bin"]

    def packages(self):
        return ["kokoro-onnx", "onnxruntime-gpu", "onnxruntime", "phonemizer", "espeakng-loader", "misaki", "spacy"]

    def load(self):
        kokoro_onnx = need("kokoro_onnx", "pip install --no-deps kokoro-onnx==0.6.1")
        model, voices = self.model_files()
        sess, info = ort_session(model, self.a.device, self.a.ort_conv_search, self.a.threads)
        self.k = kokoro_onnx.Kokoro.from_session(sess, str(voices))
        self.voice = self.a.voice or "af_heart"
        self.misaki = None
        if self.a.g2p == "misaki":
            need("misaki", "pip install --no-deps misaki==0.9.4 (see tts_requirements.txt)")
            from misaki import en, espeak
            self.misaki = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False))
        info.update(voice=self.voice, g2p=self.a.g2p)
        return info

    def synth(self, text, lang, cancel):
        t0 = now_ns()
        if self.misaki is not None and lang == "en":
            ph, _ = self.misaki(text)
        else:
            ph = self.k.tokenizer.phonemize(text, self.LANG.get(lang, "en-us"))
        self.extras["g2p_ms"] = self.extras.get("g2p_ms", 0.0) + ms(now_ns() - t0)
        audio, _ = self.k.create(ph, voice=self.voice, speed=self.a.speed, is_phonemes=True)
        yield np.asarray(audio, dtype=np.float32)


class KokoroTorch(Engine):
    """Kokoro-82M v1.0 through hexgrad's `kokoro` package (PyTorch, misaki G2P with espeak-ng fallback)."""

    name = "kokoro-torch"
    languages = frozenset({"en", "es", "fr", "it", "pt-br", "hi", "ja", "zh"})
    LANG = {"en": "a", "es": "e", "fr": "f", "it": "i", "pt-br": "p", "hi": "h", "ja": "j", "zh": "z"}

    def engine_id(self):
        return f"kokoro-torch-{self.a.device}"

    def packages(self):
        return ["kokoro", "misaki", "torch", "spacy", "phonemizer", "espeakng-loader"]

    def model_files(self):
        with contextlib.suppress(Exception):
            from huggingface_hub import try_to_load_from_cache
            p = try_to_load_from_cache("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
            v = try_to_load_from_cache("hexgrad/Kokoro-82M", f"voices/{self.a.voice or 'af_heart'}.pt")
            return [Path(x) for x in (p, v) if isinstance(x, str)]
        return []

    def load(self):
        self.torch = need("torch", "pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130")
        kokoro = need("kokoro", "pip install --no-deps kokoro==0.9.4 misaki==0.9.4 (see tts_requirements.txt)")
        self.pipes: dict[str, Any] = {}
        self.model = kokoro.KModel(repo_id="hexgrad/Kokoro-82M").to(self.a.device).eval()
        self.voice = self.a.voice or "af_heart"
        self._pipe("en")
        return {"torch": self.torch.__version__, "cuda": self.torch.version.cuda, "device": self.a.device,
                "voice": self.voice}

    def _pipe(self, lang):
        if lang not in self.pipes:
            from kokoro import KPipeline
            self.pipes[lang] = KPipeline(lang_code=self.LANG.get(lang, "a"), repo_id="hexgrad/Kokoro-82M", model=self.model)
        return self.pipes[lang]

    def synth(self, text, lang, cancel):
        pipe = self._pipe(lang)
        with self.torch.inference_mode():
            for r in pipe(text, voice=self.voice, speed=self.a.speed, split_pattern=None):
                if r.audio is not None:
                    yield r.audio.detach().float().cpu().numpy().ravel()
                if cancel.is_set():
                    return

    def gpu_memory(self):
        if self.a.device != "cuda":
            return {}
        c = self.torch.cuda
        return {"torch_max_allocated_mb": round(c.max_memory_allocated() / 2**20, 1),
                "torch_max_reserved_mb": round(c.max_memory_reserved() / 2**20, 1)}


class PiperEngine(Engine):
    """Piper (piper-tts 1.8.0, GPL-3.0-or-later) VITS voices on onnxruntime; one chunk per sentence."""

    name = "piper"
    native_streaming = True
    default_chunking = "clause"

    def engine_id(self):
        return f"piper-{self.a.voice or 'en_US-lessac-medium'}-{self.a.device}"

    def model_files(self):
        v = self.a.voice or "en_US-lessac-medium"
        return [MODELS / "piper" / f"{v}.onnx", MODELS / "piper" / f"{v}.onnx.json"]

    def packages(self):
        return ["piper-tts", "onnxruntime-gpu", "onnxruntime"]

    def load(self):
        piper = need("piper", "pip install --no-deps piper-tts==1.8.0 pathvalidate")
        model, cfg = self.model_files()
        self.v = piper.PiperVoice.load(model, cfg, use_cuda=self.a.device == "cuda")
        self.sample_rate = self.v.config.sample_rate
        return {"providers_active": self.v.session.get_providers(), "sample_rate": self.sample_rate}

    def synth(self, text, lang, cancel):
        for chunk in self.v.synthesize(text):
            yield np.asarray(chunk.audio_float_array, dtype=np.float32)
            if cancel.is_set():
                return


class SherpaEngine(Engine):
    """sherpa-onnx 1.13.8 (Apache-2.0) offline TTS; it runs several families from one runtime and calls back once per
    sentence, and the callback can stop generation (true mid-reply cancellation). CPU wheel; CUDA needs sherpa's own
    CUDA build. --model picks a directory under tools/tts/sherpa/."""

    name = "sherpa"
    native_streaming = True
    default_chunking = "clause"
    MODELS = {
        "kokoro-int8-multi-lang-v1_0": "kokoro",
        "kokoro-multi-lang-v1_0": "kokoro",
        "kitten-nano-en-v0_8-int8": "kitten",
        "kitten-mini-en-v0_8": "kitten",
        "kitten-micro-en-v0_8": "kitten",
        "sherpa-onnx-supertonic-3-tts-int8-2026-05-11": "supertonic",
        "vits-piper-en_US-lessac-medium": "vits",
    }

    def engine_id(self):
        return f"sherpa-{self.a.model.replace('sherpa-onnx-', '')}-{self.a.device}"

    def _dir(self) -> Path:
        return MODELS / "sherpa" / self.a.model

    def model_files(self):
        d = self._dir()
        return sorted(p for p in d.glob("*") if p.suffix in (".onnx", ".bin", ".json") and p.is_file())

    def packages(self):
        return ["sherpa-onnx", "sherpa-onnx-core"]

    def languages_for(self):
        return self.languages

    def load(self):
        so = need("sherpa_onnx", "pip install --no-deps sherpa-onnx==1.13.8 sherpa-onnx-core==1.13.8")
        fam = self.MODELS.get(self.a.model)
        if fam is None:
            raise MissingDependency(f"unknown sherpa model {self.a.model!r}; known: {sorted(self.MODELS)}")
        d = self._dir()
        if not d.is_dir():
            raise MissingDependency(f"{d} is missing; download {self.a.model}.tar.bz2 from the sherpa-onnx tts-models release")

        def pick(*names):
            for n in names:
                hits = sorted(d.glob(n))
                if hits:
                    return str(hits[0])
            return ""

        mc = so.OfflineTtsModelConfig(num_threads=self.a.threads or 4, provider=self.a.device, debug=False)
        if fam == "kokoro":
            lex = ",".join(x for x in (pick("lexicon-us-en.txt"), pick("lexicon-zh.txt")) if x)
            mc.kokoro = so.OfflineTtsKokoroModelConfig(model=pick("model.int8.onnx", "model.onnx", "*.onnx"),
                                                       voices=pick("voices.bin"), tokens=pick("tokens.txt"),
                                                       lexicon=lex, data_dir=pick("espeak-ng-data"),
                                                       dict_dir=pick("dict"))
            self.languages = frozenset({"en", "zh"})
        elif fam == "kitten":
            mc.kitten = so.OfflineTtsKittenModelConfig(model=pick("model.int8.onnx", "model.fp16.onnx", "model.onnx",
                                                                  "*.onnx"),
                                                       voices=pick("voices.bin"), tokens=pick("tokens.txt"),
                                                       data_dir=pick("espeak-ng-data"))
        elif fam == "supertonic":
            mc.supertonic = so.OfflineTtsSupertonicModelConfig(
                duration_predictor=pick("duration_predictor*.onnx"), text_encoder=pick("text_encoder*.onnx"),
                vector_estimator=pick("vector_estimator*.onnx"), vocoder=pick("vocoder*.onnx"),
                tts_json=pick("tts.json", "*.json"), unicode_indexer=pick("unicode_indexer*"),
                voice_style=pick("voice_styles/*.json", "voice*.bin", "*.bin"))
        elif fam == "vits":
            mc.vits = so.OfflineTtsVitsModelConfig(model=pick("*.onnx"), tokens=pick("tokens.txt"),
                                                   data_dir=pick("espeak-ng-data"))
        cfg = so.OfflineTtsConfig(model=mc, max_num_sentences=1)
        if not cfg.validate():
            raise MissingDependency(f"sherpa-onnx rejected the config for {d} (files: {[p.name for p in d.iterdir()]})")
        self.tts = so.OfflineTts(cfg)
        self.sample_rate = self.tts.sample_rate
        self.sid = int(self.a.sid) if self.a.sid is not None else 0
        return {"family": fam, "sample_rate": self.sample_rate, "num_speakers": self.tts.num_speakers,
                "sid": self.sid, "provider": self.a.device, "num_threads": self.a.threads or 4}

    def synth(self, text, lang, cancel):
        q: queue.Queue = queue.Queue()
        stop_value = self.a.sherpa_stop_value
        closed = threading.Event()

        def cb(samples, progress):
            q.put(np.array(samples, dtype=np.float32))
            return stop_value if cancel.is_set() or closed.is_set() else 1 - stop_value

        def work():
            try:
                self.tts.generate(text, sid=self.sid, speed=self.a.speed, callback=cb)
            except Exception as exc:  # hand it to the consumer
                q.put(exc)
            q.put(None)

        th = threading.Thread(target=work, daemon=True)
        th.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            # a cancelled call has stopped only when generate() returns (after the sentence in flight): wait for
            # it, so stop_ms counts the engine's remaining compute and the next call does not overlap it
            closed.set()
            th.join()


class PocketTts(Engine):
    """Kyutai Pocket TTS (pocket-tts, PyTorch, CPU-first) with its own streaming generator."""

    name = "pocket-tts"
    native_streaming = True
    default_chunking = "native"

    def engine_id(self):
        return f"pocket-tts{'-int8' if self.a.variant == 'int8' else ''}-{self.a.device}"

    def packages(self):
        return ["pocket-tts", "torch"]

    def load(self):
        self.torch = need("torch", "pip install torch (CPU is enough)")
        pocket = need("pocket_tts", "pip install --no-deps pocket-tts==3.3.0")
        self.model = pocket.TTSModel.load_model(quantize=self.a.variant == "int8")
        if self.a.device == "cuda" and hasattr(self.model, "to"):
            self.model = self.model.to("cuda")
        self.voice = self.a.voice or "alba"
        self.state = self.model.get_state_for_audio_prompt(self.voice)
        self.sample_rate = int(getattr(self.model, "sample_rate", 24000))
        self.stream_api = hasattr(self.model, "generate_audio_stream")
        return {"voice": self.voice, "sample_rate": self.sample_rate, "streaming_api": self.stream_api,
                "torch": self.torch.__version__}

    def synth(self, text, lang, cancel):
        # no inference_mode here: the stream generates and decodes on its own threads, which would then meet
        # inference tensors outside inference mode (RuntimeError on the KV-cache update)
        if self.stream_api:
            for chunk in self.model.generate_audio_stream(self.state, text, stop=cancel):
                yield chunk.detach().float().cpu().numpy().ravel()
                if cancel.is_set():
                    return
        else:
            yield self.model.generate_audio(self.state, text).detach().float().cpu().numpy().ravel()


ENGINES: dict[str, type[Engine]] = {e.name: e for e in (SineEngine, KokoroOnnx, KokoroTorch, PiperEngine, SherpaEngine,
                                                        PocketTts)}


# ---------------------------------------------------------------------------------------------------------------------
# measurement helpers: GPU sampler, snapshots, environment
# ---------------------------------------------------------------------------------------------------------------------

class GpuSampler(threading.Thread):
    """NVML every `period` s: device memory used, SM and memory clocks, utilisation, power, temperature, P-state,
    plus system-wide CPU %. Phases are time windows; summaries are computed per window."""

    def __init__(self, period: float = 0.05):
        super().__init__(daemon=True)
        self.period, self.rows, self.ok = period, [], False
        self._halt = threading.Event()
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv, self.h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
            self.ok = True
        except Exception as exc:  # no NVML: the run still works, without GPU samples
            self.err = repr(exc)
        try:
            import psutil
            self.ps = psutil
            psutil.cpu_percent(None)
        except ImportError:
            self.ps = None

    def read(self) -> dict:
        nv, h = self.nv, self.h
        mem = nv.nvmlDeviceGetMemoryInfo(h)
        util = nv.nvmlDeviceGetUtilizationRates(h)
        return {"mem_mb": mem.used / 2**20, "sm_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                "memclk_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM), "util": util.gpu,
                "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000, "temp_c": nv.nvmlDeviceGetTemperature(h, 0),
                "pstate": nv.nvmlDeviceGetPerformanceState(h)}

    def static(self) -> dict:
        if not self.ok:
            return {"nvml": False, "error": getattr(self, "err", "")}
        nv, h = self.nv, self.h
        out = {"nvml": True}
        for k, f in (("name", lambda: nv.nvmlDeviceGetName(h)), ("driver", nv.nvmlSystemGetDriverVersion),
                     ("cuda_driver", lambda: nv.nvmlSystemGetCudaDriverVersion_v2()),
                     ("mem_total_mb", lambda: nv.nvmlDeviceGetMemoryInfo(h).total / 2**20),
                     ("sm_max_mhz", lambda: nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_SM)),
                     ("mem_max_mhz", lambda: nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_MEM)),
                     ("power_limit_w", lambda: nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1000)):
            with contextlib.suppress(Exception):
                v = f()
                out[k] = v.decode() if isinstance(v, bytes) else v
        return out

    def run(self):
        while not self._halt.is_set():
            row = {"t": now_ns()}
            if self.ok:
                with contextlib.suppress(Exception):
                    row.update(self.read())
            if self.ps:
                row["cpu_sys_pct"] = self.ps.cpu_percent(None)
            self.rows.append(row)
            self._halt.wait(self.period)

    def stop(self):
        self._halt.set()
        self.join(timeout=2)

    def summary(self, t0: int, t1: int) -> dict:
        rows = [r for r in self.rows if t0 <= r["t"] <= t1]
        if not rows:
            return {"n": 0}
        out: dict[str, Any] = {"n": len(rows)}

        def col(k):
            return [r[k] for r in rows if k in r]

        for k in ("mem_mb", "sm_mhz", "memclk_mhz", "util", "power_w", "temp_c", "cpu_sys_pct"):
            v = col(k)
            if v:
                out[k] = {"min": round(min(v), 1), "median": round(statistics.median(v), 1), "max": round(max(v), 1)}
        p = col("pstate")
        if p:
            out["pstates"] = sorted(set(p))
        return out


def run_cmd(cmd: list[str], timeout: float = 60) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as exc:
        return f"error: {exc!r}"


def nvidia_smi() -> dict:
    q = ("name,driver_version,pstate,memory.used,memory.total,utilization.gpu,clocks.sm,clocks.mem,clocks.max.sm,"
         "power.draw,power.limit,temperature.gpu")
    out = run_cmd(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    vals = [v.strip() for v in out.split(",")]
    keys = q.split(",")
    return dict(zip(keys, vals)) if len(vals) == len(keys) else {"raw": out}


def gpu_lock(*args: str) -> dict | str:
    if not GPU_LOCK.exists():
        return {"error": f"{GPU_LOCK} not found"}
    out = run_cmd([sys.executable, str(GPU_LOCK), *args], timeout=3600)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return out


def own_vram_mb(pid: int | None = None) -> dict:
    """This process's dedicated and shared GPU memory from the Windows performance counters (NVML cannot report
    per-process memory under WDDM). Returns {} off Windows or on failure."""
    if os.name != "nt":
        return {}
    pid = pid or os.getpid()
    ps = (f"(Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage','\\GPU Process Memory(pid_{pid}_*)\\"
          "Shared Usage' -EA SilentlyContinue).CounterSamples | % { '{0}|{1}' -f $_.Path.Split('\\')[-1],"
          "($_.CookedValue/1MB) }")
    out = run_cmd(["powershell", "-NoProfile", "-Command", ps], timeout=60)
    res: dict[str, float] = {}
    for line in out.splitlines():
        if "|" in line:
            k, v = line.rsplit("|", 1)
            key = "dedicated_mb" if "dedicated" in k.lower() else "shared_mb"
            with contextlib.suppress(ValueError):
                res[key] = round(res.get(key, 0.0) + float(v.replace(",", ".")), 1)
    return res


def sha256_cached(p: Path) -> str | None:
    if not p.is_file():
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    idx_path = CACHE / "sha256.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    st = p.stat()
    key = f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    if key not in idx:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        idx[key] = h.hexdigest()
        idx_path.write_text(json.dumps(idx, indent=0))
    return idx[key]


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def environment(engine: Engine | None) -> dict:
    env: dict[str, Any] = {"python": sys.version.split()[0], "executable": rel(Path(sys.executable)),
                           "platform": platform.platform(), "machine": platform.machine()}
    if os.name == "nt":
        with contextlib.suppress(Exception):
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            env["cpu"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        env["power_scheme"] = run_cmd(["powercfg", "/getactivescheme"])
    with contextlib.suppress(Exception):
        import psutil
        env["logical_cpus"] = psutil.cpu_count()
        env["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        b = psutil.sensors_battery()
        env["on_ac_power"] = None if b is None else bool(b.power_plugged)
    wanted = {"numpy", "onnxruntime", "onnxruntime-gpu", "torch", "kokoro", "kokoro-onnx", "misaki", "phonemizer",
              "espeakng-loader", "piper-tts", "sherpa-onnx", "sherpa-onnx-core", "pocket-tts", "faster-whisper",
              "ctranslate2", "nvidia-ml-py", "sounddevice", "soundfile", "spacy", "transformers", "nvidia-cudnn-cu13",
              "nvidia-cublas", "nvidia-cuda-runtime", "scipy"}
    if engine:
        wanted |= set(engine.packages())
    pk = {}
    for d in md.distributions():
        n = (d.metadata["Name"] or "").lower()
        if n in wanted:
            pk[n] = d.version
    env["packages"] = dict(sorted(pk.items()))
    env["git_commit"] = run_cmd(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"])
    env["git_dirty"] = bool(run_cmd(["git", "-C", str(ROOT), "status", "--porcelain", "--", "experiments", "data"]))
    return env


def pct(xs: list[float], q: float) -> float | None:
    return round(float(np.percentile(xs, q)), 2) if xs else None


def dist(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "p50": pct(xs, 50), "p95": pct(xs, 95), "mean": round(statistics.fmean(xs), 2),
            "min": round(min(xs), 2), "max": round(max(xs), 2)}


# ---------------------------------------------------------------------------------------------------------------------
# one call, timed
# ---------------------------------------------------------------------------------------------------------------------

def timed_call(eng: Engine, text: str, lang: str, chunking: str, cancel: threading.Event | None = None,
               pace_lookahead_ms: float | None = None, on_first=None) -> tuple[dict, np.ndarray]:
    """Run one utterance and time it. With pace_lookahead_ms the consumer behaves like a 1x player with a bounded
    buffer: it waits whenever more than that much synthesised audio is queued ahead of playback."""
    cancel = cancel or threading.Event()
    eng.extras = {}
    sr = eng.sample_rate
    pieces = split_text(text, chunking if not (chunking == "native" and not eng.native_streaming) else "none")
    arrivals: list[int] = []
    lengths: list[int] = []
    chunks: list[np.ndarray] = []
    cpu0, t0 = time.process_time(), now_ns()
    cancelled_at_piece = None
    for i, piece in enumerate(pieces):
        gen = eng.synth(piece, lang, cancel)
        try:
            for audio in gen:
                t = now_ns()
                a = np.asarray(audio, dtype=np.float32).ravel()
                if not a.size:
                    continue
                arrivals.append(t)
                lengths.append(a.size)
                chunks.append(a)
                if len(arrivals) == 1 and on_first:
                    on_first(t)
                if pace_lookahead_ms is not None:
                    played_ns = now_ns() - arrivals[0]
                    queued_ms = sum(lengths) / sr * 1000 - played_ns / 1e6
                    while queued_ms > pace_lookahead_ms and not cancel.is_set():
                        time.sleep(min(0.01, (queued_ms - pace_lookahead_ms) / 1000))
                        queued_ms = sum(lengths) / sr * 1000 - (now_ns() - arrivals[0]) / 1e6
                if cancel.is_set():
                    break
        finally:
            gen.close()
        if cancel.is_set():
            cancelled_at_piece = i
            break
    t_end = now_ns()
    cpu_s = time.process_time() - cpu0
    audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    wall_ms = ms(t_end - t0)
    row: dict[str, Any] = {"chars": len(text), "pieces": len(pieces), "n_chunks": len(arrivals), "total_ms": wall_ms,
                           "audio_ms": round(audio.size / sr * 1000, 1), "cpu_s": round(cpu_s, 4),
                           "cpu_cores": round(cpu_s / max(wall_ms / 1000, 1e-9), 2)}
    if arrivals:
        row["ttfa_ms"] = ms(arrivals[0] - t0)
        row["rtf"] = round(wall_ms / max(row["audio_ms"], 1e-9), 4)
        gaps = [ms(b - a) for a, b in zip(arrivals, arrivals[1:])]
        row["gap_max_ms"] = max(gaps) if gaps else 0.0
        if len(gaps) <= 24:
            row["gaps_ms"] = gaps
        # a 1x player starting at the first chunk: chunk k is needed when the audio before it has played
        need_at = np.cumsum([0] + lengths[:-1]) / sr * 1e9 + arrivals[0]
        stalls = np.maximum(0, np.array(arrivals) - need_at)
        # each stall delays everything after it, so accumulate the running delay
        delay, total = 0.0, 0.0
        for arr, need_t in zip(arrivals, need_at):
            late = arr - (need_t + delay)
            if late > 0:
                delay += late
                total += late
        row["stall_ms"] = round(total / 1e6, 2)
        row["stalls"] = int((stalls > 0).sum())
    for k, v in eng.extras.items():
        row[k] = round(v, 3)
    if cancelled_at_piece is not None:
        row["cancelled_at_piece"] = cancelled_at_piece
    return row, audio


# ---------------------------------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------------------------------

def load_utterances(tiers: set[str]) -> tuple[list[dict], dict]:
    data = json.loads(UTTS.read_text(encoding="utf-8"))
    rows = [u for u in data["utterances"] if u["tier"] in tiers]
    return rows, data


def make_engine(a: argparse.Namespace) -> Engine:
    if a.engine not in ENGINES:
        raise SystemExit(f"unknown engine {a.engine!r}; known: {', '.join(ENGINES)}")
    eng = ENGINES[a.engine](a)
    if a.chunking == "auto":
        a.chunking = eng.default_chunking
    return eng


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1, 1), sr, subtype="PCM_16")


def cmd_cold_child(a: argparse.Namespace) -> int:
    """One fresh process: import, load, first call. Prints one JSON line."""
    t0 = now_ns()
    eng = make_engine(a)
    try:
        info = eng.load()
    except MissingDependency as exc:
        print(json.dumps({"error": str(exc)}))
        return EXIT_DEPS
    t1 = now_ns()
    utts, _ = load_utterances({"clip", "short"})
    call_epoch = time.time()
    row, _ = timed_call(eng, utts[0]["text"], "en", a.chunking)
    out = {"load_ms": ms(t1 - t0), "first_call": row, "providers": info.get("providers_active")}
    with contextlib.suppress(Exception):
        import psutil
        # process creation -> first audio: interpreter start, imports, model load and the first call together
        out["process_to_first_audio_ms"] = round((call_epoch - psutil.Process().create_time()) * 1000
                                                 + row.get("ttfa_ms", 0.0), 1)
    print(json.dumps(out))
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    eng = make_engine(a)
    tiers = set(a.tiers.split(","))
    utts, data = load_utterances(tiers)
    if a.limit:
        utts = utts[: a.limit]
    run_name = f"{eng.engine_id()}_{a.tag}"
    out_path = OUT_DIR / f"{run_name}.json"
    audio_dir = CACHE / run_name
    kind = "smoke" if a.smoke else "timed"
    reasons: list[str] = ["smoke test: a functional check, not a timing"] if a.smoke else []

    lock_note = None
    if a.lock != "none" and a.device == "cuda":
        st = gpu_lock("status")
        holder = (st.get("lock") or {}).get("owner") if isinstance(st, dict) else None
        stale = isinstance(st, dict) and st.get("stale")
        if a.lock == "check" and holder and holder != a.lock_owner and not stale:
            print(f"GPU lock is held by {holder}; not running (exit {EXIT_LOCKED})", file=sys.stderr)
            return EXIT_LOCKED
        if a.lock == "acquire":
            r = run_cmd([sys.executable, str(GPU_LOCK), "acquire", "--owner", a.lock_owner, "--purpose",
                         f"tts_bench {run_name}", "--minutes", str(a.lock_minutes), "--wait", "--pid", str(os.getpid())],
                        timeout=24 * 3600)
            lock_note = {"acquired": r[-300:]}
        else:
            lock_note = {"status": st}

    sampler = GpuSampler()
    static_gpu = sampler.static()
    before = {"nvidia_smi": nvidia_smi(), "gpu_lock_snapshot": gpu_lock("snapshot"), "own_vram": own_vram_mb()}
    others = [p for p in (before["gpu_lock_snapshot"].get("processes", []) if isinstance(before["gpu_lock_snapshot"], dict)
                          else []) if p.get("pid") != os.getpid() and p.get("kind") == "dedicated" and p.get("mb", 0) > 1024]
    if others and not a.smoke:
        reasons.append("other processes hold GPU memory at the start: "
                       + ", ".join(f"{p['name']} {p['mb']} MB" for p in others))
    sampler.start()
    time.sleep(0.5)
    phases: dict[str, tuple[int, int]] = {}
    rows: list[dict] = []
    t_idle0 = now_ns()
    time.sleep(1.0)
    phases["baseline"] = (t_idle0, now_ns())
    base = sampler.summary(*phases["baseline"])
    base_mem = base.get("mem_mb", {}).get("median")
    if not a.smoke and (base.get("util", {}).get("median") or 0) > 10:
        reasons.append("GPU utilisation above 10 % before the engine loaded")
    if not a.smoke and (base.get("cpu_sys_pct", {}).get("median") or 0) > a.cpu_idle_max:
        reasons.append(f"system CPU at {base['cpu_sys_pct']['median']} % before the engine loaded "
                       f"(limit {a.cpu_idle_max} %)")

    contender = Contender(a.contend_url) if a.contend_url else None

    exit_code = 0
    result: dict[str, Any] = {}
    try:
        t0 = now_ns()
        try:
            load_info = eng.load()
        except MissingDependency as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_DEPS
        t1 = now_ns()
        phases["load"] = (t0, t1)
        after_load_vram = own_vram_mb()
        load_mem = sampler.summary(t1 - 200_000_000, now_ns()).get("mem_mb", {}).get("median")

        # cold: first call after load
        tc = now_ns()
        row, _ = timed_call(eng, utts[0]["text"], utts[0].get("lang", "en"), a.chunking)
        rows.append({"phase": "cold", "utt": utts[0]["id"], "tier": utts[0]["tier"], "rep": 0, **row})
        for w in WARMUP_TEXTS:
            timed_call(eng, w, "en", a.chunking)
        phases["cold+warmup"] = (tc, now_ns())

        if contender:
            contender.start()

        # first: each utterance once; its audio is kept for scoring
        tf = now_ns()
        audio_dir.mkdir(parents=True, exist_ok=True)
        for u in utts:
            lang = u.get("lang", "en")
            if lang != "en" and lang not in eng.languages:
                continue
            row, audio = timed_call(eng, u["text"], lang, a.chunking)
            rows.append({"phase": "first", "utt": u["id"], "tier": u["tier"], "rep": 0, **row})
            write_wav(audio_dir / f"{u['id']}.wav", audio, eng.sample_rate)
        phases["first"] = (tf, now_ns())

        # repeat: reps 1..N-1, interleaved
        tr = now_ns()
        for rep in range(1, a.reps):
            for u in utts:
                lang = u.get("lang", "en")
                if u["tier"] == "barge" or (lang != "en" and lang not in eng.languages):
                    continue
                row, _ = timed_call(eng, u["text"], lang, a.chunking)
                rows.append({"phase": "repeat", "utt": u["id"], "tier": u["tier"], "rep": rep, **row})
        phases["repeat"] = (tr, now_ns())

        # idle: GPU clocks drop within seconds on a laptop; the next command pays for the ramp
        ti = now_ns()
        short = [u for u in utts if u["tier"] in ("clip", "short")][:2]
        for gap in [float(g) for g in a.idle_gaps.split(",") if g.strip()]:
            for u in short:
                time.sleep(gap)
                pre = sampler.read() if sampler.ok else {}
                row, _ = timed_call(eng, u["text"], "en", a.chunking)
                rows.append({"phase": "idle", "utt": u["id"], "tier": u["tier"], "rep": 0, "idle_s": gap,
                             "sm_mhz_at_call": pre.get("sm_mhz"), "pstate_at_call": pre.get("pstate"), **row})
        phases["idle"] = (ti, now_ns())

        # barge-in
        tb = now_ns()
        barge = [u for u in load_utterances({"barge"})[0]]
        if barge and a.barge_trials:
            for k in range(a.barge_trials):
                rows.append({"phase": "barge", "utt": barge[0]["id"], "tier": "barge", "rep": k,
                             **barge_trial(eng, barge[0]["text"], a)})
        phases["barge"] = (tb, now_ns())

        if contender:
            contender.stop()
        end_vram = own_vram_mb()
        gpu_mem = eng.gpu_memory()
    except MissingDependency:
        raise
    except Exception:  # record what happened, keep the partial rows
        import traceback
        result["error"] = traceback.format_exc()[-4000:]
        print(result["error"], file=sys.stderr)
        exit_code = EXIT_ENGINE
        load_info = locals().get("load_info", {})
        after_load_vram = locals().get("after_load_vram", {})
        end_vram, gpu_mem, load_mem = {}, {}, locals().get("load_mem")
    finally:
        sampler.stop()
        eng.close()
        if a.lock == "acquire" and a.device == "cuda":
            run_cmd([sys.executable, str(GPU_LOCK), "release", "--owner", a.lock_owner])
    after = {"nvidia_smi": nvidia_smi(), "gpu_lock_snapshot": gpu_lock("snapshot")}

    cold_runs = []
    if a.cold_runs and exit_code == 0:
        for _ in range(a.cold_runs):
            cmd = [sys.executable, str(Path(__file__).resolve()), "_cold", *child_args(a)]
            out = run_cmd(cmd, timeout=900)
            with contextlib.suppress(json.JSONDecodeError, IndexError):
                cold_runs.append(json.loads(out.strip().splitlines()[-1]))

    peak_mem = sampler.summary(phases.get("load", (0, 0))[0], now_ns()).get("mem_mb", {}).get("max")
    spill = max((end_vram or {}).get("shared_mb", 0.0), (after_load_vram or {}).get("shared_mb", 0.0))
    if a.device == "cuda" and spill > a.spill_max_mb:
        reasons.append(f"this process holds {spill} MB of shared (system) GPU memory: VRAM spilled, timings invalid")
    # Under WDDM an oversubscribed card does not fail: it evicts other processes' memory to system RAM. If this
    # process holds much more dedicated memory than the device total grew by, someone else was pushed out.
    own_ded = (end_vram or {}).get("dedicated_mb")
    evicted = round(own_ded - (peak_mem - base_mem), 1) if own_ded and peak_mem and base_mem else None
    if a.device == "cuda" and evicted is not None and evicted > a.evict_max_mb:
        reasons.append(f"VRAM oversubscribed: this process holds {own_ded} MB but the device total rose by only "
                       f"{round(peak_mem - base_mem, 1)} MB, so ~{evicted} MB of other processes was evicted")
    vram = {"device_used_before_mb": base_mem, "device_used_after_load_mb": load_mem, "device_used_peak_mb": peak_mem,
            "delta_after_load_mb": round(load_mem - base_mem, 1) if load_mem and base_mem else None,
            "delta_peak_mb": round(peak_mem - base_mem, 1) if peak_mem and base_mem else None,
            "own_process_after_load": after_load_vram, "own_process_end": end_vram, "evicted_others_mb": evicted,
            **gpu_mem,
            "note": "device deltas include this process's CUDA context and anything else that allocated meanwhile; "
                    "own_process values come from the Windows GPU performance counters"}

    model_files = [{"path": rel(p), "bytes": p.stat().st_size if p.exists() else None, "sha256": sha256_cached(p)}
                   for p in eng.model_files()]
    kept = []
    if a.keep_wavs and exit_code == 0:
        sample_dir = OUT_DIR / "samples" / run_name
        for uid in a.keep_ids.split(",")[: a.keep_wavs]:
            src = audio_dir / f"{uid}.wav"
            if src.exists():
                sample_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, sample_dir / src.name)
                kept.append(rel(sample_dir / src.name))

    result.update({
        "harness": {"script": rel(Path(__file__)), "version": HARNESS_VERSION, "sha256": sha256_file(Path(__file__)),
                    "argv": sys.argv[1:]},
        "run": {"name": run_name, "engine": a.engine, "engine_id": eng.engine_id(), "tag": a.tag, "kind": kind,
                "timing_valid": not reasons and exit_code == 0, "invalid_reasons": reasons,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "chunking": a.chunking, "reps": a.reps,
                "tiers": sorted(tiers), "contended_by": a.contend_url or None},
        "data": {"utterances": rel(UTTS), "sha256": sha256_file(UTTS), "n": len(utts)},
        "engine": {"load_ms": ms(phases["load"][1] - phases["load"][0]) if "load" in phases else None,
                   "info": load_info, "model_files": model_files, "sample_rate": eng.sample_rate,
                   "native_streaming": eng.native_streaming},
        "environment": {**environment(eng), "gpu": static_gpu},
        "gpu": {"before": before, "after": after, "lock": lock_note, "vram": vram,
                "phases": {k: sampler.summary(*v) for k, v in phases.items()}},
        "contention": contender.report() if contender else None,
        "cold_runs": cold_runs,
        "summary": summarize(rows, cold_runs),
        "audio": {"scoring_dir": rel(audio_dir), "kept_samples": kept},
        "rows": rows,
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"wrote {rel(out_path)}  ({'SMOKE' if a.smoke else 'timed'}, timing_valid={result['run']['timing_valid']})")
    print(json.dumps(result["summary"].get("headline", {}), indent=1))
    if a.score and exit_code == 0:
        return cmd_score(argparse.Namespace(result=str(out_path), asr_model=a.asr_model, asr_device=a.asr_device,
                                            asr_compute=a.asr_compute, mos=a.mos))
    return exit_code


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def child_args(a: argparse.Namespace) -> list[str]:
    out = ["--engine", a.engine, "--device", a.device, "--chunking", a.chunking, "--speed", str(a.speed),
           "--ort-conv-search", a.ort_conv_search, "--threads", str(a.threads), "--sherpa-stop-value",
           str(a.sherpa_stop_value)]
    for k in ("variant", "voice", "model", "sid", "g2p"):
        v = getattr(a, k)
        if v is not None:
            out += [f"--{k}", str(v)]
    return out


def barge_trial(eng: Engine, text: str, a: argparse.Namespace) -> dict:
    """Play the long passage at 1x (bounded lookahead), cancel `barge_after_ms` after its first audio, then start the
    next utterance at once. stop_ms: cancel -> the engine's call returned. next_ttfa_ms: cancel -> next first audio."""
    cancel = threading.Event()
    t_cancel = {}

    def on_first(t_first):
        def fire():
            t_cancel["t"] = now_ns()
            cancel.set()
        threading.Timer(a.barge_after_ms / 1000, fire).start()

    row, audio = timed_call(eng, text, "en", a.chunking, cancel=cancel, pace_lookahead_ms=a.lookahead_ms,
                            on_first=on_first)
    t_stop = now_ns()
    out = {k: row.get(k) for k in ("ttfa_ms", "audio_ms", "n_chunks", "pieces", "cancelled_at_piece")}
    if "t" not in t_cancel:
        out["note"] = "the passage finished before the cancel fired"
        return out
    out["stop_ms"] = ms(t_stop - t_cancel["t"])
    t_next = now_ns()
    nxt, _ = timed_call(eng, "Okay, stopping.", "en", a.chunking)
    out["next_ttfa_ms"] = round(ms(t_next - t_cancel["t"]) + nxt.get("ttfa_ms", 0.0), 3)
    out["next_call_ttfa_ms"] = nxt.get("ttfa_ms")
    return out


class Contender:
    """Optional: keep a llama-server decoding while the TTS runs (the realistic Theioma state), and record its tokens
    per second before and during. Only with --contend-url, only on a machine you hold the GPU lock for."""

    def __init__(self, url: str):
        self.url, self.rates, self.base = url.rstrip("/"), [], []
        self._halt = threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)
        for _ in range(3):
            r = self._one()
            if r:
                self.base.append(r)

    def _one(self) -> float | None:
        import urllib.request
        body = json.dumps({"prompt": "Write a short paragraph about the sea.", "n_predict": 64, "cache_prompt": False,
                           "temperature": 0.7}).encode()
        req = urllib.request.Request(self.url + "/completion", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                t = json.loads(resp.read()).get("timings", {})
                return t.get("predicted_per_second")
        except Exception:
            return None

    def _loop(self):
        while not self._halt.is_set():
            r = self._one()
            if r:
                self.rates.append(r)

    def start(self):
        self.th.start()

    def stop(self):
        self._halt.set()
        self.th.join(timeout=90)

    def report(self) -> dict:
        return {"url": self.url, "llm_tok_s_before": dist(self.base), "llm_tok_s_during": dist(self.rates)}


def summarize(rows: list[dict], cold_runs: list[dict]) -> dict:
    out: dict[str, Any] = {}

    def sel(phase=None, tiers=None):
        return [r for r in rows if (phase is None or r["phase"] in phase) and (tiers is None or r["tier"] in tiers)]

    for phase in ("cold", "first", "repeat", "idle"):
        rs = sel({phase})
        if not rs:
            continue
        by_tier = {}
        for tier in sorted({r["tier"] for r in rs}):
            t = [r for r in rs if r["tier"] == tier]
            by_tier[tier] = {"ttfa_ms": dist([r.get("ttfa_ms") for r in t]), "rtf": dist([r.get("rtf") for r in t]),
                             "stall_ms": dist([r.get("stall_ms") for r in t]),
                             "gap_max_ms": dist([r.get("gap_max_ms") for r in t]),
                             "cpu_cores": dist([r.get("cpu_cores") for r in t])}
            if any("g2p_ms" in r for r in t):
                by_tier[tier]["g2p_ms"] = dist([r.get("g2p_ms") for r in t])
        out[phase] = by_tier
    b = sel({"barge"})
    if b:
        out["barge"] = {"stop_ms": dist([r.get("stop_ms") for r in b]),
                        "next_ttfa_ms": dist([r.get("next_ttfa_ms") for r in b]), "n": len(b)}
    if cold_runs:
        ok = [c for c in cold_runs if "first_call" in c]
        out["cold_process"] = {"load_ms": dist([c["load_ms"] for c in ok]),
                               "first_ttfa_ms": dist([c["first_call"].get("ttfa_ms") for c in ok]),
                               "process_to_first_audio_ms": dist([c.get("process_to_first_audio_ms") for c in ok]),
                               "errors": [c.get("error") for c in cold_runs if "error" in c]}
    dyn = sel({"first"}, {"short", "medium", "long", "stress"})
    rep = sel({"repeat"}, {"short", "medium", "long", "stress"})
    out["headline"] = {
        "cold_first_call_ttfa_ms": (sel({"cold"}) or [{}])[0].get("ttfa_ms"),
        "clip_first_ttfa_ms": dist([r.get("ttfa_ms") for r in sel({"first"}, {"clip"})]),
        "dynamic_first_ttfa_ms": dist([r.get("ttfa_ms") for r in dyn]),
        "dynamic_repeat_ttfa_ms": dist([r.get("ttfa_ms") for r in rep]),
        "rtf_long_first": dist([r.get("rtf") for r in sel({"first"}, {"long"})]),
        "stall_ms_long_first": dist([r.get("stall_ms") for r in sel({"first"}, {"long"})]),
        "idle_ttfa_ms": dist([r.get("ttfa_ms") for r in sel({"idle"})]),
        "barge_stop_ms": out.get("barge", {}).get("stop_ms"),
    }
    return out


# ---------------------------------------------------------------------------------------------------------------------
# score: intelligibility (ASR WER) and a naturalness proxy (UTMOS22-strong)
# ---------------------------------------------------------------------------------------------------------------------

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + num_words(n % 100))
    if n < 1_000_000:
        return num_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + num_words(n % 1000))
    return " ".join(_ONES[int(d)] for d in str(n))


def normalise(s: str) -> list[str]:
    """Lower-case words for WER, with digits spelled out (whisper writes "3:45 PM", the reference says "three forty
    five p m"). Deliberately simple; the stress tier is reported apart from the rest because of it."""
    s = s.lower().replace("’", "'")
    s = re.sub(r"\$(\d[\d,]*)\.(\d\d)\b", lambda m: f"{m.group(1)} dollars and {m.group(2)} cents", s)
    s = re.sub(r"\$(\d[\d,]*)", r"\1 dollars", s)
    s = re.sub(r"(\d+):(\d\d)", lambda m: f"{m.group(1)} {m.group(2)}", s)
    s = re.sub(r"(\d+)\.(\d+)", lambda m: f"{m.group(1)} point " + " ".join(m.group(2)), s)
    s = s.replace("%", " percent").replace("/", " slash ").replace("dr.", "doctor")
    s = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", s)
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
    s = re.sub(r"(\d+)([a-z])\b", r"\1 \2", s)
    s = re.sub(r"\d+", lambda m: num_words(int(m.group(0))), s)
    s = re.sub(r"\b([a-z])\.(?=[a-z]\.)", r"\1 ", s)
    s = re.sub(r"\b(p\.?m|a\.?m)\b\.?", lambda m: " ".join(m.group(0).replace(".", "")), s)
    s = re.sub(r"(?<=[a-z])-(?=[a-z])", "", s)  # lo-fi / lofi, e-mail / email
    s = s.replace("-", " ")
    s = re.sub(r"(?<=[a-z]{2})is(e|ed|es|ing)\b", r"iz\1", s)  # synthesised / synthesized
    s = re.sub(r"[^a-z0-9' àáâãçéêíóôõúñü]+", " ", s)
    return s.split()


def wer(ref: list[str], hyp: list[str]) -> float:
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    d[:, 0] = np.arange(len(ref) + 1)
    d[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    return float(d[len(ref), len(hyp)] / max(1, len(ref)))


def to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return audio.astype(np.float32)
    try:
        from scipy.signal import resample_poly
        g = np.gcd(16000, sr)
        return resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
    except ImportError:
        x = np.linspace(0, len(audio), int(len(audio) * 16000 / sr), endpoint=False)
        return np.interp(x, np.arange(len(audio)), audio).astype(np.float32)


def torchaudio_shim() -> bool:
    """SpeechMOS declares torchaudio and uses it only to resample input that is not 16 kHz. torchaudio has no build
    for torch 2.14, and this harness always passes 16 kHz, so a stub that refuses to resample stands in for it."""
    import importlib.machinery
    import importlib.util
    import types
    if importlib.util.find_spec("torchaudio") is not None:
        return False

    def resample(wave, orig_freq, new_freq):
        if int(orig_freq) == int(new_freq):  # SpeechMOS calls resample even when the rate already matches
            return wave
        raise RuntimeError("tts_bench passes 16 kHz audio to UTMOS; the torchaudio stub does not resample")

    ta, fn = types.ModuleType("torchaudio"), types.ModuleType("torchaudio.functional")
    ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    fn.__spec__ = importlib.machinery.ModuleSpec("torchaudio.functional", None)
    fn.resample, ta.functional = resample, fn
    sys.modules["torchaudio"], sys.modules["torchaudio.functional"] = ta, fn
    return True


def cmd_score(a: argparse.Namespace) -> int:
    import soundfile as sf
    path = Path(a.result)
    res = json.loads(path.read_text(encoding="utf-8"))
    adir = ROOT / res["audio"]["scoring_dir"]
    utts = {u["id"]: u for u in json.loads(UTTS.read_text(encoding="utf-8"))["utterances"]}
    files = sorted(adir.glob("*.wav"))
    if not files:
        print(f"no audio in {adir}", file=sys.stderr)
        return EXIT_ENGINE
    scores: dict[str, Any] = {"scored": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "per_utt": {}}
    wav16 = {}
    for f in files:
        audio, sr = sf.read(str(f), dtype="float32")
        wav16[f.stem] = to_16k(audio if audio.ndim == 1 else audio.mean(1), sr)

    if a.asr_model != "none":
        from faster_whisper import WhisperModel
        m = WhisperModel(a.asr_model, device=a.asr_device, compute_type=a.asr_compute)
        scores["asr"] = {"engine": "faster-whisper", "version": md.version("faster-whisper"), "model": a.asr_model,
                         "device": a.asr_device, "compute": a.asr_compute, "beam_size": 5}
        for uid, w in wav16.items():
            u = utts.get(uid)
            if not u:
                continue
            lang = u.get("lang", "en").split("-")[0]
            segs, _ = m.transcribe(w, beam_size=5, language=lang if not a.asr_model.endswith(".en") else "en",
                                   condition_on_previous_text=False, vad_filter=False)
            hyp = " ".join(s.text.strip() for s in segs)
            e = wer(normalise(u["spoken"]), normalise(hyp))
            scores["per_utt"].setdefault(uid, {}).update({"tier": u["tier"], "asr": hyp, "wer": round(e, 4)})

    if a.mos == "utmos22":
        try:
            import torch
            shim = torchaudio_shim()
            predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
            scores["mos"] = {"predictor": "UTMOS22-strong", "source": "torch.hub tarepan/SpeechMOS:v1.2.0",
                             "input": "16 kHz mono", "torchaudio_shim": shim}
            for uid, w in wav16.items():
                with torch.inference_mode():
                    s = float(predictor(torch.from_numpy(w).unsqueeze(0), 16000).item())
                scores["per_utt"].setdefault(uid, {})["utmos"] = round(s, 3)
        except Exception as exc:
            scores["mos"] = {"error": repr(exc)[:500]}

    per = scores["per_utt"]
    agg: dict[str, Any] = {}
    groups = {"plain": {"clip", "short", "medium", "long", "barge"}, "stress": {"stress"}, "i18n": {"i18n"}}
    for g, tiers in groups.items():
        ids = [k for k, v in per.items() if utts.get(k, {}).get("tier") in tiers]
        if not ids:
            continue
        ref_words = sum(len(normalise(utts[k]["spoken"])) for k in ids)
        errs = sum(per[k]["wer"] * len(normalise(utts[k]["spoken"])) for k in ids if "wer" in per[k])
        agg[g] = {"n": len(ids), "wer": round(errs / max(1, ref_words), 4) if any("wer" in per[k] for k in ids) else None,
                  "exact": sum(1 for k in ids if per[k].get("wer") == 0.0),
                  "utmos_mean": round(statistics.fmean([per[k]["utmos"] for k in ids if "utmos" in per[k]]), 3)
                  if any("utmos" in per[k] for k in ids) else None}
    scores["aggregate"] = agg
    res["scores"] = scores
    path.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(f"scored {rel(path)}: {json.dumps(agg)}")
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# playback: the earcon / cached-clip tier (no synthesis; how soon a clip in RAM reaches the output device)
# ---------------------------------------------------------------------------------------------------------------------

def cmd_playback(a: argparse.Namespace) -> int:
    """Open one WASAPI shared-mode output stream (as tez-voice would keep open), trigger each earcon `trials` times and
    record trigger -> the callback that starts writing it, plus the device's reported output latency. --mute writes
    zeros through the same path (same timing, no sound). Acoustic latency needs a loopback recording; not done here."""
    sd = need("sounddevice", "pip install sounddevice")
    data = json.loads(UTTS.read_text(encoding="utf-8"))
    # The default device usually belongs to MME; WASAPI settings only apply to a WASAPI device, so open the WASAPI
    # host API's default output at its shared-mode mix rate.
    kw: dict[str, Any] = {}
    wasapi = next((i for i, h in enumerate(sd.query_hostapis()) if "WASAPI" in h["name"]), None)
    if wasapi is not None and sd.query_hostapis(wasapi)["default_output_device"] >= 0:
        kw["device"] = sd.query_hostapis(wasapi)["default_output_device"]
        kw["extra_settings"] = sd.WasapiSettings(exclusive=False)
    sr = a.play_rate or int(sd.query_devices(kw.get("device"), "output")["default_samplerate"])
    clips = {}
    for e in data["earcons"]:
        n = int(sr * e["ms"] / 1000)
        t = np.arange(n) / sr
        f = np.repeat(e["tones_hz"], int(np.ceil(n / len(e["tones_hz"]))))[:n]
        env = np.minimum(1, np.minimum(t, t[::-1]) / 0.005)
        clips[e["id"]] = (0.2 * env * np.sin(2 * np.pi * f * t)).astype(np.float32)
    state = {"pending": None, "pos": 0, "clip": None}
    marks: list[dict] = []
    lock = threading.Lock()

    def cb(outdata, frames, time_info, status):
        outdata.fill(0)
        with lock:
            if state["pending"] is not None and state["clip"] is None:
                cid, t_trig = state["pending"]
                state.update(pending=None, clip=clips[cid], pos=0)
                marks.append({"clip": cid, "trigger_to_callback_ms": ms(now_ns() - t_trig),
                              "callback_to_dac_ms": round((time_info.outputBufferDacTime - time_info.currentTime) * 1000, 3)})
            if state["clip"] is not None:
                c, p = state["clip"], state["pos"]
                k = min(frames, len(c) - p)
                if not a.mute:
                    outdata[:k, 0] = c[p:p + k]
                state["pos"] = p + k
                if state["pos"] >= len(c):
                    state["clip"] = None

    blocksize = int(sr * a.block_ms / 1000)
    with sd.OutputStream(samplerate=sr, channels=1, dtype="float32", blocksize=blocksize, latency="low", callback=cb,
                         **kw) as stream:
        time.sleep(0.5)
        for i in range(a.trials):
            for cid in clips:
                with lock:
                    state["pending"] = (cid, now_ns())
                time.sleep(0.25 + (i % 7) * 0.013)  # jitter the trigger against the buffer period
        reported = stream.latency
        dev = sd.query_devices(stream.device) if hasattr(stream, "device") else {}
    tot = [m["trigger_to_callback_ms"] + max(0.0, m["callback_to_dac_ms"]) for m in marks]
    res = {"harness": {"script": rel(Path(__file__)), "argv": sys.argv[1:]}, "kind": "smoke" if a.mute else "timed",
           "mute": a.mute, "sample_rate": sr, "block_ms": a.block_ms, "device": dev.get("name") if dev else None,
           "hostapi": "WASAPI shared" if "extra_settings" in kw else "default",
           "reported_output_latency_ms": round(reported * 1000, 2),
           "trigger_to_callback_ms": dist([m["trigger_to_callback_ms"] for m in marks]),
           "callback_to_dac_ms": dist([m["callback_to_dac_ms"] for m in marks]),
           "trigger_to_dac_estimate_ms": dist(tot), "n": len(marks),
           "note": "software-side estimate: trigger -> callback + the callback's reported DAC time; Bluetooth adds more"}
    out = OUT_DIR / f"playback_{a.tag}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {rel(out)}: trigger->dac p50 {res['trigger_to_dac_estimate_ms'].get('p50')} ms")
    return 0


# The configurations of the plan's round 1 (docs/plans/voice-tts-plan.md section 4). One command per engine; `suite`
# prints them (--print) or runs them one after another, so a scheduler can call either the suite or single lines.
SUITES: dict[str, list[list[str]]] = {
    "control": [["--engine", "sine", "--device", "cpu"]],
    "gpu": [
        ["--engine", "kokoro-onnx", "--variant", "fp32", "--device", "cuda"],
        ["--engine", "kokoro-onnx", "--variant", "fp32", "--device", "cuda", "--g2p", "misaki"],
        ["--engine", "kokoro-onnx", "--variant", "fp16", "--device", "cuda"],
        # Theioma's sidecar as configured today: the fp16 export, onnxruntime's default cuDNN search
        ["--engine", "kokoro-onnx", "--variant", "fp16", "--device", "cuda", "--ort-conv-search", "EXHAUSTIVE"],
        ["--engine", "kokoro-torch", "--device", "cuda"],
    ],
    "cpu": [
        ["--engine", "kokoro-onnx", "--variant", "int8", "--device", "cpu", "--threads", "4"],
        ["--engine", "kokoro-onnx", "--variant", "int8", "--device", "cpu", "--threads", "4", "--g2p", "misaki"],
        ["--engine", "kokoro-torch", "--device", "cpu"],
        ["--engine", "sherpa", "--model", "kokoro-int8-multi-lang-v1_0", "--device", "cpu", "--threads", "4"],
        ["--engine", "sherpa", "--model", "sherpa-onnx-supertonic-3-tts-int8-2026-05-11", "--device", "cpu",
         "--threads", "4"],
        ["--engine", "sherpa", "--model", "kitten-mini-en-v0_8", "--device", "cpu", "--threads", "4"],
        ["--engine", "sherpa", "--model", "kitten-nano-en-v0_8-int8", "--device", "cpu", "--threads", "4"],
        ["--engine", "pocket-tts", "--device", "cpu"],
        ["--engine", "pocket-tts", "--variant", "int8", "--device", "cpu"],
        ["--engine", "piper", "--device", "cpu"],
    ],
}
SUITE_RUN_ARGS = ["--reps", "10", "--cold-runs", "5", "--idle-gaps", "3,15", "--barge-trials", "5", "--score"]


def cmd_suite(a: argparse.Namespace) -> int:
    names = ["control", "gpu", "cpu"] if a.name == "all" else a.name.split(",")
    cmds = []
    for n in names:
        for cfg in SUITES[n]:
            extra = ["--contend-url", a.contend_url] if a.contend_url and "cuda" in cfg else []
            lock = ["--lock", a.lock] if "cuda" in cfg else ["--lock", "none"]
            cmds.append([sys.executable, str(Path(__file__).resolve()), "run", *cfg, "--tag", a.tag, *SUITE_RUN_ARGS,
                         *lock, *extra, *a.extra])
    if a.print:
        for c in cmds:
            print(" ".join(f'"{x}"' if " " in x else x for x in c))
        return 0
    codes = []
    for c in cmds:
        print("$ " + " ".join(c[2:]), flush=True)
        codes.append(subprocess.run(c, cwd=ROOT).returncode)
    print("exit codes:", codes)
    return 0 if all(x == 0 for x in codes) else EXIT_ENGINE


def cmd_list(a: argparse.Namespace) -> int:
    for name, cls in ENGINES.items():
        ns = argparse.Namespace(**{**vars(a), "engine": name})
        e = cls(ns)
        files = e.model_files()
        missing = [rel(p) for p in files if not p.exists()]
        print(f"{name:13s} {cls.__doc__.strip().splitlines()[0][:90]}")
        note = f"; missing {missing}" if missing else ""
        print(f"{'':13s} model files: {len(files) - len(missing)}/{len(files)} present{note}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def engine_args(p):
        p.add_argument("--engine", default="kokoro-onnx", help=", ".join(ENGINES))
        p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
        p.add_argument("--variant", default=None, help="kokoro-onnx: fp32 | fp16 | int8")
        p.add_argument("--voice", default=None, help="engine voice (kokoro: af_heart; piper: en_US-lessac-medium)")
        p.add_argument("--model", default="kokoro-int8-multi-lang-v1_0", help="sherpa: a directory under tools/tts/sherpa")
        p.add_argument("--sid", default=None, help="sherpa: speaker id")
        p.add_argument("--speed", type=float, default=1.0)
        p.add_argument("--chunking", default="auto", choices=["auto", "none", "native", "sentence", "clause",
                                                                "first-clause"])
        p.add_argument("--ort-conv-search", default="HEURISTIC", choices=["EXHAUSTIVE", "HEURISTIC", "DEFAULT"],
                       help="onnxruntime CUDA EP cuDNN algorithm search (onnxruntime's default is EXHAUSTIVE)")
        p.add_argument("--threads", type=int, default=0, help="CPU threads for onnxruntime / sherpa (0 = default)")
        p.add_argument("--g2p", default="espeak", choices=["espeak", "misaki"],
                       help="kokoro-onnx: phonemiser for English (misaki = what hexgrad's kokoro package uses)")
        p.add_argument("--sherpa-stop-value", type=int, default=0, choices=[0, 1],
                       help="the value sherpa's callback returns to stop generation")
        p.add_argument("--sine-rtf", type=float, default=0.05)

    r = sub.add_parser("run", help="time one engine on the utterance set")
    engine_args(r)
    r.add_argument("--tag", required=True, help="run label; the output is results/tts/<engine-id>_<tag>.json")
    r.add_argument("--tiers", default="clip,short,medium,long,stress,i18n")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--reps", type=int, default=5, help="calls per utterance: 1 'first' + reps-1 'repeat'")
    r.add_argument("--cold-runs", type=int, default=0, help="extra fresh processes for the cold distribution")
    r.add_argument("--idle-gaps", default="3,15", help="seconds of idle before an idle-phase call; '' to skip")
    r.add_argument("--barge-trials", type=int, default=5)
    r.add_argument("--barge-after-ms", type=float, default=400)
    r.add_argument("--lookahead-ms", type=float, default=500, help="player buffer during barge trials")
    r.add_argument("--contend-url", default="", help="keep this llama-server decoding during the run (opt-in)")
    r.add_argument("--lock", default="check", choices=["none", "check", "acquire"])
    r.add_argument("--lock-owner", default="tts-bench")
    r.add_argument("--lock-minutes", type=float, default=30)
    r.add_argument("--smoke", action="store_true", help="functional check: the result is marked timing_valid=false")
    r.add_argument("--cpu-idle-max", type=float, default=25.0,
                   help="system CPU %% above which, before the load, the run is marked timing_valid=false")
    r.add_argument("--spill-max-mb", type=float, default=160.0,
                   help="shared GPU memory (MB) held by this process above which the run is marked invalid (a CUDA "
                        "process on this laptop shows ~76-82 MB of shared memory with no pressure at all)")
    r.add_argument("--evict-max-mb", type=float, default=200.0,
                   help="own dedicated VRAM minus the device's growth above which other processes were evicted")
    r.add_argument("--keep-wavs", type=int, default=3)
    r.add_argument("--keep-ids", default="clip_open_spotify,med_agents,stress_time,long_summary")
    r.add_argument("--score", action="store_true", help="score the audio after the timing (CPU)")
    for p in (r,):
        p.add_argument("--asr-model", default="large-v3")
        p.add_argument("--asr-device", default="cpu")
        p.add_argument("--asr-compute", default="int8")
        p.add_argument("--mos", default="utmos22", choices=["utmos22", "none"])
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("_cold", help=argparse.SUPPRESS)
    engine_args(c)
    c.set_defaults(fn=cmd_cold_child)

    s = sub.add_parser("score", help="add WER and UTMOS to a result file")
    s.add_argument("result")
    s.add_argument("--asr-model", default="large-v3")
    s.add_argument("--asr-device", default="cpu")
    s.add_argument("--asr-compute", default="int8")
    s.add_argument("--mos", default="utmos22", choices=["utmos22", "none"])
    s.set_defaults(fn=cmd_score)

    pb = sub.add_parser("playback", help="earcon / cached-clip playback latency through one open WASAPI stream")
    pb.add_argument("--tag", required=True)
    pb.add_argument("--trials", type=int, default=20)
    pb.add_argument("--play-rate", type=int, default=0, help="0 = the device's shared-mode rate")
    pb.add_argument("--block-ms", type=float, default=10)
    pb.add_argument("--mute", action=argparse.BooleanOptionalAction, default=True)
    pb.set_defaults(fn=cmd_playback)

    su = sub.add_parser("suite", help="run (or --print) the plan's engine configurations, one command per engine")
    su.add_argument("--name", default="all", help="control, gpu, cpu (comma-separated) or all")
    su.add_argument("--tag", required=True)
    su.add_argument("--lock", default="acquire", choices=["none", "check", "acquire"],
                    help="for the GPU commands; use none when the caller already holds the GPU lock")
    su.add_argument("--contend-url", default="", help="GPU commands only: keep this llama-server decoding")
    su.add_argument("--print", action="store_true", help="print the commands instead of running them")
    su.add_argument("extra", nargs="*", help="extra arguments for every command (after --)")
    su.set_defaults(fn=cmd_suite)

    ls = sub.add_parser("list", help="engines and whether their model files are present")
    engine_args(ls)
    ls.set_defaults(fn=cmd_list)

    a = ap.parse_args()
    try:
        return a.fn(a)
    except MissingDependency as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_DEPS


if __name__ == "__main__":
    sys.exit(main())
