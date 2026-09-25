"""Streaming speech recognisers for the voice replay harness (experiments/voice_replay.py), behind one interface.

    asr = make_asr("nemocpp:C:/temp/asr-models/nemotron-speech-streaming-en-0.6b.q8_0.gguf", right_context=0)
    asr.reset()
    for frame in frames_of_10ms:            # float32, 16 kHz mono
        for e in asr.push(frame):           # Emission(text, audio_s, compute_ms): the transcript after a step
            ...
    for e in asr.finish(): ...

Adapters
  nemocpp:GGUF   NVIDIA NeMo-Speech.cpp v0.1.0 (ggml, CUDA on this laptop) through its stable C ABI (nemo_speech/asr.h),
                 cache-aware streaming: right_context 0 = 80 ms steps, 1 = 160 ms (the library's default). Its ggml and
                 llama DLLs share file names with llama.cpp b11100's but not its version, so this adapter must not share
                 a process with tez.inproc: the replay harness runs it in a worker process (asr_worker below).
  sherpa:DIR     sherpa-onnx OnlineRecognizer over a transducer export (encoder/decoder/joiner + tokens.txt), on ONNX
                 Runtime (CPU in the PyPI wheel): no ggml, so it can share the Tez process.
  whisper:NAME   faster-whisper re-decoding the whole buffer every `step` seconds (Tez's experiment loop, the baseline).
Every adapter reports, per step that changed the text, the transcript so far, the audio position at the end of the
step's input (audio_s) and the step's compute time.
"""
from __future__ import annotations

import ctypes as C
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SR = 16000
FRAME_SAMPLES = 160                      # 10 ms feature hop
NEMO_SPEECH_DIR = os.environ.get("NEMO_SPEECH_DIR", r"C:\temp\nemo-speech\x\bin")


@dataclass
class Emission:
    text: str            # the transcript so far (normalised: lower case, no punctuation)
    audio_s: float       # audio consumed when the step ran (the end of its input)
    compute_ms: float    # the step's compute
    t: float = 0.0       # perf_counter when the step finished
    final: bool = False


def add_nvidia_dlls() -> None:
    """CTranslate2 (faster-whisper) on CUDA wants the cuBLAS/cuDNN DLLs of the nvidia-* pip wheels, which Windows does
    not find unless their directories are registered (experiments/asr_bench.py does the same)."""
    import os
    import site
    for sp in set(site.getsitepackages() + [site.getusersitepackages()]):
        nv = Path(sp) / "nvidia"
        if nv.is_dir():
            for d in nv.glob("*/bin"):
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]


def normalise(text: str) -> str:
    """Lower case, words only (the Tez command sets are written that way)."""
    text = text.lower().replace("-", " ")
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", text).split())


class Asr:
    name = "asr"
    chunk_s = 0.08

    def reset(self) -> None: ...

    def push(self, x: np.ndarray) -> list[Emission]: ...

    def finish(self) -> list[Emission]: ...

    def close(self) -> None: ...

    def info(self) -> dict:
        return {"name": self.name, "chunk_s": self.chunk_s}


# ---------------------------------------------------------------------------------------------- NeMo-Speech.cpp
class _Backend(C.Structure):
    _fields_ = [("size", C.c_size_t), ("gpu", C.c_int32)]


class _Model(C.Structure):
    _fields_ = [("size", C.c_size_t), ("path", C.c_char_p), ("name", C.c_char_p)]


class _Streaming(C.Structure):
    _fields_ = [("size", C.c_size_t), ("chunk_size", C.c_float), ("ctc_left_padding", C.c_float),
                ("ctc_right_padding", C.c_float), ("rnnt_right_context", C.c_int32)]


class _Endpointing(C.Structure):
    _fields_ = [("size", C.c_size_t), ("enable", C.c_bool), ("vad_based", C.c_bool), ("stop_history_eou_ms", C.c_int32)]


class _RecognizerConfig(C.Structure):
    _fields_ = [("size", C.c_size_t), ("backend", C.POINTER(_Backend)), ("model", C.POINTER(_Model)),
                ("streaming", C.POINTER(_Streaming)), ("decoder", C.c_void_p), ("vad", C.c_void_p),
                ("endpointing", C.POINTER(_Endpointing)), ("postproc", C.c_void_p), ("diar", C.c_void_p),
                ("batching", C.c_void_p)]


class _Options(C.Structure):
    _fields_ = [("size", C.c_size_t), ("request_id", C.c_char_p), ("language_code", C.c_char_p),
                ("interim_results", C.c_bool), ("enable_word_time_offsets", C.c_bool),
                ("enable_automatic_punctuation", C.c_bool), ("verbatim_transcripts", C.c_bool),
                ("profanity_filter", C.c_bool), ("stop_history_eou_ms", C.c_int32), ("speech_contexts", C.c_void_p),
                ("speech_context_count", C.c_size_t), ("max_alternatives", C.c_int32),
                ("enable_speaker_diarization", C.c_bool), ("max_speaker_count", C.c_int32)]


class NemoSpeechCpp(Asr):
    """NeMo-Speech.cpp's streaming recogniser (cache-aware RNNT). One recogniser, one stream per utterance."""

    def __init__(self, model: str, right_context: int = 0, gpu: int = 0, lib_dir: str = NEMO_SPEECH_DIR,
                 punctuation: bool = False):
        self.lib_dir = lib_dir
        self._dll_dir = os.add_dll_directory(lib_dir)
        lib = C.CDLL(str(Path(lib_dir) / "nemo_speech_asr_c.dll"))
        P = C.c_void_p
        sig = {
            "nemo_speech_asr_recognition_options_default": (_Options, []),
            "nemo_speech_asr_create": (C.c_int, [C.POINTER(_RecognizerConfig), C.POINTER(P)]),
            "nemo_speech_asr_destroy": (None, [P]),
            "nemo_speech_asr_streaming_recognize": (C.c_int, [P, C.POINTER(_Options), C.POINTER(P)]),
            "nemo_speech_asr_stream_push_f32": (C.c_int, [P, C.POINTER(C.c_float), C.c_size_t, C.c_int32]),
            "nemo_speech_asr_stream_finish": (C.c_int, [P]),
            "nemo_speech_asr_stream_next": (C.c_int, [P, C.POINTER(P)]),
            "nemo_speech_asr_stream_close": (None, [P]),
            "nemo_speech_asr_result_is_final": (C.c_bool, [P]),
            "nemo_speech_asr_result_audio_processed": (C.c_float, [P]),
            "nemo_speech_asr_result_alternative_count": (C.c_size_t, [P]),
            "nemo_speech_asr_result_transcript": (C.c_char_p, [P, C.c_size_t]),
            "nemo_speech_asr_result_destroy": (None, [P]),
            "nemo_speech_asr_last_error": (C.c_char_p, []),
            "nemo_speech_asr_version": (C.c_char_p, []),
        }
        for name, (res, args) in sig.items():
            f = getattr(lib, name)
            f.restype, f.argtypes = res, args
        self.lib = lib
        self.version = lib.nemo_speech_asr_version().decode()
        self.right_context = int(right_context)
        self.chunk_s = 0.08 * (1 + self.right_context)
        self.name = f"nemocpp-{Path(model).stem}-rc{self.right_context}"
        self.model = model
        self._keep = []
        backend = _Backend(C.sizeof(_Backend), int(gpu))
        mcfg = _Model(C.sizeof(_Model), str(model).encode(), None)
        scfg = _Streaming(C.sizeof(_Streaming), 0.16, 0.0, 0.0, self.right_context)
        ecfg = _Endpointing(C.sizeof(_Endpointing), False, False, 800)
        cfg = _RecognizerConfig(C.sizeof(_RecognizerConfig), C.pointer(backend), C.pointer(mcfg), C.pointer(scfg), None,
                                None, C.pointer(ecfg), None, None, None)
        self._keep += [backend, mcfg, scfg, ecfg, cfg]
        rec = P()
        t0 = time.perf_counter()
        self._check(lib.nemo_speech_asr_create(C.byref(cfg), C.byref(rec)), "create")
        self.load_s = time.perf_counter() - t0
        self.rec = rec
        self.opts = lib.nemo_speech_asr_recognition_options_default()
        self.opts.interim_results = True
        self.opts.enable_word_time_offsets = False
        self.opts.enable_automatic_punctuation = bool(punctuation)
        self.stream = None
        self.audio_s = 0.0

    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            err = self.lib.nemo_speech_asr_last_error()
            raise RuntimeError(f"nemo-speech {what} failed ({rc}): {err.decode() if err else '?'}")

    def reset(self) -> None:
        if self.stream is not None:
            self.lib.nemo_speech_asr_stream_close(self.stream)
        s = C.c_void_p()
        self._check(self.lib.nemo_speech_asr_streaming_recognize(self.rec, C.byref(self.opts), C.byref(s)), "stream")
        self.stream = s
        self.audio_s = 0.0
        self._finals = ""
        self._text = ""
        self.steps, self.busy_ms = [], 0.0

    def _drain(self, final_pass: bool = False) -> list[Emission]:
        out = []
        while True:
            t0 = time.perf_counter()
            r = C.c_void_p()
            self._check(self.lib.nemo_speech_asr_stream_next(self.stream, C.byref(r)), "next")
            ms = (time.perf_counter() - t0) * 1000.0
            self.busy_ms += ms
            if not r.value:
                break
            self.steps.append(ms)
            try:
                n = self.lib.nemo_speech_asr_result_alternative_count(r)
                raw = self.lib.nemo_speech_asr_result_transcript(r, 0).decode("utf-8", "replace") if n else ""
                final = bool(self.lib.nemo_speech_asr_result_is_final(r))
                audio = float(self.lib.nemo_speech_asr_result_audio_processed(r))
            finally:
                self.lib.nemo_speech_asr_result_destroy(r)
            text = normalise(" ".join(x for x in (self._finals, raw) if x))
            if final:
                self._finals = " ".join(x for x in (self._finals, raw) if x)
            if text != self._text or final_pass:
                self._text = text
                out.append(Emission(text, audio if audio > 0 else self.audio_s, ms, time.perf_counter(), final))
        return out

    def push(self, x: np.ndarray) -> list[Emission]:
        x = np.ascontiguousarray(x, dtype=np.float32)
        t0 = time.perf_counter()
        self._check(self.lib.nemo_speech_asr_stream_push_f32(self.stream, x.ctypes.data_as(C.POINTER(C.c_float)), len(x),
                                                             SR), "push")
        self.audio_s += len(x) / SR
        self.busy_ms += (time.perf_counter() - t0) * 1000.0
        return self._drain()

    def finish(self) -> list[Emission]:
        self._check(self.lib.nemo_speech_asr_stream_finish(self.stream), "finish")
        return self._drain()

    def close(self) -> None:
        if self.stream is not None:
            self.lib.nemo_speech_asr_stream_close(self.stream)
            self.stream = None
        if getattr(self, "rec", None):
            self.lib.nemo_speech_asr_destroy(self.rec)
            self.rec = None

    def info(self) -> dict:
        return {"name": self.name, "engine": "NeMo-Speech.cpp", "version": self.version, "model": self.model,
                "right_context": self.right_context, "chunk_s": self.chunk_s, "load_s": round(self.load_s, 2),
                "lib": self.lib_dir}


# ---------------------------------------------------------------------------------------------- sherpa-onnx
class Sherpa(Asr):
    """sherpa-onnx's online transducer recogniser; a step runs whenever the stream has a full chunk."""

    def __init__(self, model_dir: str, threads: int = 4, provider: str = "cpu"):
        threads = min(int(threads), 4)
        import sherpa_onnx
        d = Path(model_dir)

        def pick(stem: str) -> str:
            for name in (f"{stem}.int8.onnx", f"{stem}.onnx"):
                if (d / name).is_file():
                    return str(d / name)
            raise FileNotFoundError(f"{d}: no {stem}.onnx")

        t0 = time.perf_counter()
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"), encoder=pick("encoder"), decoder=pick("decoder"), joiner=pick("joiner"),
            num_threads=threads, sample_rate=SR, feature_dim=80, decoding_method="greedy_search", provider=provider,
            model_type="nemo_transducer" if "nemo" in d.name or "nemotron" in d.name else "")
        self.load_s = time.perf_counter() - t0
        self.name = f"sherpa-{d.name}-{provider}{threads}"
        self.model_dir, self.threads, self.provider = str(d), threads, provider
        self.version = sherpa_onnx.__version__
        s = self.rec.create_stream()             # the step size, measured: steps over 4 s of silence
        s.accept_waveform(SR, np.zeros(4 * SR, np.float32))
        n = 0
        while self.rec.is_ready(s):
            self.rec.decode_stream(s)
            n += 1
        self.chunk_s = round(4.0 / max(1, n), 3)
        self.stream = None

    def reset(self) -> None:
        self.stream = self.rec.create_stream()
        self.audio_s = 0.0
        self._text = ""
        self.steps, self.busy_ms = [], 0.0

    def _steps(self, final: bool = False) -> list[Emission]:
        out = []
        while self.rec.is_ready(self.stream):
            t0 = time.perf_counter()
            self.rec.decode_stream(self.stream)
            ms = (time.perf_counter() - t0) * 1000.0
            self.steps.append(ms)
            self.busy_ms += ms
            text = normalise(self.rec.get_result(self.stream))
            if text != self._text:
                self._text = text
                out.append(Emission(text, self.audio_s, ms, time.perf_counter(), final))
        return out

    def push(self, x: np.ndarray) -> list[Emission]:
        self.stream.accept_waveform(SR, np.ascontiguousarray(x, dtype=np.float32))
        self.audio_s += len(x) / SR
        return self._steps()

    def finish(self) -> list[Emission]:
        self.stream.accept_waveform(SR, np.zeros(int(0.3 * SR), np.float32))   # tail padding flushes the last chunk
        self.stream.input_finished()
        return self._steps(final=True)

    def info(self) -> dict:
        return {"name": self.name, "engine": "sherpa-onnx", "version": self.version, "model": self.model_dir,
                "threads": self.threads, "provider": self.provider, "chunk_s": self.chunk_s,
                "load_s": round(self.load_s, 2)}


# ---------------------------------------------------------------------------------------------- NeMo (PyTorch)
class NemoTorch(Asr):
    """A NeMo cache-aware streaming model through NeMo's own PyTorch code (conformer_stream_step), fed as audio arrives:
    whenever the audio received holds the next chunk (plus one feature frame, whose window reaches past it), the
    features of the audio so far are computed and the chunk goes through one cache-aware step. For models NeMo-Speech.cpp
    does not run (parakeet_realtime_eou_120m-v1, which also emits an end-of-utterance token)."""

    def __init__(self, path: str, right_context: int = 1, device: str = "cuda"):
        import types

        import torch
        name = "nv_one_logger.training_telemetry.integration.pytorch_lightning"
        try:                                         # NeMo 2.7.3 imports a training-telemetry module the PyPI package
            __import__(name)                         # no longer ships; inference never calls it
        except ImportError:
            stub = types.ModuleType(name)
            stub.TimeEventCallback = type("TimeEventCallback", (), {"__init__": lambda self, *a, **k: None})
            sys.modules[name] = stub
        torch.set_num_threads(min(6, torch.get_num_threads()))
        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
        from omegaconf import open_dict
        self.torch, self.Buffer = torch, CacheAwareStreamingAudioBuffer
        t0 = time.perf_counter()
        m = nemo_asr.models.ASRModel.restore_from(path, map_location=device)
        m.eval()
        if hasattr(m.encoder, "set_default_att_context_size"):
            m.encoder.set_default_att_context_size([70, int(right_context)])
        m.encoder.setup_streaming_params()
        dc = m.cfg.decoding
        with open_dict(dc):
            dc.strategy = "greedy"
            dc.greedy.max_symbols = 10
            dc.fused_batch_size = -1
            dc.compute_timestamps = False
            dc.preserve_alignments = False
        m.change_decoding_strategy(dc)
        self.m = m
        cfg = m.encoder.streaming_cfg
        shift = cfg.shift_size[1] if isinstance(cfg.shift_size, list) else cfg.shift_size
        self.chunk_s = round(shift * 0.01, 3)
        self.load_s = time.perf_counter() - t0
        self.name = f"nemo-{Path(path).stem}-rc{right_context}-{device}"
        self.path, self.right_context, self.device = path, right_context, device

    def reset(self) -> None:
        self.audio: list[np.ndarray] = []
        self.n = 0
        self.buf = self.Buffer(self.m, online_normalization=False)
        self.cache = self.m.encoder.get_initial_cache_state(batch_size=1)
        self.hyp = None
        self.pred = None
        self.step = 0
        self._text = ""
        self.eou: list[float] = []
        self.steps, self.busy_ms = [], 0.0

    def _next_end(self) -> int:
        cfg = self.buf.streaming_cfg
        size = cfg.chunk_size
        size = (size[0] if self.buf.buffer_idx == 0 else size[1]) if isinstance(size, list) else size
        return self.buf.buffer_idx + size

    def _step(self, final: bool) -> list[Emission]:
        torch = self.torch
        t0 = time.perf_counter()
        x = np.concatenate(self.audio).astype(np.float32)
        feats, _ = self.buf.preprocess_audio(x)
        if not final:
            feats = feats[:, :, :-1]                     # the last frame's window reaches past the audio
        self.buf.buffer = feats
        self.buf.streams_length = torch.tensor([feats.size(-1)], device=feats.device)
        chunk, lengths = next(iter(self.buf))
        cfg = self.m.encoder.streaming_cfg
        with torch.inference_mode():
            (self.pred, texts, c1, c2, c3, self.hyp) = self.m.conformer_stream_step(
                processed_signal=chunk, processed_signal_length=lengths, cache_last_channel=self.cache[0],
                cache_last_time=self.cache[1], cache_last_channel_len=self.cache[2],
                keep_all_outputs=final, previous_hypotheses=self.hyp, previous_pred_out=self.pred,
                drop_extra_pre_encoded=0 if self.step == 0 else cfg.drop_extra_pre_encoded, return_transcription=True)
        self.cache = (c1, c2, c3)
        self.step += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        self.steps.append(ms)
        self.busy_ms += ms
        raw = texts[0].text if texts and hasattr(texts[0], "text") else (texts[0] if texts else "")
        audio_s = min(self.buf.buffer_idx, feats.size(-1)) * 0.01
        if raw.count("<EOU>") > len(self.eou):         # the transcript is cumulative: count new end-of-utterance tokens
            self.eou.append(audio_s)
        text = normalise(raw.replace("<EOU>", " "))
        if text != self._text:
            self._text = text
            return [Emission(text, audio_s, ms, time.perf_counter(), final)]
        return []

    def push(self, x: np.ndarray) -> list[Emission]:
        self.audio.append(np.asarray(x, np.float32))
        self.n += len(x)
        out = []
        while self.n // FRAME_SAMPLES - 1 >= self._next_end():
            out += self._step(False)
        return out

    def finish(self) -> list[Emission]:
        out = []
        self.audio.append(np.zeros(int(0.3 * SR), np.float32))
        self.n += int(0.3 * SR)
        while self.n // FRAME_SAMPLES - 1 >= self._next_end():
            out += self._step(False)
        return out

    def info(self) -> dict:
        return {"name": self.name, "engine": "NeMo (PyTorch)", "model": self.path, "right_context": self.right_context,
                "chunk_s": self.chunk_s, "device": self.device, "load_s": round(self.load_s, 2)}


# ---------------------------------------------------------------------------------------------- faster-whisper
class WhisperRedecode(Asr):
    """faster-whisper re-decoding the whole buffer every `step` seconds (greedy, no VAD): experiments/asr_bench.py."""

    def __init__(self, model: str = "base.en", device: str = "cuda", compute_type: str = "float16", step: float = 0.2):
        if device == "cuda":
            add_nvidia_dlls()
        from faster_whisper import WhisperModel
        t0 = time.perf_counter()
        self.m = WhisperModel(model, device=device, compute_type=compute_type, cpu_threads=4)
        self.load_s = time.perf_counter() - t0
        self.name = f"whisper-{model}-{device}-{int(step * 1000)}ms"
        self.model, self.device, self.chunk_s = model, device, step

    def reset(self) -> None:
        self.buf = np.zeros(0, np.float32)
        self.next_s = self.chunk_s
        self._text = ""
        self.steps, self.busy_ms = [], 0.0

    def _decode(self, final: bool) -> list[Emission]:
        t0 = time.perf_counter()
        segs, _ = self.m.transcribe(self.buf, language="en", beam_size=1, vad_filter=False, without_timestamps=True,
                                    condition_on_previous_text=False)
        text = normalise(" ".join(s.text for s in segs))
        ms = (time.perf_counter() - t0) * 1000.0
        self.steps.append(ms)
        self.busy_ms += ms
        if text != self._text or final:
            self._text = text
            return [Emission(text, len(self.buf) / SR, ms, time.perf_counter(), final)]
        return []

    def push(self, x: np.ndarray) -> list[Emission]:
        self.buf = np.concatenate([self.buf, x.astype(np.float32)])
        out = []
        while len(self.buf) / SR >= self.next_s:
            self.next_s += self.chunk_s
            out += self._decode(False)
        return out

    def finish(self) -> list[Emission]:
        return self._decode(True)

    def info(self) -> dict:
        return {"name": self.name, "engine": "faster-whisper", "model": self.model, "device": self.device,
                "chunk_s": self.chunk_s, "load_s": round(self.load_s, 2)}


def make_asr(spec: str, **kw: Any) -> Asr:
    kind, _, arg = spec.partition(":")
    if kind == "nemocpp":
        return NemoSpeechCpp(arg, **kw)
    if kind == "sherpa":
        return Sherpa(arg, **kw)
    if kind == "whisper":
        return WhisperRedecode(arg or "base.en", **kw)
    if kind == "nemo":
        return NemoTorch(arg, **kw)
    raise ValueError(f"unknown recogniser {spec!r} (nemocpp:GGUF, sherpa:DIR, whisper:NAME, nemo:.nemo)")


# ---------------------------------------------------------------------------------------------- worker process
def asr_worker(conn: Any, spec: str, kw: dict) -> None:
    """Run a recogniser in its own process (multiprocessing target). Messages in: ("utt", pcm float32 bytes, t_start)
    to stream one utterance in real time from perf_counter t_start (10 ms frames, each pushed when its last sample
    would have been captured; t_start None pushes them as fast as the recogniser takes them), ("quit",). Messages out: ("ready", info), then per utterance ("emit", text, audio_s,
    compute_ms, t, final) for every step that changed the text and ("done", t_last_push). perf_counter is the system's
    QueryPerformanceCounter on Windows, so both processes read one clock."""
    if sys.platform == "win32":                       # a 1 ms timer period, as the harness sets for itself
        try:
            C.windll.winmm.timeBeginPeriod(1)
        except Exception:  # noqa: BLE001 - best effort
            pass
    asr = make_asr(spec, **kw)
    conn.send(("ready", asr.info()))
    frame = int(0.01 * SR)
    while True:
        msg = conn.recv()
        if msg[0] == "quit":
            break
        _, raw, t_start = msg
        x = np.frombuffer(raw, np.float32)
        asr.reset()
        for i in range(0, len(x), frame):
            due = t_start + (i + frame) / SR if t_start is not None else 0.0     # None: as fast as it can
            while True:
                now = time.perf_counter()
                if now >= due:
                    break
                if due - now > 0.002:
                    time.sleep(due - now - 0.0015)
            for e in asr.push(x[i: i + frame]):
                conn.send(("emit", e.text, e.audio_s, e.compute_ms, e.t, e.final))
        for e in asr.finish():
            conn.send(("emit", e.text, e.audio_s, e.compute_ms, e.t, True))
        conn.send(("done", time.perf_counter(), list(asr.steps), asr.busy_ms, len(x) / SR))
    asr.close()
    conn.close()
