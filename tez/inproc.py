"""In-process llama.cpp: the model runs inside the Tez process, through llama.cpp's C library (ctypes, no HTTP).

    tez serve --backend inproc:models/gemma-4-12b-it-Q8_0.gguf --template gemma4 --llama-lib C:/llama.cpp

The library is llama.cpp's own (llama.dll, libllama.so or libllama.dylib, with its ggml libraries and GPU backend) from
the release archive for your platform; Tez was tested with b11100 (ggml commit 7ab4ee7ba). It is looked up in the
directory given by --llama-lib (or the library file itself), else TEZ_LLAMA_LIB, else the directory of `llama-server`
on PATH. Only numpy and ctypes are needed.

What the backend reads
  letters    the logits of the bare option-letter tokens ("A", "B", ...) at the answer position, as log-probabilities
             over the whole vocabulary: nothing floored (llama-server lists only the top n_probs; --n-probs does not
             apply here), the normaliser summing every token within 25 nats of the top one (to within 4e-6 nats)
  embed      the last token's hidden state after the output norm: what llama-server --embeddings --pooling last
             returns, so probes fitted over HTTP read the same features (and the other way round)
  read_many  many prompts that share a prefix (the questions about one state, state first): the common token prefix is
             evaluated once on sequence 0 and copied to sequences 1..n with llama_memory_seq_cp (in the unified KV
             cache the copies share the cells), then every prompt's suffix goes into ONE llama_decode, with an output
             at each suffix end. The engine uses it for requests with two or more questions.
A single prompt reuses the longest token prefix it shares with the previous one on sequence 0, like llama-server's
prompt cache: by removing the tail (llama_memory_seq_rm), except on hybrid and recurrent models (Qwen3.5), whose
recurrent state cannot be rolled back; those reuse a prefix only by extending it, or through sequence copies.
  stream     prompts built as head + state + tail whose state grows between reads (a transcript, word by word):
             stream_reader() keeps the head evaluated once and the state read so far on sequences of their own, so
             each read evaluates only the new words and the tail (InprocStreamReader; tez.stream uses it).

Traps kept from the speed study (experiments/speed_llama.py, experiments/speed_multiq_inproc.py):
  * Windows: ggml loads its backends with a plain LoadLibrary, which does not find ggml-cuda.dll's own dependencies
    (cudart and cublas next to it): error 126, and the model silently runs on the CPU. The GPU backend is loaded
    through ctypes first, and a load that ends without a GPU device fails loudly (unless n_gpu_layers is 0).
  * llama_get_logits_ith and llama_get_embeddings_ith take the token's position in the batch, not the output ordinal.
  * n_outputs_max_per_seq defaults to 1: it is set explicitly.
  * With embeddings on, every token of a decode is an output (with full-vocabulary logits): embeddings are switched on
    only for the decode of the last tokens whose states are read.
  * Hybrid-memory models cannot roll back: a prefix is reused by sequence copies or extension, never by removing tokens.

The model and its KV cache live in this process: one process per GPU (llama-server, or a second Tez process, loading
the same model needs its memory again). Calls are serialised by a lock, so tez serve's worker threads take turns.
"""
from __future__ import annotations

import ctypes as C
import logging
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .backends import Backend, Embedding, LetterScores, derive_model_name
from .errors import BackendRequestError, BackendUnavailable
from .prompt import LETTERS, MAX_LETTERS, check_template

log = logging.getLogger("tez")

SCHEME = "inproc:"
TESTED_BUILDS = {"7ab4ee7ba": "b11100"}          # ggml commit -> the llama.cpp release Tez was measured with
DEFAULT_N_CTX = 4096
DEFAULT_N_UBATCH = 512
DEFAULT_N_SEQ_MAX = 64
DEFAULT_N_GPU_LAYERS = -1                        # all layers (llama.h: a negative value means all)
OPTION_NAMES = ("lib", "n_ctx", "n_batch", "n_ubatch", "n_seq_max", "n_gpu_layers", "n_outputs", "n_threads")
_GPU_LIBS = ("ggml-cuda", "ggml-hip", "ggml-vulkan", "ggml-sycl", "ggml-musa", "ggml-cann", "ggml-opencl")
_NOT_GPU = {"CPU", "BLAS", "RPC"}                # backend registries that are not a GPU
_HASHED = re.compile(r"(sha256[-:])?[0-9a-f]{32,}", re.I)


def lib_file(stem: str) -> str:
    """The platform's file name of a llama.cpp / ggml shared library: llama -> llama.dll, libllama.so, libllama.dylib."""
    if sys.platform == "win32":
        return f"{stem}.dll"
    if sys.platform == "darwin":
        return f"lib{stem}.dylib"
    return f"lib{stem}.so"


def find_library(path: str | os.PathLike | None = None, environ: Mapping[str, str] | None = None) -> Path:
    """The directory holding llama.cpp's shared library: `path` (the directory or the library file), else TEZ_LLAMA_LIB,
    else the directory of llama-server on PATH. The first of these that is set decides: a wrong explicit setting is an
    error, not a reason to pick up another build."""
    env = os.environ if environ is None else environ
    name = lib_file("llama")
    sources = (("--llama-lib", path), ("TEZ_LLAMA_LIB", env.get("TEZ_LLAMA_LIB") or None),
               ("llama-server on PATH", shutil.which("llama-server", path=env.get("PATH"))))
    for source, value in sources:
        if not value:
            continue
        p = Path(value).expanduser()
        d = p.parent if p.is_file() else p
        if (d / name).is_file():
            return d.resolve()
        raise BackendUnavailable(f"llama.cpp's library {name} is not in {d} ({source}); point --llama-lib or TEZ_LLAMA_LIB "
                                 f"at the directory of an unpacked llama.cpp release (tested: b11100)")
    raise BackendUnavailable(f"llama.cpp's library {name} was not found: pass --llama-lib DIR or set TEZ_LLAMA_LIB to the "
                             "directory of an unpacked llama.cpp release for your platform (tested: b11100), or put its "
                             "llama-server on PATH")


# ---------------------------------------------------------------------------------------------- C structs (llama.h, b11100)
class ModelParams(C.Structure):
    _fields_ = [("devices", C.c_void_p), ("tensor_buft_overrides", C.c_void_p),
                ("n_gpu_layers", C.c_int32), ("split_mode", C.c_int), ("load_mode", C.c_int), ("lazy_mode", C.c_int),
                ("main_gpu", C.c_int32), ("tensor_split", C.c_void_p), ("progress_callback", C.c_void_p),
                ("progress_callback_user_data", C.c_void_p), ("kv_overrides", C.c_void_p),
                ("vocab_only", C.c_bool), ("check_tensors", C.c_bool), ("use_extra_bufts", C.c_bool),
                ("no_host", C.c_bool), ("no_alloc", C.c_bool), ("load_mtp", C.c_bool)]


class ContextParams(C.Structure):
    _fields_ = [("n_ctx", C.c_uint32), ("n_batch", C.c_uint32), ("n_ubatch", C.c_uint32), ("n_seq_max", C.c_uint32),
                ("n_rs_seq", C.c_uint32), ("n_outputs_max", C.c_uint32), ("n_outputs_max_per_seq", C.c_uint32),
                ("n_threads", C.c_int32), ("n_threads_batch", C.c_int32),
                ("ctx_type", C.c_int), ("rope_scaling_type", C.c_int), ("pooling_type", C.c_int),
                ("attention_type", C.c_int), ("flash_attn_type", C.c_int),
                ("rope_freq_base", C.c_float), ("rope_freq_scale", C.c_float), ("yarn_ext_factor", C.c_float),
                ("yarn_attn_factor", C.c_float), ("yarn_beta_fast", C.c_float), ("yarn_beta_slow", C.c_float),
                ("yarn_orig_ctx", C.c_uint32), ("defrag_thold", C.c_float),
                ("cb_eval", C.c_void_p), ("cb_eval_user_data", C.c_void_p),
                ("type_k", C.c_int), ("type_v", C.c_int),
                ("abort_callback", C.c_void_p), ("abort_callback_data", C.c_void_p),
                ("embeddings", C.c_bool), ("offload_kqv", C.c_bool), ("no_perf", C.c_bool), ("op_offload", C.c_bool),
                ("swa_full", C.c_bool), ("kv_unified", C.c_bool),
                ("samplers", C.c_void_p), ("n_samplers", C.c_size_t), ("ctx_other", C.c_void_p)]


class Batch(C.Structure):
    _fields_ = [("n_tokens", C.c_int32), ("token", C.POINTER(C.c_int32)), ("embd", C.POINTER(C.c_float)),
                ("pos", C.POINTER(C.c_int32)), ("n_seq_id", C.POINTER(C.c_int32)),
                ("seq_id", C.POINTER(C.POINTER(C.c_int32))), ("logits", C.POINTER(C.c_int8))]


LOG_CALLBACK = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_void_p)
POOLING_NONE = 0
FLASH_ATTN_AUTO = -1

_P, _I32, _U32, _B = C.c_void_p, C.c_int32, C.c_uint32, C.c_bool
# name: (library that exports it in b11100, restype, argtypes, required)
_SIGNATURES: dict[str, tuple] = {
    "llama_backend_init": ("llama", None, [], True),
    "llama_log_set": ("llama", None, [LOG_CALLBACK, _P], True),
    "llama_version": ("llama", C.c_char_p, [], False),
    "llama_model_default_params": ("llama", ModelParams, [], True),
    "llama_context_default_params": ("llama", ContextParams, [], True),
    "llama_model_load_from_file": ("llama", _P, [C.c_char_p, ModelParams], True),
    "llama_model_free": ("llama", None, [_P], True),
    "llama_init_from_model": ("llama", _P, [_P, ContextParams], True),
    "llama_free": ("llama", None, [_P], True),
    "llama_model_get_vocab": ("llama", _P, [_P], True),
    "llama_vocab_n_tokens": ("llama", _I32, [_P], True),
    "llama_model_n_embd": ("llama", _I32, [_P], True),
    "llama_model_n_layer": ("llama", _I32, [_P], True),
    "llama_model_n_params": ("llama", C.c_uint64, [_P], True),
    "llama_model_size": ("llama", C.c_uint64, [_P], True),
    "llama_model_ftype": ("llama", C.c_int, [_P], False),
    "llama_ftype_name": ("llama", C.c_char_p, [C.c_int], False),
    "llama_model_desc": ("llama", _I32, [_P, C.c_char_p, C.c_size_t], True),
    "llama_model_chat_template": ("llama", C.c_char_p, [_P, C.c_char_p], True),
    "llama_model_is_recurrent": ("llama", _B, [_P], True),
    "llama_model_is_hybrid": ("llama", _B, [_P], False),
    "llama_n_ctx": ("llama", _U32, [_P], True),
    "llama_n_batch": ("llama", _U32, [_P], True),
    "llama_n_seq_max": ("llama", _U32, [_P], True),
    "llama_tokenize": ("llama", _I32, [_P, C.c_char_p, _I32, C.POINTER(_I32), _I32, _B, _B], True),
    "llama_token_to_piece": ("llama", _I32, [_P, _I32, C.c_char_p, _I32, _I32, _B], True),
    "llama_decode": ("llama", _I32, [_P, Batch], True),
    "llama_get_logits_ith": ("llama", C.POINTER(C.c_float), [_P, _I32], True),
    "llama_get_embeddings_ith": ("llama", C.POINTER(C.c_float), [_P, _I32], True),
    "llama_get_memory": ("llama", _P, [_P], True),
    "llama_memory_clear": ("llama", None, [_P, _B], True),
    "llama_memory_seq_rm": ("llama", _B, [_P, _I32, _I32, _I32], True),
    "llama_memory_seq_cp": ("llama", None, [_P, _I32, _I32, _I32, _I32], True),
    "llama_set_embeddings": ("llama", None, [_P, _B], True),
    "ggml_backend_load_all_from_path": ("ggml", None, [C.c_char_p], True),
    "ggml_backend_reg_count": ("ggml", C.c_size_t, [], True),
    "ggml_backend_reg_get": ("ggml", _P, [C.c_size_t], True),
    "ggml_backend_reg_name": ("ggml-base", C.c_char_p, [_P], True),
    "ggml_backend_reg_dev_count": ("ggml-base", C.c_size_t, [_P], True),
    "ggml_backend_reg_dev_get": ("ggml-base", _P, [_P, C.c_size_t], True),
    "ggml_backend_dev_name": ("ggml-base", C.c_char_p, [_P], True),
    "ggml_backend_dev_description": ("ggml-base", C.c_char_p, [_P], True),
    "ggml_backend_dev_memory": ("ggml-base", None, [_P, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)], True),
    "ggml_commit": ("ggml-base", C.c_char_p, [], False),
    "ggml_version": ("ggml-base", C.c_char_p, [], False),
}


class LlamaApi:
    """The llama.cpp and ggml functions the backend calls, bound with ctypes to the signatures in include/llama.h and
    ggml-backend.h at b11100. The GPU backend libraries are loaded through ctypes before ggml loads the backends of the
    directory (see the module docstring). Tests give InprocBackend a stand-in with the same attribute names."""

    def __init__(self, directory: str | os.PathLike, verbose: bool = False):
        self.directory = Path(directory)
        self.preload_errors: dict[str, str] = {}
        self._handles: list = []
        if sys.platform == "win32":
            self._dll_dir = os.add_dll_directory(str(self.directory))
        libs: dict[str, Any] = {}
        for stem in ("ggml-base", "ggml", "llama"):
            f = self.directory / lib_file(stem)
            if not f.is_file():
                if stem == "ggml-base":
                    continue                  # older builds had no separate ggml-base
                raise BackendUnavailable(f"{f} is missing: {self.directory} is not a complete llama.cpp release")
            try:
                libs[stem] = C.CDLL(str(f), mode=getattr(C, "RTLD_GLOBAL", 0))
            except OSError as exc:
                raise BackendUnavailable(f"cannot load {f}: {exc}") from exc
        for stem in _GPU_LIBS:                  # before ggml_backend_load_all_from_path: see the module docstring
            f = self.directory / lib_file(stem)
            if f.is_file():
                try:
                    self._handles.append(C.CDLL(str(f)))
                except OSError as exc:
                    self.preload_errors[f.name] = str(exc)
        order = list(libs.values())
        for name, (home, restype, argtypes, required) in _SIGNATURES.items():
            fn = None
            for lib in ([libs[home]] if home in libs else []) + order:
                fn = getattr(lib, name, None)
                if fn is not None:
                    break
            if fn is None:
                if required:
                    raise BackendUnavailable(f"{self.directory}: llama.cpp's library has no {name} (tested with b11100)")
                setattr(self, name, None)
                continue
            fn.restype = restype
            fn.argtypes = argtypes
            setattr(self, name, fn)
        self._log_cb = LOG_CALLBACK(_log_forward if verbose else _log_quiet)
        self.llama_log_set(self._log_cb, None)
        self.ggml_backend_load_all_from_path(str(self.directory).encode("utf-8"))
        self.llama_backend_init()


def _log_quiet(level: int, text: bytes, user_data: Any) -> None:
    return None


def _log_forward(level: int, text: bytes, user_data: Any) -> None:
    log.debug("llama.cpp: %s", (text or b"").decode("utf-8", "replace").rstrip())


_APIS: dict[str, Any] = {}
_APIS_LOCK = threading.Lock()


def load_api(directory: str | os.PathLike, verbose: bool = False) -> Any:
    """The process's LlamaApi for a library directory (loaded once; ggml's backend registry is process-wide, so one
    process loads one llama.cpp build)."""
    key = str(Path(directory).resolve())
    with _APIS_LOCK:
        if key in _APIS:
            return _APIS[key]
        if _APIS:
            other = next(iter(_APIS))
            raise BackendUnavailable(f"this process already loaded llama.cpp from {other}; it cannot load a second build "
                                     f"from {key}")
        api = LlamaApi(key, verbose=verbose)
        _APIS[key] = api
        return api


def _text(value: Any) -> str | None:
    if value is None:
        return None
    return value.decode("utf-8", "replace") if isinstance(value, (bytes, bytearray)) else str(value)


def backend_devices(api: Any) -> list[dict]:
    """The ggml backend registries and their devices: [{registry, name, description, free_mb, total_mb, gpu}]."""
    out = []
    for r in range(int(api.ggml_backend_reg_count())):
        reg = api.ggml_backend_reg_get(r)
        rname = _text(api.ggml_backend_reg_name(reg)) or "?"
        for d in range(int(api.ggml_backend_reg_dev_count(reg))):
            dev = api.ggml_backend_reg_dev_get(reg, d)
            free, total = C.c_size_t(0), C.c_size_t(0)
            api.ggml_backend_dev_memory(dev, C.byref(free), C.byref(total))
            out.append({"registry": rname, "name": _text(api.ggml_backend_dev_name(dev)),
                        "description": _text(api.ggml_backend_dev_description(dev)),
                        "free_mb": int(free.value) // 2 ** 20, "total_mb": int(total.value) // 2 ** 20,
                        "gpu": rname.upper() not in _NOT_GPU})
    return out


def check_gpu(api: Any, n_gpu_layers: int) -> list[dict]:
    """The devices; BackendUnavailable when layers should go to a GPU and ggml registered none (a GPU backend that did
    not load would otherwise leave the model on the CPU without a word)."""
    devices = backend_devices(api)
    if n_gpu_layers != 0 and not any(d["gpu"] for d in devices):
        errors = "; ".join(f"{k}: {v}" for k, v in (getattr(api, "preload_errors", None) or {}).items())
        found = ", ".join(sorted({d["registry"] for d in devices})) or "none"
        raise BackendUnavailable(
            f"llama.cpp found no GPU (backends: {found})" + (f"; {errors}" if errors else "")
            + ": the model would run on the CPU. Use the release archive for your GPU with its runtime libraries (for "
              "CUDA the cudart archive, unpacked into the same directory), or pass --n-gpu-layers 0 to run on the CPU")
    return devices


def build_label(api: Any) -> str:
    """The library's build: ggml's commit (and the llama.cpp release when it is one Tez was measured with)."""
    commit = _text(api.ggml_commit()) if getattr(api, "ggml_commit", None) else None
    version = _text(api.ggml_version()) if getattr(api, "ggml_version", None) else None
    release = TESTED_BUILDS.get(commit or "")
    head = f"{release}-{commit}" if release else (f"ggml {version} ({commit})" if commit else f"ggml {version or '?'}")
    llama = _text(api.llama_version()) if getattr(api, "llama_version", None) else None
    return head + (f", libllama {llama}" if llama else "")


# ---------------------------------------------------------------------------------------------- one model, one context
class LlamaModel:
    """One loaded model and one context: tokenisation, llama_decode over reusable numpy-backed batches, and reads of
    the output rows. Not thread-safe (InprocBackend holds the lock)."""

    def __init__(self, api: Any, path: str, *, n_ctx: int, n_batch: int, n_ubatch: int, n_seq_max: int,
                 n_gpu_layers: int, n_threads: int | None = None, n_outputs: int | None = None):
        self.api = api
        self.path = path
        mp = api.llama_model_default_params()
        mp.n_gpu_layers = int(n_gpu_layers)
        t0 = time.perf_counter()
        self.model = api.llama_model_load_from_file(str(path).encode("utf-8"), mp)
        if not self.model:
            raise BackendUnavailable(f"llama.cpp could not load {path} (not a GGUF, an architecture this build does not "
                                     "know, or not enough memory)")
        self.ctx = None
        try:
            cp = api.llama_context_default_params()
            cp.n_ctx = int(n_ctx)
            cp.n_batch = int(n_batch)
            cp.n_ubatch = int(min(n_ubatch, n_batch))
            cp.n_seq_max = int(n_seq_max)
            outputs = max(int(n_seq_max), int(n_outputs or 0))
            cp.n_outputs_max = outputs                   # the logits buffer is n_outputs_max x n_vocab floats
            cp.n_outputs_max_per_seq = outputs           # set explicitly: the default is 1
            threads = n_threads or max(1, (os.cpu_count() or 2) // 2)
            cp.n_threads = cp.n_threads_batch = int(threads)
            cp.embeddings = False                        # switched on only for the decodes whose states are read
            cp.pooling_type = POOLING_NONE               # per-token states: the last token's is read
            cp.flash_attn_type = FLASH_ATTN_AUTO
            cp.swa_full = True                           # a full-size SWA cache, so a shared prefix stays valid
            cp.kv_unified = True                         # one KV buffer: sequence copies share the prefix cells
            cp.no_perf = True
            self.ctx = api.llama_init_from_model(self.model, cp)
            if not self.ctx:
                raise BackendUnavailable(f"llama.cpp could not create a context for {path} (n_ctx {n_ctx}, n_batch "
                                         f"{n_batch}: not enough memory? try a smaller --n-ctx)")
        except BaseException:
            self.close()
            raise
        self.load_s = time.perf_counter() - t0
        self.vocab = api.llama_model_get_vocab(self.model)
        self.n_vocab = int(api.llama_vocab_n_tokens(self.vocab))
        self.n_embd = int(api.llama_model_n_embd(self.model))
        self.n_layer = int(api.llama_model_n_layer(self.model))
        self.n_params = int(api.llama_model_n_params(self.model))
        self.size = int(api.llama_model_size(self.model))
        self.recurrent = bool(api.llama_model_is_recurrent(self.model))
        self.hybrid = bool(api.llama_model_is_hybrid(self.model)) if getattr(api, "llama_model_is_hybrid", None) else False
        self.n_ctx = int(api.llama_n_ctx(self.ctx))
        self.cap = int(api.llama_n_batch(self.ctx))
        self.n_seq = int(api.llama_n_seq_max(self.ctx))
        self.n_outputs = max(int(n_seq_max), int(n_outputs or 0))
        self.mem = api.llama_get_memory(self.ctx)
        self._embd_on = False
        self._tok = np.zeros(self.cap, np.int32)
        self._pos = np.zeros(self.cap, np.int32)
        self._nseq = np.ones(self.cap, np.int32)
        self._seq = np.zeros(self.cap, np.int32)
        self._out = np.zeros(self.cap, np.int8)
        p32 = C.POINTER(C.c_int32)
        base = self._seq.ctypes.data
        self._seqptrs = (p32 * self.cap)(*[C.cast(base + 4 * i, p32) for i in range(self.cap)])

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self.api.llama_free(self.ctx)
            self.ctx = None
        if getattr(self, "model", None):
            self.api.llama_model_free(self.model)
            self.model = None

    # ------------------------------------------------------------------ metadata
    def ftype(self) -> str | None:
        api = self.api
        if getattr(api, "llama_model_ftype", None) is None or getattr(api, "llama_ftype_name", None) is None:
            desc = self.desc().split(" ", 2)
            return desc[2] if len(desc) == 3 else None
        return _text(api.llama_ftype_name(api.llama_model_ftype(self.model)))

    def desc(self) -> str:
        buf = C.create_string_buffer(256)
        self.api.llama_model_desc(self.model, buf, 256)
        return buf.value.decode("utf-8", "replace")

    def chat_template(self) -> str | None:
        return _text(self.api.llama_model_chat_template(self.model, None))

    # ------------------------------------------------------------------ tokens
    def tokenize(self, text: str, add_special: bool = True, parse_special: bool = True) -> list[int]:
        """Tokens as llama-server's /completion makes them (special tokens parsed, BOS added when the model adds it)."""
        b = text.encode("utf-8")
        n = len(b) + 16
        buf = (C.c_int32 * n)()
        k = self.api.llama_tokenize(self.vocab, b, len(b), buf, n, add_special, parse_special)
        if k < 0:
            n = -k
            buf = (C.c_int32 * n)()
            k = self.api.llama_tokenize(self.vocab, b, len(b), buf, n, add_special, parse_special)
        if k < 0:
            raise BackendRequestError(f"the prompt could not be tokenised ({k})")
        return list(buf[:k])

    def piece(self, token: int) -> str:
        buf = C.create_string_buffer(256)
        k = self.api.llama_token_to_piece(self.vocab, token, buf, 256, 0, True)
        return buf.raw[: max(k, 0)].decode("utf-8", "replace")

    # ------------------------------------------------------------------ decoding
    def decode(self, tokens: Sequence[int], pos: Sequence[int], seqs: Sequence[int], outputs: Sequence[bool]) -> list[int]:
        """One llama_decode of at most n_batch tokens, one sequence id per token. Returns the batch positions of the
        tokens flagged as outputs: llama_get_logits_ith and llama_get_embeddings_ith index by batch position."""
        n = len(tokens)
        if not 0 < n <= self.cap:
            raise ValueError(f"a decode takes 1 to {self.cap} tokens, got {n}")
        self._tok[:n] = tokens
        self._pos[:n] = pos
        self._seq[:n] = seqs
        self._out[:n] = np.asarray(outputs, dtype=np.int8)
        batch = Batch(n, self._tok.ctypes.data_as(C.POINTER(C.c_int32)), None, self._pos.ctypes.data_as(C.POINTER(C.c_int32)),
                      self._nseq.ctypes.data_as(C.POINTER(C.c_int32)), C.cast(self._seqptrs, C.POINTER(C.POINTER(C.c_int32))),
                      self._out.ctypes.data_as(C.POINTER(C.c_int8)))
        rc = int(self.api.llama_decode(self.ctx, batch))
        if rc == 1:
            raise BackendRequestError(f"the request does not fit the KV cache ({self.n_ctx} tokens): start Tez with a "
                                      "larger --n-ctx")
        if rc != 0:
            raise BackendUnavailable(f"llama_decode failed ({rc})")
        return [int(i) for i in np.flatnonzero(self._out[:n])]

    def logits_row(self, i: int) -> np.ndarray:
        """The logits of the output at batch position i of the last decode (a view, valid until the next decode)."""
        p = self.api.llama_get_logits_ith(self.ctx, i)
        if not p:
            raise BackendUnavailable(f"llama.cpp returned no logits for batch position {i}")
        return np.ctypeslib.as_array(p, shape=(self.n_vocab,))

    def embd_row(self, i: int) -> np.ndarray:
        """The hidden state (after the output norm) at batch position i of the last decode, copied."""
        p = self.api.llama_get_embeddings_ith(self.ctx, i)
        if not p:
            raise BackendUnavailable(f"llama.cpp returned no embedding for batch position {i}")
        return np.ctypeslib.as_array(p, shape=(self.n_embd,)).astype(np.float32, copy=True)

    def set_embeddings(self, on: bool) -> None:
        if on != self._embd_on:
            self.api.llama_set_embeddings(self.ctx, bool(on))
            self._embd_on = bool(on)

    # ------------------------------------------------------------------ memory
    def clear(self) -> None:
        self.api.llama_memory_clear(self.mem, False)     # metadata only: freed cells are simply overwritten

    def seq_rm(self, seq: int, p0: int = -1, p1: int = -1) -> bool:
        return bool(self.api.llama_memory_seq_rm(self.mem, seq, p0, p1))

    def seq_cp(self, src: int, dst: int) -> None:
        self.api.llama_memory_seq_cp(self.mem, src, dst, -1, -1)


def _common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


LSE_WINDOW = 25.0        # nats below the top logit beyond which tokens are left out of the normaliser


def _log_normaliser(row: np.ndarray) -> float:
    """log sum exp over the vocabulary, so that letter logits become log-probabilities. Only the tokens within
    LSE_WINDOW nats of the top one are summed: together the others move the result by less than n_vocab * e^-25 (4e-6
    nats for Gemma's 262,144 tokens; measured on Gemma 4 12B rows, within 1e-7 of the full sum), while reading a whole row
    of llama.cpp's output buffer costs milliseconds (for 50 rows: 180-560 ms against 9-43 ms)."""
    m = float(row.max())
    near = row[row > m - LSE_WINDOW]
    return m + float(np.log(np.exp(near.astype(np.float64) - m).sum()))


# ---------------------------------------------------------------------------------------------- the backend
class InprocBackend(Backend):
    """llama.cpp inside the Tez process (see the module docstring).

        InprocBackend("models/gemma-4-12b-it-Q8_0.gguf", template="gemma4", lib="C:/llama.cpp")

    model_path    the GGUF file
    lib           llama.cpp's library directory (or file); default TEZ_LLAMA_LIB, else next to llama-server on PATH
    n_ctx         KV cache size in tokens, shared by the sequences of a batched read (default 4096)
    n_batch       most tokens per llama_decode (default n_ctx, so a batched read that fits the cache is one decode)
    n_ubatch      physical batch size (default 512)
    n_seq_max     sequences in the context: prompts read together in one wave, plus the one that holds the prefix
                  (default 64); also the most outputs per decode
    n_gpu_layers  layers offloaded to the GPU (default -1: all; 0 runs on the CPU and needs no GPU backend)
    n_outputs     most outputs per decode (default n_seq_max). With embeddings on, llama.cpp makes every token of a
                  decode an output, so a stream reader that reads states wants more than n_seq_max: a hybrid model
                  (Qwen3.5) holds a recurrent state per sequence and runs best with a small n_seq_max
    cache_prompt  reuse the prefix a single read shares with the previous one (default True)
    The model is loaded on first use, or by load(); nothing is read from disk before that.
    """

    batched = True

    def __init__(self, model_path: str | os.PathLike, template: str = "gemma4", *, lib: str | os.PathLike | None = None,
                 n_ctx: int = DEFAULT_N_CTX, n_batch: int | None = None, n_ubatch: int = DEFAULT_N_UBATCH,
                 n_seq_max: int = DEFAULT_N_SEQ_MAX, n_gpu_layers: int = DEFAULT_N_GPU_LAYERS, cache_prompt: bool = True,
                 model_name: str | None = None, n_threads: int | None = None, verbose: bool = False, api: Any = None,
                 n_outputs: int | None = None):
        self.model_path = str(model_path)
        if not self.model_path:
            raise ValueError("inproc needs the path of a GGUF file: inproc:/path/to/model.gguf")
        self.template = check_template(template)
        self.url = f"{SCHEME}{self.model_path}"
        self.lib = lib
        for name, value, low in (("n_ctx", n_ctx, 16), ("n_ubatch", n_ubatch, 1), ("n_seq_max", n_seq_max, 2)):
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f"{name} must be a whole number of at least {low}, got {value!r}")
        if n_batch is not None and (isinstance(n_batch, bool) or not isinstance(n_batch, int) or n_batch < 1):
            raise ValueError(f"n_batch must be a whole number of at least 1, got {n_batch!r}")
        if isinstance(n_gpu_layers, bool) or not isinstance(n_gpu_layers, int):
            raise ValueError(f"n_gpu_layers must be a whole number (-1: all layers), got {n_gpu_layers!r}")
        if n_outputs is not None and (isinstance(n_outputs, bool) or not isinstance(n_outputs, int) or n_outputs < 1):
            raise ValueError(f"n_outputs must be a whole number of at least 1, got {n_outputs!r}")
        self.n_outputs = n_outputs
        self.n_ctx = n_ctx
        self.n_batch = n_batch or n_ctx
        self.n_ubatch = n_ubatch
        self.n_seq_max = n_seq_max
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.cache_prompt = cache_prompt
        self.verbose = verbose
        self._model_name = model_name
        self._api = api
        self._m: LlamaModel | None = None
        self._devices: list[dict] = []
        self._letter_ids: np.ndarray | None = None
        self._cached: list[int] = []            # the tokens sequence 0 holds
        self._no_rollback = False
        self._reserved: list[int] = []          # sequence ids held by stream readers (the top of the range)
        self._epoch = 0                         # bumped whenever the KV cache is cleared: stream readers rebuild
        self._lock = threading.RLock()
        self.loads = 0                          # how many times a model was loaded (tests check laziness)

    def __repr__(self) -> str:
        return (f"InprocBackend({self.model_path!r}, template={self.template!r}, n_ctx={self.n_ctx}, n_batch={self.n_batch}, "
                f"n_seq_max={self.n_seq_max}, n_gpu_layers={self.n_gpu_layers})")

    # ------------------------------------------------------------------ lifecycle
    @property
    def loaded(self) -> bool:
        return self._m is not None

    def load(self) -> InprocBackend:
        """Load the library and the model now (tez serve does this at start-up). Returns the backend."""
        with self._lock:
            self._ensure()
        return self

    def close(self) -> None:
        """Free the context and the model (the library stays loaded)."""
        with self._lock:
            if self._m is not None:
                self._m.close()
                self._m = None
            self._cached = []
            self._reserved = []
            self._epoch += 1

    def _ensure(self) -> LlamaModel:
        if self._m is not None:
            return self._m
        if not Path(self.model_path).is_file():
            raise BackendUnavailable(f"no GGUF file at {self.model_path}")
        api = self._api
        if api is None:
            api = self._api = load_api(find_library(self.lib), verbose=self.verbose)
        devices = check_gpu(api, self.n_gpu_layers)
        m = LlamaModel(api, self.model_path, n_ctx=self.n_ctx, n_batch=self.n_batch, n_ubatch=self.n_ubatch,
                       n_seq_max=self.n_seq_max, n_gpu_layers=self.n_gpu_layers, n_threads=self.n_threads,
                       n_outputs=self.n_outputs)
        try:
            ids = []
            for letter in LETTERS:
                t = m.tokenize(letter, add_special=False, parse_special=False)
                if len(t) != 1 or m.piece(t[0]) != letter:
                    raise BackendUnavailable(f"{self.model_path}: the letter {letter!r} is not a single token ({t}), so the "
                                             "letters readout cannot read it")
                ids.append(t[0])
        except BaseException:
            m.close()
            raise
        self._letter_ids = np.asarray(ids, dtype=np.int64)
        self._no_rollback = m.recurrent or m.hybrid
        self._devices = backend_devices(api) if devices else devices
        self._m = m
        self._cached = []
        self.loads += 1
        self._check_template(m)
        log.info("inproc: loaded %s in %.1f s (%s)", self.model_path, m.load_s,
                 ", ".join(d["description"] or d["name"] for d in self._devices if d["gpu"]) or "CPU")
        return m

    def _check_template(self, m: LlamaModel) -> None:
        from .doctor import detect_template
        family = detect_template(m.chat_template())
        if family is not None and family != self.template:
            log.warning("inproc: %s has %s's chat template, but the template is %s (tez --template %s?)",
                        self.model_path, family, self.template, family)

    # ------------------------------------------------------------------ model info
    def info(self) -> dict:
        """The fields LlamaCppBackend.info() reports (the model is loaded first)."""
        with self._lock:
            m = self._ensure()
            return {"model_path": self.model_path, "model_alias": None, "ftype": m.ftype(), "n_params": m.n_params,
                    "n_embd": m.n_embd, "n_ctx": m.n_ctx, "build": build_label(self._api)}

    def details(self) -> dict:
        """Everything tez doctor shows: library, build, devices, settings and the model's shape."""
        with self._lock:
            m = self._ensure()
            api = self._api
            return {"library": str(getattr(api, "directory", "")), "build": build_label(api),
                    "commit": _text(api.ggml_commit()) if getattr(api, "ggml_commit", None) else None,
                    "devices": list(self._devices), "preload_errors": dict(getattr(api, "preload_errors", {}) or {}),
                    "n_ctx": m.n_ctx, "n_batch": m.cap, "n_ubatch": self.n_ubatch, "n_seq_max": m.n_seq,
                    "n_gpu_layers": self.n_gpu_layers, "n_layer": m.n_layer, "n_embd": m.n_embd, "n_vocab": m.n_vocab,
                    "n_params": m.n_params, "size_mb": m.size // 2 ** 20, "recurrent": m.recurrent, "hybrid": m.hybrid,
                    "desc": m.desc(), "chat_template": m.chat_template(), "load_s": round(m.load_s, 2)}

    def model_name(self) -> str:
        if self._model_name:
            return self._model_name
        try:
            i = self.info()
        except BackendUnavailable:
            return "unknown"
        self._model_name = derive_model_name(i["model_path"], None, i["ftype"], i["n_params"], self.template)
        return self._model_name

    def known_model_name(self) -> str | None:
        """The name without loading anything: the given or derived name, or the GGUF file's name when it is descriptive
        (an Ollama blob's hash name needs the model's metadata)."""
        if self._model_name:
            return self._model_name
        stem = re.sub(r"\.gguf$", "", Path(self.model_path).name, flags=re.I)
        return stem.lower() if stem and not _HASHED.fullmatch(stem) else None

    def health(self) -> dict:
        if self._m is not None:
            return {"ok": True, "status": "ok"}
        return {"ok": True, "status": "not loaded"}

    def tokenize(self, text: str, add_special: bool = True) -> list[int]:
        """Token ids of `text` in the model's vocabulary (special tokens parsed, as prompts are)."""
        with self._lock:
            return self._ensure().tokenize(text, add_special=add_special)

    # ------------------------------------------------------------------ readouts
    def letters(self, prompt: str, k: int) -> LetterScores:
        if not 1 <= k <= MAX_LETTERS:
            raise ValueError(f"a letter readout reads 1 to {MAX_LETTERS} options, got {k}")
        z, _, t = self._single(prompt, k, False)
        return LetterScores(z[:k], t["tokens"], timings=t["timings"], ms=t["ms"])

    def embed(self, prompt: str) -> Embedding:
        _, v, t = self._single(prompt, 0, True)
        return Embedding(v, t["tokens"], timings=t["timings"], ms=t["ms"])

    def letters_many(self, prompts: Sequence[str], ks: Sequence[int]) -> list[LetterScores]:
        """Letter readouts of many prompts in one batched read (read_many)."""
        return [r for r, _ in self.read_many(prompts, ks)]

    def read_many(self, prompts: Sequence[str], ks: Sequence[int] | None = None,
                  embed: Sequence[bool] | None = None) -> list[tuple[LetterScores | None, Embedding | None]]:
        """Read many prompts together: per prompt the letter log-probabilities of its first ks[i] options (0: none) and,
        where embed[i], the last-token state. Prompts sharing a token prefix evaluate it once (sequence copies) and all
        suffixes go into one llama_decode (more when they do not fit n_batch, the KV cache or n_seq_max). The same
        prompt twice is read once. Each result's timings say what the batch evaluated for it (prompt_n: its suffix,
        plus the shared prefix for the first prompt; cache_n: the rest) and batch_n / batch_ms describe the batch;
        its ms is the batch's wall time divided among the results."""
        n = len(prompts)
        ks = [0] * n if ks is None else [int(k or 0) for k in ks]
        embed = [False] * n if embed is None else [bool(e) for e in embed]
        if len(ks) != n or len(embed) != n:
            raise ValueError("prompts, ks and embed must have the same length")
        for k, e in zip(ks, embed):
            if not 0 <= k <= MAX_LETTERS:
                raise ValueError(f"a letter readout reads 1 to {MAX_LETTERS} options, got {k}")
            if not k and not e:
                raise ValueError("every prompt needs a letters readout (k >= 1) or an embedding")
        if n == 0:
            return []
        index: dict[str, int] = {}
        uniq: list[list] = []                    # [prompt, k, embed]
        where = []
        for p, k, e in zip(prompts, ks, embed):
            j = index.get(p)
            if j is None:
                j = index[p] = len(uniq)
                uniq.append([p, k, e])
            else:
                uniq[j][1], uniq[j][2] = max(uniq[j][1], k), uniq[j][2] or e
            where.append(j)
        with self._lock:
            m = self._ensure()
            t0 = time.perf_counter()
            toks = [m.tokenize(u[0]) for u in uniq]
            if len(uniq) == 1:
                z, v, n_eval = self._read_one(toks[0], uniq[0][1], uniq[0][2])
                reads = [(z, v, n_eval)]
                prefix = len(toks[0]) - n_eval
            else:
                reads, prefix = self._read_batch(toks, [u[1] for u in uniq], [u[2] for u in uniq])
            wall = (time.perf_counter() - t0) * 1000.0
        records = sum(1 for k in ks if k) + sum(1 for e in embed if e)
        share = wall / max(1, records)
        evaluated = sum(r[2] for r in reads)
        out: list[tuple[LetterScores | None, Embedding | None]] = []
        for i in range(n):
            j = where[i]
            z, v, n_eval = reads[j]
            total = len(toks[j])
            timings = {"prompt_n": n_eval, "cache_n": total - n_eval, "prompt_ms": round(share, 3), "batch_n": len(uniq),
                       "batch_ms": round(wall, 3), "batch_prompt_n": evaluated, "batch_prefix_n": prefix}
            ls = LetterScores(z[: ks[i]].copy(), total, timings=dict(timings), ms=share) if ks[i] else None
            em = Embedding(v.copy(), total, timings=dict(timings), ms=share) if embed[i] else None
            out.append((ls, em))
        return out

    def stream_reader(self, parts: Sequence[tuple[str, str, int, bool]], fold: bool = False) -> InprocStreamReader:
        """A reader of prompts head + state + tail whose state grows between reads (InprocStreamReader). parts: one
        (head, tail, k, embed) per prompt read together at every step: k option letters (0: none) and whether the
        last-token state is read. fold: commit words inside the next read's decode instead of in a decode of their own
        (see InprocStreamReader)."""
        return InprocStreamReader(self, parts, fold=fold)

    def _reserve(self, n: int) -> list[int]:
        """n sequence ids for a stream reader, from the top of the range; at least one other stays free for the
        batched reads (sequence 0 is the single-prompt cache)."""
        m = self._m
        taken = set(self._reserved)
        ids = [s for s in range(m.n_seq - 1, 0, -1) if s not in taken][:n]
        if len(ids) < n or m.n_seq - 1 - len(taken) - n < 1:
            raise BackendRequestError(f"a stream reader needs {n} sequences and {m.n_seq - 1 - len(taken)} are free "
                                      f"(n_seq_max {m.n_seq}): start the backend with a larger n_seq_max")
        self._reserved += ids
        return ids

    def _release(self, ids: Sequence[int]) -> None:
        m = self._m
        for s in ids:
            if s in self._reserved:
                self._reserved.remove(s)
                if m is not None:
                    m.seq_rm(s)

    # ------------------------------------------------------------------ internals (the lock is held)
    def _single(self, prompt: str, k: int, want_embed: bool) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
        with self._lock:
            m = self._ensure()
            t0 = time.perf_counter()
            toks = m.tokenize(prompt)
            z, v, n_eval = self._read_one(toks, k, want_embed)
            ms = (time.perf_counter() - t0) * 1000.0
        timings = {"prompt_n": n_eval, "cache_n": len(toks) - n_eval, "prompt_ms": round(ms, 3)}
        return z, v, {"tokens": len(toks), "timings": timings, "ms": ms}

    def _fits(self, toks: Sequence[int]) -> None:
        if not toks:
            raise BackendRequestError("the prompt is empty")
        if len(toks) > self._m.n_ctx:
            raise BackendRequestError(f"the request exceeds the available context size: {len(toks)} tokens, n_ctx "
                                      f"{self._m.n_ctx} (start Tez with a larger --n-ctx)")

    def _reset(self) -> None:
        """After a failed read: drop every sequence, so the next read starts from a clean cache."""
        m = self._m
        if m is None:
            return
        try:
            m.set_embeddings(False)
            m.clear()
        finally:
            self._cached = []
            self._epoch += 1                     # stream readers lost their sequences too

    def _keep_on_seq0(self, toks: Sequence[int], need_output: bool) -> int:
        """Leave on sequence 0 the longest prefix of `toks` it already holds (at most len - 1 tokens when the last one
        must be evaluated for an output) and return its length. A tail is removed with llama_memory_seq_rm, except on
        hybrid and recurrent models: their state cannot roll back, so they start again unless it is an extension."""
        m = self._m
        held = self._cached
        n = _common_prefix(held, toks) if self.cache_prompt else 0
        if need_output and n == len(toks):
            n -= 1
        if n < len(held):
            if n == 0 or self._no_rollback or not m.seq_rm(0, n, -1):
                if self._reserved:
                    m.seq_rm(0)                  # stream readers hold other sequences: drop only this one
                else:
                    m.clear()
                n = 0
            self._cached = list(held[:n])
        return n

    def _eval(self, toks: Sequence[int], seq: int, start: int, output_last: bool) -> int | None:
        """Evaluate a token run on one sequence in n_batch chunks; the batch position of the last token's output."""
        m = self._m
        pos_out = None
        n = len(toks)
        for s in range(0, n, m.cap):
            chunk = toks[s: s + m.cap]
            last = output_last and s + len(chunk) == n
            outs = m.decode(chunk, range(start + s, start + s + len(chunk)), [seq] * len(chunk),
                            [False] * (len(chunk) - 1) + [last])
            if last:
                pos_out = outs[-1]
        return pos_out

    def _letters_of(self, rows: list[np.ndarray], k: int | list[int]) -> list[np.ndarray]:
        """Letter log-probabilities from output rows (views into llama.cpp's logits buffer, read before the next decode)."""
        ks = k if isinstance(k, list) else [k] * len(rows)
        ids = self._letter_ids
        return [r[ids[:kk]].astype(np.float64) - _log_normaliser(r) for r, kk in zip(rows, ks)]

    def _read_one(self, toks: list[int], k: int, want_embed: bool) -> tuple[np.ndarray | None, np.ndarray | None, int]:
        """One prompt on sequence 0, reusing what it shares with the previous one. (letters, state, tokens evaluated)."""
        m = self._m
        self._fits(toks)
        try:
            n = self._keep_on_seq0(toks, need_output=True)
            z = v = None
            if want_embed:
                if len(toks) - 1 > n:
                    self._eval(toks[n:-1], 0, n, output_last=False)
                m.set_embeddings(True)
                (p,) = m.decode([toks[-1]], [len(toks) - 1], [0], [True])
                v = m.embd_row(p)
                if k:
                    z = self._letters_of([m.logits_row(p)], k)[0]
                m.set_embeddings(False)
            else:
                p = self._eval(toks[n:], 0, n, output_last=True)
                z = self._letters_of([m.logits_row(p)], k)[0]
            self._cached = list(toks)
            return z, v, len(toks) - n
        except BaseException:
            self._reset()
            raise

    def _read_batch(self, toks: list[list[int]], ks: list[int], embeds: list[bool]) -> tuple[list[tuple], int]:
        """Prompts sharing a prefix: the prefix once on sequence 0, copied to one sequence per prompt, every suffix in one
        llama_decode (several when they do not fit). Prompts whose states are read evaluate all but their last token
        there; the last tokens then go in one decode with embeddings on. Returns ([(letters, state, tokens evaluated)]
        per prompt, the shared prefix length)."""
        m = self._m
        for t in toks:
            self._fits(t)
        U = len(toks)
        shortest = min(len(t) for t in toks)
        lp = shortest - 1                       # every prompt keeps at least its last token to evaluate
        for t in toks[1:]:
            lp = min(lp, _common_prefix(toks[0], t))
            if lp == 0:
                break
        zs: list[np.ndarray | None] = [None] * U
        vs: list[np.ndarray | None] = [None] * U
        evaluated = [len(t) - lp for t in toks]
        used: set[int] = set()
        try:
            held = self._keep_on_seq0(toks[0][:lp], need_output=False)
            if held < lp:
                self._eval(toks[0][held:lp], 0, held, output_last=False)
            self._cached = list(toks[0][:lp])
            evaluated[0] += lp - held
            waves, cur, resident = [], [], lp
            free = m.n_seq - 1 - len(self._reserved)   # sequences 1..free (stream readers hold the top ones)
            for j, t in enumerate(toks):         # the suffixes of a wave must fit the cache too
                s = len(t) - lp
                if cur and (len(cur) >= free or resident + s > m.n_ctx):
                    waves.append(cur)
                    cur, resident = [], lp
                cur.append(j)
                resident += s
            waves.append(cur)
            for wave in waves:
                seq = {j: 1 + i for i, j in enumerate(wave)}
                for j in wave:
                    if lp:
                        m.seq_cp(0, seq[j])
                    used.add(seq[j])
                self._suffixes(toks, ks, embeds, wave, seq, lp, zs)
                last = [j for j in wave if embeds[j]]
                if last:
                    m.set_embeddings(True)       # every token of these decodes is an output: only the last tokens go in
                    for s in range(0, len(last), m.n_outputs):
                        group = last[s: s + m.n_outputs]
                        outs = m.decode([toks[j][-1] for j in group], [len(toks[j]) - 1 for j in group],
                                        [seq[j] for j in group], [True] * len(group))
                        for j, p in zip(group, outs):
                            vs[j] = m.embd_row(p)
                        lj = [(j, p) for j, p in zip(group, outs) if ks[j]]
                        if lj:
                            got = self._letters_of([m.logits_row(p) for _, p in lj], [ks[j] for j, _ in lj])
                            for (j, _), z in zip(lj, got):
                                zs[j] = z
                    m.set_embeddings(False)
                for j in wave:
                    m.seq_rm(seq[j])
                    used.discard(seq[j])
        except BaseException:
            self._reset()
            raise
        finally:
            for s in used:                       # a read that failed half way leaves no sequence behind
                try:
                    m.seq_rm(s)
                except Exception:  # noqa: BLE001 - best effort during cleanup
                    pass
        return [(zs[j], vs[j], evaluated[j]) for j in range(U)], lp

    def _suffixes(self, toks: list[list[int]], ks: list[int], embeds: list[bool], wave: list[int], seq: dict, lp: int,
                  zs: list) -> None:
        """Every suffix of a wave (all but the last token for prompts whose states are read), packed into decodes of at
        most n_batch tokens and n_outputs outputs. A letters-only prompt's sequence is freed as soon as it is read."""
        m = self._m
        bt: list[int] = []
        bp: list[int] = []
        bs: list[int] = []
        bo: list[bool] = []
        owners: list[int] = []                   # prompt index of each output in the pending decode

        def flush() -> None:
            if not bt:
                return
            outs = m.decode(bt, bp, bs, bo)
            if owners:
                got = self._letters_of([m.logits_row(p) for p in outs], [ks[j] for j in owners])
                for j, z in zip(owners, got):
                    zs[j] = z
                    m.seq_rm(seq[j])
            bt.clear()
            bp.clear()
            bs.clear()
            bo.clear()
            owners.clear()

        for j in wave:
            t = toks[j]
            end = len(t) - 1 if embeds[j] else len(t)
            start = lp
            while start < end:
                piece = min(end - start, m.cap - len(bt))
                closes = start + piece == end and not embeds[j]
                if piece <= 0 or (closes and len(owners) >= m.n_outputs):
                    flush()
                    continue
                bt += t[start: start + piece]
                bp += range(start, start + piece)
                bs += [seq[j]] * piece
                bo += [False] * (piece - 1) + [closes]
                if closes:
                    owners.append(j)
                start += piece
        flush()


class InprocStreamReader:
    """Reads prompts head_j + state + tail_j (a few prompts, j = 0..n-1) while the state grows between reads: the voice
    loop's transcript, one word at a time. Per prompt it keeps three sequences of its own:
      P_j  the head, evaluated once (a checkpoint: nothing ever extends or removes it)
      A_j  a copy of P_j extended with the committed state (the words read so far)
      S_j  a scratch copy of A_j, made for each read, that receives the new words and the tail, then is dropped
    read(state) evaluates only the tokens past the committed state, for every prompt in ONE llama_decode, and returns
    their letter log-probabilities (and last-token states). commit(state) then extends A_j with the state's own tokens,
    which the caller does after acting on the read, in the gap before the next word. No sequence is ever rolled back,
    so hybrid and recurrent models (Qwen3.5) work the same way: their copies copy the recurrent state.
    When the state's tokens stop extending the committed ones (a word that merges with the next, a revised
    transcript, a new residual), A_j is rebuilt from P_j. A cleared cache (a failed read elsewhere, close()) is noticed
    through the backend's epoch and the heads are evaluated again.
    fold=True defers each commit into the next read's llama_decode: the committed tokens go to A_j and, with the new
    words and the tail, to S_j (copied from A_j before them) in the same batch. A step then costs one forward pass
    instead of two, which matters for a large attention model, whose decode costs about the same for 2 tokens as for
    12 (the weights are read either way). Hybrid and recurrent models split a batch that spans two sequences into
    several, so for them the default (a commit decode of its own, in the gap) is the faster one."""

    def __init__(self, backend: InprocBackend, parts: Sequence[tuple[str, str, int, bool]], fold: bool = False):
        if not parts:
            raise ValueError("a stream reader needs at least one prompt")
        self.backend = backend
        self.parts = [(str(h), str(t), int(k or 0), bool(e)) for h, t, k, e in parts]
        for _, tail, k, e in self.parts:
            if not 0 <= k <= MAX_LETTERS:
                raise ValueError(f"a letter readout reads 1 to {MAX_LETTERS} options, got {k}")
            if not k and not e:
                raise ValueError("every prompt needs a letters readout (k >= 1) or an embedding")
            if not tail:
                raise ValueError("the tail must not be empty: it holds the read position")
        self.want_embed = any(e for _, _, _, e in self.parts)
        self._seqs: list[tuple[int, int, int]] = []     # (P, A, S) per prompt
        self._P: list[list[int]] = []                   # head tokens per prompt
        self._C: list[list[int]] = []                   # tokens on A_j
        self._last: tuple[str, list[list[int]]] | None = None   # the state of the last read and its prompts' tokens
        self._epoch: int | None = None
        self.restored = 0                               # times A_j was rebuilt from its head
        self.head_ms = 0.0
        self.fold = bool(fold)
        self._pending: list[list[int] | None] = [None] * len(self.parts)   # fold: what A_j should hold next

    # ------------------------------------------------------------------ set-up (the backend lock is held)
    def _ensure(self) -> LlamaModel:
        b = self.backend
        m = b._ensure()
        if self._seqs and self._epoch == b._epoch:
            return m
        if not self._seqs or not all(s in b._reserved for trio in self._seqs for s in trio):
            ids = b._reserve(3 * len(self.parts))
            self._seqs = [(ids[3 * j], ids[3 * j + 1], ids[3 * j + 2]) for j in range(len(self.parts))]
        t0 = time.perf_counter()
        m.set_embeddings(False)
        self._P, self._C = [], []
        for (head, tail, _, _), (sp, sa, ss) in zip(self.parts, self._seqs):
            for s in (sp, sa, ss):
                m.seq_rm(s)
            a = m.tokenize(head + "open notes" + tail)
            c = m.tokenize(head + "what time is it" + tail)
            p = a[: max(1, _common_prefix(a, c) - 1)]     # the head's tokens, one token of margin at the seam
            b._fits(p)
            b._eval(p, sp, 0, output_last=False)
            m.seq_cp(sp, sa)
            self._P.append(p)
            self._C.append(list(p))
        self._last = None
        self._pending = [None] * len(self.parts)
        self._epoch = b._epoch
        self.head_ms = (time.perf_counter() - t0) * 1000.0
        return m

    def load(self) -> InprocStreamReader:
        """Evaluate the heads now (otherwise the first read does)."""
        with self.backend._lock:
            self._ensure()
        return self

    def _rebuild(self, m: LlamaModel, j: int) -> None:
        sp, sa, _ = self._seqs[j]
        m.seq_rm(sa)
        m.seq_cp(sp, sa)
        self._C[j] = list(self._P[j])
        self.restored += 1

    def _decode_on(self, m: LlamaModel, seq: int, toks: Sequence[int], start: int) -> None:
        """Extend one sequence without outputs (chunks small enough for the output buffer when embeddings are on)."""
        step = min(m.cap, m.n_outputs) if m._embd_on else m.cap
        for s in range(0, len(toks), step):
            chunk = toks[s: s + step]
            m.decode(chunk, range(start + s, start + s + len(chunk)), [seq] * len(chunk), [False] * len(chunk))

    # ------------------------------------------------------------------ reads
    def read(self, state: str) -> list[tuple[LetterScores | None, Embedding | None]]:
        """The prompts head_j + state + tail_j, evaluated past the committed state on scratch copies in one decode.
        `state` is the text exactly as it sits in the prompt (tez.prompt.render_state of the state)."""
        b = self.backend
        with b._lock:
            m = self._ensure()
            t0 = time.perf_counter()
            try:
                return self._read(m, state, t0)
            except BaseException:
                b._reset()
                raise

    def _read(self, m: LlamaModel, state: str, t0: float) -> list[tuple[LetterScores | None, Embedding | None]]:
        b = self.backend
        toks = [m.tokenize(head + state + tail) for head, tail, _, _ in self.parts]
        starts, rests, exts, targets = [], [], [], []
        for j, T in enumerate(toks):
            b._fits(T)
            target, self._pending[j] = self._pending[j], None
            if _common_prefix(self._C[j], T) < len(self._C[j]):
                if _common_prefix(self._P[j], T) < len(self._P[j]):
                    raise BackendRequestError("the prompt no longer starts with the stream's head")
                self._rebuild(m, j)
            c = len(self._C[j])
            if target is not None and (len(target) <= c or _common_prefix(target, T) < len(target)):
                target = None                            # a folded commit this prompt no longer extends
            _, sa, ss = self._seqs[j]
            m.seq_rm(ss)
            m.seq_cp(sa, ss)
            starts.append(c)
            rests.append(T[c:])
            exts.append(target[c:] if target is not None else [])
            targets.append(target)
        total = sum(len(r) for r in rests) + sum(len(x) for x in exts)
        m.set_embeddings(self.want_embed)
        limit = min(m.cap, m.n_outputs) if self.want_embed else m.cap
        if total <= limit:
            bt: list[int] = []
            bp: list[int] = []
            bs: list[int] = []
            bo: list[bool] = []
            for j, rest in enumerate(rests):
                if exts[j]:                              # the folded commit: A_j's new tokens, no output
                    bt += exts[j]
                    bp += range(starts[j], starts[j] + len(exts[j]))
                    bs += [self._seqs[j][1]] * len(exts[j])
                    bo += [False] * len(exts[j])
                bt += rest
                bp += range(starts[j], starts[j] + len(rest))
                bs += [self._seqs[j][2]] * len(rest)
                bo += [False] * (len(rest) - 1) + [True]
            outs = m.decode(bt, bp, bs, bo)
        else:                                            # rare (a rebuild): the bodies first, then the last tokens
            for j, rest in enumerate(rests):
                if exts[j]:
                    self._decode_on(m, self._seqs[j][1], exts[j], starts[j])
                if len(rest) > 1:
                    self._decode_on(m, self._seqs[j][2], rest[:-1], starts[j])
            n = len(rests)
            outs = m.decode([r[-1] for r in rests], [starts[j] + len(rests[j]) - 1 for j in range(n)],
                            [self._seqs[j][2] for j in range(n)], [True] * n)
        for j, target in enumerate(targets):
            if target is not None:
                self._C[j] = list(target)
        rows = []
        for (_, _, k, e), p in zip(self.parts, outs):
            z = b._letters_of([m.logits_row(p)], k)[0] if k else None
            v = m.embd_row(p) if e else None
            rows.append((z, v))
        ms = (time.perf_counter() - t0) * 1000.0
        for _, _, ss in self._seqs:
            m.seq_rm(ss)
        self._last = (state, toks)
        out: list[tuple[LetterScores | None, Embedding | None]] = []
        for j, (_, _, k, e) in enumerate(self.parts):
            timings = {"prompt_n": len(rests[j]), "cache_n": starts[j], "prompt_ms": round(ms, 3),
                       "batch_n": len(self.parts), "batch_prompt_n": total, "restored": self.restored}
            z, v = rows[j]
            out.append((LetterScores(z, len(toks[j]), timings=dict(timings), ms=ms) if k else None,
                        Embedding(v, len(toks[j]), timings=dict(timings), ms=ms) if e else None))
        return out

    def commit(self, state: str) -> int:
        """Extend A_j with the tokens the state contributes to every prompt (the state's own, up to where the tail
        starts). Called after the decision on a read, in the gap before the next one. Returns the tokens evaluated."""
        b = self.backend
        with b._lock:
            m = self._ensure()
            try:
                return self._commit(m, state)
            except BaseException:
                b._reset()
                raise

    def _commit(self, m: LlamaModel, state: str) -> int:
        b = self.backend
        if self._last is not None and self._last[0] == state:
            toks = self._last[1]
        else:
            toks = [m.tokenize(head + state + tail) for head, tail, _, _ in self.parts]
        if self.fold:                                    # the next read evaluates them, in its own decode
            for j, ((head, _, _, _), T) in enumerate(zip(self.parts, toks)):
                s = _common_prefix(m.tokenize(head + state), T)
                self._pending[j] = list(T[:s]) if s > len(self._P[j]) else None
            return 0
        m.set_embeddings(self.want_embed)
        n_eval = 0
        for j, ((head, _, _, _), T) in enumerate(zip(self.parts, toks)):
            b._fits(T)
            s = _common_prefix(m.tokenize(head + state), T)      # the state's tokens as the full prompt has them
            if _common_prefix(self._C[j], T) < len(self._C[j]) or s < len(self._C[j]):
                if _common_prefix(self._P[j], T) < len(self._P[j]):
                    raise BackendRequestError("the prompt no longer starts with the stream's head")
                self._rebuild(m, j)
            c = len(self._C[j])
            new = T[c:s]
            if new:
                self._decode_on(m, self._seqs[j][1], new, c)
                n_eval += len(new)
            self._C[j] = list(T[:s])
        return n_eval

    def reset(self) -> None:
        """Forget the committed state (a new utterance): A_j holds its head again."""
        b = self.backend
        with b._lock:
            m = self._ensure()
            for j in range(len(self.parts)):
                if len(self._C[j]) != len(self._P[j]):
                    sp, sa, _ = self._seqs[j]
                    m.seq_rm(sa)
                    m.seq_cp(sp, sa)
                    self._C[j] = list(self._P[j])
            self._last = None
            self._pending = [None] * len(self.parts)

    def close(self) -> None:
        """Free this reader's sequences."""
        b = self.backend
        with b._lock:
            ids = [s for trio in self._seqs for s in trio]
            if ids and self._epoch == b._epoch:
                b._release(ids)
            else:
                b._reserved = [s for s in b._reserved if s not in ids]
            self._seqs, self._P, self._C, self._last = [], [], [], None
            self._epoch = None

    @property
    def committed_tokens(self) -> list[int]:
        """Tokens A_j holds past its head, per prompt (for tests and traces)."""
        return [len(c) - len(p) for c, p in zip(self._C, self._P)]


def parse_spec(spec: str) -> str:
    """The GGUF path of an 'inproc:PATH' backend spec."""
    path = spec[len(SCHEME):].strip() if spec.startswith(SCHEME) else ""
    if not path:
        raise ValueError("an in-process backend is inproc:PATH, the path of a GGUF file (e.g. inproc:models/model.gguf)")
    return path


def options_from_env(env: Callable[[str], Any]) -> dict:
    """The in-process settings from TEZ_LLAMA_LIB, TEZ_N_CTX, TEZ_N_BATCH and TEZ_N_GPU_LAYERS (env: a getter such as
    tez.config.Env().get). Unset variables are left out, so the backend's defaults apply."""
    out: dict[str, Any] = {}
    if env("TEZ_LLAMA_LIB"):
        out["lib"] = env("TEZ_LLAMA_LIB")
    for key, var in (("n_ctx", "TEZ_N_CTX"), ("n_batch", "TEZ_N_BATCH"), ("n_gpu_layers", "TEZ_N_GPU_LAYERS")):
        value = env(var)
        if value is None:
            continue
        try:
            out[key] = int(str(value).strip())
        except ValueError:
            raise ValueError(f"{var} must be a whole number, got {value!r}") from None
    return out
