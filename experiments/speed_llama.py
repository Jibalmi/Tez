"""In-process llama.cpp for the speed experiments: a ctypes binding to the b11100 release's llama.dll.

No HTTP, no JSON, no top-k log-probabilities: the letter readout reads the logits of the option-letter tokens
straight from the output buffer (exact, nothing floored), and the batch API exposes what llama-server hides:
  * several sequences in ONE llama_decode call, each with its own outputs (many questions in one forward pass),
  * llama_memory_seq_cp: a state evaluated once is shared by other sequences (unified KV: no copy for attention
    layers; hybrid/recurrent models copy the small recurrent state),
  * exact-prefix extension without rollback, which hybrid Qwen3.5 GGUFs need (their recurrent layers cannot be
    rolled back, so llama-server b11100 must run them with prompt caching off).

Struct layouts follow include/llama.h at tag b11100 (the build the release reports: b11100-7ab4ee7ba). Nothing here
changes the model or the numerics: it is the same library llama-server links.

Self-test (compares the in-process letter readout with llama-server's on the same prompt; the GPU must hold both
models, so run it with a small model or with the server stopped and --no-server):
  python experiments/speed_llama.py --model tools/models/Qwen3.5-4B-Q8_0-L24.gguf --template qwen3 --no-server
"""
from __future__ import annotations

import ctypes as C
import os
import time
from typing import Iterable, Sequence

import numpy as np

DLL_DIR = r"C:\temp\llamacpp"
os.add_dll_directory(DLL_DIR)
_ggml = C.CDLL(os.path.join(DLL_DIR, "ggml.dll"))
_lib = C.CDLL(os.path.join(DLL_DIR, "llama.dll"))

POOLING = {"unspecified": -1, "none": 0, "mean": 1, "cls": 2, "last": 3, "rank": 4}


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


_LOGCB = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_void_p)


def _sig(name, res, *args):
    f = getattr(_lib, name)
    f.restype = res
    f.argtypes = list(args)
    return f


vp, i32, u32, b_ = C.c_void_p, C.c_int32, C.c_uint32, C.c_bool
_ggml.ggml_backend_load_all_from_path.argtypes = [C.c_char_p]
_ggml.ggml_backend_load_all_from_path.restype = None
llama_backend_init = _sig("llama_backend_init", None)
llama_log_set = _sig("llama_log_set", None, _LOGCB, vp)
llama_model_default_params = _sig("llama_model_default_params", ModelParams)
llama_context_default_params = _sig("llama_context_default_params", ContextParams)
llama_model_load_from_file = _sig("llama_model_load_from_file", vp, C.c_char_p, ModelParams)
llama_model_free = _sig("llama_model_free", None, vp)
llama_init_from_model = _sig("llama_init_from_model", vp, vp, ContextParams)
llama_free = _sig("llama_free", None, vp)
llama_model_get_vocab = _sig("llama_model_get_vocab", vp, vp)
llama_vocab_n_tokens = _sig("llama_vocab_n_tokens", i32, vp)
llama_model_n_embd = _sig("llama_model_n_embd", i32, vp)
llama_model_n_layer = _sig("llama_model_n_layer", i32, vp)
llama_n_ctx = _sig("llama_n_ctx", u32, vp)
llama_n_batch = _sig("llama_n_batch", u32, vp)
llama_n_seq_max = _sig("llama_n_seq_max", u32, vp)
llama_tokenize = _sig("llama_tokenize", i32, vp, C.c_char_p, i32, C.POINTER(i32), i32, b_, b_)
llama_token_to_piece = _sig("llama_token_to_piece", i32, vp, i32, C.c_char_p, i32, i32, b_)
llama_decode = _sig("llama_decode", i32, vp, Batch)
llama_get_logits_ith = _sig("llama_get_logits_ith", C.POINTER(C.c_float), vp, i32)
llama_get_embeddings_ith = _sig("llama_get_embeddings_ith", C.POINTER(C.c_float), vp, i32)
llama_get_embeddings_seq = _sig("llama_get_embeddings_seq", C.POINTER(C.c_float), vp, i32)
llama_get_memory = _sig("llama_get_memory", vp, vp)
llama_memory_clear = _sig("llama_memory_clear", None, vp, b_)
llama_memory_seq_rm = _sig("llama_memory_seq_rm", b_, vp, i32, i32, i32)
llama_memory_seq_cp = _sig("llama_memory_seq_cp", None, vp, i32, i32, i32, i32)
llama_memory_seq_pos_max = _sig("llama_memory_seq_pos_max", i32, vp, i32)
llama_memory_seq_pos_min = _sig("llama_memory_seq_pos_min", i32, vp, i32)
llama_set_embeddings = _sig("llama_set_embeddings", None, vp, b_)
llama_synchronize = _sig("llama_synchronize", None, vp)

_quiet = _LOGCB(lambda level, text, ud: None)
_initialised = False


def init(verbose: bool = False) -> None:
    global _initialised
    if _initialised:
        return
    if not verbose:
        llama_log_set(_quiet, None)
    # ggml loads backends with a plain LoadLibraryW, which (under Python's restricted DLL search) cannot resolve
    # ggml-cuda.dll's own dependencies (cudart/cublas next to it): error 126, and the model silently runs on the CPU.
    # Loading it first through ctypes (LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR) makes the later load find the module.
    global _cuda
    _cuda = C.CDLL(os.path.join(DLL_DIR, "ggml-cuda.dll"))
    _ggml.ggml_backend_load_all_from_path(DLL_DIR.encode())
    _ggml.ggml_backend_dev_count.restype = C.c_size_t
    if _ggml.ggml_backend_dev_count() < 2:
        raise RuntimeError("the CUDA backend did not load: llama.dll would run on the CPU")
    llama_backend_init()
    _initialised = True


class Llama:
    """One model + one context. Batches are numpy-backed and reused (capacity = n_batch tokens)."""

    def __init__(self, path: str, n_ctx: int = 4096, n_batch: int = 2048, n_ubatch: int = 512, n_seq_max: int = 1,
                 embeddings: bool = False, pooling: str = "none", swa_full: bool = True, kv_unified: bool = True,
                 flash_attn: int = -1, n_gpu_layers: int = 99, n_threads: int = 8, verbose: bool = False,
                 n_outputs_max: int = 64, n_outputs_max_per_seq: int = 0):
        init(verbose)
        mp = llama_model_default_params()
        mp.n_gpu_layers = n_gpu_layers
        t = time.perf_counter()
        self.model = llama_model_load_from_file(path.encode(), mp)
        if not self.model:
            raise RuntimeError(f"cannot load {path}")
        cp = llama_context_default_params()
        cp.n_ctx, cp.n_batch, cp.n_ubatch, cp.n_seq_max = n_ctx, n_batch, n_ubatch, n_seq_max
        cp.n_threads = cp.n_threads_batch = n_threads
        cp.n_outputs_max = n_outputs_max               # the logits buffer is n_outputs_max x n_vocab floats
        cp.n_outputs_max_per_seq = n_outputs_max_per_seq
        cp.embeddings = embeddings
        cp.pooling_type = POOLING[pooling]
        cp.swa_full = swa_full
        cp.kv_unified = kv_unified
        cp.flash_attn_type = flash_attn
        cp.no_perf = True
        self.ctx = llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise RuntimeError("cannot create the context")
        self.load_s = time.perf_counter() - t
        self.path = path
        self.vocab = llama_model_get_vocab(self.model)
        self.n_vocab = llama_vocab_n_tokens(self.vocab)
        self.n_embd = llama_model_n_embd(self.model)
        self.n_layer = llama_model_n_layer(self.model)
        self.mem = llama_get_memory(self.ctx)
        self.cap = int(llama_n_batch(self.ctx))
        self.n_seq_max = int(llama_n_seq_max(self.ctx))
        self.embeddings = embeddings
        self._embd_on = embeddings
        S = max(1, self.n_seq_max)
        self._tok = np.zeros(self.cap, np.int32)
        self._pos = np.zeros(self.cap, np.int32)
        self._nseq = np.ones(self.cap, np.int32)
        self._seq = np.zeros((self.cap, S), np.int32)
        self._out = np.zeros(self.cap, np.int8)
        P32 = C.POINTER(C.c_int32)
        base = self._seq.ctypes.data
        self._seqptrs = (P32 * self.cap)(*[C.cast(base + i * S * 4, P32) for i in range(self.cap)])
        self.params = dict(n_ctx=n_ctx, n_batch=n_batch, n_ubatch=n_ubatch, n_seq_max=n_seq_max, embeddings=embeddings,
                           pooling=pooling, swa_full=swa_full, kv_unified=kv_unified, flash_attn=flash_attn,
                           n_gpu_layers=n_gpu_layers, n_outputs_max=n_outputs_max,
                           n_outputs_max_per_seq=n_outputs_max_per_seq)

    def close(self) -> None:
        if getattr(self, "ctx", None):
            llama_free(self.ctx)
            self.ctx = None
        if getattr(self, "model", None):
            llama_model_free(self.model)
            self.model = None

    # ------------------------------------------------------------------ tokens
    def tokenize(self, text: str, add_special: bool = False, parse_special: bool = True) -> list[int]:
        b = text.encode("utf-8")
        n = len(b) + 16
        buf = (C.c_int32 * n)()
        k = llama_tokenize(self.vocab, b, len(b), buf, n, add_special, parse_special)
        if k < 0:
            buf = (C.c_int32 * (-k))()
            k = llama_tokenize(self.vocab, b, len(b), buf, -k, add_special, parse_special)
        return list(buf[:k])

    def piece(self, tok: int) -> str:
        buf = C.create_string_buffer(256)
        k = llama_token_to_piece(self.vocab, tok, buf, 256, 0, True)
        return buf.raw[:max(k, 0)].decode("utf-8", "replace")

    def letter_ids(self, letters: Iterable[str]) -> list[int]:
        """Token ids of the bare letters ("A", not " A"), as the HTTP readout matches them."""
        out = []
        for L in letters:
            ids = self.tokenize(L, False, False)
            if len(ids) != 1 or self.piece(ids[0]) != L:
                raise ValueError(f"letter {L!r} is not a single token: {ids}")
            out.append(ids[0])
        return out

    # ------------------------------------------------------------------ decoding
    def decode(self, tokens: Sequence[int], pos: Sequence[int], seqs: Sequence[Sequence[int]] | Sequence[int],
               outputs: Sequence[bool]) -> list[int]:
        """One llama_decode over len(tokens) <= n_batch tokens. seqs[i] is a seq id or a tuple of seq ids.
        Returns the batch positions of the tokens flagged in `outputs` (llama_get_logits_ith / _embeddings_ith index
        by batch position, not by output ordinal)."""
        n = len(tokens)
        if n > self.cap:
            raise ValueError(f"batch of {n} tokens exceeds n_batch {self.cap}")
        self._tok[:n] = tokens
        self._pos[:n] = pos
        multi = False
        if len(seqs) and not isinstance(seqs[0], (int, np.integer)):
            multi = True
        if multi:
            for i, s in enumerate(seqs):
                self._nseq[i] = len(s)
                self._seq[i, : len(s)] = s
        else:
            self._nseq[:n] = 1
            self._seq[:n, 0] = seqs
        self._out[:n] = np.asarray(outputs, np.int8)
        b = Batch(n, self._tok.ctypes.data_as(C.POINTER(C.c_int32)), None, self._pos.ctypes.data_as(C.POINTER(C.c_int32)),
                  self._nseq.ctypes.data_as(C.POINTER(C.c_int32)), C.cast(self._seqptrs, C.POINTER(C.POINTER(C.c_int32))),
                  self._out.ctypes.data_as(C.POINTER(C.c_int8)))
        rc = llama_decode(self.ctx, b)
        if rc != 0:
            raise RuntimeError(f"llama_decode returned {rc}")
        return [int(i) for i in np.flatnonzero(self._out[:n])]

    def logits_at(self, pos: int, ids: Sequence[int] | None = None) -> np.ndarray:
        """Logits of the output at batch position `pos` of the last decode (-1 = the last output)."""
        p = llama_get_logits_ith(self.ctx, pos)
        if not p:
            raise RuntimeError("no logits for this output")
        if ids is None:
            return np.ctypeslib.as_array(p, shape=(self.n_vocab,)).copy()
        return np.array([p[i] for i in ids], dtype=np.float64)

    def embd_at(self, pos: int) -> np.ndarray:
        """Hidden state (after the output norm) at batch position `pos` of the last decode (-1 = the last output)."""
        p = llama_get_embeddings_ith(self.ctx, pos)
        if not p:
            raise RuntimeError("no embeddings for this output (context created with embeddings=True?)")
        return np.ctypeslib.as_array(p, shape=(self.n_embd,)).astype(np.float32).copy()

    def set_embeddings(self, on: bool) -> None:
        """libllama marks EVERY token of a decode as an output while embeddings are on (and computes full-vocab logits
        for each), so embeddings are switched on only for the decodes whose states are read."""
        if on != getattr(self, "_embd_on", None):
            llama_set_embeddings(self.ctx, on)
            self._embd_on = on

    def last_state(self, tokens: Sequence[int], seq: int, start: int) -> np.ndarray:
        """Hidden state of the last token: the body with embeddings off, then the last token alone with them on."""
        if len(tokens) > 1:
            self.set_embeddings(False)
            self.eval_seq(tokens[:-1], seq, start, want_last=False)
        self.set_embeddings(True)
        p = self.decode([tokens[-1]], [start + len(tokens) - 1], [seq], [True])[-1]
        v = self.embd_at(p)
        self.set_embeddings(False)
        return v

    def eval_seq(self, tokens: Sequence[int], seq: int, start: int, want_last: bool = True, want_all: bool = False) -> list[int]:
        """Evaluate a token run on one sequence in chunks of n_batch; outputs only at the end (or everywhere)."""
        outs: list[int] = []
        n = len(tokens)
        for s in range(0, n, self.cap):
            chunk = tokens[s: s + self.cap]
            o = [want_all or (want_last and s + j == n - 1) for j in range(len(chunk))]
            outs = self.decode(chunk, list(range(start + s, start + s + len(chunk))), [seq] * len(chunk), o)
        return outs

    # ------------------------------------------------------------------ memory
    def clear(self) -> None:
        llama_memory_clear(self.mem, False)      # metadata only: freed cells are overwritten, no need to zero the buffers

    def seq_rm(self, seq: int, p0: int = -1, p1: int = -1) -> bool:
        return bool(llama_memory_seq_rm(self.mem, seq, p0, p1))

    def seq_cp(self, src: int, dst: int, p0: int = -1, p1: int = -1) -> None:
        llama_memory_seq_cp(self.mem, src, dst, p0, p1)

    def pos_max(self, seq: int) -> int:
        return int(llama_memory_seq_pos_max(self.mem, seq))

    def sync(self) -> None:
        llama_synchronize(self.ctx)


def _selftest():
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import speed_common as sc
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--template", default="qwen3")
    ap.add_argument("--no-server", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:8095")
    a = ap.parse_args()
    cp = llama_context_default_params()
    print({f: getattr(cp, f) for f, _ in ContextParams._fields_ if f not in ("cb_eval", "cb_eval_user_data", "abort_callback", "abort_callback_data", "samplers", "ctx_other")})
    m = Llama(a.model, n_ctx=4096, n_batch=2048, n_seq_max=4)
    print("loaded", a.model, f"{m.load_s:.1f}s", "vocab", m.n_vocab, "embd", m.n_embd, "layers", m.n_layer, "n_batch", m.cap)
    cases = sc.typed_decisions("test")[:5]
    ids = m.letter_ids(sc.LETTERS[:5])
    for c in cases:
        toks = m.tokenize(sc.prompt_today(c, a.template), add_special=True)
        m.clear()
        t = time.perf_counter()
        outs = m.eval_seq(toks, 0, 0)
        z = m.logits_at(outs[-1], ids[: len(c["options"])])
        ms = (time.perf_counter() - t) * 1000
        line = f"{c['id']:40s} n={len(toks)} {ms:6.1f} ms  pred {int(np.argmax(z))} gold {c['gold']}  z {np.round(z - z.max(), 2)}"
        if not a.no_server:
            r = sc.score_http(a.url, sc.prompt_today(c, a.template), len(c["options"]), cache=False)
            line += f" | server z {np.round(r['z'] - r['z'].max(), 2)}"
        print(line)
    m.close()


if __name__ == "__main__":
    _selftest()
