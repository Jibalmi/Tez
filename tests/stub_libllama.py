"""A stand-in for llama.cpp's library (tez.inproc.LlamaApi), for tests without a GPU or a model.

It has the functions InprocBackend calls, with their C calling conventions (ctypes structs, pointers and buffers), over
a toy model: the logits and the hidden state at a position are a hash of the whole token history of that sequence, so
a wrong position, a missed sequence copy or a row read at the wrong index changes the numbers. The KV cache is modelled
as cells shared between sequences (seq_cp shares them, as a unified cache does) with n_ctx cells in all.

It also enforces what the real library does silently or badly: a decode with embeddings on makes every token an output
(and fails the test past n_outputs_max), more outputs than n_outputs_max_per_seq in one sequence fail, a row asked for
at a batch position that is not an output is NULL, positions must continue their sequence, and a hybrid or recurrent
model records any partial llama_memory_seq_rm (which must never happen) and refuses it.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import threading
import time
from typing import Any

import numpy as np

BOS = 1
OFFSET = 3                      # token = byte + OFFSET
N_VOCAB = 256 + OFFSET
GEMMA4_TEMPLATE = "{{ bos_token }}{% for m in messages %}<|turn>{{ m.role }}\n{{ m.content }}<turn|>\n{% endfor %}"


def tokenize(text: str, add_special: bool = True) -> list[int]:
    return ([BOS] if add_special else []) + [b + OFFSET for b in text.encode("utf-8")]


def _seed(history: list[int]) -> int:
    return int.from_bytes(hashlib.blake2b(np.asarray(history, np.int64).tobytes(), digest_size=8).digest(), "little")


def logits_of(history: list[int]) -> np.ndarray:
    """The toy model's next-token logits after `history`: random, with the letters A..Z favoured."""
    z = np.random.default_rng(_seed(history)).standard_normal(N_VOCAB).astype(np.float32) * 2.0
    z[ord("A") + OFFSET: ord("Z") + OFFSET + 1] += 4.0
    z[ord("A") + OFFSET: ord("D") + OFFSET + 1] += 4.0       # a model that answers with an early letter
    return z


def state_of(history: list[int], n_embd: int) -> np.ndarray:
    return np.random.default_rng(_seed(history) ^ 0x5DEECE66D).standard_normal(n_embd).astype(np.float32)


def letters_ref(prompt: str, k: int) -> np.ndarray:
    """Exact letter log-probabilities of a prompt read from scratch."""
    z = logits_of(tokenize(prompt)).astype(np.float64)
    lse = z.max() + np.log(np.exp(z - z.max()).sum())
    return np.array([z[ord(c) + OFFSET] for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:k]]) - lse


def embed_ref(prompt: str, n_embd: int = 16) -> np.ndarray:
    return state_of(tokenize(prompt), n_embd)


class StubApi:
    def __init__(self, *, n_embd: int = 16, registries: tuple = (("CPU", 1), ("CUDA", 1)), hybrid: bool = False,
                 recurrent: bool = False, ftype: str = "Q8_0", n_params: int = 11_907_350_576,
                 chat_template: str | None = GEMMA4_TEMPLATE, preload_errors: dict | None = None, commit: str = "7ab4ee7ba",
                 decode_delay: float = 0.0):
        self.directory = "stub"
        self.n_embd = n_embd
        self.registries = list(registries)
        self.hybrid, self.recurrent = hybrid, recurrent
        self.ftype, self.n_params, self.template = ftype, n_params, chat_template
        self.preload_errors = dict(preload_errors or {})
        self.commit = commit
        self.decode_delay = decode_delay
        self.ctx_params: dict[str, Any] = {}
        self.model_params: dict[str, Any] = {}
        self.decodes: list[dict] = []          # one record per llama_decode
        self.seq_cps: list[tuple[int, int]] = []
        self.seq_rms: list[tuple[int, int, int]] = []
        self.partial_rm_refused: list[tuple[int, int, int]] = []
        self.loads = 0
        self.freed: list[str] = []
        self._seqs: dict[int, list[int]] = {}   # seq -> cell ids in position order
        self._cells: dict[int, int] = {}       # cell id -> token
        self._next_cell = 0
        self._embd_on = False
        self._rows: dict[int, np.ndarray] = {}
        self._embd: dict[int, np.ndarray] = {}
        self._busy = threading.Lock()
        self._keep: list = []

    # ------------------------------------------------------------------ strings and pointers
    def _cstr(self, text: str | None) -> bytes | None:
        return None if text is None else text.encode("utf-8")

    def _fptr(self, arr: np.ndarray) -> Any:
        self._keep.append(arr)
        return arr.ctypes.data_as(C.POINTER(C.c_float))

    # ------------------------------------------------------------------ lifecycle
    def llama_version(self) -> bytes:
        return b"0.4.1-dev"

    def ggml_commit(self) -> bytes:
        return self.commit.encode()

    def ggml_version(self) -> bytes:
        return b"0.24.0"

    def llama_model_default_params(self) -> Any:
        from tez.inproc import ModelParams
        p = ModelParams()
        p.n_gpu_layers = -1
        return p

    def llama_context_default_params(self) -> Any:
        from tez.inproc import ContextParams
        p = ContextParams()
        p.n_ctx, p.n_batch, p.n_ubatch, p.n_seq_max, p.n_outputs_max_per_seq = 512, 2048, 512, 1, 1
        return p

    def llama_model_load_from_file(self, path: bytes, mp: Any) -> int:
        self.model_params = {"path": path.decode("utf-8"), "n_gpu_layers": mp.n_gpu_layers}
        self.loads += 1
        return 1001

    def llama_init_from_model(self, model: int, cp: Any) -> int:
        self.ctx_params = {name: getattr(cp, name) for name, _ in cp._fields_}
        self._seqs, self._cells = {}, {}
        return 2002

    def llama_free(self, ctx: int) -> None:
        self.freed.append("ctx")

    def llama_model_free(self, model: int) -> None:
        self.freed.append("model")

    # ------------------------------------------------------------------ model and vocabulary
    def llama_model_get_vocab(self, model: int) -> int:
        return 3003

    def llama_vocab_n_tokens(self, vocab: int) -> int:
        return N_VOCAB

    def llama_model_n_embd(self, model: int) -> int:
        return self.n_embd

    def llama_model_n_layer(self, model: int) -> int:
        return 4

    def llama_model_n_params(self, model: int) -> int:
        return self.n_params

    def llama_model_size(self, model: int) -> int:
        return 12 * 2 ** 20

    def llama_model_ftype(self, model: int) -> int:
        return 7

    def llama_ftype_name(self, ftype: int) -> bytes:
        return self.ftype.encode()

    def llama_model_desc(self, model: int, buf: Any, size: int) -> int:
        text = f"stub 12B {self.ftype}".encode()
        C.memmove(buf, text, len(text))
        return len(text)

    def llama_model_chat_template(self, model: int, name: Any) -> bytes | None:
        return self._cstr(self.template)

    def llama_model_is_recurrent(self, model: int) -> bool:
        return self.recurrent

    def llama_model_is_hybrid(self, model: int) -> bool:
        return self.hybrid

    def llama_n_ctx(self, ctx: int) -> int:
        return self.ctx_params["n_ctx"]

    def llama_n_batch(self, ctx: int) -> int:
        return self.ctx_params["n_batch"]

    def llama_n_seq_max(self, ctx: int) -> int:
        return self.ctx_params["n_seq_max"]

    def llama_tokenize(self, vocab: int, text: bytes, n_text: int, buf: Any, n_max: int, add_special: bool,
                       parse_special: bool) -> int:
        toks = tokenize(text[:n_text].decode("utf-8"), add_special)
        if len(toks) > n_max:
            return -len(toks)
        for i, t in enumerate(toks):
            buf[i] = t
        return len(toks)

    def llama_token_to_piece(self, vocab: int, token: int, buf: Any, length: int, lstrip: int, special: bool) -> int:
        piece = b"" if token < OFFSET else bytes([token - OFFSET])
        C.memmove(buf, piece, len(piece))
        return len(piece)

    # ------------------------------------------------------------------ decoding
    def llama_decode(self, ctx: int, batch: Any) -> int:
        if not self._busy.acquire(blocking=False):
            raise AssertionError("llama_decode entered from two threads at once")
        try:
            if self.decode_delay:
                time.sleep(self.decode_delay)
            return self._decode(batch)
        finally:
            self._busy.release()

    def _decode(self, batch: Any) -> int:
        cp = self.ctx_params
        n = int(batch.n_tokens)
        if not 0 < n <= cp["n_batch"]:
            return -1
        items = []
        for i in range(n):
            if batch.n_seq_id[i] != 1:
                return -1
            items.append((int(batch.token[i]), int(batch.pos[i]), int(batch.seq_id[i][0]), bool(batch.logits[i])))
        outputs = [i for i, it in enumerate(items) if it[3] or self._embd_on]
        if len(outputs) > cp["n_outputs_max"]:
            raise AssertionError(f"{len(outputs)} outputs in one decode (n_outputs_max {cp['n_outputs_max']}): "
                                 "embeddings on for a multi-token decode?")
        per_seq: dict[int, int] = {}
        for i in outputs:
            per_seq[items[i][2]] = per_seq.get(items[i][2], 0) + 1
        if per_seq and max(per_seq.values()) > cp["n_outputs_max_per_seq"]:
            raise AssertionError("more outputs in one sequence than n_outputs_max_per_seq")
        alive = {c for cells in self._seqs.values() for c in cells}
        if len(alive) + n > cp["n_ctx"]:
            return 1
        lengths = {s: len(cells) for s, cells in self._seqs.items()}
        for _tok, pos, seq, _ in items:
            if not 0 <= seq < cp["n_seq_max"] or pos != lengths.get(seq, 0):
                raise AssertionError(f"token at position {pos} does not continue sequence {seq} "
                                     f"(it holds {lengths.get(seq, 0)} tokens)")
            lengths[seq] = pos + 1
        self._rows, self._embd = {}, {}
        for i, (tok, _pos, seq, _) in enumerate(items):
            self._cells[self._next_cell] = tok
            self._seqs.setdefault(seq, []).append(self._next_cell)
            self._next_cell += 1
            if i in outputs:
                history = [self._cells[c] for c in self._seqs[seq]]
                self._rows[i] = logits_of(history)
                if self._embd_on:
                    self._embd[i] = state_of(history, self.n_embd)
        self.decodes.append({"n": n, "outputs": len(outputs), "embeddings": self._embd_on,
                             "seqs": sorted({it[2] for it in items})})
        return 0

    def llama_get_logits_ith(self, ctx: int, i: int) -> Any:
        row = self._rows.get(i)
        return C.POINTER(C.c_float)() if row is None else self._fptr(row)

    def llama_get_embeddings_ith(self, ctx: int, i: int) -> Any:
        row = self._embd.get(i)
        return C.POINTER(C.c_float)() if row is None else self._fptr(row)

    def llama_set_embeddings(self, ctx: int, on: bool) -> None:
        self._embd_on = bool(on)

    # ------------------------------------------------------------------ memory
    def llama_get_memory(self, ctx: int) -> int:
        return 4004

    def llama_memory_clear(self, mem: int, data: bool) -> None:
        self._seqs = {}

    def llama_memory_seq_rm(self, mem: int, seq: int, p0: int, p1: int) -> bool:
        self.seq_rms.append((seq, p0, p1))
        seqs = list(self._seqs) if seq < 0 else [seq]
        whole = p0 <= 0 and p1 < 0
        if not whole and (self.hybrid or self.recurrent):
            self.partial_rm_refused.append((seq, p0, p1))
            return False
        if p1 >= 0:
            raise NotImplementedError("the stub removes whole sequences or tails only")
        for s in seqs:
            if s in self._seqs:
                self._seqs[s] = self._seqs[s][: max(p0, 0)]
        return True

    def llama_memory_seq_cp(self, mem: int, src: int, dst: int, p0: int, p1: int) -> None:
        assert (p0, p1) == (-1, -1), "the backend copies whole sequences"
        assert not self._seqs.get(dst), f"copy into sequence {dst}, which is not empty"
        self.seq_cps.append((src, dst))
        self._seqs[dst] = list(self._seqs.get(src, []))

    def held(self) -> dict[int, int]:
        """Sequence -> tokens it holds (non-empty sequences only)."""
        return {s: len(c) for s, c in self._seqs.items() if c}

    # ------------------------------------------------------------------ ggml backends
    def ggml_backend_reg_count(self) -> int:
        return len(self.registries)

    def ggml_backend_reg_get(self, i: int) -> int:
        return 100 + i

    def ggml_backend_reg_name(self, reg: int) -> bytes:
        return self.registries[reg - 100][0].encode()

    def ggml_backend_reg_dev_count(self, reg: int) -> int:
        return self.registries[reg - 100][1]

    def ggml_backend_reg_dev_get(self, reg: int, d: int) -> int:
        return (reg - 100) * 100 + d + 10_000

    def _dev(self, dev: int) -> tuple[str, int]:
        return self.registries[(dev - 10_000) // 100][0], (dev - 10_000) % 100

    def ggml_backend_dev_name(self, dev: int) -> bytes:
        name, d = self._dev(dev)
        return f"{name}{d}".encode()

    def ggml_backend_dev_description(self, dev: int) -> bytes:
        name, _ = self._dev(dev)
        return b"Stub GPU 16GB" if name != "CPU" else b"Stub CPU"

    def ggml_backend_dev_memory(self, dev: int, free: Any, total: Any) -> None:
        free._obj.value = 3 * 2 ** 30
        total._obj.value = 16 * 2 ** 30
