"""Model backends.

LlamaCppBackend talks to a llama.cpp `llama-server`:
  letters  POST /completion with one predicted token and the top-n_probs log-probabilities of the next token
           (n_probs 200 by default); the log-probabilities of the bare letters "A".."Z" are the option scores, and a
           letter missing from the top list gets the floor (the smallest listed log-probability minus 2).
  embed    POST /embedding (server started with --embeddings --pooling last): the last-token state.
InprocBackend (tez/inproc.py, "inproc:PATH.gguf") runs llama.cpp inside the Tez process and also reads many prompts
that share a prefix in one batched call (read_many).
FakeBackend is a deterministic stand-in for tests and offline demos (keyword overlap and hashed words).
Every readout also reports its wall time and, when the server sends them, llama.cpp's own timings (prompt_n,
cache_n, prompt_ms, ...), which the engine passes to hooks as per-question traces.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Callable, Mapping

import numpy as np
import requests
from requests.adapters import HTTPAdapter

from .errors import BackendRequestError, BackendUnavailable
from .prompt import HEAD_STATE_FIRST, LETTERS, MAX_LETTERS, TEMPLATES, check_template

log = logging.getLogger("tez")
DEFAULT_BACKEND = "http://127.0.0.1:8080"
DEFAULT_N_PROBS = 200
_FAMILY = {"gemma4": "gemma-4", "qwen3": "qwen3"}


@dataclass
class LetterScores:
    logits: np.ndarray            # (k,) log-probabilities of the letters A.. (letters missing from the top list are floored)
    tokens: int                   # prompt tokens as reported by the backend (or estimated)
    timings: dict | None = None   # llama.cpp's `timings` block for this call (prompt_n, cache_n, prompt_ms, ...)
    ms: float | None = None       # wall time of the call, retries included


@dataclass
class Embedding:
    vector: np.ndarray
    tokens: int
    timings: dict | None = None
    ms: float | None = None


def check_n_probs(n: Any) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 1000:
        raise ValueError(f"n_probs must be an integer from 1 to 1000, got {n!r}")
    return n


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


def letters_from_response(data: Any, k: int) -> np.ndarray | None:
    """Letter log-probabilities from a /completion response; None when it carries no token probabilities
    (the first greedy token was end-of-sequence). Letters absent from the top list get min - 2."""
    if not isinstance(data, dict):
        return None
    cps = data.get("completion_probabilities")
    if not cps:
        return None
    first = cps[0]
    if first.get("top_logprobs") is not None:
        pairs = [(t.get("token"), float(t["logprob"])) for t in first["top_logprobs"]]
    elif first.get("probs") is not None:   # older llama.cpp builds: probabilities instead of log-probabilities
        pairs = [(t.get("tok_str"), math.log(max(float(t["prob"]), 1e-30))) for t in first["probs"]]
    else:
        return None
    if not pairs:
        return None
    best: dict[str, float] = {}
    for tok, lp in pairs:
        best.setdefault(tok, lp)
    floor = min(lp for _, lp in pairs) - 2.0
    return np.array([best.get(LETTERS[i], floor) for i in range(k)], dtype=float)


def prompt_tokens(data: Any, prompt: str) -> int:
    """Total prompt tokens: `tokens_evaluated` (includes cached tokens), else timings, else an estimate."""
    if isinstance(data, dict):
        n = data.get("tokens_evaluated")
        if isinstance(n, int) and not isinstance(n, bool) and n > 0:
            return n
        t = data.get("timings") or {}
        if isinstance(t.get("prompt_n"), int):
            return int(t["prompt_n"]) + int(t.get("cache_n") or 0)
    return estimate_tokens(prompt)


def derive_model_name(path: str | None, alias: str | None, ftype: str | None, n_params: int | None, template: str) -> str:
    """A readable model name: the --alias, else the GGUF file name, else family-size-quant (Ollama blobs have
    hash names), e.g. gemma-4-12b-q8_0."""
    if alias and alias != path:
        return alias
    stem = PurePosixPath((path or "").replace("\\", "/")).name
    stem = re.sub(r"\.gguf$", "", stem, flags=re.I)
    if stem and not re.fullmatch(r"(sha256[-:])?[0-9a-f]{32,}", stem, flags=re.I):
        return stem.lower()
    parts = [_FAMILY.get(template, template)]
    if n_params:
        b = n_params / 1e9
        parts.append(f"{b:.0f}b" if b >= 1.5 else f"{b:.1f}b")
    if ftype:
        parts.append(str(ftype).lower())
    return "-".join(parts)


class Backend:
    """Interface: letters(prompt, k) -> LetterScores, embed(prompt) -> Embedding, model_name(), health().
    known_model_name() answers without any I/O (None when the name would need a call to the server). A backend with
    batched = True also has read_many(prompts, ks, embed) -> [(LetterScores | None, Embedding | None)], which the engine
    uses for requests with several questions (tez/inproc.py)."""

    url: str = ""
    template: str = "gemma4"
    batched: bool = False

    def letters(self, prompt: str, k: int) -> LetterScores:  # pragma: no cover - interface
        raise NotImplementedError

    def embed(self, prompt: str) -> Embedding:  # pragma: no cover - interface
        raise NotImplementedError

    def model_name(self) -> str:
        return "unknown"

    def known_model_name(self) -> str | None:
        return self.model_name()

    def health(self) -> dict:
        return {"ok": True}


class LlamaCppBackend(Backend):
    def __init__(self, url: str, template: str = "gemma4", cache_prompt: bool = True, n_probs: int = DEFAULT_N_PROBS,
                 timeout: float = 300.0, retries: int = 2, model_name: str | None = None):
        self.url = url.rstrip("/")
        self.template = check_template(template)
        self.cache_prompt = cache_prompt
        self.n_probs = check_n_probs(n_probs)
        self.timeout = timeout
        self.retries = retries
        self._model_name = model_name
        self._info: dict | None = None
        self._cache_checked = not cache_prompt
        self._lock = threading.Lock()
        self._session = requests.Session()
        if re.match(r"^https?://(localhost|127\.\d+\.\d+\.\d+|\[::1\])(:\d+)?(/|$)", self.url, flags=re.I):
            self._session.trust_env = False   # a local model server must never be reached through a corporate proxy
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=32)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def __repr__(self) -> str:
        return (f"LlamaCppBackend({self.url!r}, template={self.template!r}, cache_prompt={self.cache_prompt}, "
                f"n_probs={self.n_probs})")

    # ------------------------------------------------------------------ transport
    def _request(self, method: str, path: str, body: dict | None = None, timeout: float | None = None) -> Any:
        url = f"{self.url}{path}"
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = self._session.request(method, url, json=body, timeout=(5.0, timeout or self.timeout))
            except requests.exceptions.ReadTimeout as exc:   # the server accepted the request but is too slow: do not re-send
                raise BackendUnavailable(f"backend {self.url} timed out after {timeout or self.timeout:.0f}s") from exc
            except requests.ConnectionError as exc:          # refused / reset / connect timeout: retry with backoff
                last = exc
                if attempt < self.retries:
                    time.sleep(0.25 * 2 ** attempt)
                    continue
                raise BackendUnavailable(f"backend {self.url} is unreachable ({exc.__class__.__name__})") from exc
            if r.status_code == 503 and attempt < self.retries:   # llama-server answers 503 while loading the model
                time.sleep(0.5 * 2 ** attempt)
                continue
            if r.status_code >= 400:
                msg = _error_message(r)
                if r.status_code == 404:
                    raise BackendUnavailable(f"backend {self.url} has no {path} endpoint (is it a llama.cpp server?)")
                if r.status_code == 501:
                    raise BackendUnavailable(f"backend {self.url} does not support {path}: {msg}")
                if 400 <= r.status_code < 500:
                    raise BackendRequestError(f"backend rejected the request: {msg}")
                raise BackendUnavailable(f"backend {self.url} failed ({r.status_code}): {msg}")
            try:
                return r.json()
            except ValueError as exc:
                raise BackendUnavailable(f"backend {self.url} returned invalid JSON from {path}") from exc
        raise BackendUnavailable(f"backend {self.url} is unavailable ({last})")  # pragma: no cover

    # ------------------------------------------------------------------ model info
    def info(self) -> dict:
        """Model metadata from /props and /v1/models (cached after the first success)."""
        with self._lock:
            if self._info is not None:
                return self._info
        props = self._request("GET", "/props", timeout=10.0) or {}
        meta: dict = {}
        try:
            models = self._request("GET", "/v1/models", timeout=10.0) or {}
            data = models.get("data") or []
            meta = (data[0].get("meta") or {}) if data else {}
        except BackendRequestError:
            pass
        info = {
            "model_path": props.get("model_path"),
            "model_alias": props.get("model_alias"),
            "ftype": props.get("model_ftype") or meta.get("ftype"),
            "n_params": meta.get("n_params"),
            "n_embd": meta.get("n_embd"),
            "n_ctx": meta.get("n_ctx") or (props.get("default_generation_settings") or {}).get("n_ctx"),
            "build": props.get("build_info"),
        }
        with self._lock:
            self._info = info
        return info

    def model_name(self) -> str:
        if self._model_name:
            return self._model_name
        try:
            i = self.info()
        except (BackendUnavailable, BackendRequestError):
            return "unknown"
        self._model_name = derive_model_name(i["model_path"], i["model_alias"], i["ftype"], i["n_params"], self.template)
        return self._model_name

    def known_model_name(self) -> str | None:
        return self._model_name or None

    def health(self) -> dict:
        try:
            r = self._session.get(f"{self.url}/health", timeout=(2.0, 3.0))
            status = (r.json() or {}).get("status", "unknown") if r.headers.get("content-type", "").startswith("application/json") else "unknown"
            ok = r.status_code == 200 and status == "ok"
            return {"ok": ok, "status": status}
        except (requests.RequestException, ValueError) as exc:
            return {"ok": False, "status": "unreachable", "error": exc.__class__.__name__}

    def _check_cache_prompt(self) -> None:
        """llama.cpp b11100 crashes on partial prefix reuse with hybrid Qwen3.5 GGUFs: switch caching off for them."""
        self._cache_checked = True
        try:
            i = self.info()
        except (BackendUnavailable, BackendRequestError):
            self._cache_checked = False
            return
        names = " ".join(str(x) for x in (i.get("model_path"), i.get("model_alias"), self._model_name) if x)
        if re.search(r"qwen[-_ ]?3[._]?5", names, flags=re.I):
            self.cache_prompt = False
            log.warning("prompt caching switched off for %s (hybrid Qwen3.5 GGUFs crash llama.cpp on partial prefix reuse)", names)

    # ------------------------------------------------------------------ readouts
    def letters(self, prompt: str, k: int) -> LetterScores:
        if not 1 <= k <= MAX_LETTERS:
            raise ValueError(f"a letter readout reads 1 to {MAX_LETTERS} options, got {k}")
        if not self._cache_checked:
            self._check_cache_prompt()
        t0 = time.perf_counter()
        body = {"prompt": prompt, "n_predict": 1, "n_probs": self.n_probs, "temperature": 0, "samplers": [],
                "cache_prompt": self.cache_prompt}
        data = self._request("POST", "/completion", body)
        z = letters_from_response(data, k)
        if z is None:   # the greedy first token was end-of-sequence (seen on depth-pruned GGUFs): suppress it and re-read
            data = self._request("POST", "/completion", dict(body, ignore_eos=True))
            z = letters_from_response(data, k)
            if z is None:
                raise BackendUnavailable(f"backend {self.url} returned no token probabilities (n_probs unsupported?)")
        return LetterScores(z, prompt_tokens(data, prompt), timings=_timings(data), ms=(time.perf_counter() - t0) * 1000.0)

    def embed(self, prompt: str) -> Embedding:
        t0 = time.perf_counter()
        data = self._request("POST", "/embedding", {"content": prompt, "embd_normalize": -1})
        d = data[0] if isinstance(data, list) and data else data
        e = d.get("embedding") if isinstance(d, dict) else None
        if not e:
            raise BackendUnavailable(f"backend {self.url} returned no embedding (start llama-server with --embeddings --pooling last)")
        vec = e[-1] if isinstance(e[0], list) else e
        return Embedding(np.asarray(vec, dtype=np.float32), estimate_tokens(prompt), timings=_timings(d),
                         ms=(time.perf_counter() - t0) * 1000.0)


def _timings(data: Any) -> dict | None:
    t = data.get("timings") if isinstance(data, dict) else None
    return dict(t) if isinstance(t, dict) else None


def _error_message(r: requests.Response) -> str:
    try:
        d = r.json()
        err = d.get("error") if isinstance(d, dict) else None
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str):
            return err
    except ValueError:
        pass
    return (r.text or r.reason or str(r.status_code)).strip()[:300]


# ---------------------------------------------------------------------------------------------- fake
_STOP = {"a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are", "be", "it", "this", "that", "my",
         "i", "me", "we", "you", "your", "with", "has", "have", "been", "was", "no", "yes", "not", "none", "these",
         "above", "options", "option", "fits", "level", "statement", "does", "hold", "holds", "what", "how", "which"}


def _words(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _STOP or len(w) < 2:
            continue
        w = w[:-1] if len(w) > 3 and w.endswith("s") else w
        out.append(w[:6])
    return out


def _h(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def _split_prompt(prompt: str, template: str) -> tuple[list[tuple[str, str]], str]:
    """(options, state) of the final question in a Tez prompt, in either layout."""
    body = prompt
    pre, post = TEMPLATES.get(template, ("", ""))
    if post and body.endswith(post):
        body = body[: -len(post)]
    cut = body.rfind("\nInput:\n")
    if cut >= 0 and HEAD_STATE_FIRST in body[: cut + 1]:     # state_first: Input, then the question and its options
        rest = body[cut + len("\nInput:\n"):]
        q = rest.rfind("\n\nQuestion (")
        state, head = (rest[:q], rest[q:]) if q >= 0 else (rest, "")
    else:
        state = body[cut + len("\nInput:\n"):] if cut >= 0 else body
        head = body[:cut] if cut >= 0 else ""
    opts: list[tuple[str, str]] = []
    start = head.rfind("Options:\n")
    if start >= 0:
        for line in head[start + len("Options:\n"):].split("\n"):
            m = re.match(r"^([A-Z]+)\. (.*)$", line)
            if not m:
                break
            key, _, desc = m.group(2).partition(": ")
            opts.append((key, desc))
    return opts, state


def common_prefix(a: str, b: str) -> int:
    """Length of the longest common prefix of two strings. A binary search over slices, so the characters are compared
    in C (linear overall) rather than one by one in Python: a /v1/plan of 64 questions over a 50,000-character state
    compares hundreds of prompts this way."""
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n                      # a[:lo] == b[:lo] and a[:hi] != b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid
    return lo


class FakeBackend(Backend):
    """Deterministic offline backend. It is not a model:
      letters  options sharing words with the state score higher (plus a tiny hash-based tie-breaker);
      embed    a hashed bag of the state's words, so states that share vocabulary share directions.
    Pass letters_fn(prompt, k) -> k scores or embed_fn(prompt) -> vector to script it; fail=True makes every
    call raise BackendUnavailable (for testing 503s). `calls` counts backend calls. Letter readouts report
    llama.cpp-style timings from a simulated one-slot prompt cache: `cache_n` is the prefix shared with the previous
    letters prompt (in estimated tokens), `prompt_n` the rest."""

    url = "fake"

    def __init__(self, template: str = "gemma4", dim: int = 64, letters_fn: Callable | None = None,
                 embed_fn: Callable | None = None, fail: bool = False, model: str = "fake", sharpness: float = 3.0):
        self.template = check_template(template)
        self.dim = dim
        self.letters_fn = letters_fn
        self.embed_fn = embed_fn
        self.fail = fail
        self.model = model
        self.sharpness = sharpness
        self.calls = {"letters": 0, "embed": 0}
        self.prompts: list[str] = []
        self._last_prompt = ""
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"FakeBackend(template={self.template!r})"

    def model_name(self) -> str:
        return self.model

    def health(self) -> dict:
        return {"ok": not self.fail, "status": "fake" if not self.fail else "failing"}

    def _cache_timings(self, prompt: str) -> dict:
        with self._lock:
            shared = common_prefix(self._last_prompt, prompt)
            self._last_prompt = prompt
        total = estimate_tokens(prompt)
        cache_n = min(total - 1, shared // 4) if shared else 0
        return {"prompt_n": total - cache_n, "cache_n": cache_n, "prompt_ms": 0.0, "predicted_n": 1}

    def letters(self, prompt: str, k: int) -> LetterScores:
        if self.fail:
            raise BackendUnavailable("fake backend is set to fail")
        if not 1 <= k <= MAX_LETTERS:
            raise ValueError(f"a letter readout reads 1 to {MAX_LETTERS} options, got {k}")
        t0 = time.perf_counter()
        self.calls["letters"] += 1
        self.prompts.append(prompt)
        if self.letters_fn is not None:
            s = np.asarray(self.letters_fn(prompt, k), dtype=float)
        else:
            opts, state = _split_prompt(prompt, self.template)
            sw = set(_words(state))
            s = np.zeros(k)
            for i in range(k):
                key, desc = opts[i] if i < len(opts) else ("", "")
                s[i] = self.sharpness * len(set(_words(f"{key} {desc}")) & sw) + (_h(f"{i}|{prompt}") % 1000) / 1e5
        s = s - s.max()
        logp = s - np.log(np.exp(s).sum())
        return LetterScores(logp, estimate_tokens(prompt), timings=self._cache_timings(prompt),
                            ms=(time.perf_counter() - t0) * 1000.0)

    def embed(self, prompt: str) -> Embedding:
        if self.fail:
            raise BackendUnavailable("fake backend is set to fail")
        self.calls["embed"] += 1
        self.prompts.append(prompt)
        if self.embed_fn is not None:
            return Embedding(np.asarray(self.embed_fn(prompt), dtype=np.float32), estimate_tokens(prompt))
        _, state = _split_prompt(prompt, self.template)
        v = np.zeros(self.dim)
        for w in _words(state):
            h = _h(w)
            v[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        rng = np.random.default_rng(_h(prompt) % (2 ** 32))
        v = v + 0.05 * rng.standard_normal(self.dim)
        return Embedding(v.astype(np.float32), estimate_tokens(prompt))


def make_backend(spec: Any, template: str = "gemma4", cache_prompt: bool = True, model_name: str | None = None,
                 n_probs: int = DEFAULT_N_PROBS, *, inproc: Mapping[str, Any] | None = None) -> Backend:
    """A Backend from a URL, 'inproc:PATH.gguf', 'fake', or an existing Backend instance. n_probs: how many next-token
    log-probabilities a letter readout asks llama-server for (letters outside that list get the floor). inproc: the
    in-process backend's settings (lib, n_ctx, n_batch, n_ubatch, n_seq_max, n_gpu_layers; tez.inproc.InprocBackend).
    An in-process backend loads nothing until it is first used."""
    if isinstance(spec, Backend):
        return spec
    if spec is None:
        spec = DEFAULT_BACKEND
    if not isinstance(spec, str):
        raise ValueError(f"backend must be a URL, 'inproc:PATH.gguf', 'fake' or a Backend instance, got {spec!r}")
    if spec == "fake" or spec.startswith("fake:"):
        return FakeBackend(template=template, model=model_name or "fake")
    if spec.startswith("inproc:"):
        from .inproc import OPTION_NAMES, InprocBackend, parse_spec
        opts = {k: v for k, v in (inproc or {}).items() if v is not None}
        unknown = sorted(set(opts) - set(OPTION_NAMES))
        if unknown:
            raise ValueError(f"unknown in-process setting(s) {', '.join(unknown)}; use {', '.join(OPTION_NAMES)}")
        return InprocBackend(parse_spec(spec), template=template, cache_prompt=cache_prompt, model_name=model_name, **opts)
    if not re.match(r"^https?://", spec):
        spec = "http://" + spec
    return LlamaCppBackend(spec, template=template, cache_prompt=cache_prompt, model_name=model_name, n_probs=n_probs)
