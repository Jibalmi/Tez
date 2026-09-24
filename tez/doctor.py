"""`tez doctor --backend URL`: check a llama-server for what Tez needs, and print the fix for anything that is off.

Checks, in order (each ok / warn / fail / info, with a concrete fix):
  health      GET /health answers ok (503 while the model loads)
  model       GET /props: the model file, build, context size and slots
  template    the model's chat template matches --template (Gemma 4: <|turn>, Qwen3: <|im_start|>)
  n_probs     /completion returns the top-n_probs next-token log-probabilities, the option letters among them
  cache       the same prompt twice: the second is served from the prompt cache (timings.prompt_n)
  prefix      two questions about one state, state first: the second reuses the state (Gemma needs --swa-full)
  embeddings  /embedding works (fitted probes need --embeddings --pooling last, or a separate --embed-backend)
  slots       one slot (-np 1), so consecutive questions share one cache
  cache-ram   the host-memory prompt cache (--cache-ram, 8 GiB by default) copies a slot to RAM whenever a request
              keeps less than half of it, which a new state does: --cache-ram 0 is recommended
The checks send a handful of tiny prompts; nothing is changed on the server.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import requests

from .backends import DEFAULT_N_PROBS, derive_model_name, letters_from_response
from .prompt import build_prompt
from .schema import parse_question

_LOCAL = re.compile(r"^https?://(localhost|127\.\d+\.\d+\.\d+|\[::1\])(:\d+)?(/|$)", re.I)
STATE = ("Subject: payout failing\n\nHi, my payouts have failed for three days and I am losing sales. Please help, "
         "this is urgent. Order 4411 was charged twice as well.")
Q1 = parse_question("topic", {"type": "choice", "instructions": "What is the message about?",
                              "criteria": {"billing": "Payments, payouts, invoices", "technical": "Something is broken",
                                           "sales": "Pricing and plans"}})
Q2 = parse_question("urgent", {"type": "noul", "instructions": "Does the writer need a reply today?"})


@dataclass
class Check:
    name: str
    status: str               # ok | warn | fail | info
    detail: str
    fix: str | None = None


def detect_template(chat_template: str | None) -> str | None:
    """The Tez template family a GGUF chat template belongs to, or a tag for one Tez cannot drive."""
    t = chat_template or ""
    if "<|turn>" in t:
        return "gemma4"
    if "<start_of_turn>" in t:
        return "gemma3"
    if "<|im_start|>" in t:
        return "qwen3"
    return None


class Doctor:
    def __init__(self, url: str, template: str = "gemma4", n_probs: int = DEFAULT_N_PROBS, timeout: float = 60.0,
                 session: requests.Session | None = None):
        self.url = url.rstrip("/") if re.match(r"^https?://", url) else "http://" + url.rstrip("/")
        self.template = template
        self.n_probs = n_probs
        self.timeout = timeout
        self.session = session or requests.Session()
        if session is None and _LOCAL.match(self.url):
            self.session.trust_env = False
        self.checks: list[Check] = []
        self.props: dict = {}

    def add(self, name: str, status: str, detail: str, fix: str | None = None) -> Check:
        c = Check(name, status, detail, fix)
        self.checks.append(c)
        return c

    def get(self, path: str) -> requests.Response:
        return self.session.get(self.url + path, timeout=(5.0, self.timeout))

    def post(self, path: str, body: dict) -> requests.Response:
        return self.session.post(self.url + path, json=body, timeout=(5.0, self.timeout))

    def completion(self, prompt: str) -> dict:
        r = self.post("/completion", {"prompt": prompt, "n_predict": 1, "n_probs": self.n_probs, "temperature": 0,
                                      "samplers": [], "cache_prompt": True})
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------------------------------ checks
    def run(self) -> list[Check]:
        if not self.check_health():
            return self.checks
        self.check_props()
        if self.check_n_probs():
            self.check_cache()
        self.check_embeddings()
        self.check_slots()
        self.add("cache-ram", "info", "llama.cpp's host-memory prompt cache (--cache-ram, 8 GiB by default) first copies "
                 "a slot to RAM whenever a request keeps less than half of its tokens, which every new state does",
                 "start llama-server with --cache-ram 0")
        return self.checks

    def check_health(self) -> bool:
        try:
            r = self.get("/health")
        except requests.RequestException as exc:
            self.add("health", "fail", f"{self.url} is unreachable ({exc.__class__.__name__})",
                     "start llama-server, e.g. llama-server -m <model.gguf> -ngl 99 -c 4096 -np 1 --swa-full --cache-ram 0 "
                     "--port 8080 (and check --backend)")
            return False
        status = None
        try:
            status = (r.json() or {}).get("status")
        except ValueError:
            pass
        if r.status_code == 503:
            self.add("health", "warn", "the server is up but still loading the model (503)", "wait and run tez doctor again")
            return False
        if r.status_code != 200:
            self.add("health", "fail", f"/health answered {r.status_code}; is {self.url} a llama.cpp server?",
                     "point --backend at llama-server (default port 8080)")
            return False
        self.add("health", "ok", f"/health: {status or 'ok'}")
        return True

    def check_props(self) -> None:
        try:
            r = self.get("/props")
            r.raise_for_status()
            self.props = r.json() or {}
        except (requests.RequestException, ValueError) as exc:
            self.add("model", "warn", f"/props is not available ({exc.__class__.__name__}); the model cannot be checked")
            return
        p = self.props
        gen = p.get("default_generation_settings") or {}
        n_ctx = gen.get("n_ctx") or p.get("n_ctx")
        name = derive_model_name(p.get("model_path"), p.get("model_alias"), p.get("model_ftype"), None, self.template)
        slots = p.get("total_slots")
        detail = f"{name}" + (f", build {p['build_info']}" if p.get("build_info") else "") + \
                 (f", context {n_ctx}" if n_ctx else "") + (f", {slots} slot(s)" if slots else "")
        self.add("model", "ok", detail)
        if isinstance(n_ctx, int) and n_ctx < 2048:
            self.add("context", "warn", f"a context of {n_ctx} tokens is short for a state plus a question",
                     "start llama-server with -c 4096")
        family = detect_template(p.get("chat_template"))
        if family is None:
            self.add("template", "info", "the model's chat template was not recognised; check --template by hand")
        elif family == "gemma3":
            self.add("template", "fail", "the model uses Gemma 3's chat template (<start_of_turn>), which Tez does not drive",
                     "serve Gemma 4 (template gemma4) or a Qwen3 model (template qwen3)")
        elif family != self.template:
            self.add("template", "fail", f"the model's chat template is {family}'s, --template is {self.template}",
                     f"tez serve --template {family}")
        else:
            self.add("template", "ok", f"chat template matches --template {self.template}")

    def check_n_probs(self) -> bool:
        prompt = build_prompt(Q1, STATE, self.template, layout="state_first")
        try:
            data = self.completion(prompt)
        except (requests.RequestException, ValueError) as exc:
            self.add("n_probs", "fail", f"/completion failed ({exc.__class__.__name__})",
                     "check that --backend is llama-server's native API (not an OpenAI-compatible proxy)")
            return False
        cps = data.get("completion_probabilities") or [{}]
        tops = cps[0].get("top_logprobs") if cps else None
        if tops is None:
            tops = cps[0].get("probs") if cps else None
        z = letters_from_response(data, 3)
        if not tops or z is None:
            self.add("n_probs", "fail", "the server returns no next-token probabilities",
                     "use llama-server's /completion (Ollama's OpenAI endpoint drops log-probabilities)")
            return False
        listed = {t.get("token") or t.get("tok_str") for t in tops}
        missing = [c for c in "ABC" if c not in listed]
        if len(tops) < self.n_probs:
            self.add("n_probs", "warn", f"asked for {self.n_probs} log-probabilities, got {len(tops)}; letters outside them "
                     "get the floor", "use llama-server (Ollama caps top log-probabilities at 20) or tez serve --n-probs "
                     f"{len(tops)}")
        elif missing:
            self.add("n_probs", "warn", f"the option letters {', '.join(missing)} are not among the top {len(tops)}",
                     "check --template: the model may not be answering with a letter")
        else:
            self.add("n_probs", "ok", f"top {len(tops)} log-probabilities, the option letters among them")
        self._first = (prompt, data)
        return True

    def check_cache(self) -> None:
        prompt, first = self._first
        t0 = (first.get("timings") or {})
        try:
            again = self.completion(prompt)
            other = self.completion(build_prompt(Q2, STATE, self.template, layout="state_first"))
        except (requests.RequestException, ValueError) as exc:
            self.add("cache", "warn", f"could not repeat a prompt ({exc.__class__.__name__})")
            return
        t1, t2 = again.get("timings") or {}, other.get("timings") or {}
        if "prompt_n" not in t1:
            self.add("cache", "info", "the server reports no timings; prompt caching cannot be checked")
            return
        total = int(t0.get("prompt_n") or 0) + int(t0.get("cache_n") or 0) or int(first.get("tokens_evaluated") or 0)
        gemma = "gemma" in " ".join(str(self.props.get(k) or "") for k in ("model_path", "model_alias")).lower() or \
            self.template == "gemma4"
        if int(t1.get("prompt_n") or 0) > max(4, total // 4):
            self.add("cache", "fail", f"the same prompt was evaluated again ({t1.get('prompt_n')} of {total} tokens)",
                     "do not start llama-server with --no-cache-prompt, and use -np 1"
                     + ("; Gemma also needs --swa-full" if gemma else ""))
            return
        self.add("cache", "ok", f"a repeated prompt is served from the cache ({t1.get('prompt_n')} new tokens)")
        new2 = int(t2.get("prompt_n") or 0)
        total2 = new2 + int(t2.get("cache_n") or 0)
        if total2 and new2 >= 0.8 * total2:
            name = str(self.props.get("model_path") or self.props.get("model_alias") or "")
            if gemma:
                fix = "start llama-server with --swa-full (Gemma's sliding-window attention otherwise drops the cache)"
            elif re.search(r"qwen[-_ ]?3[._]?5", name, flags=re.I):
                fix = ("hybrid Qwen3.5 models cannot reuse part of a prompt: read one question per state (--layout "
                       "question_first) or accept the full cost")
            else:
                fix = "start llama-server with --swa-full if the model uses sliding-window attention"
            self.add("prefix", "warn", f"a second question about the same state re-read the whole prompt ({new2} of {total2} "
                     "tokens): state-first requests get no speed-up", fix)
        else:
            self.add("prefix", "ok", f"a second question about the same state reuses it ({new2} of {total2} tokens new)")

    def check_embeddings(self) -> None:
        try:
            r = self.post("/embedding", {"content": "hello", "embd_normalize": -1})
        except requests.RequestException as exc:
            self.add("embeddings", "info", f"/embedding failed ({exc.__class__.__name__})")
            return
        if r.status_code in (404, 501):
            self.add("embeddings", "info", "embeddings are off: the letters readout works, fitted probes do not",
                     "for probes, start llama-server with --embeddings --pooling last (or give tez --embed-backend)")
            return
        try:
            d = r.json()
            d = d[0] if isinstance(d, list) and d else d
            e = d.get("embedding")
            dim = len(e[-1] if isinstance(e[0], list) else e)
        except (ValueError, AttributeError, TypeError, IndexError, KeyError):
            self.add("embeddings", "warn", f"/embedding answered {r.status_code} without a vector",
                     "start llama-server with --embeddings --pooling last")
            return
        self.add("embeddings", "ok", f"/embedding works ({dim} numbers): fitted probes can be served")

    def check_slots(self) -> None:
        slots = self.props.get("total_slots")
        if slots is None:
            try:
                r = self.get("/slots")
                slots = len(r.json()) if r.status_code == 200 and isinstance(r.json(), list) else None
            except (requests.RequestException, ValueError):
                slots = None
        if slots is None:
            self.add("slots", "info", "the slot count is not reported")
        elif slots > 1:
            self.add("slots", "warn", f"{slots} slots: consecutive questions may land in different slots and miss the cache",
                     "start llama-server with -np 1")
        else:
            self.add("slots", "ok", "one slot: consecutive questions share one prompt cache")


def run_doctor(url: str, template: str = "gemma4", n_probs: int = DEFAULT_N_PROBS, timeout: float = 60.0,
               session: requests.Session | None = None) -> list[Check]:
    """Run every check against a llama-server; see the module docstring."""
    return Doctor(url, template, n_probs, timeout, session).run()


def report(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        lines.append(f"{c.status:<5} {c.name:<11} {c.detail}")
        if c.fix and c.status != "ok":
            lines.append(f"{'':<5} {'':<11} fix: {c.fix}")
    fails = sum(c.status == "fail" for c in checks)
    warns = sum(c.status == "warn" for c in checks)
    lines.append(f"{fails} failed, {warns} warning(s)")
    return "\n".join(lines)


def as_json(checks: list[Check]) -> list[dict[str, Any]]:
    return [asdict(c) for c in checks]
