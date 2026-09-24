"""A stub llama-server for `tez doctor` tests: /health, /props, /completion (with llama.cpp-style timings from a simulated
one-slot prompt cache), /embedding and /slots, on 127.0.0.1 with an ephemeral port. No model: letters get fixed
log-probabilities."""
from __future__ import annotations

import json
import string
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

GEMMA4_TEMPLATE = "{{ bos_token }}{% for m in messages %}<|turn>{{ m.role }}\n{{ m.content }}<turn|>\n{% endfor %}"
QWEN_TEMPLATE = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"


class StubLlama:
    """cache: "prefix" (reuses the longest common prefix, like Gemma with --swa-full), "exact" (only an identical or
    extended prompt, like a sliding-window model without --swa-full), or "off". top: how many log-probabilities it
    returns at most (Ollama caps at 20)."""

    def __init__(self, template: str = GEMMA4_TEMPLATE, model_path: str = "/models/gemma-4-12b-it-Q8_0.gguf",
                 cache: str = "prefix", top: int = 1000, embeddings: bool = True, slots: int = 1, health: int = 200,
                 timings: bool = True, n_ctx: int = 4096):
        self.template, self.model_path, self.cache, self.top = template, model_path, cache, top
        self.embeddings, self.slots, self.health, self.timings, self.n_ctx = embeddings, slots, health, timings, n_ctx
        self.last = ""
        self.requests: list[tuple[str, str, Any]] = []
        self._httpd: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def completion(self, body: dict) -> dict:
        prompt = body.get("prompt", "")
        total = max(1, len(prompt) // 4)
        shared = 0
        if self.cache == "prefix":
            n = 0
            for a, b in zip(self.last, prompt):
                if a != b:
                    break
                n += 1
            shared = n // 4
        elif self.cache == "exact" and self.last and prompt.startswith(self.last):
            shared = len(self.last) // 4
        cache_n = min(total - 1, shared)
        self.last = prompt
        n_probs = int(body.get("n_probs") or 0)
        letters = list(string.ascii_uppercase)
        tops = [{"id": i, "token": t, "logprob": -0.5 - i, "bytes": []} for i, t in enumerate(letters + [f"t{i}" for i in range(1000)])]
        tops = tops[: min(n_probs, self.top)]
        out = {"content": "A", "tokens_evaluated": total,
               "completion_probabilities": [{"id": 0, "token": "A", "logprob": -0.5, "top_logprobs": tops}]}
        if self.timings:
            out["timings"] = {"prompt_n": total - cache_n, "cache_n": cache_n, "prompt_ms": 1.0, "predicted_n": 1}
        return out

    def __enter__(self) -> StubLlama:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                pass

            def reply(self, status: int, body: Any) -> None:
                raw = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                stub.requests.append(("GET", self.path, None))
                if self.path == "/health":
                    self.reply(stub.health, {"status": "ok" if stub.health == 200 else "loading model"})
                elif self.path == "/props":
                    self.reply(200, {"model_path": stub.model_path, "chat_template": stub.template, "build_info": "b11100",
                                     "total_slots": stub.slots, "default_generation_settings": {"n_ctx": stub.n_ctx}})
                elif self.path == "/slots":
                    self.reply(200, [{"id": i} for i in range(stub.slots)])
                elif self.path == "/v1/models":
                    self.reply(200, {"object": "list", "data": [{"id": stub.model_path, "meta": {
                        "n_params": 11_907_350_576, "n_embd": 3840, "n_ctx": stub.n_ctx, "ftype": "Q8_0"}}]})
                else:
                    self.reply(404, {"error": {"code": 404, "message": "not found"}})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                stub.requests.append(("POST", self.path, body))
                if self.path == "/completion":
                    self.reply(200, stub.completion(body))
                elif self.path == "/embedding":
                    if stub.embeddings:
                        self.reply(200, [{"index": 0, "embedding": [[0.1, 0.2, 0.3, 0.4]]}])
                    else:
                        self.reply(501, {"error": {"code": 501, "message": "This server does not support embeddings."}})
                else:
                    self.reply(404, {"error": {"code": 404, "message": "not found"}})

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
