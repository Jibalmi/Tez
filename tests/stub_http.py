"""A stub /v1/systemone server for client tests: standard library only, on 127.0.0.1 with an ephemeral port.

By default it serves an in-process Tez engine (docs/API.md routes and error bodies, like tez.server), so a remote
client can be checked against the engine's own answers. `routes` overrides any (method, path) with a function
(body, headers) -> (status, body[, headers]) for error, redirect and slow-server cases. Every request is recorded.
"""
from __future__ import annotations

import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import unquote, urlsplit

from tez.errors import NotFound, TezError

Route = Callable[[Any, dict], tuple]


class StubServer:
    def __init__(self, engine: Any = None, routes: dict[tuple[str, str], Route] | None = None, api_key: str | None = None):
        self.engine = engine
        self.routes = dict(routes or {})
        self.api_key = api_key
        self.requests: list[dict] = []
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> StubServer:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                pass

            def _reply(self, status: int, body: Any, headers: dict | None = None) -> None:
                raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
                self.send_response(status)
                ctype = "application/json" if not isinstance(body, bytes) else "text/html; charset=utf-8"
                self.send_header("Content-Type", ctype)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _handle(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = raw
                path = urlsplit(self.path).path
                headers = {k.lower(): v for k, v in self.headers.items()}
                stub.requests.append({"method": method, "path": path, "headers": headers, "body": body})
                route = stub.routes.get((method, path))
                if route is not None:
                    self._reply(*route(body, headers))
                    return
                status, out = stub.serve(method, path, body, headers)
                self._reply(status, out)

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def serve(self, method: str, path: str, body: Any, headers: dict) -> tuple[int, Any]:
        """The engine behind docs/API.md's routes, with the server's error bodies."""
        def err(status: int, etype: str, message: str) -> tuple[int, dict]:
            return status, {"error": {"type": etype, "message": message}}

        if self.api_key and path != "/healthz":
            auth = headers.get("authorization", "")
            token = auth[7:].strip() if auth.lower().startswith("bearer ") else headers.get("x-api-key", "")
            if not token or not hmac.compare_digest(token.encode(), self.api_key.encode()):
                return err(401, "unauthorized", "missing or invalid API key")
        if self.engine is None:
            return err(404, "not_found", f"no route {path}")
        try:
            if method == "POST" and path == "/v1/systemone":
                return 200, self.engine.handle(body)
            if method == "POST" and path == "/v1/feedback":
                return 200, self.engine.record_feedback(body)
            if method == "GET" and path == "/v1/models":
                return 200, self.engine.models()
            if method == "GET" and path == "/healthz":
                return 200, self.engine.health()
            if method == "GET" and path == "/v1/schemas":
                return 200, self.engine.schema_summaries()
            if method == "GET" and path.startswith("/v1/schemas/"):
                return 200, self.engine.schema_detail(unquote(path[len("/v1/schemas/"):]))
            raise NotFound(f"no route {method} {path}")
        except TezError as exc:
            return exc.status, exc.body()
