"""HTTP server: TypeSafe's /v1/systemone wire format plus Tez's schema and feedback endpoints (docs/API.md).

    POST /v1/systemone        decide (Jev-compatible)
    POST /v1/systemone/batch  decide many states against the same questions
    GET  /v1/models           model aliases (Jev-compatible)
    GET  /healthz             liveness, backend and readout status
    GET  /v1/schemas          loaded schemas
    GET  /v1/schemas/{name}   one schema: questions, probe status, calibration
    POST /v1/feedback         record a correct label for a past decision

Every response carries x-typesafe-request-id and server-timing; decisions also carry X-Tez-Run-Id (the id hooks see).
Errors: {"error": {"type": ..., "message": ...}} with 401 (only with an API key), 404, 413, 422, 500 or 503.
"""
from __future__ import annotations

import hmac
import inspect
import json
import logging
import time
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException

from ._version import __version__
from .engine import Limits, Tez
from .errors import InternalError, InvalidRequest, PayloadTooLarge, TezError, Unauthorized
from .hooks import DecisionContext, new_run_id

log = logging.getLogger("tez")
_HTTP_TYPES = {404: "not_found", 405: "method_not_allowed", 413: "payload_too_large", 422: "invalid_request"}
EXPOSED_HEADERS = ["x-tez-run-id", "x-typesafe-request-id", "x-tez-layout", "server-timing"]


def _error(status: int, type_: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"type": type_, "message": message}})


class _RequestMeta:
    """Outermost: every response gets x-typesafe-request-id (a decision's run id, a fresh id for anything else) and
    server-timing (`total`, plus `tez` and `backend` for decisions); decision endpoints add X-Tez-Run-Id and
    X-Tez-Layout. The id is in request.state.tez_run_id before the endpoint runs."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        state = scope.setdefault("state", {})
        rid = new_run_id()
        state["tez_run_id"] = rid
        t0 = time.perf_counter()

        async def send_with_meta(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["x-typesafe-request-id"] = rid
                timing = list(state.get("tez_timing") or [])
                timing.append(f"total;dur={(time.perf_counter() - t0) * 1000.0:.1f}")
                headers["server-timing"] = ", ".join(timing)
                if state.get("tez_decision"):
                    headers["x-tez-run-id"] = rid
                    if state.get("tez_layout"):
                        headers["x-tez-layout"] = state["tez_layout"]
            await send(message)

        return await self.app(scope, receive, send_with_meta)


class _PrivateNetworkAccess:
    """Every CORS preflight answers `Access-Control-Allow-Private-Network: true`. Chrome's Private Network Access
    sends such a preflight when a public page (the website playground) calls a server on this machine."""

    HEADER = b"access-control-allow-private-network"

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        names = {k.lower() for k, _ in scope.get("headers", [])} if scope["type"] == "http" else set()
        preflight = scope.get("method") == "OPTIONS" and b"origin" in names and b"access-control-request-method" in names
        if not preflight:
            return await self.app(scope, receive, send)

        async def send_with_header(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                if not any(k.lower() == self.HEADER for k, _ in headers):
                    headers.append((self.HEADER, b"true"))
                message = {**message, "headers": headers}
            await send(message)

        return await self.app(scope, receive, send_with_header)


def create_app(tez: Tez, api_key: str | None = None, cors: bool = True, limits: Limits | None = None) -> FastAPI:
    """The FastAPI application around a Tez engine. limits: request limits (default: Limits(), tez serve's
    defaults: 2 MiB bodies, 64 questions, 50,000-character states, 64 states per batch)."""
    app = FastAPI(title="Tez", version=__version__,
                  description="Open local System One decision engine (Jev-compatible /v1/systemone).")
    app.state.tez = tez
    limits = Limits() if limits is None else limits
    app.state.limits = limits

    if cors:
        kwargs: dict[str, Any] = dict(allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"],
                                      expose_headers=EXPOSED_HEADERS, max_age=600)
        if "allow_private_network" in inspect.signature(CORSMiddleware.__init__).parameters:
            kwargs["allow_private_network"] = True     # otherwise Starlette rejects a preflight that asks for it
        app.add_middleware(CORSMiddleware, **kwargs)
        app.add_middleware(_PrivateNetworkAccess)       # outside CORS, so it also sees the preflights CORS answers
    app.add_middleware(_RequestMeta)                    # outermost: every response, preflights included

    @app.exception_handler(TezError)
    async def _tez_error(request: Request, exc: TezError):
        return JSONResponse(status_code=exc.status, content=exc.body())

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        return _error(exc.status_code, _HTTP_TYPES.get(exc.status_code, "error"), str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        return _error(422, "invalid_request", "; ".join(str(e.get("msg", e)) for e in exc.errors()) or "invalid request")

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):   # pragma: no cover - defensive
        log.exception("unhandled error")
        return _error(500, "internal_error", f"{exc.__class__.__name__}: {exc}")

    def authorize(request: Request) -> None:
        if not api_key:
            return
        header = request.headers.get("authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else (request.headers.get("x-api-key") or "")
        if not token or not hmac.compare_digest(token.encode("utf-8"), api_key.encode("utf-8")):
            raise Unauthorized("missing or invalid API key (send Authorization: Bearer <key>)")

    def too_large() -> PayloadTooLarge:
        return PayloadTooLarge(f"the request body is over {limits.max_body_bytes:,} bytes (tez serve --max-body-bytes)")

    async def read_body(request: Request) -> bytes:
        """The body, refusing to buffer more than max_body_bytes whatever the framing (a Content-Length that says so
        is refused before reading; a chunked body is abandoned as soon as it passes the limit)."""
        cap = limits.max_body_bytes
        if cap is None:
            return await request.body()
        length = request.headers.get("content-length", "")
        if length.strip().isdigit() and int(length) > cap:
            raise too_large()
        chunks, total = [], 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > cap:
                raise too_large()
            chunks.append(chunk)
        return b"".join(chunks)

    async def read_json(request: Request) -> Any:
        raw = await read_body(request)
        if not raw.strip():
            raise InvalidRequest("the request body must be a JSON object")
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidRequest(f"the request body is not valid JSON ({exc.__class__.__name__})") from exc

    async def call(fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Run an engine call in the thread pool. A failure that is not a TezError (a hook raising, say) becomes a
        500 internal_error answered inside the app, so it still carries CORS and request-id headers."""
        try:
            return await run_in_threadpool(fn, *args, **kwargs)
        except TezError:
            raise
        except Exception as exc:
            log.exception("unexpected error in %s", getattr(fn, "__name__", fn))
            raise InternalError(f"{exc.__class__.__name__}: {exc}") from exc

    def decided(request: Request, ctx: DecisionContext) -> dict:
        request.state.tez_decision = True
        request.state.tez_layout = ctx.layout
        timing = [f"tez;dur={ctx.latency_ms or 0.0:.1f}"]
        if ctx.traces:
            timing.append(f"backend;dur={tez.backend_ms(ctx):.1f}")
        request.state.tez_timing = timing
        return ctx.response

    @app.get("/", include_in_schema=False)
    def index():
        return {"name": "tez", "version": __version__,
                "endpoints": ["POST /v1/systemone", "POST /v1/systemone/batch", "GET /v1/models", "GET /healthz",
                              "GET /v1/schemas", "GET /v1/schemas/{name}", "POST /v1/feedback"]}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        """Decide one state against many typed questions (Jev's wire format; Tez extensions in `schema` and `tez`)."""
        authorize(request)
        body = await read_json(request)
        request.state.tez_decision = True
        ctx = await call(tez.execute, body, run_id=request.state.tez_run_id, limits=limits)
        return decided(request, ctx)

    @app.post("/v1/systemone/batch")
    async def systemone_batch(request: Request):
        """Decide many states against the same questions: {"states": [...], "questions" | "schema", "tez"}."""
        authorize(request)
        body = await read_json(request)
        request.state.tez_decision = True
        res = await call(tez.handle_batch, body, run_id=request.state.tez_run_id, limits=limits)
        request.state.tez_timing = [f"tez;dur={res['tez']['latency_ms']:.1f}"]
        return res

    @app.get("/v1/models")
    async def models(request: Request):
        authorize(request)
        return await call(tez.models)

    @app.get("/healthz")
    async def healthz():
        return await call(tez.health)

    @app.get("/v1/schemas")
    async def schemas(request: Request):
        authorize(request)
        return tez.schema_summaries()

    @app.get("/v1/schemas/{name}")
    async def schema(name: str, request: Request):
        authorize(request)
        return tez.schema_detail(name)

    @app.post("/v1/feedback")
    async def feedback(request: Request):
        authorize(request)
        body = await read_json(request)
        if isinstance(body, dict) and isinstance(body.get("state"), (str, dict, list)):
            limits.check_state(body["state"])
        return await call(tez.record_feedback, body)

    return app
