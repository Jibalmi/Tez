"""HTTP server: TypeSafe's /v1/systemone wire format plus Tez's schema and feedback endpoints (docs/API.md).

    POST /v1/systemone      decide (Jev-compatible)
    GET  /v1/models         model aliases (Jev-compatible)
    GET  /healthz           liveness, backend and readout status
    GET  /v1/schemas        loaded schemas
    GET  /v1/schemas/{name} one schema: questions, probe status, calibration
    POST /v1/feedback       record a correct label for a past decision

Errors: {"error": {"type": ..., "message": ...}} with 401 (only with an API key), 404, 422 or 503.
"""
from __future__ import annotations

import hmac
import inspect
import json
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from ._version import __version__
from .engine import Tez
from .errors import InvalidRequest, TezError, Unauthorized

log = logging.getLogger("tez")
_HTTP_TYPES = {404: "not_found", 405: "method_not_allowed", 413: "invalid_request", 422: "invalid_request"}


def _error(status: int, type_: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"type": type_, "message": message}})


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


def create_app(tez: Tez, api_key: str | None = None, cors: bool = True) -> FastAPI:
    """The FastAPI application around a Tez engine."""
    app = FastAPI(title="Tez", version=__version__,
                  description="Open local System One decision engine (Jev-compatible /v1/systemone).")
    app.state.tez = tez

    if cors:
        kwargs: dict[str, Any] = dict(allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"],
                                      expose_headers=["*"], max_age=600)
        if "allow_private_network" in inspect.signature(CORSMiddleware.__init__).parameters:
            kwargs["allow_private_network"] = True     # otherwise Starlette rejects a preflight that asks for it
        app.add_middleware(CORSMiddleware, **kwargs)
        app.add_middleware(_PrivateNetworkAccess)       # outermost, so it also sees the preflights CORS answers

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

    async def read_json(request: Request) -> Any:
        raw = await request.body()
        if not raw.strip():
            raise InvalidRequest("the request body must be a JSON object")
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidRequest(f"the request body is not valid JSON ({exc.__class__.__name__})") from exc

    @app.get("/", include_in_schema=False)
    def index():
        return {"name": "tez", "version": __version__,
                "endpoints": ["POST /v1/systemone", "GET /v1/models", "GET /healthz", "GET /v1/schemas",
                              "GET /v1/schemas/{name}", "POST /v1/feedback"]}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        """Decide one state against many typed questions (Jev's wire format; Tez extensions in `schema` and `tez`)."""
        authorize(request)
        body = await read_json(request)
        return await run_in_threadpool(tez.handle, body)

    @app.get("/v1/models")
    async def models(request: Request):
        authorize(request)
        return await run_in_threadpool(tez.models)

    @app.get("/healthz")
    async def healthz():
        return await run_in_threadpool(tez.health)

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
        return await run_in_threadpool(tez.record_feedback, body)

    return app
