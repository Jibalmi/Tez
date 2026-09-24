"""RemoteTez: the decision part of Tez's Python API over HTTP, for any server that speaks the /v1/systemone wire
format (docs/API.md): `tez serve` on this machine or another one, or a Jev-compatible endpoint.

    from tez.integrations import RemoteTez
    tez = RemoteTez("http://127.0.0.1:8787", api_key=None, timeout=30)
    tez.decide("Help! My payouts have been failing for 3 days.",
               questions={"topic": {"type": "choice", "instructions": "What is the message about?",
                                    "criteria": {"billing": None, "technical": None}}})

The method names and return values match the in-process engine (tez.Tez): decide, handle, decide_many,
decide_batch, handle_batch, extract, plan, health, models, schema_summaries, schema_detail and record_feedback, so every
integration accepts either. Error responses raise the
TezError subclasses the engine raises (InvalidRequest 422, NotFound 404, Unauthorized 401, BackendUnavailable 503);
a server that cannot be reached, times out or answers with something other than JSON raises BackendUnavailable.

Only non-default Tez extensions are sent (`schema`, and `tez` when readout, abstain or a gate is set), so a strict
Jev-compatible server sees a plain request. Redirects are followed only within the same origin (scheme, host and
port) and only when the method and body survive the hop (307 and 308, or any redirect of a GET). A redirect to another
origin is refused before anything is sent there, so a state or an API key never goes to a host you did not name.
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping
from urllib.parse import quote, urljoin, urlsplit

import requests

from ..errors import (BackendUnavailable, Forbidden, InternalError, InvalidRequest, NotFound, PayloadTooLarge, TezError,
                      Unauthorized)

DEFAULT_URL = "http://127.0.0.1:8787"
MAX_REDIRECTS = 5
_BY_TYPE = {"unauthorized": Unauthorized, "invalid_request": InvalidRequest, "not_found": NotFound,
            "backend_unavailable": BackendUnavailable, "forbidden": Forbidden, "payload_too_large": PayloadTooLarge,
            "internal_error": InternalError}
_BY_STATUS = {401: Unauthorized, 403: Forbidden, 404: NotFound, 413: PayloadTooLarge, 422: InvalidRequest,
              500: InternalError, 502: BackendUnavailable, 503: BackendUnavailable, 504: BackendUnavailable}
_LOCAL = re.compile(r"^https?://(localhost|127\.\d+\.\d+\.\d+|\[::1\])(:\d+)?(/|$)", re.I)


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    return scheme, (parts.hostname or "").lower(), parts.port or {"http": 80, "https": 443}.get(scheme)


def _tez_options(readout: str | None, abstain: bool, alpha: float | None, gate: Any, layout: str | None) -> dict:
    tez: dict[str, Any] = {}
    if readout not in (None, "auto"):
        tez["readout"] = readout
    if abstain:
        tez["abstain"] = True
    if alpha is not None:
        tez["gate"] = {"alpha": alpha}
    elif gate is not None:
        tez["gate"] = gate
    if layout is not None:
        tez["layout"] = layout
    return tez


def _with(err: TezError, status: int | None = None, type_: str | None = None) -> TezError:
    """Override a TezError's HTTP status or error type (both are class defaults)."""
    if status is not None:
        err.status = status
    if type_ is not None:
        err.type = type_
    return err


def error_from_response(r: requests.Response) -> TezError:
    """The TezError for an HTTP error response. The wire contract's body is {"error": {"type", "message"}}; other
    bodies (a proxy's HTML page, FastAPI's {"detail": ...}) are mapped by status code."""
    etype = message = None
    try:
        body = r.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            etype = err.get("type") if isinstance(err.get("type"), str) else None
            message = err.get("message") if isinstance(err.get("message"), str) else None
        elif isinstance(err, str):
            message = err
        elif isinstance(body.get("detail"), str):
            message = body["detail"]
    if not message:
        message = (r.text or r.reason or "").strip()[:300] or f"HTTP {r.status_code}"
    cls = _BY_TYPE.get(etype or "") or _BY_STATUS.get(r.status_code) or TezError
    return _with(cls(message), status=r.status_code, type_=etype or (cls.type if cls is not TezError else "http_error"))


class RemoteTez:
    """A Tez server over HTTP, with the in-process engine's decide / schema methods.

    base_url  the server, default http://127.0.0.1:8787 (a bare host:port gets http://)
    api_key   sent as Authorization: Bearer <key> (only needed when the server runs with --api-key)
    timeout   seconds for connecting and for the answer
    session   an optional requests.Session (proxies, certificates); a local base_url never uses proxy settings
    """

    def __init__(self, base_url: str = DEFAULT_URL, api_key: str | None = None, timeout: float = 30.0,
                 session: requests.Session | None = None):
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a URL such as http://127.0.0.1:8787")
        url = base_url.strip()
        if not re.match(r"^https?://", url, flags=re.I):
            if "://" in url:
                raise ValueError(f"base_url must be an http:// or https:// URL, got {base_url!r}")
            url = "http://" + url
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not timeout > 0:
            raise ValueError(f"timeout must be a positive number of seconds, got {timeout!r}")
        self.base_url = url.rstrip("/")
        self.timeout = float(timeout)
        self._api_key = api_key or None
        self._session = session if session is not None else requests.Session()
        if session is None and _LOCAL.match(self.base_url):
            self._session.trust_env = False   # a local server must never be reached through a corporate proxy

    def __repr__(self) -> str:
        return f"RemoteTez({self.base_url!r}{', api_key=***' if self._api_key else ''})"

    def __enter__(self) -> RemoteTez:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._session.close()

    # ---------------------------------------------------------------------------------- transport
    def _request(self, method: str, path: str, body: Any = None) -> Any:
        url = self.base_url + path
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        for _ in range(MAX_REDIRECTS + 1):
            try:
                r = self._session.request(method, url, data=data, headers=headers, timeout=self.timeout,
                                          allow_redirects=False)
            except requests.Timeout as exc:
                raise BackendUnavailable(f"Tez server {self.base_url} timed out after {self.timeout:g}s") from exc
            except requests.RequestException as exc:
                raise BackendUnavailable(f"Tez server {self.base_url} is unreachable ({exc.__class__.__name__})") from exc
            if r.is_redirect:
                target = urljoin(url, r.headers["location"])
                r.close()
                if _origin(target) != _origin(url):
                    raise _with(TezError(f"refused a redirect from {url} to another origin ({target}); set base_url to "
                                         "the server itself"), status=r.status_code, type_="redirect_refused")
                if r.status_code not in (307, 308) and method != "GET":
                    raise _with(TezError(f"{url} answered {r.status_code}: a redirect that would drop the {method} body; "
                                         "set base_url to the server itself"), status=r.status_code, type_="redirect_refused")
                url = target
                continue
            if r.status_code >= 400:
                raise error_from_response(r)
            try:
                return r.json()
            except ValueError as exc:
                raise _with(BackendUnavailable(f"{url} did not answer with JSON (HTTP {r.status_code})"),
                            type_="invalid_response") from exc
        raise _with(TezError(f"more than {MAX_REDIRECTS} redirects from {self.base_url}{path}"), status=310,
                    type_="redirect_refused")

    # ---------------------------------------------------------------------------------- the engine's API
    def handle(self, body: Any) -> dict:
        """POST a wire-format request body to /v1/systemone and return the response body."""
        return self._request("POST", "/v1/systemone", body)

    def decide(self, state: Any, questions: Mapping | None = None, schema: Any = None, readout: str | None = "auto",
               abstain: bool = False, alpha: float | None = None, gate: Any = None, model: str | None = None, *,
               layout: str | None = None) -> dict:
        """Decide one state; the same arguments as tez.Tez.decide. `model` is sent only when given."""
        return self.handle(self.request_body(state, questions, schema, readout, abstain, alpha, gate, model, layout))

    def request_body(self, state: Any, questions: Mapping | None = None, schema: Any = None, readout: str | None = "auto",
                     abstain: bool = False, alpha: float | None = None, gate: Any = None, model: str | None = None,
                     layout: str | None = None) -> dict:
        """The minimal wire-format body: Tez extensions only when they differ from the defaults."""
        body: dict[str, Any] = {"state": state}
        if model is not None:
            body["model"] = model
        if questions is not None:
            body["questions"] = {qid: (q.to_wire() if hasattr(q, "to_wire") else q) for qid, q in questions.items()}
        if schema is not None:
            body["schema"] = schema if isinstance(schema, str) else getattr(schema, "name", schema)
        tez = _tez_options(readout, abstain, alpha, gate, layout)
        if tez:
            body["tez"] = tez
        return body

    def extract(self, state: Any, schema_or_model: Any, *, alpha: float | None = None, return_details: bool = False,
                readout: str | None = "auto", layout: str | None = None) -> Any:
        """tez.Tez.extract over HTTP: a JSON schema or a pydantic model is sent as json_schema; a schema name uses the
        server's schema (its questions are fetched once, to read the answers back as values)."""
        from ..extract import finish, prepare
        from ..schema import Schema, parse_questions

        def loaded(name: str) -> Schema | None:
            if not isinstance(schema_or_model, str) or name != schema_or_model:
                return None
            detail = self.schema_detail(name)
            return Schema(name=name, questions=parse_questions(detail.get("questions") or {}))

        extraction, fields = prepare(schema_or_model, loaded)
        body = {**self.request_body(state, None, None, readout, False, alpha, None, None, layout), **fields}
        return finish(extraction, self.handle(body), alpha, return_details)

    def plan(self, body: Any) -> dict:
        """POST /v1/plan: what a /v1/systemone request body would do, without calling the model (tez.Tez.plan)."""
        return self._request("POST", "/v1/plan", body)

    def handle_batch(self, body: Any) -> dict:
        """POST a wire-format batch body to /v1/systemone/batch and return the batch response."""
        return self._request("POST", "/v1/systemone/batch", body)

    def decide_batch(self, states: list, questions: Mapping | None = None, schema: Any = None,
                     readout: str | None = "auto", abstain: bool = False, alpha: float | None = None, gate: Any = None,
                     model: str | None = None, *, layout: str | None = None) -> dict:
        """Decide many states against the same questions (POST /v1/systemone/batch): {"model", "results", "usage",
        "tez"}, one result per state in order, each a response or {"error": {...}}."""
        body = self.request_body(None, questions, schema, readout, abstain, alpha, gate, model, layout)
        body.pop("state")
        body["states"] = list(states)
        return self.handle_batch(body)

    def decide_many(self, states: list, questions: Mapping | None = None, schema: Any = None,
                    readout: str | None = "auto", abstain: bool = False, alpha: float | None = None, gate: Any = None,
                    model: str | None = None, *, layout: str | None = None) -> list[dict]:
        """The results of decide_batch: one per state, in order (tez.Tez.decide_many over HTTP)."""
        return self.decide_batch(states, questions, schema, readout, abstain, alpha, gate, model, layout=layout)["results"]

    def health(self) -> dict:
        return self._request("GET", "/healthz")

    def models(self) -> dict:
        return self._request("GET", "/v1/models")

    def schema_summaries(self) -> dict:
        return self._request("GET", "/v1/schemas")

    def schema_detail(self, name: str) -> dict:
        return self._request("GET", "/v1/schemas/" + quote(str(name), safe=""))

    def record_feedback(self, body: Mapping) -> dict:
        """POST /v1/feedback: {"schema", "question", "state", "label"}."""
        return self._request("POST", "/v1/feedback", dict(body))
