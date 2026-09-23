"""Errors that map onto the wire contract's status codes (docs/API.md, "Errors")."""
from __future__ import annotations


class TezError(Exception):
    """Base class. `status` is the HTTP code, `type` the error type in the response body."""

    status = 500
    type = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def body(self) -> dict:
        return {"error": {"type": self.type, "message": self.message}}


class InvalidRequest(TezError):
    """The request (or a schema / label file) does not match the contract: 422."""

    status = 422
    type = "invalid_request"


class NotFound(TezError):
    status = 404
    type = "not_found"


class Unauthorized(TezError):
    status = 401
    type = "unauthorized"


class BackendUnavailable(TezError):
    """The model server cannot be reached, is loading, or failed: 503."""

    status = 503
    type = "backend_unavailable"


class BackendRequestError(TezError):
    """The model server rejected this particular input (for example a prompt longer than its context): 422."""

    status = 422
    type = "invalid_request"
