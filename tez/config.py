"""Server defaults shared by the engine, the HTTP server, the MCP server and the CLI (importing this pulls in nothing
heavy): request limits, the browser origins CORS allows, and how TEZ_* environment variables are read."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import PayloadTooLarge, TezError

# The website playground (https://jibalmi.github.io) and pages served from this machine on any port.
DEFAULT_CORS_ORIGINS = ("http://127.0.0.1:*", "http://localhost:*", "https://jibalmi.github.io")


@dataclass(frozen=True)
class Limits:
    """Request limits (413 payload_too_large past any of them). The defaults are tez serve's; the Python API applies
    none unless given limits=. 0 or None switches one off (as --max-* 0 does)."""

    max_body_bytes: int | None = 2 * 1024 * 1024
    max_questions: int | None = 64
    max_state_chars: int | None = 50_000
    max_batch: int | None = 64

    def __post_init__(self) -> None:
        for name in ("max_body_bytes", "max_questions", "max_state_chars", "max_batch"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a whole number of at least 0 (0 or None: no limit), got {value!r}")

    @classmethod
    def unlimited(cls) -> Limits:
        return cls(None, None, None, None)

    def check_questions(self, n: int) -> None:
        if self.max_questions and n > self.max_questions:
            raise PayloadTooLarge(f"too many questions: {n} (the limit is {self.max_questions}; tez serve --max-questions)")

    def check_state(self, state: Any, where: str = "state") -> None:
        if not self.max_state_chars:
            return
        n = len(state) if isinstance(state, str) else len(json.dumps(state, ensure_ascii=False))
        if n > self.max_state_chars:
            raise PayloadTooLarge(f"{where} is too large: {n:,} characters (the limit is {self.max_state_chars:,}; "
                                  f"tez serve --max-state-chars)")

    def check_batch(self, n: int) -> None:
        if self.max_batch and n > self.max_batch:
            raise PayloadTooLarge(f"too many states in one batch: {n} (the limit is {self.max_batch}; tez serve --max-batch)")


class Env:
    """The TEZ_* environment: VAR, or the contents of the file that VAR_FILE names (Docker and Compose secrets; the
    surrounding whitespace is stripped). An empty VAR counts as unset; VAR and VAR_FILE together, an unreadable file
    or an empty one raise TezError."""

    def __init__(self, environ: Mapping[str, str] | None = None):
        self.environ = os.environ if environ is None else environ

    def get(self, name: str, default: Any = None) -> Any:
        value = self.environ.get(name) or None
        path = self.environ.get(f"{name}_FILE") or None
        if path is not None:
            if value is not None:
                raise TezError(f"set {name} or {name}_FILE, not both")
            try:
                value = Path(path).read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise TezError(f"cannot read {name}_FILE={path}: {exc.strerror or exc}") from exc
            if not value:
                raise TezError(f"{name}_FILE={path} is empty")
        return default if value is None else value

    def flag(self, name: str) -> bool:
        """A yes/no variable: 1, true, yes or on."""
        return str(self.get(name, "")).strip().lower() in ("1", "true", "yes", "on")
