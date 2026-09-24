"""Container entrypoint: `tez serve`, followed by any arguments given to the container (docs/DOCKER.md).

Configuration is by environment variables, which `tez serve` reads itself: TEZ_BACKEND, TEZ_TEMPLATE, TEZ_SCHEMAS,
TEZ_DATA_DIR, TEZ_PORT, TEZ_HOST, TEZ_API_KEY, TEZ_LOG_LEVEL, TEZ_CORS_ORIGINS, TEZ_PRESETS, TEZ_LAYOUT and
TEZ_EMBED_BACKEND, each also as VAR_FILE for Docker and Compose secrets (TEZ_API_KEY_FILE=/run/secrets/tez_api_key).
The image sets TEZ_HOST=0.0.0.0, TEZ_PORT=8787 and TEZ_DATA_DIR=/data. The process is replaced by tez (exec), so it
receives docker stop's SIGTERM directly.
"""
from __future__ import annotations

import os
import sys
from typing import Sequence


def serve_argv(extra: Sequence[str] = ()) -> list[str]:
    """The command the container runs."""
    return ["tez", "serve", *extra]


def main(argv: Sequence[str] | None = None) -> int:
    cmd = serve_argv(sys.argv[1:] if argv is None else argv)
    os.execvp(cmd[0], cmd)
    return 0  # pragma: no cover - exec does not return


if __name__ == "__main__":
    sys.exit(main())
