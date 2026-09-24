"""Container entrypoint: `tez serve`, configured from environment variables (docs/DOCKER.md).

  TEZ_BACKEND        llama-server URL, or "fake" for the offline demo backend   (read by tez serve itself)
  TEZ_TEMPLATE       prompt template of the model: gemma4 | qwen3               (read by tez serve itself)
  TEZ_API_KEY        require Authorization: Bearer <key> (/healthz stays open)  (read by tez serve itself)
  TEZ_EMBED_BACKEND  separate llama-server for probe features                   (read by tez serve itself)
  TEZ_SCHEMAS        directory of *.yaml schemas                                -> --schemas
  TEZ_DATA_DIR       where POST /v1/feedback writes (default <schemas>/.tez)    -> --data-dir
  TEZ_PORT           port to listen on (default 8787)                           -> --port
  TEZ_HOST           address to bind (default 0.0.0.0)                          -> --host

Each of them can be read from a file instead, for Docker and Compose secrets: TEZ_API_KEY_FILE=/run/secrets/tez_api_key.
Surrounding whitespace (the trailing newline) is stripped; setting both VAR and VAR_FILE is an error. Arguments given
to the container are appended to `tez serve`. The process is replaced by tez (exec), so it receives docker stop's
SIGTERM directly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

FILE_VARS = ("TEZ_BACKEND", "TEZ_TEMPLATE", "TEZ_API_KEY", "TEZ_EMBED_BACKEND", "TEZ_SCHEMAS", "TEZ_DATA_DIR",
             "TEZ_PORT", "TEZ_HOST")
DEFAULT_HOST = "0.0.0.0"   # noqa: S104 - inside a container; publish the port on 127.0.0.1 on the host
DEFAULT_PORT = "8787"


class ConfigError(Exception):
    pass


def resolve_env(environ: Mapping[str, str]) -> dict[str, str]:
    """The environment with every VAR_FILE read into VAR (and VAR_FILE removed)."""
    env = dict(environ)
    for name in FILE_VARS:
        path = env.pop(f"{name}_FILE", None)
        if not path:
            continue
        if env.get(name):
            raise ConfigError(f"set {name} or {name}_FILE, not both")
        try:
            env[name] = Path(path).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ConfigError(f"cannot read {name}_FILE={path}: {exc.strerror or exc}") from exc
        if not env[name]:
            raise ConfigError(f"{name}_FILE={path} is empty")
    return env


def serve_argv(env: Mapping[str, str], extra: Sequence[str] = ()) -> list[str]:
    """The `tez serve` command line for this environment."""
    port = (env.get("TEZ_PORT") or DEFAULT_PORT).strip()
    if not port.isdigit() or not 0 < int(port) < 65536:
        raise ConfigError(f"TEZ_PORT must be a port number, got {port!r}")
    argv = ["tez", "serve", "--host", env.get("TEZ_HOST") or DEFAULT_HOST, "--port", port]
    if env.get("TEZ_SCHEMAS"):
        argv += ["--schemas", env["TEZ_SCHEMAS"]]
    if env.get("TEZ_DATA_DIR"):
        argv += ["--data-dir", env["TEZ_DATA_DIR"]]
    return argv + list(extra)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        env = resolve_env(os.environ)
        cmd = serve_argv(env, sys.argv[1:] if argv is None else argv)
    except ConfigError as exc:
        print(f"tez-entrypoint: {exc}", file=sys.stderr)
        return 2
    os.execvpe(cmd[0], cmd, env)
    return 0  # pragma: no cover - exec does not return


if __name__ == "__main__":
    sys.exit(main())
