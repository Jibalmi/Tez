"""MCP server for Tez (stdio): an agent asks typed questions about a state and gets calibrated answers, with an explicit
`escalate` for the cases it should not handle alone.

    pip install "tez-decisions[mcp]"
    tez-mcp                                   # speaks MCP on stdin/stdout; configured by environment variables

  TEZ_URL        forward every call to a running `tez serve` (TEZ_API_KEY is then sent as the bearer token)
  TEZ_BACKEND    otherwise run Tez in this process against this llama-server (default http://127.0.0.1:8080;
                 "fake" for an offline demo whose answers mean nothing)
  TEZ_TEMPLATE   gemma4 | qwen3 (default gemma4)
  TEZ_SCHEMAS    a directory of schemas to load;  TEZ_PRESETS=1 loads the built-in presets too
  TEZ_DATA_DIR   where tez_feedback writes (default <schemas>/.tez)

Tools: tez_decide, tez_schemas, tez_schema, tez_feedback, tez_status. In process, requests get tez serve's limits
(64 questions, 50,000-character states). The MCP SDK (1.x's FastMCP or 2.x's MCPServer) is imported only when the
server is built, so the tool functions (TezTools) work and are tested without it.
"""
# No `from __future__ import annotations` here: FastMCP builds each tool's input schema from its annotations at run time,
# and the tool functions are defined inside build_server with names that only exist there.
import sys
from typing import Any, Mapping

from ._version import __version__
from .config import Env, Limits
from .engine import Tez, check_state_depth
from .errors import TezError

ESCALATE = ("A gate decision of `escalate` means: do not act on that answer yourself; hand the case to a person or to "
            "a larger model. `act` means the answer is certified at the requested error rate.")
INSTRUCTIONS = (
    "Tez answers typed questions about a state (a text, or a JSON object or array) with calibrated probabilities from "
    "one forward pass of a local model; it does not generate text. Use tez_decide for yes/no (noul), one-of-N (choice) "
    "and ordered-scale (score) decisions, tez_schemas / tez_schema to find ready-made questions, and tez_feedback to "
    "record the correct label when you learn it. " + ESCALATE)

DESCRIPTIONS = {
    "tez_decide": (
        "Decide one state against typed questions, all read from a local model in one pass each, without generating "
        "text. Give `questions` as {id: {type, instructions, criteria}}: type `noul` (yes/no; the answer's `noul` is "
        "P(yes)), `choice` (criteria maps each option label to a description or null; 2 to 255 options) or `score` "
        "(criteria is a list of 2 to 10 ordered level descriptions). Or name a loaded `schema` (see tez_schemas) to use "
        "its questions, fitted probes and calibration. `readout`: auto | letters | probe. `abstain: true` adds a "
        "`__none__` option to choices ('none of these fits'). `alpha` (for example 0.05, the target error rate among "
        "acted decisions; needs a fitted schema) adds a `decision` per question in tez.questions. " + ESCALATE +
        " Without a fitted calibration every gated answer is `escalate`."),
    "tez_schemas": "List the schemas this Tez has loaded: name, description, question types, calibration and probes.",
    "tez_schema": ("One schema: its questions in wire form (to reuse in tez_decide), the fit status of each question "
                   "(probe ready / stale / none) and its calibration id."),
    "tez_feedback": ("Record the correct label for a past decision about a schema question (noul: true/false; choice: "
                     "the option label or __none__; score: the level number). tez fit learns from these rows."),
    "tez_status": "Tez's version, whether it runs in process or forwards to a server, backend health and loaded schemas.",
}


def engine_from_env(environ: Mapping[str, str] | None = None) -> Any:
    """A RemoteTez for TEZ_URL, otherwise an in-process Tez from TEZ_BACKEND, TEZ_TEMPLATE, TEZ_SCHEMAS, TEZ_DATA_DIR,
    TEZ_LAYOUT and TEZ_PRESETS (each also as VAR_FILE, like tez serve)."""
    env = Env(environ)
    if env.get("TEZ_URL"):
        from .integrations.remote import RemoteTez
        return RemoteTez(env.get("TEZ_URL"), api_key=env.get("TEZ_API_KEY"), timeout=float(env.get("TEZ_TIMEOUT", 120.0)))
    from .backends import DEFAULT_BACKEND
    tez = Tez(backend=env.get("TEZ_BACKEND", DEFAULT_BACKEND), template=env.get("TEZ_TEMPLATE", "gemma4"),
              schemas=env.get("TEZ_SCHEMAS"), data_dir=env.get("TEZ_DATA_DIR"), layout=env.get("TEZ_LAYOUT", "auto"))
    if env.flag("TEZ_PRESETS"):
        tez.add_presets()
    return tez


class TezTools:
    """The MCP tools as plain functions over a Tez engine (in process) or a RemoteTez (forwarding)."""

    def __init__(self, engine: Any, limits: Limits | None = None):
        self.engine = engine
        self.limits = Limits() if limits is None else limits

    @property
    def remote(self) -> bool:
        return not isinstance(self.engine, Tez)

    def tez_decide(self, state: Any, questions: dict | None = None, schema: str | None = None, readout: str = "auto",
                   abstain: bool = False, alpha: float | None = None) -> dict:
        body = self.engine.request_body(state, questions, schema, readout, abstain, alpha)
        if self.remote:
            return self.engine.handle(body)
        return self.engine.handle(body, limits=self.limits)

    def tez_schemas(self) -> dict:
        return self.engine.schema_summaries()

    def tez_schema(self, name: str) -> dict:
        detail = dict(self.engine.schema_detail(name))
        detail.pop("calibration", None)          # per-question thresholds and metrics: not for the agent
        detail.pop("manifest", None)
        return detail

    def tez_feedback(self, schema: str, question: str, state: Any, label: Any) -> dict:
        body = {"schema": schema, "question": question, "state": state, "label": label}
        if not self.remote:
            if isinstance(state, (dict, list)):
                check_state_depth(state)
            self.limits.check_state(state)
        return self.engine.record_feedback(body)

    def tez_status(self) -> dict:
        health = self.engine.health()
        return {"version": __version__, "mode": "remote" if self.remote else "in-process",
                "server": getattr(self.engine, "base_url", None), "health": health,
                "schemas": [s["name"] for s in self.engine.schema_summaries().get("schemas", [])]}


def _server_class() -> Any:
    """The MCP SDK's high-level server: FastMCP in mcp 1.x, MCPServer in mcp 2.x (the same constructor, add_tool and
    run("stdio") for what Tez uses)."""
    try:
        from mcp.server.fastmcp import FastMCP
        return FastMCP
    except ImportError:
        pass
    try:
        from mcp.server.mcpserver import MCPServer
        return MCPServer
    except ImportError as exc:
        raise ImportError('the Tez MCP server needs the MCP SDK: pip install "tez-decisions[mcp]"') from exc


def build_server(tools: TezTools) -> Any:
    """The MCP server around the tools (imports the MCP SDK, 1.x or 2.x)."""
    FastMCP = _server_class()               # noqa: N806 - a class chosen at run time
    from typing import Annotated, Literal, Optional, Union

    from pydantic import Field

    server = FastMCP("tez", instructions=INSTRUCTIONS)

    def guard(fn):
        def run(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except TezError as exc:          # the tool result carries the wire format's error type and message
                raise RuntimeError(f"{exc.type}: {exc.message}") from exc
        return run

    State = Union[str, dict, list]          # noqa: N806 - a type alias

    def tez_decide(state: Annotated[State, Field(description="The state: a text, or a JSON object or array.")],
                   questions: Annotated[Optional[dict], Field(description="{id: {type, instructions, criteria}}; "
                                                              "optional when schema is given.")] = None,
                   schema: Annotated[Optional[str], Field(description="Name of a loaded schema.")] = None,
                   readout: Annotated[Literal["auto", "letters", "probe"], Field(description="auto uses a fitted "
                                      "probe when there is one.")] = "auto",
                   abstain: Annotated[bool, Field(description="Add a __none__ option to every choice question.")] = False,
                   alpha: Annotated[Optional[float], Field(description="Gate at this error rate among acted decisions "
                                    "(0 < alpha < 1); needs a fitted schema.")] = None) -> dict:
        return guard(tools.tez_decide)(state, questions, schema, readout, abstain, alpha)

    def tez_schemas() -> dict:
        return guard(tools.tez_schemas)()

    def tez_schema(name: Annotated[str, Field(description="The schema's name (from tez_schemas).")]) -> dict:
        return guard(tools.tez_schema)(name)

    def tez_feedback(schema: Annotated[str, Field(description="The schema's name.")],
                     question: Annotated[str, Field(description="The question's id in that schema.")],
                     state: Annotated[State, Field(description="The state that was decided.")],
                     label: Annotated[Union[str, int, bool], Field(description="The correct label.")]) -> dict:
        return guard(tools.tez_feedback)(schema, question, state, label)

    def tez_status() -> dict:
        return guard(tools.tez_status)()

    for fn in (tez_decide, tez_schemas, tez_schema, tez_feedback, tez_status):
        server.add_tool(fn, name=fn.__name__, description=DESCRIPTIONS[fn.__name__])
    return server


def main(argv: list[str] | None = None) -> int:
    """Console script `tez-mcp`: serve the tools on stdio."""
    import logging
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="tez-mcp: %(message)s")
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if args and args[0] == "--version":
        print(f"tez-mcp {__version__}")
        return 0
    try:
        server = build_server(TezTools(engine_from_env()))
    except (ImportError, TezError, ValueError) as exc:
        print(f"tez-mcp: {getattr(exc, 'message', exc)}", file=sys.stderr)
        return 2
    server.run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
