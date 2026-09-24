"""The MCP server (tez/mcp_server.py): the tool functions against the FakeBackend, in process and forwarded to a server,
the FastMCP wiring, and the stdio transport (skipped without the MCP SDK)."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from conftest import DOCS_REQUEST
from stub_http import StubServer
from tez import FakeBackend, InvalidRequest, Tez
from tez.errors import PayloadTooLarge
from tez.integrations import RemoteTez
from tez.mcp_server import DESCRIPTIONS, INSTRUCTIONS, TezTools, build_server, engine_from_env, main

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = DOCS_REQUEST["questions"]
ESCALATE = "hand the case to a person or to a larger model"


@pytest.fixture
def tools(schema_dir):
    return TezTools(Tez(backend=FakeBackend(), schemas=schema_dir))


def test_decide_tool(tools):
    res = tools.tez_decide(DOCS_REQUEST["state"], questions=QUESTIONS)
    assert set(res["answers"]) == set(QUESTIONS) and res["answers"]["topic"]["choice"] == "billing"
    gated = tools.tez_decide("The app crashes", schema="support-triage", alpha=0.05)
    assert all(m["decision"] == "escalate" for m in gated["tez"]["questions"].values())     # nothing fitted yet
    abstained = tools.tez_decide("x", questions={"t": QUESTIONS["topic"]}, abstain=True, readout="letters")
    assert "__none__" in abstained["answers"]["t"]["probabilities"]
    with pytest.raises(InvalidRequest, match="type must be one of"):
        tools.tez_decide("x", questions={"q": {"type": "maybe", "instructions": "x"}})
    with pytest.raises(PayloadTooLarge, match="too many questions"):
        tools.tez_decide("x", questions={f"q{i}": QUESTIONS["is_urgent"] for i in range(65)})


def test_schema_feedback_and_status_tools(tools, schema_dir):
    assert [s["name"] for s in tools.tez_schemas()["schemas"]] == ["support-triage"]
    detail = tools.tez_schema("support-triage")
    assert set(detail["questions"]) == {"topic", "is_urgent", "anger"} and "calibration" not in detail and "manifest" not in detail
    assert tools.tez_feedback("support-triage", "topic", "I was charged twice", "billing")["label"] == "billing"
    rows = (schema_dir / ".tez" / "feedback" / "support-triage.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(rows[-1])["state"] == "I was charged twice"
    status = tools.tez_status()
    assert status["mode"] == "in-process" and status["health"]["status"] == "ok" and status["schemas"] == ["support-triage"]


def test_descriptions_say_what_escalate_means():
    assert ESCALATE in DESCRIPTIONS["tez_decide"] and ESCALATE in INSTRUCTIONS
    assert set(DESCRIPTIONS) == {"tez_decide", "tez_schemas", "tez_schema", "tez_feedback", "tez_status"}


def test_engine_from_env(schema_dir):
    tez = engine_from_env({"TEZ_BACKEND": "fake", "TEZ_TEMPLATE": "qwen3", "TEZ_SCHEMAS": str(schema_dir), "TEZ_PRESETS": "1"})
    assert isinstance(tez, Tez) and tez.template == "qwen3" and "support-triage" in tez.schemas and "topic-news" in tez.schemas
    assert tez.schemas["support-triage"].builtin is False                   # the directory's schema wins over the preset
    remote = engine_from_env({"TEZ_URL": "http://127.0.0.1:8787", "TEZ_API_KEY": "k", "TEZ_BACKEND": "ignored"})
    assert isinstance(remote, RemoteTez) and repr(remote).endswith("api_key=***)")


def test_forwarding_to_a_server(schema_dir):
    engine = Tez(backend=FakeBackend(), schemas=schema_dir)
    with StubServer(engine) as srv:
        tools = TezTools(RemoteTez(srv.url))
        res = tools.tez_decide("The app crashes", schema="support-triage", readout="letters")
        assert res["answers"]["topic"]["choice"] == engine.decide("The app crashes", schema="support-triage")["answers"]["topic"]["choice"]
        assert tools.tez_status()["mode"] == "remote" and tools.tez_status()["server"] == srv.url
        assert tools.tez_schema("support-triage")["name"] == "support-triage"
    assert srv.requests[0]["body"] == {"state": "The app crashes", "schema": "support-triage", "tez": {"readout": "letters"}}


def test_main_version_and_help(capsys):
    assert main(["--version"]) == 0 and capsys.readouterr().out.startswith("tez-mcp ")
    assert main(["--help"]) == 0 and "TEZ_URL" in capsys.readouterr().out


def test_missing_sdk_names_the_extra(monkeypatch, tools):
    for name in ("mcp", "mcp.server", "mcp.server.fastmcp"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match=r'tez-decisions\[mcp\]'):
        build_server(tools)


# ---------------------------------------------------------------------------------------------- with the MCP SDK
def test_fastmcp_wiring(tools):
    pytest.importorskip("mcp.server.fastmcp")
    server = build_server(tools)
    listed = asyncio.run(server.list_tools())
    assert [t.name for t in listed] == ["tez_decide", "tez_schemas", "tez_schema", "tez_feedback", "tez_status"]
    decide = listed[0]
    assert ESCALATE in decide.description
    assert set(decide.inputSchema["properties"]) == {"state", "questions", "schema", "readout", "abstain", "alpha"}
    assert decide.inputSchema["required"] == ["state"]
    out = asyncio.run(server.call_tool("tez_decide", {"state": "Something is broken again", "schema": "support-triage"}))
    content = out[0] if isinstance(out, tuple) else out
    assert json.loads(content[0].text)["answers"]["topic"]["choice"] == "technical"
    with pytest.raises(Exception, match="invalid_request: unknown schema 'nope'"):
        asyncio.run(server.call_tool("tez_decide", {"state": "x", "schema": "nope"}))


def test_stdio_transport(schema_dir):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = {"TEZ_BACKEND": "fake", "TEZ_SCHEMAS": str(schema_dir), "PYTHONPATH": str(ROOT),
           "PYTHONIOENCODING": "utf-8", **({"SYSTEMROOT": os.environ["SYSTEMROOT"]} if "SYSTEMROOT" in os.environ else {})}
    params = StdioServerParameters(command=sys.executable, args=["-m", "tez.mcp_server"], env=env, cwd=str(ROOT))

    async def session_run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = [t.name for t in (await session.list_tools()).tools]
                result = await session.call_tool("tez_decide", {"state": "Help! My payouts have been failing",
                                                                "schema": "support-triage", "alpha": 0.05})
                status = await session.call_tool("tez_status", {})
                return names, result, status

    names, result, status = asyncio.run(asyncio.wait_for(session_run(), timeout=60))
    assert "tez_decide" in names and not result.isError
    body = json.loads(result.content[0].text)
    assert body["answers"]["topic"]["choice"] == "billing"
    assert body["tez"]["questions"]["topic"]["decision"] == "escalate"
    assert json.loads(status.content[0].text)["mode"] == "in-process"
