# AGENTS.md

How to work in this repository: for coding agents, and for people.

## Map

| Path | What |
|---|---|
| `tez/` | The runtime: engine, HTTP server, CLI, schemas, backends, readouts, fit, gate. The wheel contains only this. |
| `tez/prompt.py` | Prompt layouts (`question_first`, `state_first`), templates, control-token neutralisation, fingerprints |
| `tez/hooks.py` | Hooks around every decision and the built-ins (DecisionLog, Redact, Cache, Metrics, OTelHook); [`docs/HOOKS.md`](docs/HOOKS.md) |
| `tez/extract.py` | JSON schema / pydantic model -> questions, answers -> values (`Tez.extract`, the wire's `json_schema`) |
| `tez/presets/` | The ten use-case schemas as package data (copies of `examples/usecases/*/schema.yaml`; a test keeps them equal) |
| `tez/mcp_server.py` | The MCP server (`tez-mcp`, the `mcp` extra) |
| `tez/doctor.py`, `tez/config.py`, `tez/state.py` | `tez doctor`; server defaults (limits, CORS origins, `TEZ_*` environment); email state helpers (adapted from Laya, Apache-2.0) |
| `tez/integrations/` | `RemoteTez` (the Python API over HTTP) and the LangChain / LangGraph components |
| `tests/` | pytest suite; `test_live.py` needs a running llama-server; `stub_http.py` and `stub_llama.py` stand in for servers |
| `clients/ts/` | TypeScript client, npm package `tez-client` |
| `docker/`, `compose.yaml`, `compose.cpu.yaml` | Containers ([`docs/DOCKER.md`](docs/DOCKER.md)) |
| `docs/API.md` | The wire format: the contract every server and client here follows |
| `experiments/`, `results/`, `BENCHMARKS.md`, `docs/REPORT.md` | The research, its raw outputs and write-up |
| `site/` | The website (deployed by `.github/workflows/pages.yml`) |
| `.github/workflows/` | `ci.yml`, `release.yml` ([`docs/RELEASING.md`](docs/RELEASING.md)), `security.yml`, `pages.yml` |

## Build and test

```
python -m pip install -e ".[test,langchain,mcp,otel]" langgraph opentelemetry-sdk    # Python 3.10 or newer
python -m pytest -q -m "not live"                         # the offline suite: FakeBackend and stub servers, no model
ruff check .                                              # conservative rule set in ruff.toml
python scripts/check_banned_calls.py                      # no pickle, torch.load, unsafe YAML, eval or exec in tez/
python -m build && python scripts/check_dist.py dist      # the wheel holds tez/ and the licence files, nothing else
cd clients/ts && npm ci && npm test                       # builds dist/, type checks, runs node:test
```

Without the `langchain`, `mcp` or `otel` extras their tests skip (the MCP tool functions and the OTel hook with a
stand-in tracer still run); everything else must still pass. Behind a TLS-inspecting proxy, scope any workaround to the
one command that needs it (`pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org ...`,
`npm --strict-ssl=false ...`); never switch verification off globally.

`tez doctor` is the quickest check of a llama-server's settings (template, `n_probs`, prompt caching, `--swa-full`,
embeddings, slots, `--cache-ram 0`); it sends a few tiny prompts. `tez plan` explains what a request would do without
calling the model.

## Live tests

- Marked `live` (declared in `pyproject.toml`). `tests/test_live.py` talks to a llama-server at
  `http://127.0.0.1:8091` (override with `TEZ_LIVE_BACKEND`) serving Gemma 4 12B with template `gemma4`, started with
  `--embeddings --pooling last`. They skip themselves when nothing answers there; CI never runs them.
- Run them on purpose: `python -m pytest -q -m live`.
- They are read-only. Never restart, reload or reconfigure a llama-server you did not start: someone may be measuring
  on it. Check which model it serves (`curl http://127.0.0.1:8091/props`, `model_path`) before trusting any result.

## Where numbers come from

Every number in `README.md`, `BENCHMARKS.md`, `docs/`, `site/`, examples or docstrings comes from a recorded run:

- raw rows and a manifest in `results/` (data SHA-256, the model path read from the server's `/props`, settings,
  latency), produced by a script in `experiments/` (or `tez eval`), and consolidated in `BENCHMARKS.md`;
- with what it was measured on: model and quantisation, llama.cpp build (b11100 for the current tables), GPU, dataset
  and number of rows;
- never estimated, extrapolated, rounded from memory or carried over from elsewhere. Other people's figures (Jev's,
  SemIf's published ones) are marked as published by them;
- anything not measured that way is labelled "not separately measured": another GGUF of the same model, the Docker
  images, a CPU;
- results that go against the story are recorded too, and a number found wrong is corrected everywhere it appears,
  with a note (see `docs/REPORT.md` section 3.7).

Tests check behaviour, never benchmark numbers.

## Commits and pull requests

- **No AI or assistant attribution anywhere**: no `Co-Authored-By:` trailer naming an AI tool, no "Generated with ..."
  line, in commit messages, pull request descriptions, code, comments or docs.
- Do not commit GGUFs or llama.cpp binaries (`tools/`), feature caches (`*.npy`, `*.npz`), `dist/`, `node_modules/` or
  `clients/ts/dist/`.
- The wire format changes as one unit: `docs/API.md`, `tests/test_wire.py` and `tests/test_server.py`, the TypeScript
  types (`clients/ts/src/types.ts`) and `tez/integrations/remote.py` move together. `RemoteTez` keeps the engine's
  method names (`decide`, `decide_many`, `decide_batch`, `extract`, ...), so integrations accept either.
- A change to prompts (`tez/prompt.py`) changes fingerprints: fits made before it show as `stale`. `question_first`
  must keep hashing exactly as it does, or every existing fit goes stale.
- `tez/presets/*.yaml` are copies of `examples/usecases/*/schema.yaml`: edit both (a test fails when they differ).
- Code adapted from elsewhere (Laya, Apache-2.0) carries the attribution header and is listed in
  `THIRD_PARTY_NOTICES.md`; the wheel ships `LICENSES/Apache-2.0.txt` and that file.
- `tez/` stays free of pickle, `torch.load`, unsafe YAML loading, `eval` and `exec` (`security.yml` enforces it).

## Conventions

- Python 3.10+, `from __future__ import annotations` (except `tez/mcp_server.py`, whose tool annotations FastMCP reads
  at run time), lines up to about 130 characters, docstrings that say what a thing does and why.
- Optional dependencies stay optional: import them lazily and fail with a message naming the extra
  (`pip install "tez-decisions[langchain]"`). `import tez` needs only the base dependencies and never imports
  pydantic, LangChain, the MCP SDK or OpenTelemetry.
- Errors that reach HTTP clients are `TezError` subclasses (`tez/errors.py`) and keep the wire format's error body;
  anything else becomes a `500 internal_error` inside the app, so it keeps the CORS and request-id headers.
- Server settings come from flags or `TEZ_*` environment variables (`VAR_FILE` for secrets), read by the CLI
  (`tez/config.py`); the Docker entrypoint only runs `tez serve`.
- The version lives in one place, `tez/_version.py`; `pyproject.toml` reads it (`dynamic = ["version"]`) and
  `scripts/check_version.py` fails CI and releases when that changes.
