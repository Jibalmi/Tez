# Contributing to Tez

Tez is a small runtime plus the research behind it. The most useful contributions keep both honest: code with tests,
and numbers with the runs behind them.

## Set up

```
git clone https://github.com/Jibalmi/Tez && cd Tez
python -m venv .venv
. .venv/bin/activate                          # Windows: .venv\Scripts\activate
python -m pip install -e ".[test,langchain]" langgraph
python -m pytest -q -m "not live"
```

The offline suite runs on `FakeBackend`, a deterministic stand-in for a model, so it needs no GPU and no download. To run
Tez on a real model, follow "Use it" in the [README](README.md) or [`docs/DOCKER.md`](docs/DOCKER.md).

## Before you open a pull request

- `python -m pytest -q -m "not live"` passes. If you changed prompts or readouts and have a llama-server, run
  `python -m pytest -q -m live` too ([`AGENTS.md`](AGENTS.md) explains the live tests).
- `ruff check .` passes.
- Changes under `tez/`: `python scripts/check_banned_calls.py` passes (no pickle, `torch.load`, unsafe YAML, `eval`
  or `exec`).
- Changes to the wire format: [`docs/API.md`](docs/API.md), the tests, the TypeScript types in `clients/ts/src/types.ts`
  and `tez/integrations/remote.py` change together.
- Changes to `clients/ts`: `cd clients/ts && npm ci && npm test`.
- Changes to the Docker files: `docker build -f docker/Dockerfile .` and `docker compose config` (CI also builds the
  image and runs it against the fake backend).
- New or changed numbers: the raw rows and a manifest go in `results/`, the script in `experiments/`, the table in
  `BENCHMARKS.md`. "Where numbers come from" in [`AGENTS.md`](AGENTS.md) has the rules.

## Pull requests and commits

- One topic per pull request; say what changed and how you checked it.
- Commit messages say what and why. No AI or assistant attribution lines: no `Co-Authored-By:` for tools, no
  "Generated with ...".
- CI (`ci.yml` and `security.yml`) must pass.

## Code from elsewhere

- Copied or adapted code needs a licence compatible with MIT (MIT, BSD, Apache-2.0, ISC), a header in the file naming
  its origin and your changes, and an entry in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The Apache-2.0
  text is in [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).
- No data or model outputs whose terms forbid this use. Nothing here is trained on or derived from Jev's outputs, and it
  stays that way.

## Security

Please report vulnerabilities privately through GitHub's *Security* tab (*Report a vulnerability*), not in a public
issue.

## Licence

Contributions are accepted under the repository's MIT licence.
