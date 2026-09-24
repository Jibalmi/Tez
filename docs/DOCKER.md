# Tez in Docker

Two containers: llama.cpp's server holds the model, and Tez reads each decision out of it and serves the
`/v1/systemone` wire format ([`API.md`](API.md)) on `http://127.0.0.1:8787`. `compose.yaml` runs both on an NVIDIA GPU;
`compose.cpu.yaml` runs both on the CPU.

> **Status.** These files are checked for syntax (YAML and the Compose specification's JSON schema), the entrypoint's
> configuration logic is unit-tested (`tests/test_packaging.py`), and CI builds the Tez image and runs it against the
> offline fake backend (`.github/workflows/ci.yml`, job `docker`). The GPU stack has not yet been run end to end, and
> nothing measured in [`BENCHMARKS.md`](../BENCHMARKS.md) was measured in a container.

## What you need

- Docker Engine with Compose v2 (`docker compose version`).
- For `compose.yaml`: an NVIDIA GPU, a driver recent enough for CUDA 12.8, and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  The measurements used a 16 GB GPU (RTX 5080 Laptop).
- The model file (next step).

## 1. Download the model

```
pip install -U huggingface_hub
hf download unsloth/gemma-4-12b-it-GGUF gemma-4-12b-it-Q8_0.gguf --local-dir models
```

This is Hugging Face repository `unsloth/gemma-4-12b-it-GGUF`, file `gemma-4-12b-it-Q8_0.gguf`, the public equivalent of
the weights behind every Gemma 4 12B Q8_0 number in this repository. Those numbers were measured with Ollama's
`gemma4:12b-it-q8_0`; the Hugging Face file has not been separately measured. To serve the exact measured file, set
`TEZ_MODELS_DIR` to Ollama's `models/blobs` directory and `TEZ_GGUF` to the blob's file name (`sha256-...`).

## 2. Start

```
docker compose up -d --build
docker compose ps                  # llama: starting -> healthy once the model is loaded; then tez starts
curl http://127.0.0.1:8787/healthz
```

The first decision:

```
curl http://127.0.0.1:8787/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {"topic": {"type": "choice", "instructions": "What is the message about?",
    "criteria": {"billing": "Payments, payouts, invoices", "technical": "Something is broken", "sales": null}}}}'
```

Only Tez is published, and only on the host's loopback address. llama.cpp stays on the Compose network; to reach it
from the host (for `tez fit` run outside Docker, say), add `ports: ["127.0.0.1:8080:8080"]` to the `llama` service.

## Configuration

Compose reads these from the shell or from a `.env` file next to `compose.yaml`:

| Variable | Default | Meaning |
|---|---|---|
| `TEZ_MODELS_DIR` | `./models` | Host directory mounted read-only at `/models` |
| `TEZ_GGUF` | `gemma-4-12b-it-Q8_0.gguf` | Model file inside it |
| `TEZ_TEMPLATE` | `gemma4` | Prompt template of the model: `gemma4` or `qwen3` |
| `TEZ_HOST_PORT` | `8787` | Host port for Tez (bound to 127.0.0.1) |
| `LLAMA_IMAGE` | `ghcr.io/ggml-org/llama.cpp:server-cuda-b11096` | llama.cpp image (`compose.cpu.yaml`: `server-b11096`) |

The Tez image itself (`docker/Dockerfile`) is configured by environment; `docker/entrypoint.py` turns it into
`tez serve` flags:

| Variable | Default in the image | Meaning |
|---|---|---|
| `TEZ_BACKEND` | the CLI's `http://127.0.0.1:8080` (compose sets `http://llama:8080`) | llama-server URL, or `fake` for the offline demo backend |
| `TEZ_TEMPLATE` | `gemma4` | Prompt template |
| `TEZ_SCHEMAS` | unset | Directory of `*.yaml` schemas (fitted artefacts in `<dir>/.tez/`) |
| `TEZ_DATA_DIR` | `/data` | Where `POST /v1/feedback` writes |
| `TEZ_PORT` | `8787` | Listening port |
| `TEZ_HOST` | `0.0.0.0` | Listening address inside the container |
| `TEZ_API_KEY` | unset | Require `Authorization: Bearer <key>` (`/healthz` stays open) |
| `TEZ_EMBED_BACKEND` | unset | Separate llama-server for probe features |

Every variable in the second table can be read from a file instead: `TEZ_API_KEY_FILE=/run/secrets/tez_api_key`. The
file's surrounding whitespace is stripped, and setting both `X` and `X_FILE` is an error. Extra container arguments are
appended to `tez serve` (for example `--log-level warning`). The process runs as user 10001, not root.

### Schemas, feedback and an API key

Uncomment the matching lines in `compose.yaml`:

```yaml
services:
  tez:
    environment:
      TEZ_SCHEMAS: /schemas
      TEZ_API_KEY_FILE: /run/secrets/tez_api_key
    volumes:
      - ./schemas:/schemas:ro          # schema files, plus .tez/<name>/ from `tez fit`
      - tez-data:/data                 # rows recorded by POST /v1/feedback
    secrets:
      - tez_api_key
volumes:
  tez-data:
secrets:
  tez_api_key:
    file: ./secrets/tez_api_key.txt
```

Fit a schema outside the container (`pip install "tez-decisions[fit]"`, then `tez fit --backend http://127.0.0.1:8080
--schema schemas/<name>.yaml --labels labels.jsonl` with llama's port published), or build the image with the fit extra
and run it in there:

```
docker build -f docker/Dockerfile --build-arg EXTRAS=fit -t tez:fit .
```

A fit is used only while the model and the prompt match what `tez fit` saw ([`API.md`](API.md)), so fit against the
same llama.cpp image and model file you serve.

## CPU only

```
docker compose -f compose.cpu.yaml up -d --build
```

The same two services with llama.cpp's CPU image and no GPU reservation. A 12B model is much slower on a CPU; no CPU
setup has been measured.

## Without Compose

```
docker build -f docker/Dockerfile -t tez .
docker run --rm -p 127.0.0.1:8787:8787 -e TEZ_BACKEND=fake tez                 # offline smoke test, answers mean nothing
docker run --rm -p 127.0.0.1:8787:8787 -e TEZ_BACKEND=http://host.docker.internal:8080 \
  --add-host=host.docker.internal:host-gateway tez                              # a llama-server running on the host
```

## Notes

- **Keep `--swa-full`.** Gemma 4 uses sliding-window attention; without the flag llama-server silently re-reads the
  whole prompt on every request and the cached-prefix speed-ups disappear (README, "Traps").
- **The llama.cpp image.** The measured build is b11100 (the Windows `win-cuda-13.4` binary). No image was published
  for b11100, so the compose files pin b11096, the closest published build before it. `server-cuda-b11096` is the
  CUDA 12.8 build; `server-cuda13-b11096` is the CUDA 13 build. Build tags are not meant to move, but for an immutable
  pin add the digest seen when these files were written:
  `server-cuda-b11096@sha256:0192ab2545efcbe79c240645e34abd8fffbe4813aedcef5a0e3a886ef6d6d82f`,
  `server-cuda13-b11096@sha256:a2f0d414ecca5b41e9d71e06e2df6dfa5260cc4e85c1b45fbf62c58e5ec28332`,
  `server-b11096@sha256:82b5129742c04e12c4b710ae1908fd951f6b227339d45f0aa71f1165df353f70`.
- **Start-up.** The `llama` healthcheck allows ten minutes to load the model before a failed check counts; Tez starts
  only when llama.cpp reports healthy. `docker compose logs -f llama` shows the loading progress.
- **Health.** `/healthz` answers 200 with `"status": "degraded"` when Tez cannot reach llama.cpp; the container
  healthcheck tests that Tez itself is up.
