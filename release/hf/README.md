---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
tags:
- gguf
- llama.cpp
- decision-model
- classification
---

# Qwen3.5-4B Q8_0, first 24 of 32 blocks (GGUF)

A depth-pruned GGUF of [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) for reading decisions from the middle of
the network: the first 24 of its 32 transformer blocks, plus the output norm and the output head (which shares the
token-embedding matrix in this model), at Q8_0. It is built for the probe readout of
[Tez](https://github.com/Jibalmi/Tez), an open, local "System One" decision engine. It is not a chat model, and it is
the wrong file for zero-shot letter readouts on most tasks (see [Limitations](#limitations)).

## What it is

| | |
|---|---|
| File | `Qwen3.5-4B-Q8_0-L24.gguf`, 3,533,398,528 bytes |
| SHA-256 | `f6ab76cf738d0293392488e18da7088d9952b59f0c11f3ff4ab2360a61469fd6` |
| Source | `Qwen3.5-4B-Q8_0.gguf` from [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF), revision `e87f176479d0855a907a41277aca2f8ee7a09523`, SHA-256 `10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1` (4,482,403,488 bytes) |
| Change | transformer blocks 24 to 31 removed; `qwen35.block_count` set to 24 and the per-layer metadata arrays cut to match; every kept tensor copied byte for byte (no re-quantisation); 320 of the source's 426 tensors (`token_embd`, `output_norm` and blocks 0 to 23) |
| Tool | [`experiments/gguf_truncate.py`](https://github.com/Jibalmi/Tez/blob/main/experiments/gguf_truncate.py) |
| Architecture | `qwen35`, a hybrid: a full-attention block every fourth layer (`full_attention_interval = 4`) with recurrent (`ssm.*`) blocks between; text only, no vision projector |

Because the output norm and head are kept, the letter logits this file produces are the logit lens at layer 24, and
llama-server's `/embedding` with `--pooling last` returns the output-normed layer-24 state of the prompt's last token.
That state is what a Tez probe reads.

## Why cut at 24

A linear probe on the frozen Qwen3.5-4B's last-token state is as accurate from layer 20 to layer 28 as it gets; the
top of the network does not improve it (0.772 at the last layer). Probe accuracy by layer on typed-decisions (2,000
test decisions, in-process bf16 model, [BENCHMARKS.md §4b](https://github.com/Jibalmi/Tez/blob/main/BENCHMARKS.md)):

| layer | 12 | 14 | 16 | 18 | 20 to 28 | 30 | 32 (top) |
|---|---|---|---|---|---|---|---|
| probe accuracy | 0.574 | 0.661 | 0.737 | 0.777 | 0.790 to 0.793 | 0.787 | 0.772 |

Serving only the first 24 blocks keeps the probe's accuracy, shrinks the file by 21 % and cuts the probe's request
time by about a third.

## Measured results

One machine: RTX 5080 Laptop GPU (16 GB), llama.cpp b11100 (`win-cuda-13.4` release build), prompt caching off.
Accuracy on the 2,000 test decisions of [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions);
latency is the p50 of dedicated runs of 100 typed-decisions prompts per server, one endpoint at a time.
Sources: [BENCHMARKS.md §4c](https://github.com/Jibalmi/Tez/blob/main/BENCHMARKS.md) (depth-pruned GGUFs, latency
breakdown), `results/pruned_server_qwen35-4b.json`, `results/latency_breakdown.json`.

| blocks kept | file | probe on the served state | zero-shot letters | ms per decision, letters | ms per decision, probe |
|---|---|---|---|---|---|
| 20 of 32 | 3.06 GB | 0.787 | 0.269 | 85 | 48 |
| **24 of 32 (this file)** | **3.53 GB** | **0.793** | 0.580 | 104 | **58** |
| 29 of 32 | 4.13 GB | 0.781 | 0.580 | 125 | 70 |
| 32 of 32 (source) | 4.48 GB | 0.776 | 0.483 | 136 | 84 |

- **Probe**: one logistic regression per question (C = 0.05) on the raw served state (2,560 numbers), trained on the
  benchmark's 6,000 train decisions (300 per question), tested on its 2,000 test decisions.
- **Letters**: `/completion` with one token and the top-200 log-probabilities, softmax over the option letters.
- For reference on the same test decisions: Gemma 4 12B Q8_0 zero-shot letters score 0.704 and take 207 ms per
  decision (p50, same latency runs, same GPU); Laya's checkpoint fine-tuned on this benchmark's train split scores
  0.766; Jev's published figure is 0.727 (Jev was never run by us).

Where a decision's time goes (same 100 prompts, p50 / p90 ms, `results/latency_breakdown.json`):

| model | `/completion`, n_probs 0 | n_probs 20 | n_probs 200 | `/embedding` (probe) |
|---|---|---|---|---|
| this file (24 blocks) | 112 / 121 | 100 / 117 | 104 / 114 | 58 / 89 |
| source (32 blocks) | 128 / 148 | 130 / 139 | 136 / 159 | 84 / 109 |

### Use it for probes, not for zero-shot letters

On typed-decisions the cut raises zero-shot letter accuracy (0.483 to 0.580). That does not generalise. On Laya's
public suite (28 task and language rows, the same rows as Tez's head-to-head) the cut **lowers zero-shot letter
accuracy by 17.4 points on average** (BENCHMARKS.md §4c, "The cut 4B on Laya's public suite";
`results/h2h_q4b_L24/summary.json` against `results/h2h_q4b_L32/summary.json`):

| task | 32 blocks | 24 blocks |
|---|---|---|
| MASSIVE intent, English | 0.810 | 0.490 |
| BoolQ | 0.755 | 0.487 |
| Banking77 | 0.532 | 0.410 |
| XNLI, English | 0.790 | 0.690 |
| AG News | 0.797 | 0.762 |
| typed-decisions | 0.483 | 0.580 |

## Serve it with llama.cpp

```bash
llama-server -m Qwen3.5-4B-Q8_0-L24.gguf -ngl 99 -c 4096 -b 512 -np 1 \
  --embeddings --pooling last --host 127.0.0.1 --port 8092 --no-webui
```

These are the flags of the typed-decisions accuracy and latency runs above (the public-suite letter runs used `-c 8192`
and no `--embeddings --pooling last`). **Prompt caching must be off for this hybrid model on llama.cpp b11100**:
reusing part of a cached prompt crashes the server. The measurements sent `"cache_prompt": false` in every
`/completion` request; b11100 also accepts `--no-cache-prompt` at start-up, which we did not time.

Probe features, one request per decision:

```bash
curl http://127.0.0.1:8092/embedding \
  -d '{"content": "<|im_start|>user\n...question, options, then the input...<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", "embd_normalize": -1}'
```

The prompt is the Qwen3.5 chat format with thinking switched off (an empty think block), the constant material
(instructions and options) first and the input last. The last vector of the response's `embedding` field is the
2,560-number state a probe reads. A depth-pruned model sometimes puts end-of-sequence first when read greedily; if
`/completion` returns no `completion_probabilities`, repeat the request with `"ignore_eos": true`.

## How Tez uses it

Tez serves typed questions (`noul`, `choice`, `score`) over TypeSafe's `/v1/systemone` wire format and has two
readouts ([docs/API.md](https://github.com/Jibalmi/Tez/blob/main/docs/API.md)):

- **letters**, zero-shot, from Gemma 4 12B Q8_0 (typed-decisions 0.704);
- **probe**, from this file: `tez fit` trains one logistic probe per question from at least 25 to 50 labels per
  question and stores it in `schemas/.tez/<name>/probes.npz`; at request time Tez sends the prompt to this server's
  `/embedding` and applies the probe (`tez.readout: probe`, or `auto` once a probe is trained).

With the 12B on port 8091 and this file on port 8092:

```bash
tez serve --schemas schemas --backend http://127.0.0.1:8091 --template gemma4 \
  --embed-backend http://127.0.0.1:8092 --embed-template qwen3
```

How many labels a probe needs (layer 26 of the in-process bf16 model, 3 seeds; not re-measured on this file): 10 rows
per question 0.669, 25 rows 0.707, 50 rows 0.741, 100 rows 0.760, 200 rows 0.788, 300 rows 0.793. A logistic probe at
layer 26 of the same model comes out calibrated (ECE 0.023 as read, BENCHMARKS.md §4c).

## Limitations

- **Not a chat model.** The top eight blocks are gone; generated text is not meaningful.
- **Zero-shot letters drop by 17.4 points on average across Laya's public suite.** Validate per task before using the
  letters from this file for anything.
- **The probe needs labels** (25 to 50 per question at least) and is measured on one benchmark with this file
  (typed-decisions). Probes on other tasks (Banking77, BoolQ, MASSIVE and others) were measured in-process on the full
  model at layers 18, 22, 26 and the top, not on this file.
- **Keep the option order fixed.** A probe trained with the options in one order loses 18 points at layer 24 when
  they are reversed (0.791 to 0.612, measured in-process; BENCHMARKS.md §4c, "Decide, then bind").
- **Only tested on llama.cpp b11100** (Windows CUDA build, RTX 5080 Laptop GPU). Other builds, backends and CPUs are
  not measured, and prompt caching must be off (see above).

## Reproduce

From a clone of [Tez](https://github.com/Jibalmi/Tez) with `gguf` 0.19.0 (gguf-py) installed:

```bash
python experiments/gguf_truncate.py Qwen3.5-4B-Q8_0.gguf 24 Qwen3.5-4B-Q8_0-L24.gguf
```

The package command `tez truncate Qwen3.5-4B-Q8_0.gguf 24 Qwen3.5-4B-Q8_0-L24.gguf` does the same. Rebuilding from
the source file named above with either command gave the same SHA-256 as this file (checked on 23 September 2026).

## Licence and attribution

- **Licence: Apache-2.0**, inherited from [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) by the Qwen team,
  Alibaba Cloud ([LICENSE](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE)). A copy of that licence file is
  in this repository.
- GGUF conversion and Q8_0 quantisation by [Unsloth](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF).
- **Modification notice**: this file is Unsloth's `Qwen3.5-4B-Q8_0.gguf` with transformer blocks 24 to 31 removed and
  the block count and per-layer metadata changed to match. No weight was changed, retrained or re-quantised.
- Tez is MIT-licensed. This repository is not affiliated with or endorsed by the Qwen team, Alibaba Cloud or Unsloth.
