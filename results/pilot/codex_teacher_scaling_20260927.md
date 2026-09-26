# Codex Astra label throughput

Started 2026-09-26T22:11:56.734523+00:00, with codex-cli 0.156.1, `gpt-6-astra`, low reasoning effort.
160 saved states, 800 decisions, 8 states per request. Both worker counts received identical inputs.
Independent Codex processes used the existing ChatGPT login. No GPU work was launched for Codex.

| Workers | Trial times for 800 decisions | Pooled decisions/s | Valid outputs |
|---|---|---:|---:|
| 4 | 50.03 s | 15.99 | 800/800 |
| 8 | 35.85 s | 22.31 | 800/800 |
| 20 | 35.00 s | 22.86 | 800/800 |

20 workers delivered **1.43x** the throughput of 4 workers in this test.
Timing includes process startup, model requests, parsing, and answer validation. No request failed or used tools.
Worker order was 4, 8, 20; raw token usage is in the manifest.
Cached input tokens by worker count: 4: 51,712, 8: 84,480, 20: 96,000.
Cache and service conditions can change between trials; this is a short scaling test.
Local client CPU affinity: [23]. Local startup/processing can limit scaling; these results do not establish a provider concurrency ceiling.

## GPU comparison

Gemma 4 12B Q8_0 took **78.1904 seconds of saved inference** for these same 160 states (10.23 decisions/s). That timing excludes cooling, loading, and queueing.
GPU observed during this comparison: NVIDIA GeForce RTX 5080 Laptop GPU. Saved teacher used llama.cpp build b11100, context 2048.
The recovered GPU pipeline separately completed **600 states / 3,000 decisions in 670 seconds** (4.48 decisions/s), including cooling and checkpoints.
That 600-state reference is almost entirely English and mainly FineWeb. It is operating context, not a matched wall-time arm.
Its saved inference totaled 246.1043 seconds; cooling totaled 365.1253 seconds. Model loading and GPU queue time are excluded.

**Outputs differ:** Codex selects one answer per question in one order; Gemma saves complete logits in both original and reversed option orders. Hard labels do not replace those probability targets without changing the training experiment.

## Validation

Every trial returned all 800 answers with valid option indices. Gold labels and teacher outputs were excluded from prompts.
Teacher agreement uses the pilot's temperature-adjusted, order-averaged target argmax; agreement is not accuracy.

| Trial | Workers | Agreement with Gemma | Gold correct |
|---|---:|---:|---:|
| trial-1-workers-4 | 4 | 82.12% | 75/80 |
| trial-2-workers-8 | 8 | 82.12% | 74/80 |
| trial-3-workers-20 | 20 | 82.00% | 75/80 |

There are only **80 unique gold decisions**, reused across trials. Gemma's saved target argmax got 68/80 of these right. This is a small quality check, not a quality ranking.
The sample balances sources, rather than matching training prevalence. It contains 89 English states and 71 others; alphabetical language selection limits coverage. Full-dataset throughput and long-running account limits were not measured.

## Artifacts

- Manifest and per-batch measurements: [codex_teacher_scaling_20260927.json](codex_teacher_scaling_20260927.json)
- Ignored raw prompts, answers, events, and sample: `runs\pilot\codex_teacher_scaling_20260927`
- Harness: `experiments/small_model/teacher_speed.py`; offline validator tests: `test_teacher_speed.py`.
- All generated labels remain separate from training shards. The existing GPU teacher continues.

Runner uses [Codex non-interactive structured output](https://learn.chatgpt.com/docs/non-interactive-mode).
