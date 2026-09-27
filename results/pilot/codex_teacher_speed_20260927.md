# Codex Astra label throughput

Latest assessment: [are the students competitive?](../pilot-improvement-20260927/COMPETITIVENESS.md)
summarizes the matched quality comparisons, the fresh rule test, and the outstanding release gates.

Continuation: the teacher and first student runs are documented in the
[student pilot report](../pilot-student-h200-20260927/README.md). Current student quality comparisons
with released alternatives are in the [matched comparison](../pilot-improvement-20260927/COMPARISON.md)
and [research record](../pilot-improvement-20260927/RESEARCH.md).

Started 2026-09-26T22:01:35.864221+00:00, with codex-cli 0.156.1, `gpt-6-astra`, low reasoning effort.
32 saved states, 160 decisions, 8 states per request. Both worker counts received identical inputs.
Independent Codex processes used the existing ChatGPT login. No GPU work was launched for Codex.

| Workers | Trial times for 160 decisions | Pooled decisions/s | Valid outputs |
|---|---|---:|---:|
| 1 | 37.48 s, 35.69 s | 4.37 | 320/320 |
| 4 | 11.07 s, 10.98 s | 14.52 | 320/320 |

Four workers delivered **3.32x** the throughput of one worker in this test.
Timing includes process startup, model requests, parsing, and answer validation. No request failed or used tools.
Worker order was 1, 4, 4, 1; raw cache and token usage is recorded in the JSON manifest.
Each worker-count arm used 8,192 / 8,192 cached input tokens (one / four workers respectively).

## GPU comparison

Gemma 4 12B Q8_0 took **15.5811 seconds of saved inference** for these same 32 states (10.27 decisions/s). That timing excludes cooling, loading, and queueing.
GPU observed during this comparison: NVIDIA GeForce RTX 5080 Laptop GPU. Saved teacher used llama.cpp build b11100, context 2048.
The recovered GPU pipeline separately completed **600 states / 3,000 decisions in 670 seconds** (4.48 decisions/s), including cooling and checkpoints.
That 600-state reference is almost entirely English and mainly FineWeb. It is operating context, not a matched wall-time arm.
Its saved inference totaled 246.1043 seconds; cooling totaled 365.1253 seconds. Model loading and GPU queue time are excluded.

**Outputs differ:** Codex selects one answer per question in one order; Gemma saves complete logits in both original and reversed option orders. Hard labels do not replace those probability targets without changing the training experiment.

## Validation

Every trial returned all 160 answers with valid option indices. Gold labels and teacher outputs were excluded from prompts.
Teacher agreement uses the pilot's temperature-adjusted, order-averaged target argmax; agreement is not accuracy.

| Trial | Workers | Agreement with Gemma | Gold correct |
|---|---:|---:|---:|
| trial-1-workers-1 | 1 | 79.38% | 15/16 |
| trial-2-workers-4 | 4 | 80.00% | 15/16 |
| trial-3-workers-4 | 4 | 80.00% | 15/16 |
| trial-4-workers-1 | 1 | 80.00% | 15/16 |

There are only **16 unique gold decisions**, reused across trials. Gemma's saved target argmax got 13/16 of these right. This is a small quality check, not a quality ranking.
The sample balances sources, rather than matching training prevalence. It contains 17 English states and 15 others; alphabetical language selection limits coverage. Full-dataset throughput and long-running account limits were not measured.

## Artifacts

- Manifest and per-batch measurements: [codex_teacher_speed_20260927.json](codex_teacher_speed_20260927.json)
- Ignored raw prompts, answers, events, and sample: `runs\pilot\codex_teacher_speed_20260927`
- Harness: `experiments/small_model/teacher_speed.py`; offline validator tests: `test_teacher_speed.py`.
- All generated labels remain separate from training shards. The later teacher and student runs are linked above.

Runner uses [Codex non-interactive structured output](https://learn.chatgpt.com/docs/non-interactive-mode).
