# v3: a larger student on the same data

**Outcome (2026-09-28): capacity is the stronger lever, and the student now matches Decider overall; G1 is still
not met.** The unchanged Qwen control recipe on **Qwen3.5-2B** instead of 0.8B, trained on the same 155,835 v2
teacher decisions, reaches 53.55% typed-decision accuracy (0.8B v2: 52.20%; Decider 0.8B: 53.40%) and a 72.73%
MASSIVE-11 mean (0.8B v2: 62.91%; Decider: 74.45%). The multilingual jump of +9.8 points is the largest single
improvement of the pilot, and it came from model size, not from data: same data, same recipe, same evaluation.

Held-out teacher agreement on the unchanged validation split rose 0.7835 (0.8B, 30k) -> 0.8158 (0.8B, v2) ->
**0.8345** (2B, v2), so both levers help the distillation objective; only capacity moved the multilingual public tasks.

## Public comparison (accuracy, %; identical cases per column)

| Task | Qwen 0.8B 30k | Qwen 0.8B v2 | **Qwen 2B v2** | Decider 0.8B |
|---|---:|---:|---:|---:|
| Typed decisions | 50.15 | 52.20 | **53.55** | 53.40 |
| MASSIVE-11 mean | 64.09 | 62.91 | **72.73** | 74.45 |
| Routing | 89.22 | 87.22 | **89.22** | 84.46 |
| Jailbreak | 85.75 | 87.75 | 91.25 | **92.50** |
| Toxicity | 60.25 | 57.25 | 63.50 | **75.25** |
| Email spam | 87.25 | 86.75 | 90.00 | **95.50** |
| Phishing | 74.25 | 76.50 | 79.25 | **92.50** |
| RAG relevance | 51.50 | 50.00 | 54.00 | **65.00** |
| Support triage | 43.50 | 44.25 | 44.25 | **44.75** |
| SemIf authored144 | 68.75 | 68.75 | **77.78** | 77.08 |
| SemIf perturbations108 | 50.00 | 44.44 | 62.96 | **71.30** |

Typed-decision ECE: 0.055 (2B) against 0.070 (0.8B v2) and 0.134 (Decider); the student stays the better-calibrated
system. Retention against the 12B on typed decisions: 0.761 (2B), 0.742 (0.8B v2), 0.759 (Decider).

Paired over identical rows (bootstrap, 2,000 resamples, seed 13):

| Comparison | Difference | 95% interval |
|---|---:|---|
| 2B v2 - 0.8B v2, typed | +1.35 | -0.70 to +3.60 |
| 2B v2 - Decider, typed | +0.15 | -2.35 to +2.65 |
| 2B v2 - 0.8B v2, MASSIVE-11 pooled | **+9.82** | +7.27 to +12.36 |
| 2B v2 - Decider, MASSIVE-11 pooled | -1.73 | -4.09 to +0.64 |

Only the multilingual gain over the 0.8B is separated from zero. Typed parity with Decider, and the remaining
multilingual gap, are both within noise. These public benchmarks were already observed, so the comparison is
exploratory.

## G1 and what remains

G1 needs typed accuracy >= 0.563 and soft accuracy >= 0.46; the 2B student reaches 0.5355 and 0.4044, so **G1 is
still not met**, and no arm of this pilot has met it. What the three rounds establish:

- More teacher data of this kind is a weak lever (+2.05 typed, no multilingual gain, about 5x the data).
- Student capacity is a strong lever for multilingual work (+9.8 MASSIVE-11 at 2.5x the parameters) and a weak one
  for typed decisions (+1.35, not separated from zero).
- The residual gap to Decider is concentrated in the content-safety and retrieval application tasks (toxicity,
  spam, phishing, RAG), not in typed decisions or routing. That points at Decider's training exposure on those task
  families rather than at scale.
- Cost of the 2B round: one RunPod H100, 9,740 steps in 6,039 s, $11.16.

The G3 latency gate is unmeasured for the 2B student; at 2.27B parameters it is further from the 10 ms GPU target
than the 0.8B, whose H200 single-question p50 was already 60.40 ms.

## Execution

`train_c.py` defaults (LoRA r 16, alpha 32, all-linear, lr 2e-4, batch 16, one epoch, seed 13), backbone selected by
`TEZ_PILOT_QWEN_BASE=Qwen/Qwen3.5-2B` at pinned revision `15852e8c`, `TEZ_PILOT_TRAIN_ONLY_PREFIX=v2-` so the
validation split stays the original 278 states / 1,390 decisions. Peak VRAM 11.47 GB, no thermal pauses, max 54 C.
Evaluated with the 30k run's frozen suites and scorer ([results](eval_c-qwen2b-v2.json), [rows](rows/)). The adapter
is under ignored `runs/student-data-v2/models/c-qwen2b-v2`, not in Git.
