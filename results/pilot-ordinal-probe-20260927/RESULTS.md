# Ordinal question-type and answer-position diagnostic

This secondary experiment keeps state facts, instructions and semantic answers fixed.
It varies only the declared question type and cyclic presentation order. Every condition
uses one question per call, including the recomputed unmodified score/original-order baseline.
The 144 English records were already observed in the earlier diagnostic; this is not fresh confirmation.

| Model | Type/order | Accuracy | Most frequent semantic answer | Share | Difference from score/012 | Paired 95% interval |
|---|---|---:|---|---:|---:|---:|
| a-base-12ep | score-012 | 33.33% | 1 | 100.00% | +0.00 points | [+0.00, +0.00] |
| a-base-12ep | score-120 | 43.06% | 1 | 73.61% | +9.72 points | [+2.78, +15.97] |
| a-base-12ep | score-201 | 33.33% | 1 | 100.00% | +0.00 points | [+0.00, +0.00] |
| a-base-12ep | choice-012 | 33.33% | 1 | 100.00% | +0.00 points | [+0.00, +0.00] |
| a-base-12ep | choice-120 | 40.97% | 1 | 69.44% | +7.64 points | [+0.00, +15.28] |
| a-base-12ep | choice-201 | 33.33% | 1 | 100.00% | +0.00 points | [+0.00, +0.00] |
| c-qwen08-30k | score-012 | 33.33% | 1 | 100.00% | +0.00 points | [+0.00, +0.00] |
| c-qwen08-30k | score-120 | 33.33% | 1 | 100.00% | +0.00 points | [+0.00, +0.00] |
| c-qwen08-30k | score-201 | 34.03% | 1 | 95.83% | +0.69 points | [-1.39, +3.47] |
| c-qwen08-30k | choice-012 | 33.33% | 1 | 100.00% | +0.00 points | [+0.00, +0.00] |
| c-qwen08-30k | choice-120 | 33.33% | 1 | 100.00% | +0.00 points | [+0.00, +0.00] |
| c-qwen08-30k | choice-201 | 33.33% | 1 | 88.89% | +0.00 points | [-4.17, +4.17] |

Semantic keys: 0 = below lower limit; 1 = between the inclusive limits; 2 = above upper limit.
Each correct label appears equally often; constant guesses score 33.33%. Intervals resample paired
states 2,000 times with seed 13 and have no multiplicity adjustment. Native tie order is preserved.

| Model | Score: same semantic prediction across orders | Choice: same semantic prediction across orders |
|---|---:|---:|
| a-base-12ep | 73.61% | 69.44% |
| c-qwen08-30k | 95.83% | 88.89% |

No model was trained, calibrated or promoted. No new RunPod resource was used.
Inference runs under the shared laptop compute lock and unchanged thermal governor; elapsed times
do not establish a latency gate. A follow-up change requires separate representative validation
and fresh confirmation; these diagnostic examples must not select or train it.

[Registration](registration.json) · [Verified results and receipts](summary.json)
