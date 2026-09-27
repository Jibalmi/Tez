# Qwen probability-parity result

Candidate: c-qwen08-restricted. All 278 original teacher-validation states and 1,390 decisions.
The archived inputs contain validation cases only; no model weights or training examples are exported.

| Measure | Observed | Unchanged requirement |
|---|---:|---:|
| Argmax agreement | 1.00000000 | >= 0.99 |
| Maximum probability change | 0.00364906 | <= 0.04 |
| Mean total variation | 0.00000263 | <= 0.005 |

**Parity passed.** No runtime promotion or measured speedup follows from this result.
Separately valid target-laptop latency remains required. No thresholds were relaxed and no cloud compute was used.

An independent reader checked every probability's shape, range, normalization, unique identifier and validation membership,
then reproduced all three metrics. The archive contains raw predictions, validation inputs, receipts and executed sources.
