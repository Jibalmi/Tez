# Paired structured-record results

This English synthetic diagnostic measures frozen models on identical facts in flat and nested JSON.
It does not substitute for the original public accuracy, calibration or laptop speed gates.

| Model | Flat accuracy | Nested accuracy | Nested minus flat | State-bootstrap 95% interval | Label flips |
|---|---:|---:|---:|---:|---:|
| Encoder, original | 51.39% | 52.22% | +0.83 points | [-1.67, +3.47] | 14.17% |
| Encoder, longer run | 60.56% | 65.14% | +4.58 points | [+2.22, +7.08] | 11.81% |
| Qwen student | 71.25% | 69.58% | -1.67 points | [-2.64, -0.83] | 1.67% |

All five questions for a state stay together across 2,000 bootstrap resamples. These are
exploratory unadjusted intervals. Container keys and punctuation differ between representations;
all leaf keys, values and their traversal order, questions, options and answers are preserved.

## Results by operation

Each cell is flat / nested accuracy.

| Operation | Uniform random | Schema majority label | Encoder, original | Encoder, longer run | Qwen student |
|---|---:|---:|---:|---:|---:|
| lookup | 33.33% | 33.33% | 97.22% / 100.00% | 95.14% / 99.31% | 100.00% / 100.00% |
| threshold | 50.00% | 66.67% | 38.89% / 47.92% | 46.53% / 52.78% | 66.67% / 66.67% |
| conjunction | 50.00% | 66.67% | 50.69% / 33.33% | 56.94% / 56.25% | 56.25% / 47.92% |
| band | 33.33% | 33.33% | 33.33% / 33.33% | 33.33% / 33.33% | 33.33% / 33.33% |
| missing | 25.00% | 25.00% | 36.81% / 46.53% | 70.83% / 84.03% | 100.00% / 100.00% |

The majority baseline chooses the most frequent semantic answer for each schema while ignoring
state values. It is a descriptive class-balance baseline, calculated from the diagnostic labels,
not a fitted deployed model. In particular, the Boolean tasks have unequal answer frequencies.

## Provenance and limits

The generator authored these exact-answer records without reading training or benchmark inputs.
This set stays evaluation-only. It repeats five rule families in four settings and does not prove
generalization to arbitrary JSON workflows, natural documents or other languages. Any later training
candidate requires disjoint validation and fresh confirmation; this probe cannot select it.

Completed runs validated every prediction, finite normalized probability vector, gold label and
argmax, plus the exact previously evaluated weight hashes. Input token checks are recorded in
each execution receipt. The laptop thermal governor and shared lock remained enabled; these
elapsed times are not usable latency measurements. No additional RunPod resources were used.

See [registration and method](README.md), [registration](registration.json), and [full results](summary.json).
