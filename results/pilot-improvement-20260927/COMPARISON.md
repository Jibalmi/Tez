# Matched small-model comparison

All values below come from recorded pilot evaluations.

Typed accuracy and MASSIVE are percentages; ECE is shown in percentage points (lower is better).
The MASSIVE mean requires all 11 languages. No composite rank is constructed across unlike tasks.

| System | Typed accuracy | MASSIVE 11 | Typed ECE, raw | Typed ECE, shipped/validation calibration |
|---|---:|---:|---:|---:|
| laya-td | 76.65 | 34.55 | 21.37 | — |
| tez-12b | 70.40 | 88.45 | 26.21 | — |
| decider08-native | 53.40 | 74.45 | 13.39 | — |
| c-qwen08-30k | 50.15 | 64.09 | 6.12 | 12.54 |
| kev08-native | 49.10 | 61.27 | 14.73 | 6.40 |
| a-base-12ep | 39.05 | 51.73 | 12.83 | 24.01 |
| gliclass-instruct-base-v1.0 | 39.05 | 24.73 | 50.14 | — |
| gliclass-multilang-mini | 37.20 | 62.18 | 30.25 | — |
| laya-en | 36.25 | 35.36 | 17.37 | — |
| a-small-30k | 36.10 | 44.45 | 14.26 | 20.93 |
| a-base-30k | 35.95 | 52.73 | 12.62 | 20.55 |
| laya-ml | 35.20 | 52.36 | 31.58 | — |
| qwen08-frozen | 35.10 | 13.18 | 14.29 | — |
| a-base-10k | 32.50 | 46.27 | 18.33 | 25.91 |
| gliclass-x-base | 27.35 | 29.91 | 30.38 | — |
| gliclass-instruct-edge-v1.0 | 25.50 | 4.64 | 64.71 | — |

Laya-td trained on typed-decisions' training split. Tez students excluded the entire dataset.
Released competitors may have benchmark overlap; unknown overlap is not evidence of a held-out test.
Raw T=1 is primary. Calibration uses only teacher-validation labels or the released temperature.
Evaluation-label refits are excluded. Follow-up gains require fresh independent confirmation.

## Application tasks

All seven tasks are shown to preserve strengths and losses.

| System | Routing | Jailbreak | Toxicity | Spam | Phishing | Relevance | Triage |
|---|---:|---:|---:|---:|---:|---:|---:|
| laya-td | 65.91 | 76.25 | 53.00 | 95.75 | 94.00 | 62.50 | 50.50 |
| tez-12b | 96.99 | 86.50 | 71.25 | 96.75 | 89.75 | 61.00 | 40.75 |
| decider08-native | 84.46 | 92.50 | 75.25 | 95.50 | 92.50 | 65.00 | 44.75 |
| c-qwen08-30k | 89.22 | 85.75 | 60.25 | 87.25 | 74.25 | 51.50 | 43.50 |
| kev08-native | 84.96 | 89.00 | 64.25 | 66.25 | 65.75 | 59.25 | 39.50 |
| a-base-12ep | 58.65 | 85.50 | 53.75 | 80.75 | 68.00 | 50.00 | 40.25 |
| gliclass-instruct-base-v1.0 | 38.60 | 77.00 | 49.75 | 47.00 | 55.75 | 50.00 | 40.75 |
| gliclass-multilang-mini | 64.16 | 54.25 | 49.25 | 35.50 | 72.00 | 50.75 | 37.25 |
| laya-en | 63.91 | 70.75 | 53.00 | 99.25 | 98.00 | 62.50 | 50.25 |
| a-small-30k | 53.63 | 83.00 | 62.25 | 69.50 | 51.75 | 49.75 | 33.50 |
| a-base-30k | 59.65 | 89.25 | 56.25 | 78.75 | 63.25 | 50.00 | 35.75 |
| laya-ml | 44.11 | 80.50 | 53.50 | 99.25 | 99.25 | 67.25 | 54.00 |
| qwen08-frozen | 34.59 | 24.75 | 49.50 | 53.00 | 38.75 | 49.50 | 28.50 |
| a-base-10k | 57.89 | 78.75 | 53.00 | 61.00 | 44.00 | 50.00 | 32.25 |
| gliclass-x-base | 26.57 | 67.00 | 51.50 | 55.50 | 60.00 | 51.50 | 34.50 |
| gliclass-instruct-edge-v1.0 | 7.52 | 77.00 | 50.75 | 44.75 | 45.00 | 54.50 | 28.25 |

## SemIf generalization

Mean family balanced accuracy, using SemIf's own evaluator.

| System | Authored 144 | Perturbations 108 |
|---|---:|---:|
| tez-12b | 94.31 | 99.21 |
| decider08-native | 75.61 | 69.75 |
| c-qwen08-30k | 70.52 | 50.12 |
| kev08-native | 73.21 | 67.81 |
| a-base-12ep | 49.52 | 48.34 |
| gliclass-instruct-base-v1.0 | 53.55 | 44.46 |
| gliclass-multilang-mini | 46.03 | 54.94 |
| a-small-30k | 35.21 | 44.23 |
| a-base-30k | 50.04 | 41.38 |
| qwen08-frozen | 33.46 | 33.65 |
| a-base-10k | 46.96 | 42.59 |
| gliclass-x-base | 48.43 | 54.94 |
| gliclass-instruct-edge-v1.0 | 41.39 | 31.64 |

## Paired uncertainty on typed decisions

Difference from the original Qwen student, in accuracy points. Resampling keeps each state's five
questions together (2,000 draws, seed 13). Intervals are exploratory and unadjusted for multiple comparisons.

| Candidate minus original Qwen student | Difference | 95% state-bootstrap interval |
|---|---:|---:|
| a-base-10k | -17.65 | [-20.45, -14.85] |
| a-base-30k | -14.20 | [-16.40, -12.15] |
| a-small-30k | -14.05 | [-16.75, -11.40] |
| gliclass-x-base | -22.80 | [-25.60, -20.00] |
| a-base-12ep | -11.10 | [-13.90, -8.35] |
| decider08-native | +3.25 | [+1.20, +5.20] |
| gliclass-instruct-base-v1.0 | -11.10 | [-15.00, -7.20] |
| gliclass-instruct-edge-v1.0 | -24.65 | [-28.70, -20.75] |
| gliclass-multilang-mini | -12.95 | [-15.95, -10.15] |
| kev08-native | -1.05 | [-3.65, +1.55] |
| qwen08-frozen | -15.05 | [-17.90, -12.15] |

## Evidence and limits

[Source inventory and training distinctions](RESEARCH.md), [machine-readable comparison](comparison.json).
Published Kev/Decider/JevBench scores use different protocols and do not enter these tables.
Missing evaluations are not zero scores. GPU, CPU, and different hardware timings must remain separate.
The registered speed and quality gates still govern model selection; task wins alone do not pass them.

## H200 timings

Same fresh-state protocol; 5 warmups, 40 repetitions for one question, 15 for 50.
Kev and Decider use their native eager path. Their optimized serving implementations were not measured.
Merge/cache variants require their own parity decision; timing alone does not approve them.

| Arm | 1 question p50, ms | 50 questions p50, ms | ms/question at 50 | Clean telemetry |
|---|---:|---:|---:|---|
| c-qwen08-30k | 43.42 | 200.12 | 4.002 | True |
| c-qwen08-merged | 36.39 | 135.68 | 2.714 | True |
| gliclass-multilang-mini | 8.48 | 50.10 | 1.002 | True |
| gliclass-instruct-edge-v1.0 | 5.98 | 45.54 | 0.911 | True |
| a-base-12ep | 11.64 | 12.71 | 0.254 | True |
| c-qwen08-cacheoff | 53.66 | 248.13 | 4.963 | True |
| kev08-native | 40.16 | 78.72 | 1.574 | True |
| decider08-native | 30.53 | 82.61 | 1.652 | True |

These H200 timings do not satisfy the laptop speed gate.

![Same-trial quality and speed trade-offs](quality_speed.png)

[Download vector figure](quality_speed.svg). Only candidates with separately measured accuracy and clean timings appear.
