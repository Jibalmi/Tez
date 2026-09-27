# Rule-supervision continuation results

All three frozen models completed the authored confirmation and original public suites.
Raw probabilities, aligned labels and all 23,853 primary predictions were independently validated.

## Fresh authored confirmation

360 states, 1,800 decisions per model; two new domains, recurring logic families.
Selection used separate validation data and completed before these labels were opened.

| Model | Selected step | Accuracy |
|---|---:|---:|
| a-base-12ep | starting model | 35.61% |
| a-base-replay150 | 0 | 35.61% |
| a-base-rules150 | 150 | 54.67% |

Rules minus a-base-12ep: +19.06 points, state-bootstrap 95% interval [+16.44, +21.67].

Rules minus a-base-replay150: +19.06 points, state-bootstrap 95% interval [+16.44, +21.67].

| Confirmation subset | Starting model | Replay | Rule supervision |
|---|---:|---:|---:|
| compare | 50.00% | 50.00% | 60.28% |
| music lending | 35.11% | 35.11% | 55.44% |
| combined | 26.11% | 26.11% | 72.50% |
| policy | 33.33% | 33.33% | 49.72% |
| count | 33.61% | 33.61% | 54.72% |
| largest | 35.00% | 35.00% | 36.11% |
| community swimming | 36.11% | 36.11% | 53.89% |

## Matched public comparisons

Same frozen decisions; raw T=1. Continuations were evaluated on the same laptop.
Competitor quality comes from the earlier pinned H200 runs. Hardware timings are not pooled.

| Model | Typed | MASSIVE 11 | Routing | Jailbreak | Toxicity | Typed raw ECE |
|---|---:|---:|---:|---:|---:|---:|
| a-base-12ep | 39.05 | 51.73 | 58.65 | 85.50 | 53.75 | 12.83 |
| a-base-replay150 | 39.05 | 51.73 | 58.65 | 85.50 | 53.75 | 12.83 |
| a-base-rules150 | 38.80 | 53.18 | 56.89 | 85.50 | 53.50 | 14.00 |
| tez-12b | 70.40 | 88.45 | 96.99 | 86.50 | 71.25 | 26.21 |
| laya-td | 76.65 | 34.55 | 65.91 | 76.25 | 53.00 | 21.37 |
| decider08-native | 53.40 | 74.45 | 84.46 | 92.50 | 75.25 | 13.39 |
| c-qwen08-30k | 50.15 | 64.09 | 89.22 | 85.75 | 60.25 | 6.12 |
| kev08-native | 49.10 | 61.27 | 84.96 | 89.00 | 64.25 | 14.73 |

Laya-td trained on this benchmark's training split; these Tez models excluded the whole dataset.
Other released models' training overlap is not established. These distinctions prevent a blanket ranking claim.

Rules minus a-base-12ep, typed: -0.25 points, exploratory state-bootstrap interval [-1.00, +0.45].
Rules minus a-base-replay150, typed: -0.25 points, exploratory state-bootstrap interval [-1.00, +0.45].

## Public regressions and gate checks

| Model | Typed soft accuracy | Packing delta, points | Khmer | SemIf authored | SemIf perturbations |
|---|---:|---:|---:|---:|---:|
| a-base-12ep | 34.73 | +0.15 | 29.00 | 49.52 | 48.34 |
| a-base-replay150 | 34.73 | +0.15 | 29.00 | 49.52 | 48.34 |
| a-base-rules150 | 34.76 | -0.20 | 28.00 | 45.25 | 46.81 |

Every task, language, workflow, option-order check and probability file is retained in the evaluation reports and artifact bundle.
Packing and reversed-order figures are evaluator summaries; primary rows, not those extra diagnostic predictions, were independently audited.
The G1 typed target remains 56.3%, soft accuracy 46%, and MASSIVE 70%. Passing these alone would not pass G1.
No valid target-laptop latency is claimed. The unchanged thermal governor was used for quality work.
No model is promoted to the runtime by this report. Fresh confirmation cannot replace the original gates.
The inherited public scorer also computes two-fold evaluation-label temperature diagnostics (`ece_refit`, `nll_refit`).
Those values are excluded from comparisons and selection; all primary predictions remain T=1.
This qualifies the driver's overly broad 'no calibration fitting' scope text: no fit changes the measured primary model.

One seed, 150 optimizer steps per arm; no convergence claim. The intervention changes content and exact-label supervision together.
The replay model selected step zero. Its saved head archive has a different file hash; prediction parity is measured in summary.json.
All cloud Pods remain terminated; this round used the laptop and added no RunPod charge.
