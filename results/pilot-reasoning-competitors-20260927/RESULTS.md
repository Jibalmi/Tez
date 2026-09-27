# Frozen competitors on the authored rule test

360 states and 1,800 decisions per model. Same facts, instructions, options and gold labels.
Native model rendering; no truncation, tuning or calibration fitting. All probability rows are validated.
The competitor addition was registered after the encoder confirmation result was observed.

| Model | Overall | Compare | Combined | Policy | Count | Largest |
|---|---:|---:|---:|---:|---:|---:|
| a-base-12ep | 35.61 | 50.00 | 26.11 | 33.33 | 33.61 | 35.00 |
| a-base-replay150 | 35.61 | 50.00 | 26.11 | 33.33 | 33.61 | 35.00 |
| a-base-rules150 | 54.67 | 60.28 | 72.50 | 49.72 | 54.72 | 36.11 |
| c-qwen08-30k | 70.50 | 69.72 | 51.11 | 83.06 | 66.67 | 81.94 |
| decider08-native | 76.33 | 61.11 | 73.89 | 83.33 | 65.56 | 97.78 |

Values are percentages. Original encoder confirmations are reused, with their checksums verified.
The rule-supervision model trained on independently authored examples of compare, combined and count.
These logic families recur; the domains and states are fresh. Choice-schema overlap filtering removed policy/largest from added training.
Different training exposure means this is a utility comparison, not a controlled architecture experiment.

| Model minus rule-supervision encoder | Difference, points | 95% state-bootstrap interval |
|---|---:|---:|
| a-base-12ep | -19.06 | [-21.67, -16.44] |
| a-base-replay150 | -19.06 | [-21.67, -16.44] |
| c-qwen08-30k | +15.83 | [+12.72, +18.94] |
| decider08-native | +21.67 | [+19.39, +23.83] |

Exploratory intervals use 2,000 state resamples, seed 13, without multiplicity correction.
Decider retains native independent questions, isolated score levels and four-decimal API rounding; Qwen retains its existing Tez letter prompt.
These are quality measurements, not valid laptop speed measurements. No extra cloud compute was used.
[Original public regression results](../pilot-reasoning-round2-20260927/RESULTS.md) remain separate and continue to govern the gates.
