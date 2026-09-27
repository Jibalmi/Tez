# Tev1 paired typed-decision comparison

This supplements the completed native comparison. All five decisions for each of the 400 states stay together in resampling.
Intervals use 2,000 paired state resamples and seed 13, matching the earlier analysis method.
They are exploratory, without multiplicity correction; training exposures differ.

| Tev1 minus reference | Difference, percentage points | 95% state-bootstrap interval |
|---|---:|---:|
| qwen | +1.30 | [-1.30, +3.95] |
| decider | -1.95 | [-4.30, +0.35] |

No thresholds, prompts, checkpoint choices or model routing are fitted from these cases.
[Complete native evaluation](../pilot-tev1-eval-20260927/RESULTS.md), [paired results](paired.json).
