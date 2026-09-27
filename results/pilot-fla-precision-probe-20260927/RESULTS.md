# FLA precision probe result

All 24 probability vectors were independently checked across three arms and two orders of four non-benchmark inputs.
The reader recomputed the numerical checks and verified registered source hashes and 36 attention calls per arm.

| Arm versus Torch fallback | Argmax agreement | Maximum probability change | Mean total variation | Passed |
|---|---:|---:|---:|---|
| fla_native | 1.000000 | 0.00761181 | 0.00297663 | True |
| fla_fp32_inputs | 1.000000 | 0.01399583 | 0.00632615 | False |

Unchanged limits: agreement at least 0.99, maximum change at most 0.04, mean TV at most 0.005.
FP32 inputs did not resolve drift and this candidate does not advance. FP32 dtype is not IEEE-only kernel arithmetic.
Native FLA passing these four cases does not replace its [failed full validation](../pilot-fla-qwen-attempt2-20260927/RESULTS.md).
The cause of the broader drift is not isolated by this experiment. No latency or quality improvement was measured.
[Full analysis](analysis.json), [raw vectors and dispatch receipt](probe.json), [registration](registration.json).
