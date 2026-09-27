# FLA precision compatibility probe

The [registered probe](../../experiments/small_model/fla_precision_registration.json) compares
three attention paths in the same frozen Qwen model: the Torch fallback, native FLA and FLA
with FP32 inputs. It uses four existing non-benchmark inputs in two batch orders. It is queued
under the shared laptop compute lock; no result or speed improvement has been measured yet.

The previous [full FLA validation](../pilot-fla-qwen-attempt2-20260927/RESULTS.md) failed the
unchanged probability limits. Inspection of the pinned local sources found that the Torch
fallback converts query/key tensors to FP32 before normalization, whereas FLA's normalization
output defaults to the input dtype. This is a possible source of drift, not an established
explanation. Promoting FLA inputs also changes later arithmetic; it does not isolate normalization.
The outer BF16 autocast remains active on the Torch reference, so its FP32 tensor conversions
alone do not establish that all matrix multiplication executes in FP32.

The candidate casts attention inputs to FP32, disables autocast inside that operator and casts
its output back to the incoming query dtype. The rest of the model, its full vocabulary head,
unmerged adapter, prompts and cache setting stay fixed. Convolution remains the Torch reference.
The isolated FLA backend is forced to its native implementation for all FLA arms. The test does
not edit installed dependencies or shared inference sources.

FP32 inputs do not imply IEEE-only arithmetic. The pinned FLA triangular solve explicitly uses
TF32 on supported hardware, and other Triton dot products may use the compiler default. The
candidate therefore still needs empirical probability verification. No environment-variable
precision shortcut is assumed. A passing smoke test must be followed by all 1,390 original
validation decisions before a separately valid laptop speed measurement.

The worker has an eight-minute bound including compilation, four threads, 80/76 C thermal
pacing and the wrapper's 84 C ceiling. It creates no paid resources and cannot satisfy a release
gate. CPU-only tests check malformed probabilities and each independent numerical acceptance limit.
