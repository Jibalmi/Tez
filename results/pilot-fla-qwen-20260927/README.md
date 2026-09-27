# Isolated FLA attention parity

The first attempt failed before producing candidate probabilities: Triton could not find a C
compiler. The bundled TinyCC exists in the overlay, but the default discovery uses the baseline
environment's `sysconfig` package path. FLA dispatch was selected and invoked; kernel compilation
failed. This is an incomplete parity test, not a failed numerical-parity verdict.

A [second attempt](../pilot-fla-qwen-attempt2-20260927/README.md) explicitly selects the existing,
hash-verified compiler and CUDA bundle through child-process environment variables. No baseline
packages, trained weights, inputs or probability thresholds change. The original failure is retained.

This [registered experiment](../../experiments/small_model/windows_kernel_registration.json)
compares the original Qwen student with the same checkpoint using the staged Windows FLA kernels.
It retains the full output projection, unmerged LoRA, prompts, option order, batch size 16, cache
and raw temperature. It does not combine the restricted letter projection with this change.

The baseline is the completed laptop reference from the
[restricted-projection trial](../pilot-restricted-qwen-20260927/RESULTS.md), pinned by receipt and
raw-probability hashes. Every one of its 1,390 validation decisions is checked, with the same
99% minimum argmax agreement, 0.04 maximum probability delta and 0.005 maximum mean total variation.
No thresholds or calibration are fitted to the result.

Pinned packages run only in a child process through an isolated import overlay. The original
environment stays unchanged. The worker verifies all staged files, requires FLA function dispatch,
counts actual CUDA calls, and refuses a silent reference fallback. Compilation and inference run
under the shared compute lock with four threads, the original thermal rules and a 12-minute limit.
The existing teacher-data, checkpoint and inference-source hashes must match the reference.

This checks numerical parity and compatibility. It does not measure valid serving latency or
change the Tez runtime. Promotion also requires improved target-laptop latency under the original
temperature and competing-process limits. Failures are retained. No cloud resource is created.
