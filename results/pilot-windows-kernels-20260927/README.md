# Isolated Windows attention-kernel preparation

Pinned wheels for `triton-windows 3.6.0.post26`, `fla-core 0.5.2` and `einops 0.8.2`
were downloaded from PyPI, checked against their published SHA-256 hashes and installed into
`runs/pilot-windows-kernels-20260927/site-packages`. Every installed file is hashed in
[preparation.json](preparation.json). The existing pilot environment is unchanged, and the queued
competitor and restricted-projection runs continue to use it. No kernel was imported or executed
as part of staging; compilation, numerical parity and speed are untested.

The baseline uses Python 3.14.3, PyTorch 2.10.0+cu130 and Transformers 5.17.0 on an RTX 5080 Laptop
GPU. Its Qwen implementation falls back to reference PyTorch attention because the FLA package
is absent. Installed Transformers source explicitly attempts the FLA import and uses its fallback
when import fails. This locates a plausible optimization path; it is not a measured bottleneck share.

The Triton Windows project's compatibility table pairs PyTorch 2.10 with Triton 3.6 and lists
consumer Blackwell support. PyPI provides a CPython 3.14 Windows wheel for the pinned version.
[Compatibility source](https://github.com/triton-lang/triton-windows),
[wheel metadata](https://pypi.org/pypi/triton-windows/3.6.0.post26/json).
FLA separates its kernels into `fla-core`; backend dependencies can be installed explicitly.
The overlay preserves the existing PyTorch build instead of asking pip to resolve another backend.
[FLA installation instructions](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md).

Next, under the shared compute lock, verify actual optimized-function dispatch and compilation,
then compare every original teacher-validation probability against the unchanged baseline.
Use the existing parity thresholds and preserve failures. Any candidate also needs valid target-
laptop latency under the original temperature and contamination rules. Neither package installation
nor a warm diagnostic timing satisfies those requirements. No extra cloud compute was used.
