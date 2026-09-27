# Additional Qwen kernel options

This is a primary-source metadata audit, not an installation or performance result.
[Recorded metadata](discovery.json) and the [dependency follow-up](dependency_check.json)
include request URLs, content hashes and wheel-tag checks for this Python 3.14 Windows host.

[FlashQLA at the inspected revision](https://github.com/QwenLM/FlashQLA/tree/cdfcb99061daa61d78aa07bb254a6cf9ddfa8450)
lists SM120 forward support and provides a Gated Delta Rule API. Its source setup pins TileLang
0.1.12. The PyPI FlashQLA 0.1.2 package instead pins TileLang 0.1.9 and apache-tvm-ffi 0.1.9.
These are distinct dependency combinations despite the same base version number; substituting
one for the other needs a separate pinned preparation.

The wheel metadata has no TileLang 0.1.9 build compatible with this host's tags. TileLang 0.1.12
does have a compatible Windows ABI3 wheel, as do apache-tvm-ffi 0.1.9 and 0.1.11. Compatible
wheel tags establish only packaging eligibility, not successful loading, compiler availability
or correct CUDA execution. The [TileLang installation documentation](https://github.com/tile-ai/tilelang/blob/main/docs/get_started/Installation.md)
also describes Windows source builds. Do not infer that Windows is categorically unsupported.

The [Transformers 5.17 Qwen documentation](https://huggingface.co/docs/transformers/v5.17.0/model_doc/qwen3_5)
describes an Atlas GDN Hub-kernel route gated to SM121. The installed `hub_kernels.py` also
sets both its minimum and maximum capability to 121. This is not the default automatic route
for the RTX 5080 Laptop's SM120 hardware. Its published GB10 performance is not a local result.

The current priorities remain completion of Tev1's matched comparison, the already queued CUDA
graph compatibility check, and the [FLA precision probe](../pilot-fla-precision-probe-20260927/README.md).
A future FlashQLA test should use a separate pinned source/dependency overlay, then verify native
dispatch, complete validation probabilities and the original laptop latency conditions. No
packages were installed, no model weights loaded and no paid resources created in this audit.
