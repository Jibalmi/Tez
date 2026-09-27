# Isolated FLA parity: compiler discovery correction

The [first attempt](../pilot-fla-qwen-20260927/README.md) selected the FLA implementation but
stopped on its first CUDA call because Triton's C-compiler discovery searched the baseline
environment. Both the compiler and CUDA support files already exist in the verified overlay.

The [amended registration](../../experiments/small_model/windows_kernel_registration_attempt2.json)
sets `CC` and `CUDA_PATH` only in the worker process, pointing to those existing files. The
compiler hash is checked before launch and again in the worker. The package overlay, Qwen
checkpoint, full output projection, unmerged adapter, prompts, batch size, cache configuration,
1,390 validation decisions and all three parity thresholds remain unchanged.

Execution retains the shared compute lock, four threads, original thermal rules and 12-minute
worker deadline. This fixes a diagnosed terminal failure; it does not restart a live job or
overwrite its evidence. No additional cloud resource or speed claim is involved.
