# Qwen CUDA graph compatibility probe

The first execution failed before inference with `IncompleteSnapshotError`: the pinned
cached snapshot lacked `.gitattributes`, `LICENSE` and `README.md`. It produced no probability
rows and made no thermal/GPU inference calls. This does not establish graph incompatibility.
The failure, log and executed source remain here. The separate
[metadata repair](../pilot-qwen-cache-metadata-20260927/repair.json) verifies the three downloaded
Git objects, an intact offline snapshot and all ten original files unchanged. A retry requires
its own output directory and registration; original probability limits remain unchanged.

The [registered probe](../../experiments/small_model/qwen_cuda_graph_registration.json) tests
four existing non-benchmark interface cases in original, reversed and restored batch order.
It retains the frozen base, unmerged adapter and attention kernels. It uses the previously
validated restricted output projection, explicitly disables the unused output cache, and builds
native masks before capture. Static buffers are updated for each replay.

[PyTorch 2.10 guidance](https://docs.pytorch.org/docs/2.10/notes/cuda.html#cuda-graphs) requires
fixed shapes and stable memory addresses, prohibits CPU/GPU synchronization inside capture,
and calls for warmup on a side stream. The pinned Transformers recurrent-mask helper performs
a tensor-value check; preparing that mask outside capture avoids that particular synchronization.
Other capture compatibility remains unproven until this probe executes.

Before execution, the resource profile was tightened after a host reboot: process-local
temperature checks at 80/76 C and a shared-wrapper ceiling of 84 C. The original unexecuted
plan is preserved in `pre_reboot_plan/`. No model, probability threshold or quality gate changed.

A pass is only permission to investigate broader parity and then speed. It does not establish
the full 1,390-decision validation comparison, a speedup, or any release gate. End-to-end timing
must include tokenization, mask construction and buffer copies. The original four-case full
readout is retained alongside every candidate vector; captured and identical uncaptured paths
must also agree within the fixed tolerance.
