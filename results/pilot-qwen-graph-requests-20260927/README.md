# Complete request graph-cache check

Implementation and nine focused CPU tests are ready. **This new cache has not run on the GPU,
and its protocol is not registered yet.** Registration requires independently verified passes
for both full native-prefix graph arms. Batch 16 passed; the completed
[single-question arm failed](../pilot-qwen-prefix-single-20260927/README.md) its unchanged
mean-total-variation limit. The proposed cache therefore remains unregistered and unexecuted.
The input audit and implementation are retained without weakening that requirement.

The proposed check covers the complete existing fresh-state timing workload: 1, 5, 10 and
50 questions, ten planned warmup calls per size, 100 subsequent calls at each smaller size
and 30 at size 50. The [one-thread CPU preflight](preflight.json) has now audited all 370
requests and 3,760 decision inputs without loading a GPU model. Every prompt and token
sequence has a saved content hash. The longest input is 258 tokens and none is truncated.
The six occupied shapes are 1x256, 2x512, 5x512, 10x512, 16x256 and 16x512.
These are verified input counts and shapes, not predictions or latency measurements.

The reference uses the unchanged full output head and native batch-16 inference. The candidate
keeps one CUDA graph and private memory pool for each audited shape, replaying sequentially.
It preserves native character-length sorting and real batch sizes; 50 questions produce batches
of 16, 16, 16 and 2. There are no repeated filler questions, cross-request prompt caches,
shared graph pools or unregistered shape captures. Every call builds its prompts, tokenizes,
constructs masks, transfers buffers and copies outputs to owned CPU memory. All returned
vectors remain alive until graph release so later replays cannot silently overwrite them.

Each of the four request sizes must independently meet the unchanged full-reference limits:
argmax agreement at least 0.99, maximum probability change at most 0.04 and mean total variation
at most 0.005. Every captured forward also has an identical uncaptured comparison, requiring
maximum change at most 0.00001. Passing individual graph shapes cannot substitute for this
multi-graph lifetime and complete-request check.

The independent reader rejects incomplete requests, changed input hashes, wrong native
membership, lost graph residency, incorrect replay counts and malformed probabilities. Its
CPU tests and the batch-order tests pass. They do not establish GPU correctness or speed.
The proposed GPU worker retains the shared lock, four threads, 80/76 C pacing, 84 C outer
ceiling, 18-minute soft deadline and 20-minute hard child timeout under a 22-minute lease.
Failures and partial results remain evidence. No paid resource is used.

This check does not measure latency. A subsequent timing protocol must separate model load,
first graph captures and warm replay, include all request processing, and use actual GPU
identity plus the original temperature and memory-eligibility rules. Current desktop GPU
allocations exceed the [clean-timing limit](../pilot-windows-gpu-telemetry-20260927/README.md).
No live/pinned graph worker or previous result is changed by this preparation.
