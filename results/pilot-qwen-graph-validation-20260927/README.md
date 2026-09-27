# Full validation of length-bucketed Qwen CUDA graphs

The [registered experiment](../../experiments/small_model/qwen_graph_validation_registration.json)
extends the successful [four-case compatibility check](../pilot-qwen-cuda-graph-attempt2-20260927/RESULTS.md)
to all original teacher-validation inputs. The [batch-16 result](batch-16/RESULTS.md) is complete
and independently verified: it fails the unchanged mean-total-variation limit. The batch-one arm
is queued separately under the shared laptop compute lock. No speed result is established.

Across all 1,390 decisions, batch 16 has 99.5683% reference argmax agreement and maximum probability
change 0.03596536, both within their limits. Mean total variation is 0.00623822, exceeding 0.005.
The graph and the identical uncaptured path produce exactly equal saved probabilities on every
decision. Thus the observed discrepancy already exists outside graph capture; this experiment
does not isolate which combination of padding, regrouping or other surrounding inference changes
caused it. Five captures and 89 replays were verified. The complete failure is retained in the
[1,324,962-byte evidence archive](batch-16/artifact_receipt.json), with 37 verified files.

The failed arm will not advance to timing. The next candidate should retain the original batch
composition and real-token positions, add trailing padding only after each native batch, and read
the last real token rather than the padded end. That requires a separate tested implementation
and registration; it is not a demonstrated repair. Probability limits remain unchanged. The
existing batch-one arm continues with its original frozen source and protocol.

The one-thread [input preflight](preflight.json) verified all 278 states and 1,390 decisions
against the frozen reference membership, with complete prompts up to 1,607 tokens. The candidate
left-pads each input to the first strictly larger length bucket. This ensures the native attention
masks remain explicit tensors. The registered batch sizes are one and 16, each evaluated separately.
At batch 16, the plan has 89 batches and 34 duplicate filler rows; only the 1,390 real outputs count.
The five occupied lengths are 128, 256, 512, 1,024 and 2,048. A 4,096 bucket is supported by the
plan but receives no inputs here and is not validated by this dataset.

One captured graph owns fixed input, mask and option buffers for one batch shape. Subsequent
questions update those buffers and replay the graph. Only one graph is resident at a time;
changing the length destroys it before capturing the next. This follows the fixed-shape and
memory-lifetime requirements in the [PyTorch 2.10 CUDA graph guidance](https://docs.pytorch.org/docs/2.10/notes/cuda.html#cuda-graphs).
It does not establish concurrent graph-cache residency or arbitrary request-order performance.

The original base and unmerged adapter stay frozen. Native SDPA and recurrence remain in use,
with cache disabled and the previously validated 26-letter output projection. Every real graph
prediction is compared with the saved full-head reference and an identical uncaptured forward.
Each batch-size arm independently requires argmax agreement at least 0.99, maximum probability
change at most 0.04, mean total variation at most 0.005, and captured versus uncaptured maximum
change at most 0.00001. The independent analysis is pinned before execution and retains either
a passing or failing completed result. No limits will be relaxed after seeing the result.

CPU tests cover bucket boundaries, complete-input preservation, final-row duplication, varying
option counts, and real tokens equal to the padding ID. Analysis tests reject changed probabilities,
option masking, input provenance, missing predictions and duplicate outputs. They do not exercise
CUDA or prove model parity. GPU workers retain raw probabilities, actual captures and replays,
source hashes, package versions, memory and thermal receipts. Each arm has a separate bounded
lock lease; no paid resource is created.

Any later latency experiment must include tokenization, padding, native mask creation, buffer
updates, output copying and capture for cache misses. Operational run durations from this parity
test cannot establish a speedup, public accuracy, a laptop gate pass or runtime readiness.
