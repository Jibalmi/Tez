# Qwen graphs with unchanged native batch prefixes

The [registered candidate](../../experiments/small_model/qwen_prefix_validation_registration.json)
follows the retained [left-padding graph failure](../pilot-qwen-graph-validation-20260927/batch-16/RESULTS.md).
That earlier graph matched its identical uncaptured forward exactly, but the surrounding inference
changes failed the original reference's mean-total-variation limit. This candidate changes how inputs
fit a graph; it does not relax the reference or numerical thresholds. GPU results are pending and the
batch-16 arm is queued under the shared laptop compute lock.

Each native batch is formed by the original prompt-character-length sort, including its short final
batch. Its left-padded input IDs, masks and real-token positions remain unchanged. Extra masked tokens
are appended after the native batch, and a CUDA index buffer selects its original final-token column
for the letter readout. The graph never reads the appended padding as the answer position. Whole
batches are grouped by graph shape for capture reuse, without regrouping their individual decisions.
There are no filler or duplicated rows. Only one graph remains resident at a time.

The [CPU preflight](preflight.json) compared every native batch with the actual tokenizer output.
All 1,390 decisions retain identical prefix IDs, attention masks and final-token positions at both
registered batch sizes. Inputs are complete, with at most 1,607 tokens. Nominal batch 16 has 87
native batches: 86 with 16 decisions and one with 14. Its five occupied graph shapes are 16x256,
16x512, 16x1024, 16x2048 and 14x1024. Batch one has 1,390 real single-decision batches and five
occupied lengths from 128 through 2,048. The planned 4,096 bucket has no inputs and remains untested.

Eight CPU tests check native grouping, positions, original pad-token values, dynamic final-column
selection, incomplete input rejection, probability arithmetic, and row provenance. These tests and
the full-input preflight establish input placement, not Qwen probability equivalence or CUDA behavior.
Added trailing computation may still change floating-point kernel results.

Both arms compare all original decisions with the same frozen full-head batch-16 validation snapshot:
argmax agreement must be at least 0.99, maximum probability difference at most 0.04, and mean total
variation at most 0.005. Batch one's comparison therefore also includes changing the reference batch
size. Every captured output is also compared with an identical uncaptured forward, requiring maximum
difference at most 0.00001. Each arm must pass independently; failures remain preserved.

Original weights, unmerged LoRA, native attention, temperature one and the restricted letter head
remain frozen. No public labels or new training enter this trial. No latency is measured here.
Any later speed check must account for tokenization, padding, masks, buffer copies, owned output
storage, and capture on a shape-cache miss. Passing parity is necessary, not a promotion or speed claim.
