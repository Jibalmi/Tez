# Descriptive analysis of the prefix-graph probability drift

The analysis reread and verified every member of both completed graph evidence archives.
Their 1,390 input decisions and frozen batch-16 reference predictions are identical. The
recomputed full-set metrics match both independent verification reports. All subgroup results
are post-hoc descriptions; they neither change the acceptance test nor identify a cause.

| Question type | Decisions | Batch-16 graph mean TV | Batch-one graph mean TV |
|---|---:|---:|---:|
| Choice | 526 | 0.00483942 | 0.00834300 |
| Yes/no | 491 | 0.00199119 | 0.00363365 |
| Score | 373 | 0.00562316 | 0.00821754 |
| All | 1,390 | 0.00404363 | 0.00664581 |

Every value compares with the same frozen native full-head batch-16 reference. The complete
batch-one graph still fails the unchanged 0.005 mean-TV limit. There are eight changed argmax
predictions in that comparison, versus four for the batch-16 graph. Comparing the two graph
candidates directly gives mean TV 0.00677305 and ten changed argmax predictions. This does not
tell us which prediction is more accurate; there is no new gold-quality measurement here.

Drift is present in every input-length group. The 92 decisions with **no left padding** in
their native batch-16 input have batch-one mean TV 0.00801130, versus 0.00664581 overall.
Left-padding amount alone therefore does not explain which examples drift. Different batch
sizes, padded graph shapes, masks and floating-point kernel choices are still entangled.
The separately [registered native batch-one control](../pilot-qwen-native-single-20260927/README.md)
remains necessary; this analysis does not substitute for it.

[PyTorch 2.10's numerical-accuracy documentation](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html)
explains that batched and individual computations can differ numerically. It also documents
the BF16 reduced-precision accumulation setting. A separate
[read-only fresh-process check](fresh_process_precision_defaults.json) found that setting
enabled in this installed build, with no CUDA context initialized. That is a possible variable
for a later controlled experiment, not evidence of the actual kernel used in earlier workers,
the cause of this failure, or a fix. No live worker or precision setting was changed.

The [raw per-decision distances](drift_rows.jsonl) and [all 21 descriptive groups](diagnostics.json)
are retained. The [161,745-byte archive](artifact_receipt.json), SHA-256
`71b738e03ab1cc649a5656bfccc67b46be73b6709b2eed987bd1fa45e857ac25`,
contains 12 verified members, including the analysis source and original verification receipts.
The full probability inputs remain in the two separately verified source archives. This README
and the supplemental fresh-process defaults check were added after the immutable drift archive.
Analysis used one CPU thread and no model inference. Ruff passed, and aggregate arithmetic was
checked against both existing independent reports. There is no new speed or release claim.
