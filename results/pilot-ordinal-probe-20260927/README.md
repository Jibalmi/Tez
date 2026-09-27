# Ordinal cue diagnostic

Changing the question type or rotating the answers does not provide a convincing fix for numeric
rule failures. Both frozen models completed all six registered conditions: **1,728 predictions**
were validated against the original records, semantic label mappings and exact weight hashes.

The longer-run encoder scores 33.33% in the original ordering and 43.06% when the middle-band answer
is presented first. Qwen ranges from 33.33% to 34.03%. Merely changing `score` to `choice` leaves both
models at 33.33% in the original ordering. Both still overwhelmingly choose the middle semantic
answer after it moves to another position. There is some order sensitivity, especially for the
encoder, but it does not explain away the underlying content failure.

The [full table](RESULTS.md) reports every condition, semantic prediction counts, order invariance
and exploratory paired intervals. The one better encoder ordering is not selected for deployment.
These are secondary measurements on previously observed diagnostic records; they cannot serve as
fresh confirmation, training data or checkpoint selection.

All conditions use one question per group and recompute the original-order baseline in that same
protocol. Thus the score/type contrast does not accidentally compare the earlier packed encoder
evaluation with a singleton one. Questions, state facts and option descriptions remain unchanged;
only the declared type and cyclic presentation order change. Each semantic answer is correct for
exactly one third of the states. Probability vectors are validated in both presented and canonical
semantic order, including native tie handling. Gold labels never enter model inputs.

The runs finished with exit code zero, released the shared laptop compute lock, and used the
unchanged thermal governor. There was no additional RunPod spending, no fitting and no production
runtime change. The original quality and latency gates remain unmet. Elapsed time is operational
bookkeeping rather than a usable latency measurement.

This evidence supports testing independently authored rule-based supervision with a matched
continuation control and disjoint validation. Data must pass the existing state/schema leakage
guards before training. Both the original public benchmark and these diagnostics remain secondary
after informing the research direction; any promoted change still needs fresh confirmation.

The [measurement archive](measurement-artifacts.tar.gz) preserves transformed inputs, the source
corpus, raw probability vectors, execution receipts, logs and executed source snapshots. See the
[archive receipt](artifact_receipt.json) and [per-file hashes](artifact_manifest.json). The pilot
unit suite has 59 passing tests; the four new Python files pass Ruff.
