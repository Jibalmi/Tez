# Native GLiNER decision comparison

The [registered protocol](../../experiments/small_model/gliner2_registration.json) freezes two
released checkpoints, dependencies, source hashes, full public input membership and scoring.
Both native interfaces passed four non-benchmark probability checks before this registration.
The full evaluation first requires those checks to survive batching; any failure is retained.

Each arm covers 6,151 original decisions plus 1,100 reversed-option multilingual intent cases.
All class probabilities are recovered from native logits and checked against the native answer.
Original FP32 precision, eager attention, instructions and option descriptions are preserved.
No temperatures, prompts or weights are fitted to these evaluation labels.

Jobs use the shared local compute lock, four threads and unchanged thermal rules. The 30-minute
worker deadline bounds setup and inference; elapsed job time is not a latency benchmark.
This extends the [matched comparison](../pilot-improvement-20260927/COMPETITIVENESS.md), with
unknown competitor training exposure. The public tasks have already been observed in this
project, so they are exploratory regression checks, not new held-out confirmation. No cloud
resources or runtime changes are part of this experiment.

Both first attempts stopped during the non-benchmark smoke check, before any public prediction.
The guard required byte-identical native text, but GLiNER's pinned `_collate_batch` appends a
period when a text does not end in `.`, `!` or `?`. The original contents remain intact. This
is an audit compatibility failure, not a measured model quality failure. Receipts, logs and
executed sources are preserved. A separately registered retry will permit only that exact
native punctuation normalization and will continue to reject dropped words and altered labels.
