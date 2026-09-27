# Exact-rule supervision preparation

The existing leakage checks retained **1,800 authored training states and 5,400 decisions**.
Generation produced 9,000 candidate decisions; all 3,600 choice questions were removed because their
option-label sets overlap the new held-out schemas. This exception was not waived. The remaining
questions cover numeric comparison, conjunction/negation and counting satisfied conditions.

No states were removed by the exact/normalized hashes, MinHash candidate screen, MiniLM similarity
or domain-centroid checks. The maximum measured cosine against the evaluation and new held-out
states was 0.72307, below the original 0.95 cutoff. The current embedder reproduced the cached
guard's anchor vectors with minimum cosine 0.9999995. MinHash uses the same approximate candidate
search as the existing pipeline; its reported maximum is among returned candidates, not an exhaustive
all-pairs bound. Passing these checks does not prove absence of all semantic overlap.

The generator reads no benchmark, diagnostic or teacher data. It authors deterministic states and
derives exact labels from explicit rules. These labels are not fabricated model probabilities.
Validation contains 240 separate states/1,200 decisions; confirmation contains 360/1,800. Domains
and domain-specific field names differ across splits, but logical operations recur. The generic
ranking schema and choice labels also recur in the raw candidate files; the filter removes those
schemas from the added training examples. Do not claim unseen logic families.

The [raw generation manifest](raw_generation_manifest.json) records candidate hashes and counts;
its `unfiltered_do_not_train` status refers to the original candidate files. The separate
[completed filter summary](filter_summary.json) identifies the approved `train.filtered.json` and
the full local receipt. Training requires that receipt and checks the exact filtered-file hash.
Raw training examples, embedding caches and model weights remain under ignored `runs/` paths.

The [registered continuation](../../experiments/small_model/reasoning_round2.json) compares replay
with this added supervision using fixed optimizer settings, equal added decision counts and the
same step budget. Checkpoint selection uses the original and new validation sets with explicit
original-task retention limits. Confirmation stays closed until both arms finish selection.
No original public or diagnostic labels select a checkpoint, and the original release gates remain.

Generation and rule semantics have two passing unit tests. Filtering completed under the shared
laptop compute lock and unchanged thermal governor. No RunPod resource or spend was added.
