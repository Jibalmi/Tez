# Separate structured choice candidates

The [generation receipt](generation.json) records 1,200 candidate training states with 2,400
choice decisions, plus 200 validation states/400 decisions and 300 confirmation states/600
decisions. These files remain under ignored `runs/`. The [registered filter completed](filter_summary.json)
and retained all 1,200 training states and 2,400 decisions. The filtered training hash equals
the frozen candidate hash. This data is eligible for a separately registered training experiment;
it remains excluded from the already registered original-data Qwen continuation.

The earlier rule-data filter removed every choice decision because its labels overlapped the
new held-out schemas. This new generator assigns distinct domains, field names, action labels
and bin labels to each split. It still exercises priority rules and largest-number selection
with explicit tie breaks. Logical families, generic JSON structure and wording patterns recur;
the held-outs cannot establish unseen-family reasoning.

Answers follow exact rules. They are separate hard-label supervision, not Gemma probabilities
or model-generated answers. The generator reads no external examples. An independent predicate/
sorting implementation verifies the labels from visible facts. Tests cover equality, rule
priority, ties, option reversal, deterministic generation, split isolation and corrupted inputs.
The raw candidate generator and files are frozen before filtering; failed guards will be retained.

The [registered filter](../../experiments/small_model/structured_choice_filter_registration.json)
keeps the original exact/normalized hash, MinHash 0.8, choice overlap 0.5, MiniLM cosine 0.95
and domain-centroid checks. It also guards the earlier authored validation/confirmation sets,
the earlier nested-state diagnostic and the original teacher-validation inputs. Its GPU and
multicore work ran under `run_heavy.py`; no cloud resource was added. The maximum measured
embedding similarity was 0.8252, below the unchanged 0.95 cutoff. Exact/normalized hash,
MinHash, choice-label overlap and domain-centroid stages also retained all candidates.
Passing these automated guards does not establish absence of every form of semantic overlap.

The [filter integrity check](filter_verification.json) verifies the executed sources, frozen
inputs, output hash, counts and exact labels. It does not independently rerun embedding or
centroid inference. Training examples and feature caches are excluded from the evidence archive.

No training or model evaluation has used these candidates. The confirmation set is reserved
for after a subsequent registered training run and validation-only checkpoint selection.
