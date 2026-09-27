# Paired structure diagnostic

All three frozen models completed the [paired diagnostic](RESULTS.md), with 4,320 validated
predictions. The results weaken JSON nesting alone as an explanation of the original quality gap.
The stronger signal is difficulty applying numeric and compound rules, despite strong field lookup.

| Model | Flat accuracy | Nested accuracy |
|---|---:|---:|
| Encoder, original | 51.39% | 52.22% |
| Encoder, longer run | 60.56% | 65.14% |
| Qwen student | 71.25% | 69.58% |

Both encoders improved slightly or substantially on the nested representation. Qwen fell by
1.67 points, with an exploratory paired state-bootstrap interval of -2.64 to -0.83 points. That
small effect is concentrated in conjunction questions and does not establish nesting as the cause
of its public benchmark gap. The broader goal remains unmet; these are diagnostic scores only.

The [behavior audit](behavior_audit.json) found that every model selected the middle ordinal band
for all 144 states in both formats. Qwen also always answered true to the threshold question.
Thus its 66.67% threshold accuracy merely matches that task's majority-class baseline. In contrast,
Qwen scored 100% on both lookup and explicit missing-value handling. The rendered ordinal prompt
contains the correct values, boundaries and all three option descriptions; no truncation occurred.

The [registration](registration.json) freezes 144 newly authored English records across observatory,
greenhouse, ceramics and theater settings. Each has five exact-answer questions: field lookup,
numeric threshold, two-condition conjunction, ordinal band and missing-value handling. Each record
is presented as both flat and nested JSON, giving 720 paired decisions and 1,440 predictions per model.
Every leaf key, value, traversal order, instruction, option order and answer is identical across the
pair. Extra container keys and JSON punctuation necessarily differ. Both forms use five packed
questions for the encoder and the unchanged per-question prompt protocol for Qwen.

Answers come from deterministic rules, with explicit inclusive boundaries and missing/null behavior.
The generator reads no training or public benchmark files. The corpus is evaluation-only and must
never enter fitting or checkpoint selection. These are repeated synthetic rule families, not an
independent certification of broad generalization. The two Boolean tasks have unequal class counts;
raw accuracy must be interpreted alongside their majority-class baselines.

The frozen three-epoch encoder, longer-run encoder and Qwen student are registered at their exact
previously evaluated weight hashes. Primary results are nested-minus-flat accuracy and prediction
flip rate; uncertainty resamples whole states, retaining all five questions. Per-operation accuracy
distinguishes basic lookup failures from threshold, conjunction and missing-value failures. This
diagnostic does not relax or replace any original quality, calibration or laptop speed gate.

Before any inference, a preparation review found that the optional field moved in leaf traversal
order across the two forms. The unexecuted preparation is preserved locally at
`runs/pilot/structure-probe-preflight-v1/`. The final registered generator flattens the nested record
in traversal order and rejects any paired content or order difference. No model results informed
this correction.

The evidence now favors reasoning coverage over wrapper-only augmentation. Before committing to a
training intervention, the constant middle-band prediction also warrants a fixed question-type and
option-position diagnostic. That can distinguish answer-position or score-type cues from content
difficulty. Any such follow-up is secondary analysis, because these probe results have now been seen.

Any training follow-up needs disjoint schemas and validation examples, matched training budgets,
unchanged original benchmark reporting and a fresh confirmation set. This probe must not become its
selection set. The original public comparisons remain exploratory after informing research choices.

No paid resources were launched. The prior RunPod estimate remains $9.15 against the authorized
$10 combined cap. Laptop inference uses the unchanged thermal governor and shared lock. Its elapsed
time is operational bookkeeping, not a valid latency measurement or evidence that the speed gate passes.

The first Qwen attempt loaded the model but failed in the metadata audit: an offline snapshot request
also requested uncached repository documentation. The retry restricts that audit to weight files,
preserving model settings, pinned revision and input hashes. The failed attempt produced no
predictions and remains in the archive. Its parent execution record correctly remains a partial
failure; the recovery and complete model receipts are separate. The outer PowerShell retry command
reported exit code 1; the model wrote its complete receipt and all expected checksum-verified rows,
and the shared lock was released. This shell-level status is preserved rather than treated as a clean
process exit. The Qwen thermal governor paused as needed; no timing promotion is claimed.

The [measurement archive](measurement-artifacts.tar.gz) retains the registered corpus, raw rows,
executed source snapshots, logs, receipts, unsuccessful metadata audit and unexecuted first
preparation. Its [receipt](artifact_receipt.json) and [per-file hashes](artifact_manifest.json) verify
the export. Preparation had 56 passing pilot unit tests; a separate fixed-blend test also passed.
Changed Python files pass Ruff. No production runtime or trained weights were changed.
