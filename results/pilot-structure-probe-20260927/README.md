# Paired structure diagnostic

The original training data contains no nested records. This follow-up tests whether the frozen
students are sensitive to serialization before committing to another training run. Inference is
queued under the laptop's shared compute lock; no model result is claimed yet.

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

The next training intervention depends on the results:

- A material nesting penalty supports testing wrapper augmentation on existing training states,
  preserving the existing teacher targets and checking meaning before transformation.
- Low accuracy in both forms supports adding independently authored multi-field reasoning examples;
  changing JSON wrappers alone would not address that evidence.
- Strong performance in both forms would weaken this probe as an explanation of the public benchmark
  gap and require a different diagnosis.

Any training follow-up needs disjoint schemas and validation examples, matched training budgets,
unchanged original benchmark reporting and a fresh confirmation set. This probe must not become its
selection set. The original public comparisons remain exploratory after informing research choices.

No paid resources were launched. The prior RunPod estimate remains $9.15 against the authorized
$10 combined cap. Laptop inference uses the unchanged thermal governor and shared lock. Its elapsed
time is operational bookkeeping, not a valid latency measurement or evidence that the speed gate passes.

Preparation checks: 56 pilot unit tests passed, and all four new Python files passed Ruff.
