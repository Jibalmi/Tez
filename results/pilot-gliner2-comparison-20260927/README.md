# Complete GLiNER comparison assembly

The [assembly registration](../../experiments/small_model/gliner2_comparison_registration.json)
pins the separately registered English batch-one run and multilingual batch-four recovery.
Both must finish all 6,151 primary and 1,100 reversed-option decisions before the report can
be generated. Incomplete and interrupted runs remain in their own archived directories.
Run `experiments/small_model/report_gliner2.py --attempt 4` only after both complete.

The original worker's `macro_f1` aggregates option indices. Those indices are not consistent
semantic classes when each record has a different option order, as in MASSIVE. The analysis
retains that value explicitly as `option_index_macro_f1_diagnostic`, excluded from selection
and gates. Supplemental `semantic_macro_f1` for app and intent tasks uses actual option keys.
Typed tasks mix label meanings, and SemIf retains its pinned official per-family evaluator.
Accuracy, probability calibration, model predictions and all release criteria are unchanged.

The metric clarification was made after partial English and multilingual task results had been
seen. It is an analysis correction, not a prospective new quality claim. These public tasks
remain exploratory regression comparisons with unknown competitor training overlap.
