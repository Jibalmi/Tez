# Confirmation class-coverage audit

This descriptive audit uses already completed predictions; it changes no model or selection.
The majority reference picks each operation's most frequent confirmation gold answer after the fact.
It is a class-balance check, not a deployable oracle or independent baseline fitted on training data.

| Operation | Post-hoc majority | Starting model | Rule supervision | Rule balanced accuracy |
|---|---:|---:|---:|---:|
| combined | 51.67% | 26.11% | 72.50% | 72.61% |
| compare | 50.00% | 50.00% | 60.28% | 60.28% |
| count | 33.33% | 33.61% | 54.72% | 54.72% |
| largest | 35.00% | 35.00% | 36.11% | 34.49% |
| policy | 50.00% | 33.33% | 49.72% | 44.26% |

Compare, combined and count improve beyond their majority references. These are the three operations retained in added training.
Policy remains below its 50% majority reference. Largest remains near its majority reference and heavily favors one answer.
Policy and largest were removed from added training by the choice-schema overlap guard; their weak scores do not establish a causal explanation.
Counting is still uneven across classes. The full answer histograms and class recalls are in [bias_audit.json](bias_audit.json).
The overall gain should not be described as general rule mastery or broad competitiveness.
