# Complete English GLiNER result

All 6,151 original decisions and 1,100 reversed-option intent decisions completed. Independent
verification checked case membership, full probability vectors, native winner/confidence,
option logits, task metrics, SemIf evaluator output, source/model hashes and archive contents.

Pinned fastino/GLiNER2.5-Decide, revision `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6`,
native FP32 eager implementation, batch one, RTX 5080 Laptop GPU. No fitting or truncation.

| Matched task | Accuracy |
|---|---:|
| Typed decisions | 40.30% |
| Intent, mean of 11 languages | 41.55% |
| Routing | 60.90% |
| Jailbreak | 76.00% |
| Toxicity | 45.75% |

Raw typed ECE is 0.1351 and soft accuracy 0.3649. Official SemIf mean family balanced accuracy
is 0.6783 on authored144 and 0.5492 on perturbations108. All tasks and semantic app/intent F1
are retained in [validation_english.json](validation_english.json).

This English checkpoint trails our Qwen student on the five public accuracy columns above.
Its multilingual result is not a measurement of the separate multilingual checkpoint.
Training exposures differ and these already-observed public tasks are exploratory comparisons.
Option-index F1 is diagnostic only. Execution elapsed time includes thermal pauses and is not
a latency result; no release gate or runtime changed.

[Archive receipt](artifact_receipt_english.json), [frozen registration](../../experiments/small_model/gliner2_registration_serial.json),
and [broader comparison](../pilot-improvement-20260927/COMPETITIVENESS.md).
