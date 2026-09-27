# Step-50 validation diagnostics

This descriptive analysis rereads both immutable validation archives and recomputes their
registered full-set metrics before grouping the same decisions. It does not change the
training schedule, checkpoint selector, temperatures, data weights or release gates.

| Question type | Decisions | Step-0 teacher agreement | Step-50 teacher agreement | Step-0 mean KL | Step-50 mean KL |
|---|---:|---:|---:|---:|---:|
| Choice | 526 | 76.0456% | 75.2852% | 0.21826211 | 0.22682443 |
| Yes/no | 491 | 93.0754% | 91.8534% | 0.04712936 | 0.04891335 |
| Score | 373 | 64.6113% | 64.0751% | 0.15239856 | 0.16618550 |

Teacher agreement and mean KL worsen in all three main types. This descriptive pattern does
not establish a cause or statistical significance. The six choice decisions with more than
ten options have unchanged teacher agreement and lower KL; that subset is too small to
establish a broad improvement or compensate for the complete-set regression.

Mean maximum probability over all 1,390 decisions declines from 0.66964960 to 0.66735400,
while mean entropy rises from 0.86823457 to 0.88098740. The observed regression therefore does
not accompany a broad increase in confidence. This does not isolate whether optimization,
sampling, supervision, or another factor caused it.

Only 21 validation decisions have original gold labels, all in the choice group. Accuracy
on those changes from 14/21 to 13/21. Yes/no and score rows have no original gold labels here.
Teacher agreement and divergence must not be presented as broad human-label task accuracy;
the separately registered public comparison remains necessary.

[Raw diagnostics](diagnostics.json) retain exact metrics, counts, confidence and entropy by
type and option-count group. The [529,857-byte archive](artifact_receipt.json) contains 13
verified files, including both complete validation prediction sets, their verification and
artifact receipts, the registration, and the executed analysis source. No model or GPU was
loaded. This README was added after the immutable archive.
