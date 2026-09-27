# GLiNER comparison: native punctuation audit correction

The [first attempts](../pilot-gliner2-eval-20260927/README.md) both terminated during the
four-case non-benchmark smoke check. Their artifacts retain the failure; no public case was
scored. The input guard mistook GLiNER's native appended period for a changed source text.

This [registered retry](../../experiments/small_model/gliner2_registration_attempt2.json)
accepts only the pinned processor's exact terminal punctuation rule before checking the full
word sequence. It still checks every label, option description and instruction. The processor
now raises on malformed input instead of returning fallback dummy records, which the audit
would also reject. Original cases, native rendering, weights, FP32 precision, batch size four
and all smoke tolerances are unchanged. These are guard corrections, not fitted model changes.

Both arms must repeat the frozen four-case batch smoke before the full 6,151 primary decisions
and 1,100 reversed-option intent checks. Local compute uses the shared lock, four threads and
unchanged thermal rules, with a 30-minute worker bound. No latency or release claim follows
from job elapsed time. No new cloud spending is involved.
