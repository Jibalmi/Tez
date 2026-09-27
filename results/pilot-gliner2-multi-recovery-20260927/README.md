# Recovery of the interrupted multilingual comparison

The earlier session handle is missing and a scoped OS process check found no corresponding
worker. Its [interruption record](../pilot-gliner2-eval-attempt2-20260927/interruption_multilingual.json)
preserves that evidence; the exact interruption cause is unrecorded. Its last-written running
receipts and 3,327 partial predictions are archived, not treated as a complete benchmark.

This [new registration](../../experiments/small_model/gliner2_registration_multi_recovery.json)
repeats all original cases in a new directory, retaining the same checkpoint, FP32 precision,
batch size four, native rendering, probability checks and 30-minute worker bound. It does not
splice partial results or overwrite earlier evidence. The shared compute lock and thermal
rules remain unchanged. No cloud spending or runtime changes are involved.
