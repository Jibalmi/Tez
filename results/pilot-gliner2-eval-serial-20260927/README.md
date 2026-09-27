# GLiNER comparison with bounded English batch memory

**Completed and independently verified:** all 6,151 primary decisions and 1,100 reversed-option
intent decisions. [Results](RESULTS.md), [validation](validation_english.json), and
[verified artifact receipt](artifact_receipt_english.json) preserve the complete serial arm.
The multilingual recovery and final combined report remain separate.

The previous English run passed its probability smoke and completed four tasks, then slowed
on longer emails while total GPU memory was repeatedly near 15,943 of 16,303 MiB. Its owned
worker processes were explicitly terminated and their absence verified; the driver recorded
terminal exit 15. [The cancellation record](../pilot-gliner2-eval-attempt2-20260927/cancellation_english.json)
retains process identity, telemetry and progress. This was a resource-driven stop, not an
observation timeout or a failed probability check. Partial predictions remain preserved.

The [new registration](../../experiments/small_model/gliner2_registration_serial.json) changes
only English batch size from four to one. It repeats every case in original FP32 with the same
native model, input rendering, full probability readout and fixed smoke tolerances. No partial
predictions are spliced into the new run. The separate driver preserves the already queued
multilingual source unchanged; that arm retains its original batch size four and registration.

The final report requires both complete arms, validates each against its own frozen sources,
and compares all 6,151 primary decisions plus 1,100 reversed-option intent cases per arm. It
cannot use the unfinished English run as a complete result. Job elapsed time is not a latency
benchmark. Both jobs use the shared laptop lock and original thermal rules; no cloud spending.
