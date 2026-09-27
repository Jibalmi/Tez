# Multilingual GLiNER recovery after a Windows reboot

The recovery completed all 6,151 original and 1,100 reversed-option decisions. The
[combined independent analysis](../pilot-gliner2-comparison-20260927/RESULTS.md) verifies its
raw rows, probability vectors, option-order checks and official SemIf metrics. Its
[artifact receipt](../pilot-gliner2-comparison-20260927/artifact_receipt.json) covers this run.
The worker recorded 841.41 seconds including setup and 18.07 seconds of thermal pauses,
with a maximum sampled temperature of 81 C. This is operational elapsed time, not latency.

The preceding batch-four run stopped after 3,327 rows. Its worker and queue handles disappeared,
and scoped OS process inspection found no matching workers. Windows recorded an unexpected
reboot at 08:59:52 UTC with bugcheck `0x20001` (`HYPERVISOR_ERROR`). This identifies a hypervisor
failure; it does not establish that GPU memory, temperature or this workload caused it.
[Windows evidence](../pilot-gliner2-multi-recovery-20260927/windows_reboot.json) and
[preserved partial artifact](../pilot-gliner2-multi-recovery-20260927/artifact_receipt_interrupted.json)
retain the failure. [Microsoft's definition](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x20001--hypervisor-error)
does not diagnose the trigger.

This [separate registration](../../experiments/small_model/gliner2_registration_multilingual_serial.json)
repeats all inputs at batch one, with native FP32 weights and unchanged quality checks.
The English batch-one run completed on this host. Multilingual pacing is also more conservative:
check temperature before every record, pause at 80 C until 76 C, and retain the 60-second cooling
deadline. The shared wrapper's maximum temperature is tightened to 84 C; its hardware-throttle
and free-memory protections remain active. This changes resource pacing, not a quality gate.

All 6,151 original and 1,100 reversed-option decisions must complete. No partial runs are spliced.
No system settings, unrelated processes, cloud resources or runtime code are changed.
