# Preserved step-100 validation thermal stop

The first step-100 validation loaded the verified step-100 checkpoint and began the registered
held-out pass, then stopped on the worker's own thermal guard: the GPU was still at 77 C after
the registered 60-second cooldown (target 76 C). The laptop was heat-soaked from other shared
jobs; the maximum sampled temperature was 80 C over 139 checks and three pauses. The operation
ran 141 seconds and exited with the recorded `ThermalCooldownTimeout`.

It had written 1,296 of the 1,390 held-out decisions. The registered validation requires the
complete set, so this partial file is retained as failure evidence only: it supplies no metric,
no checkpoint selection input and no quality result. The checkpoint, data, selection rule,
probability limits and thermal thresholds are unchanged. This is a resource stop, not an
implementation defect or a model-quality result.

The retry uses a new attempt number from the same verified step-100 checkpoint. To avoid
starting a registered operation on a hot GPU, the shared wrapper now waits, while holding the
compute lock, until the GPU is at or below 70 C, and releases the lock without starting the
operation if it does not cool within eight minutes. The worker's own 80/76 C pacing and the
84 C wrapper ceiling are unchanged. Eight of the twenty registered attempt identifiers are now
used.

The [failure archive](artifact_receipt.json) holds 35 verified files in 374,879 bytes, SHA-256
`bce31b0e8def4d4d65028c1e31c7c42450e8feb6610cbf8bdec0f39169ba0151`, including the partial rows,
log and executed sources. This README was added after the archive.
