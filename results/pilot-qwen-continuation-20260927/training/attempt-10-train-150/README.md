# Preserved training-segment thermal stop

The first segment toward the step-150 boundary resumed the verified, validated step-100
checkpoint (`attempt-09-step-100.pt`) after the wrapper saw the GPU at 69 C, then completed nine
registered updates (steps 101-109) before the worker's own thermal guard stopped it: the GPU was
still at 80 C after the registered 60-second cooldown (target 76 C). The worker recorded 81
thermal pauses totaling 255.3 seconds over 380.6 seconds, with a maximum sampled temperature of
82 C. The laptop was heat-soaked by other shared work (system CPU about 36 %, antivirus scanning,
desktop GPU clients).

Checkpoints are written every ten updates, so no step-110 checkpoint exists and the nine updates
are discarded. They supply no checkpoint, validation input or quality result. The retry resumes
the same validated step-100 checkpoint under a new attempt number, so the registered update
sequence is unchanged. The data, schedule, optimizer, selection rule, probability limits and
thermal thresholds are unchanged. This is a resource stop, not an implementation defect.

To stop spending registered attempts on a heat-soaked machine, the retry is not queued until the
GPU is at or below 62 C and system CPU averages under 30 % over a minute; the wrapper then also
waits, holding the compute lock, for 62 C before starting. Ten of the twenty registered attempt
identifiers are now used.

The [failure archive](artifact_receipt.json) holds 34 verified files in 140,387 bytes, SHA-256
`d9d067340fb665ce4734184981fe7d646053495ed836f612bc67e85868b81455`, including the update receipts,
log and executed sources. This README was added after the archive.
