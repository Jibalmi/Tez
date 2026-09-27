# Qwen continuation through step 75

The worker resumed the validated step-50 adapter, optimizer and random states, then completed
the next 25 registered updates. Training has reached 75 updates, 1,200 decisions and 346,292
unpadded tokens. The [independent verifier](verification.json) checked the step-60, step-70
and step-75 checkpoints, all 372 finite changed adapter tensors, optimizer state, update
receipts, schedule accounting, source hashes and retained step-zero/step-50 validation history.

The final private checkpoint is `runs/pilot-qwen-continuation-20260927/training/attempt-06-step-075.pt`,
130,343,749 bytes, SHA-256
`ea2dcbcc77098465264b0f4f01217a8c1af504ea9a3248d83e6815a2007746ea`.
Attempt seven is queued from this checkpoint to the registered step-100 validation boundary.
This establishes successful bounded GPU resumption, not equivalence to uninterrupted training.

There is no step-75 quality measurement. Step 50 regressed on both validation metrics;
the unchanged protocol still evaluates steps 100 and 150 before choosing among all four
checkpoints, including the original. No public evaluation or quality promotion follows from
this training result alone.

The [immutable evidence archive](artifact_receipt.json) contains 42 verified files in 158,614
bytes, SHA-256 `36506598b2c25e083a87467f36dd50b7c60fc0c4c437dd312b9337bd177dbf6e`.
It excludes private weights and training examples. This README was added after the archive.
The worker's 672.3249627 operational seconds include 200 thermal pauses totaling 327.2206747
seconds, with maximum sampled temperature 82 C. These are execution accounting, not latency
or training-throughput measurements.
