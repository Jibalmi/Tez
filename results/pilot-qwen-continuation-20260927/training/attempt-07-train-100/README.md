# Qwen continuation through step 100

The worker resumed the verified step-75 adapter, optimizer and random states, then completed
the next 25 registered updates. Training has reached 100 updates, 1,600 decisions and 443,798
unpadded tokens. The [independent verifier](verification.json) checked the step-80, step-90
and step-100 checkpoints, all 372 finite changed adapter tensors, optimizer state, update
receipts, schedule accounting, source hashes and retained step-zero/step-50 validation history.

The final private checkpoint is `runs/pilot-qwen-continuation-20260927/training/attempt-07-step-100.pt`,
130,343,749 bytes, SHA-256
`0fb6224a6d065f1046176d55d2be82fbfc142b51358f829899066bab297ccbfe`.
Attempt eight validates this checkpoint on all 1,390 held-out decisions at the registered step-100
boundary. This establishes a second successful bounded GPU resumption, not equivalence to
uninterrupted training.

There is no step-100 quality measurement yet. Step 50 regressed on both validation metrics;
the unchanged protocol still evaluates steps 100 and 150 before choosing among all four
checkpoints, including the original. The saved per-update mean losses (0.150 at step 76, 0.063
at step 100, mean 0.093) are training diagnostics on single accumulated batches, not validation.

The [immutable evidence archive](artifact_receipt.json) contains 42 verified files in 158,623
bytes, SHA-256 `66ac2ef0ef2d37d12026156470dc7efa3e01e9d1a7cde389f6e4a8ed33f5f719`.
It excludes private weights and training examples. This README was added after the archive.
The worker's 837.6865610 operational seconds include 119 thermal pauses totaling 507.6519215
seconds, with maximum sampled temperature 81 C and peak allocated VRAM 1.92 GB. These are
execution accounting, not latency or training-throughput measurements.
