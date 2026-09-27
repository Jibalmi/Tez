# First Qwen continuation training segment

The registered worker completed 25 optimizer updates from the independently verified step-zero
checkpoint. That is 400 original training decisions and 107,396 unpadded input tokens. It stopped
at the predeclared segment limit, saved step 25 and released the shared compute lock. The target
interval at step 50 is unfinished. All updates had finite losses and gradient norms.

The [independent CPU verifier](verification.json) checked saved checkpoints at steps 10, 20 and
25: all 372 adapter tensors had changed from the original, all saved values and optimizer moments
were finite, and optimizer steps, token counts, parent checksum, registration and validation history
were consistent. It did not replay GPU updates or measure new validation quality.

The final private checkpoint is `runs/pilot-qwen-continuation-20260927/training/attempt-03-step-025.pt`,
SHA-256 `86efb12b93b02859dbe7cf46df313aa766bd85f7949fd9b64bc021f36becddb0`.
It preserves adapter, optimizer and CPU/CUDA random state. Attempt four is queued to resume from
that exact checkpoint. Weights and original training examples remain outside the public archive.

The [artifact receipt](artifact_receipt.json) records a verified 158,582-byte archive with 42 files,
SHA-256 `13d17bf551f0830569a59ddd758ad90de1af25c99883311ab712b8e975d3ca43`.
This README was added after that immutable evidence snapshot. [Work receipts](work.json) retain
every update, resource metadata and checkpoint hash. No public accuracy, speedup or gate pass is
claimed from a completed training segment; validation is scheduled at steps 50, 100 and 150.
