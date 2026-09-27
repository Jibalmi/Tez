# Qwen continuation through step 50

The worker resumed the verified step-25 adapter, optimizer and CPU/CUDA random state, then completed
another 25 updates. The original registered schedule has now reached 50 updates, 800 training
decisions and 221,201 unpadded tokens. All update losses and gradient norms were finite. This is
successful GPU resumption, not a comparison against an uninterrupted GPU training trajectory.

The [independent verifier](verification.json) checked checkpoints at steps 30, 40 and 50, including
all 372 adapter tensors, optimizer moments and update counts, schedule tokens, parent checksum,
registration and retained step-zero validation history. [Work receipts](work.json) preserve every
new update and the actual execution settings. The final private checkpoint is
`runs/pilot-qwen-continuation-20260927/training/attempt-04-step-050.pt`, SHA-256
`c4f103b93bf1df2c7ee09732b8a3885c753080b26c1c618552c7fcf46f4ae919`.

Attempt five is queued to validate this checkpoint on all 1,390 original held-out decisions.
Training toward step 100 must use the new checkpoint written by that completed validation,
so its validation history cannot be skipped. No post-training quality result is available yet.

The [immutable evidence archive](artifact_receipt.json) is 158,600 bytes with 42 verified files,
SHA-256 `d7d323eb317ffca7ebf358ce10cd2e2469d7fcb16246b18270d134fb8abfd22c`.
Weights and training examples remain outside that archive. This README was added afterward.
