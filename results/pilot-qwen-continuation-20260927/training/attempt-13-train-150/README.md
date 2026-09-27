# Qwen continuation through step 125 (RunPod H100)

Under the [recorded deviation](../README.md#recorded-deviation-steps-101-150-trained-on-a-runpod-gpu), the
worker resumed the verified, validated step-100 checkpoint (`attempt-09-step-100.pt`) on one RunPod
NVIDIA H100 80GB HBM3 (driver 580.126.09, CUDA 13.0, Python 3.14.7 against the laptop's 3.14.3, the registered
torch/transformers/peft/tokenizers/huggingface-hub versions, no FLA or causal-conv1d, four threads) and
completed 25 registered updates. Training has reached 125 updates and 2,000 decisions. The unchanged
[independent verifier](verification.json) checked the step-110, step-120 and step-125 checkpoints, all 372
finite changed adapter tensors, optimizer state, update receipts, schedule accounting, source hashes and the
retained step-zero/step-50/step-100 validation history.

The final private checkpoint is `runs/pilot-qwen-continuation-20260927/training/attempt-13-step-125.pt`,
130,343,941 bytes, SHA-256 `eda7b02ec43f7f79823f5699d1cc4eac9388b60ca520a6d2ff03de8c6e637369`.
Saved per-update mean losses (0.073 at step 101, 0.071 at step 125, mean 0.080) are training diagnostics,
not validation. The worker's 604.2 operational seconds (most of it loading and hashing on the Pod's network
volume) include no thermal pauses; the maximum sampled temperature was 30 C and peak allocated VRAM 1.86 GB.
These are execution accounting, not throughput.

The [evidence archive](artifact_receipt.json) holds 42 verified files in 158,262 bytes, SHA-256
`1e85d1937bf61bd9eb86859c4cecb637668e25ff07bf4a0420dd1d16b8730cbb`. This README was added after the archive.
