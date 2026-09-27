# Qwen continuation through step 150 (RunPod H100)

On the same Pod and software as [attempt 13](../attempt-13-train-150/README.md), the worker resumed the
verified step-125 checkpoint and completed the final 25 registered updates. Training has reached all 150
updates and 2,400 decisions: the frozen schedule is complete. The unchanged [independent verifier](verification.json)
checked the step-130, step-140 and step-150 checkpoints, all 372 finite changed adapter tensors, optimizer state,
update receipts, schedule accounting, source hashes and the retained validation history.

The final private checkpoint is `runs/pilot-qwen-continuation-20260927/training/attempt-14-step-150.pt`,
130,343,941 bytes, SHA-256 `8dabf1eb0fb722570713adc4bb246997fb72fc46fa19e47970e761d8b249072d`.
Attempt fifteen validates it on the laptop, like steps 0, 50 and 100. Saved per-update mean losses (0.226 at
step 126, 0.029 at step 150, mean 0.097) are training diagnostics, not validation. The worker's 363.2
operational seconds include no thermal pauses (maximum 30 C, peak allocated VRAM 1.96 GB).

The [evidence archive](artifact_receipt.json) holds 42 verified files in 158,246 bytes, SHA-256
`bd3301891fd110056688946e47de3fd78408f0240c031d26de4e648da564dd05`. This README was added after the archive.
The Pod was allocated for 66 minutes at $3.49/hour (about $3.86 GPU cost plus storage) and then deleted after
the checkpoints and logs were downloaded and checksum-verified.
