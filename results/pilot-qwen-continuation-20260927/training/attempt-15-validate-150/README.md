# Step-150 continuation validation

The registered validation completed on all 278 original held-out states and 1,390 decisions, on the
same RTX 5080 Laptop GPU and software as steps 0, 50 and 100. The checkpoint was trained on a RunPod H100
under the [recorded deviation](../README.md#recorded-deviation-steps-101-150-trained-on-a-runpod-gpu).
The [independent verifier](verification.json) checked complete input reconstruction, every probability
vector, the reported metrics, source/input hashes, adapter/optimizer state and checkpoint history.

| Original teacher-validation metric | Step 0 | Step 50 | Step 100 | Step 150 |
|---|---:|---:|---:|---:|
| Teacher argmax agreement | 0.7899280575539568 | 0.781294964028777 | 0.7906474820143885 | 0.7683453237410072 |
| Mean KL divergence, lower is better | 0.1401374435902039 | 0.14770740977013652 | 0.14817777051879652 | 0.14848698115125392 |
| Decisions | 1,390 | 1,390 | 1,390 | 1,390 |

Mean KL rose at every validated step, and step 150 loses 0.0216 agreement against step zero, beyond the
registered 0.01 allowance. The registered selector therefore retains the unchanged step-zero checkpoint:
this continuation shows no validation gain. No learning rate, selection threshold, temperature, data
schedule or public-test procedure changes in response to this result.

The new private checkpoint is `runs/pilot-qwen-continuation-20260927/training/attempt-15-step-150.pt`,
130,344,197 bytes, SHA-256 `14514e6bb17f3b7b2f0ec2cb08ef2366b451f41b5fd0b1b68a0e33be873849d3`, carrying
all four validations. Validation rows SHA-256:
`098a41c5e1310774ea6589c50f2dfdb118c12230454384dea29642ff92722ce6`.

The [889,396-byte evidence archive](artifact_receipt.json) contains 44 verified files, including all
validation predictions and executed sources, without private weights or training rows. This README was
added after the immutable archive. The worker recorded 145.2426 operational seconds, including 21 thermal
pauses totaling 26.5025 seconds, with a maximum sampled temperature of 82 C and peak allocated VRAM
5.11 GB. The run started once the shared GPU was at or below 65 C with system CPU under 35 %.
