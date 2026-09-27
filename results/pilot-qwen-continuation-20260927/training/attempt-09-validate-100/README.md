# Step-100 continuation validation

The registered validation completed on all 278 original held-out states and 1,390 decisions.
The [independent verifier](verification.json) checked complete input reconstruction, every
probability vector, the reported metrics, source/input hashes, adapter/optimizer state and
checkpoint history. All 372 trained adapter tensors remain finite and differ from the original.
This is the retry of the [thermally stopped attempt eight](../attempt-08-validate-100/README.md),
from the same verified step-100 checkpoint, started once the GPU was at 70 C.

| Original teacher-validation metric | Unchanged step 0 | Step 50 | Step 100 |
|---|---:|---:|---:|
| Teacher argmax agreement | 0.7899280575539568 | 0.781294964028777 | 0.7906474820143885 |
| Mean KL divergence, lower is better | 0.1401374435902039 | 0.14770740977013652 | 0.14817777051879652 |
| Decisions | 1,390 | 1,390 | 1,390 |

Step 100 recovers step 50's agreement loss (one more agreeing decision than step zero) but its
mean KL is the worst of the three. Under the registered selector, which minimizes mean KL among
checkpoints within 0.01 agreement of step zero, the unchanged step-zero checkpoint still leads.
This does not establish a quality gain. The predeclared experiment still checks step 150 before
selection. No learning rate, selection threshold, temperature, data schedule or public-test
procedure changes in response to this result.

The new private checkpoint is `runs/pilot-qwen-continuation-20260927/training/attempt-09-step-100.pt`,
130,343,941 bytes, SHA-256
`98557b0436470ee252615f14244f0255aeb143ec6615a9e3215b717e4e1048df`.
It carries the step-zero, step-50 and step-100 validation history and the existing
optimizer/random state. The next training segment starts from this validated checkpoint.

The [888,425-byte evidence archive](artifact_receipt.json) contains 44 verified files, including
all validation predictions and executed sources, without private weights or training rows.
Validation rows SHA-256:
`235390341d6992bcbf246c009d46f45cb4a8aca4282b26f95fd35b92d8569fc0`.
The verifier reread all archive bytes. This README was added after the immutable archive.

The worker recorded 175.7078 operational seconds, including 43 thermal pauses totaling 82.9479
seconds, with a maximum sampled temperature of 83 C and peak allocated VRAM 5.11 GB. These are
execution-accounting values, not inference latency or training throughput. Public accuracy,
selected-model reload parity and the original release gates remain unmeasured for a selected
continuation checkpoint.
