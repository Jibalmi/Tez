# Step-50 continuation validation

The registered validation completed on all 278 original held-out states and 1,390 decisions.
The [independent verifier](verification.json) checked complete input reconstruction, every
probability vector, the reported metrics, source/input hashes, adapter/optimizer state and
checkpoint history. All 372 trained adapter tensors remain finite and differ from the original.

| Original teacher-validation metric | Unchanged step 0 | Step 50 |
|---|---:|---:|
| Teacher argmax agreement | 0.7899280575539568 | 0.781294964028777 |
| Mean KL divergence, lower is better | 0.1401374435902039 | 0.14770740977013652 |
| Decisions | 1,390 | 1,390 |

This first trained checkpoint is worse on both registered validation metrics. It does not
establish a quality gain. The predeclared experiment still checks steps 100 and 150 before
selection, with the original checkpoint eligible. No learning rate, selection threshold,
temperature, data schedule or public-test procedure changes in response to this result.

The new private checkpoint is `runs/pilot-qwen-continuation-20260927/training/attempt-05-step-050.pt`,
130,343,749 bytes, SHA-256
`ac5a3271f481106889a3b32dd51a8984e945b67b44d359a1631653a742f6edc2`.
It carries both step-zero and step-50 validation history and the existing optimizer/random state.
The next training segment uses this validated checkpoint, rather than the earlier pre-validation
step-50 save. Attempt six is queued toward step 100, with the unchanged 25-update segment bound.

The [889,021-byte evidence archive](artifact_receipt.json) contains 44 verified files, including
all validation predictions and executed sources, without private weights or training rows.
Validation rows SHA-256:
`537f4614af48d8124e6a756739b068be40c4f8050912611e9d3bdef0f3b36223`.
The verifier reread all archive bytes. This README was added after the immutable archive.

The worker recorded 108.8668 operational seconds, including 18 thermal pauses totaling 19.2155
seconds, with a maximum sampled temperature of 82 C. These are execution-accounting values,
not inference latency or training throughput. Public accuracy, selected-model reload parity,
and the original release gates remain unmeasured for a selected continuation checkpoint.
