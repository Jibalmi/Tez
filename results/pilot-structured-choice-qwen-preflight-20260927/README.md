# Qwen input audit for the separate structured-choice data

All eligible training and validation inputs fit the existing 2,048-token context without
truncation. This is preparation for a possible separate quality experiment; no model has
trained on or predicted these examples. The current original-data continuation is unchanged.

| Split | States | Distinct decisions | Encoded sequences | Longest input |
|---|---:|---:|---:|---:|
| Filtered training | 1,200 | 2,400 | 4,800 | 324 tokens |
| Validation | 200 | 400 | 800 | 303 tokens |

Each decision was tokenized in both its primary and reversed option order, hence two encoded
sequences per decision. The independent exact-rule evaluator checked that the gold option
key maps to the correct index in both orders. Gold keys and indices were never passed to the
prompt builder. Option letters A through G are distinct single tokenizer tokens. All prompts
and token sequences have content hashes in the private audit.

The worker verified the completed [filter receipt](../pilot-structured-choice-20260927/filter_verification.json),
the unchanged generator, and the actual filtered-training/validation file hashes before reading
the examples. It reran exact-label and split-vocabulary validation on those two splits. The
confirmation file was not opened. Its reserved model evaluation remains after a future fixed
training protocol and validation-only checkpoint selection. Repeated logical families across
splits still limit claims about general reasoning.

The [preflight](preflight.json) records the pinned Qwen tokenizer revision, input counts,
label-position counts, token totals, source hashes and private audit hash/size. The complete
audit is `runs/pilot-structured-choice-qwen-preflight-20260927/input_audit.json`; training tokens
and labels are excluded from Git and the public evidence archive.

The [35,668-byte archive](artifact_receipt.json), SHA-256
`126ecda8939fb54bbe547e580dbbc2f9908f7f7267b3282242e2872160640b59`,
contains 12 verified files with the preparation source and filtering/generation receipts.
Ruff passed. All label checks and tokenization ran with one CPU thread and CUDA hidden;
the worker confirmed that no CUDA context was initialized. These inputs are exact hard-label
supervision, not substitute Gemma probability targets. No training protocol, checkpoint or
quality improvement is claimed from this audit.
