# Preserved Pod source-hash stop

The first RunPod attempt stopped 0.03 seconds into the worker, before loading a model: `Registered source
changed`. The registration hashes the laptop's working-copy bytes, and on Windows several registered files
carry CRLF line endings that the repository stores as LF; the first Pod bundle shipped `git archive` output
(LF), so those hashes differed. No update, checkpoint or validation input exists. The bundle builder now
ships working-copy bytes ([fix](../../../../experiments/small_model/continuation_pod_bundle.py)), and the
Pod received the laptop's exact bytes for the 322 affected files before the next attempt.

The [failure archive](artifact_receipt.json) holds 8 verified files in 11,432 bytes, SHA-256
`0fb96e65086e5dbbf44560e3a5682acd2cca61d9b7370fb9c23f37876bc4bb74`. This README was added after the archive.
