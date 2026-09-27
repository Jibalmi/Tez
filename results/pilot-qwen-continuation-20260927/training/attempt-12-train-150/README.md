# Preserved Pod missing-prerequisite stop

The second RunPod attempt passed the registration and source hashes, then stopped after 4.8 seconds, before
loading a model, because the ignored prerequisite `results/pilot-restricted-qwen-20260927/reference_rows.jsonl`
was not in the bundle. No update, checkpoint or validation input exists. Before the next attempt, the Pod
received that file and the other ignored validation-row files, and a check on the Pod confirmed that every
registered source and prerequisite hash, every pinned package version and the resume checkpoint's hash matched.

The [failure archive](artifact_receipt.json) holds 34 verified files in 139,196 bytes, SHA-256
`0171bc85f0ad85f5a8d3395aa5fed3e289495de75372bcbde94ff64b167fab07`. This README was added after the archive.
