# Pinned Qwen cache metadata repair

The graph probe stopped before inference because the local snapshot lacked three text files.
The [repair receipt](repair.json) verifies `.gitattributes`, `LICENSE` and `README.md` against
their remote Git object hashes at the already pinned revision. All ten existing model and
tokenizer files remain byte-for-byte unchanged. Full offline snapshot lookup now succeeds.

This was a small metadata download, with no model loading, inference, package change or paid
resource. The queued training and FLA probes use this same snapshot; their code and experimental
settings are unchanged. The failed graph run remains preserved in its original directory.
