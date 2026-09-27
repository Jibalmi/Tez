# Tev1 competitor discovery

[Together's Tev1 0.8B card](https://huggingface.co/togethercomputer/Tev1-0.8B-experimental)
identifies a Qwen3.5-0.8B supervised decision fine-tune. It is a relevant same-base comparison
for our Qwen student. Its intended interface uses a system instruction and a JSON decision
with state, question, and lettered options, with thinking disabled. Native letter logits can
support complete option probabilities, but that interface still needs a local verification.
The card does not establish broad multilingual quality or calibrated confidence.

The [pinned inventory](inventory.json) records the public API metadata, file checksums and
source commit. The 0.8B checkpoint revision is `6bb2dff14b38fea90ddb14d870166ccaf77374e9`;
its safetensors metadata lists 873,438,784 parameters, including the multimodal components.
The full checkpoint and the text-only inference parameter counts must not be conflated.
All 6,151 frozen public decisions have 2–20 options and nonempty descriptions, within the
native helper's 2–24 option contract. No test cases need to be dropped to satisfy that contract.

[The official code](https://github.com/togethercomputer/tev1) exposes its decision format and
data recipe. Its source registry names MultiNLI, BoolQ, Banking77, AG News, SST-5 and synthetic
policy/routing/research examples. Exact released-checkpoint training membership is not reconstructed.
The 4B release's published development scores are reused development results; they do not
describe the new 0.8B checkpoint or belong in our matched results table.

The 0.8B preparation downloads the pinned public files under the shared compute lock,
verifies each remote object and leaves baseline packages unchanged. No benchmark has run yet.
The 4B release is metadata-only and has no download or inference job. The cards say the fine-tuned
weight license is still being finalized; discovery does not treat it as a deployment-ready release.
No paid endpoint, training upload, weight redistribution or cloud resource is involved.

Preparation completed: all 11 downloaded files match their pinned remote objects, and baseline
package versions are unchanged. See [preparation evidence](preparation.json). The next step is
the [four-case native probability check](../pilot-tev1-probe-20260927/README.md), before any
full benchmark. The discovery source archive contains metadata and licensed source only.
