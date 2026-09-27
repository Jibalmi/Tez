# Additional available decision systems

These candidates were discovered after the first matched comparison. No local GLiNER accuracy,
calibration or latency score has been measured here. Keep their published results separate from
the [16-configuration comparison](../pilot-improvement-20260927/COMPARISON.md).

## GLiNER2.5-Decide

Fastino releases English, multilingual and larger English decision checkpoints. Their native API
accepts task instructions and label descriptions. It uses a classification head rather than
generated answer text. This makes the family relevant to our routing, safety and typed-decision
comparison; it is distinct from the GLiClass family already tested.
[English card](https://huggingface.co/fastino/GLiNER2.5-Decide),
[multilingual card](https://huggingface.co/fastino/GLiNER2.5-multi-Decide),
[1B card](https://huggingface.co/fastino/GLiNER2.5-Decide-1B).

The [pinned inventory](inventory.json) records exact revisions, declared Apache-2.0 licenses,
model-card hashes and metadata. Safetensors metadata reports 486,444,053 parameters for English,
287,355,159 for multilingual and 1,188,796,693 for the larger model. The English card calls its
model 340M; these figures are different counting claims, not reconciled by this discovery audit.
Only the first two checkpoints are selected for local preparation. No training-set independence
or reproduction of the publisher's benchmark is claimed.

Preparation is complete: [the second-attempt receipt](preparation_attempt2.json) verifies both
checkpoints against upstream weight hashes and records every installed overlay file. The baseline
environment is unchanged. The [native interface probe](../pilot-gliner2-probe-20260927/README.md)
is prepared separately; successful preparation does not claim that inference has passed.

The unpacked `gliner2 2.0.0` wheel was checked against its PyPI SHA-256;
[source-file hashes](package_inspection.json) record what was inspected. Its local extra requires
Transformers below version 5, unlike the Qwen pilot's 5.17.0. The first preparation stopped at a
missing Pydantic dependency before installation. [That failure](preparation.json) is retained.
The amended preparer stages compatible packages, including Pydantic and its required dependencies,
in a separate import overlay. It must verify downloaded weight hashes and leave the baseline
environment unchanged. Staging is not an inference test.
[Package metadata](https://pypi.org/pypi/gliner2/2.0.0/json).

Source inspection found native per-label logits and single-label softmax in
`gliner2/inference/runtime.py::_extract_classification_result`. The ordinary classification API
returns the winning label and its confidence. A comparison adapter must preserve every option
and instruction, capture the full head output, and check it against that native result. The
separate constrained `Classifier` facade rejects some punctuation in schema strings; silently
stripping those characters would change our evaluation inputs. Truncation, label alignment and
complete probability output need explicit checks before evaluation.

## Other systems serve different roles

[LocalJev](https://github.com/githubnext/localjev) documents a bridge that asks a generative model
for JSON probability estimates. These are self-reported values, not selected-token logits.
Its published evaluation uses an Apple M5 Max and a different small set of tasks, so those scores
do not enter our table or establish our laptop speed. It is not a replacement for the existing
teacher's soft targets without changing the experiment.

[Ollaya](https://github.com/ollaya-dev/ollaya) is a local runtime for several existing decision
models, including Laya and Decider. It is a deployment alternative worth evaluating separately,
not another trained checkpoint whose quality can be assumed better than the same weights.
No Ollaya speed or export-parity measurement has been run in this pilot.

Next: verify native probability extraction on non-benchmark inputs,
freeze the adapter and evaluation protocol, then run the same complete suites under the shared
compute lock. No additional paid resource is created by this preparation.
