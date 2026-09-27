# GLiNER native probability interface probe

The multilingual checkpoint completed all four interface checks in FP32. The English checkpoint
failed while printing its configuration: the library emits an emoji that Windows' default
CP1252 output encoding cannot represent. No English probability was produced. Both outcomes
are retained; a [separate English retry](../pilot-gliner2-probe-attempt2-20260927/README.md) selects
UTF-8 for the child process's text output. No weight or inference change is involved.

Four fixed, newly written inputs check English and multilingual classification, a yes/no question,
and an ordinal scale. Parentheses in instructions/descriptions exercise an input that the separate
constrained-classification facade would reject. There are no benchmark cases, gold labels,
accuracy scores or latency measurements in this probe.

Each checkpoint runs its original `AutoExtractor.classify_text` implementation with full inputs,
single-label softmax, FP32 weights and the same option order. A read-only PyTorch hook captures
all head logits. Their probability argmax and maximum must match the native label/confidence
within 0.000001. Each call must encode once, emit every option and use a collator without text
truncation. This verifies an interface, not calibration quality or export fidelity.

Workers use the isolated package overlay, frozen local weights, four threads, the unchanged
thermal governor and the shared compute lock. Each worker has a three-minute limit. Failures
remain recorded. The [preparation receipts](../pilot-gliner2-discovery-20260927/README.md) preserve
dependency and weight hashes. No cloud compute is used.
