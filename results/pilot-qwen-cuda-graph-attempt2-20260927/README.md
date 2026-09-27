# Qwen CUDA graph retry

The [first attempt](../pilot-qwen-cuda-graph-probe-20260927/README.md) stopped before inference on missing cache metadata. The [verified repair](../pilot-qwen-cache-metadata-20260927/repair.json) preserves all existing model and tokenizer files.

The [retry registration](../../experiments/small_model/qwen_cuda_graph_attempt2_registration.json) retains the same four cases, graph implementation, frozen weights, probability limits and bounded compute settings. Only output-directory and registration constants differ in the worker. A registration is not a successful compatibility result; full validation parity and separate latency testing remain necessary.
