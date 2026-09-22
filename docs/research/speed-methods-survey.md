# Survey: making the single-pass readout faster or cheaper (22 Sept 2026)

Commissioned literature survey (web-verified arXiv ids). Baseline attacked: ~2.7 ms per uncached prompt token, 15 ms HTTP, 30 ms/word streaming with a cached option prefix, 57–250 ms when a long state is the variable suffix; Gemma 4 12B Q8 on llama.cpp, 16 GB.

## 1. Early exit / intermediate-layer readout
- **CALM** (2207.07061): per-token confidence-gated exit, up to 3×; needs a trained exit classifier. For one-shot readout it collapses to "truncate depth".
- **LayerSkip** (2404.16710): requires training the base; its early-exit-draft / full-depth-verify trick is the template for cascades.
- **Beyond Greedy Exits / UAT** (2509.23666, NeurIPS'25): bandit-tuned exit thresholds online; 1.7–2.1× with < 2 % drop on classification. Optimises accuracy, not probability fidelity — use a KL-vs-full-model reward.
- **Two-dimensional early exit** (2604.18592): process the input sentence-by-sentence while activating deeper layers; 1.4–2.3× on Llama 3 / Gemma / Qwen3-8B classification. Matches streaming: stop reading state once the option distribution has stabilised.
- **SimLens / SimExit** (2507.17618): training-free, single-token decision tasks, entropy-gated exit 1.15× iso-accuracy, 1.40× at −1 pt. Closest analogue to Tez's readout.
- **Confident Layer Decoding** (2606.21906): final layers can perturb refined predictions; near-final layers may be safely readable early.
- Warnings: **Calibration Across Layers** (2511.00280) — confidence is recalibrated in upper layers after the decision forms, so intermediate readouts are over-confident; **layer-wise dynamics** (2507.06722) — raw intermediate confidence is a weak gate. Fix: **tuned lens** (2303.08112) affine probe per layer, or a probe trained to reproduce the full-depth option distribution.
- Engine: llama.cpp has no exit; try `--override-kv <arch>.block_count=int:N` (unverified for Gemma 4) or transformers `output_hidden_states` + final norm + lm_head.

## 2. Depth pruning
- **Unreasonable Ineffectiveness of the Deeper Layers** (2403.17887), **ShortGPT** (2403.03853: drop 25 % of layers, 91.6 % retention), **Rethinking Layer Redundancy** (2604.24938: calibration data matters more than the search), **When Fewer Layers Break More Chains** (2510.22228: classification survives, long-chain reasoning collapses). Expected 20–30 % layers removed → ~1.25–1.4× prefill and ~3 GB VRAM freed; needs GGUF re-export; recalibrate; prune per attention type (Gemma local/global).

## 3. KV reuse across questions and permutations
- **SGLang RadixAttention** (2312.07104), vLLM automatic prefix caching: exact, prefixes only. **Hydragen** (2402.05099): shared-prefix attention batching, up to 32×; llama.cpp equivalent = prefill once, `llama_kv_cache_seq_cp` into N sequences, batch the N suffixes.
- Position-independent caching: **CacheBlend** (2405.16444), **EPIC** (2410.15332), **KV Packet** (2604.13226), **SemPIC** (2607.28069), **AdapShot** (2605.03644), **C2KV** (2607.17715); **KVShareArena** (2609.10266) is the sobering benchmark — unrepaired caches can be worse than recompute.
- The big win is layout, not a new method: for many questions on one state, make the state the prefix and (question + options) the suffix — > 5–10× for a 500-token state, exact, works today.

## 4. Speculation for scoring
- **Speculative Cascades** (2405.19261, Gemma 2B→27B): deferral rules Chow / Diff / OPT (TV distance) / token-level; 1.2–1.95× at matched quality. Nothing accelerates a *single* forward pass by drafting; the only speculative lever is a cascade.

## 5. Prompt compression / prefill pruning
- **LLMLingua-2** (2403.12968): 2–5× compression; **empirical study** (2505.00019). **LazyLLM** (2407.14057): compute KV only for tokens the last token attends to, 2.34× prefill. **Probe and Skip** (2601.13155): 2.46× prefill. **FastE** (2609.08407). None in llama.cpp.

## 6. Probe heads on frozen hidden states
- **LinC** (2401.12406): affine calibration from 5 samples. **Calibrating LLM judges with linear probes** (2512.22245). **Can linear probes measure uncertainty** (2510.04108). **LLM Microscope** (2510.04013). **Frozen-LLM probing** (2606.28798). A probe at layer L predicting the full-depth option distribution is the exit calibrator. Risk: probes learn format (2606.02907).

## 7. Quantisation and probability fidelity
- **Accuracy is Not All You Need** (2407.09141): 0–2 % accuracy deltas hide up to 13.6 % answer flips; only W8 flip-free. **Unified llama.cpp evaluation** (2601.14277): Q8_0 −0.09 %, Q4_K_M −0.46 %. GPU prefill is compute-bound, so lower weight quant buys little speed. KV cache: Gemma 4 is unusually KV-quant-sensitive (q8_0 KV KL 0.1–0.4; localbench); **statistical KV quantisation** (2605.08114): K errors are amplified super-linearly, V never touches the softmax.

## Ranked

| # | Method | Expected gain here | Cost | Risk |
|---|---|---|---|---|
| 1 | State-as-prefix KV reuse across questions | 5–10× per question when > 2 questions per state | Low | Exact; layout changes the distribution — recalibrate |
| 2 | Depth pruning 20–30 % + probe/temperature | 1.25–1.4× prefill, frees ~3 GB | Medium (GGUF surgery) | Over-confidence; local/global layer mix |
| 3 | Small-model cascade with OPT/Diff deferral | 1.5–3× average | Medium | Draft's calibration on non-deferred items (our 4B cascade was a negative result) |
| 4 | Sentence-wise early stop of state streaming | 1.4–2.3× on long states | Low | Premature stability |
| 5 | Prefill token pruning (LazyLLM, Probe-and-Skip) | 2.3–2.5× prefill | High | Mass shift |
| 6 | LLMLingua-2 state compression | 2–5× fewer state tokens | Low | Drops decision-critical tokens |
| 7 | Shared-prefix permutation batching (Hydragen pattern) | removes N−1 prefix recomputes | Low | none |
| 8 | UAT bandit thresholds on 2/4 | +10–20 % on the gated method | Low | needs KL reward |
| 9 | Position-independent caching | only if option-first layout must stay | High | approximate |
| 10 | LinC / probe calibration | enables 2, 3, 4 | Very low | template learning |
| 11 | Weight quant < Q8 / KV quant | ~0 speed on GPU | Low | flips up to 13.6 % |

## Top 5 to try first
1. Flip the layout for multi-question states (state prefix, questions as batched suffixes via seq_cp); measure ms and KL vs option-first on 500 decisions.
2. Depth-truncate and re-read: GGUFs with the lowest-BI 10/20/30 % layers removed, calibrated on our prompts; prefill ms, argmax agreement, KL before/after a LinC fix.
3. Tuned-lens exit probe: hidden states at 60/70/80 % depth for 5k decisions, affine probes to the full-depth distribution; pick the shallowest L under a KL budget.
4. Cascade with OPT deferral on the smallest Gemma 4 variant; deferral rate vs KL-to-12B vs latency.
5. Sentence-wise early stop on streamed state; UAT bandit with negative-KL reward.
