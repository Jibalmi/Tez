# Research notes — commissioned literature surveys (September 2026)

Seven parallel surveys were run to ground `docs/REPORT.md`. They are reproduced here as the primary
source material, lightly reformatted. **Caveat:** these are research-assistant outputs. Pre-2026
papers cited are well known; arXiv identifiers dated 2026 were surfaced by the surveys and, except
where noted, were not individually opened. Click before relying on any specific number.

---

## 1. The mechanism — symbol-logit scoring and its known biases

### Foundations
- **Schick & Schütze, PET** — [arXiv:2001.07676](https://arxiv.org/abs/2001.07676) · EACL 2021. Origin of the pattern–verbalizer pair: a *pattern* turns the input into a cloze, a *verbalizer* maps each class to a label token; you read that token's probability. Models predict *semantically meaningful* verbalizers better than bare numbered labels — `A/B/C` is the weakest verbalizer.
- **Gao, Fisch & Chen, LM-BFF** — [arXiv:2012.15723](https://arxiv.org/abs/2012.15723). Automates template + label-word search; label-word choice alone swings accuracy by many points.

### The known biases
- **Holtzman et al., Surface Form Competition** — [arXiv:2104.08315](https://arxiv.org/abs/2104.08315) · EMNLP 2021. Scoring option *text* splits probability across synonymous surface forms. Fix = PMI-DC: `log p(option|ctx) − log p(option|domain_premise)`, one extra pass per option. Symbol scoring largely *sidesteps* this bias.
- **Zhao et al., Calibrate Before Use** — [arXiv:2102.09690](https://arxiv.org/abs/2102.09690) · ICML 2021. Majority/recency/common-token bias swings few-shot accuracy up to 30 pts. Contextual calibration: run a content-free input ("N/A"), get `p_cf`, apply `W = diag(p_cf)^-1`. One extra pass per template. *(Our experiment §3.2 of the report: harmful when an "insufficient" option is a valid answer to empty input.)*
- **Min et al., Noisy Channel Prompting** — [arXiv:2108.04106](https://arxiv.org/abs/2108.04106). Score `p(input|label)`; lower variance, better with imbalanced labels; one full pass per option — incompatible with one-pass symbol reading.
- **Robinson & Wingate, MCSB** — [arXiv:2210.12353](https://arxiv.org/abs/2210.12353) · ICLR 2023. Multiple-choice symbol binding: the ability to bind `A` to option A's content. Models vary hugely; high-MCSB models do far better with symbol scoring than per-option likelihood. **Measure MCSB on the chosen model before committing.**
- **Zheng et al., "LLMs Are Not Robust Multiple Choice Selectors"** — [arXiv:2309.03882](https://arxiv.org/abs/2309.03882) · ICLR 2024 Spotlight · [code](https://github.com/chujiezheng/LLM-MCQ-Bias). Selection bias is mostly *token* bias over option-ID tokens, not position. **PriDe**: label-free, inference-time; estimate the ID prior by permuting option contents on ~5 % of samples, subtract for the rest. Near-zero amortised cost.
- Newer: [arXiv:2406.01026](https://arxiv.org/pdf/2406.01026) (strengthened symbol binding); [arXiv:2407.15018](https://arxiv.org/abs/2407.15018) (mechanistic account, middle layers); **Wong, Nouwen & Gatt, "When Models Decide and When They Bind"** [arXiv:2601.03914](https://arxiv.org/abs/2601.03914) — *verified directly*: two-stage computation, winner selected in content space then bound to a symbol; symbol-logit reading taps only stage two. **Lee & Son** [arXiv:2604.14634](https://arxiv.org/abs/2604.14634): position bias toward early options grows sharply with option count. [arXiv:2608.11947](https://arxiv.org/abs/2608.11947): accuracy and order-sensitivity diverge under label-free strategies — measure both.

### Scoring variants
| Variant | Cost | When it wins |
|---|---|---|
| Softmax over symbol logits | 1 pass total | High-MCSB models; many options; latency-critical |
| Raw option-text log-likelihood | 1 pass/option | Low-MCSB models |
| Length-normalised (`acc_norm`) | 1 pass/option | Unequal-length options; improves calibration, often hurts accuracy ([EleutherAI](https://blog.eleuther.ai/multiple-choice-normalization/)) |
| PMI-DC | 2 passes/option | Free-form / synonym-rich options |
| Channel | 1 pass/option | Imbalanced labels |

No universal winner — Tsvilodub et al. [arXiv:2403.00998](https://arxiv.org/abs/2403.00998). Tokenizer boundary: score the exact token id for the symbol as it appears (leading space, newline) — [arXiv:2509.15020](https://arxiv.org/pdf/2509.15020).

### Jev, documented vs inferred
Marketing blog only ([typesafe.ai/blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)): "System One model", non-autoregressive, "parallel sampler", typed outputs, calibrated confidence, trained with **RLCD**, ~200×/~400× claims, public evals at evals.typesafe.ai. **No paper, no model card, no technical report, no training details, no calibration curves** as of Sept 2026. Mechanism is inferred by analogy with SemIf.

### Recommended scoring recipe
1. Measure MCSB first. 2. One forward pass, softmax over symbol-token logits restricted to the declared set; score the exact token id. 3. Cheap fixes: contextual calibration *(with the caveat found experimentally)*, PriDe, per-workload temperature. 4. Expensive only if justified: full permutation voting, PMI-DC, channel. 5. Evaluate on accuracy *and* order-robustness; stress-test with more options than production.

---

## 2. RL for calibration, and what Jev's "parallel sampler" plausibly is

### Training-time calibration
- **RLCR** ([2507.16806](https://arxiv.org/abs/2507.16806), ICLR 2026): GRPO with Brier-on-verbalised-confidence reward; HotpotQA ECE 0.03 vs 0.37 (binary RLVR); but calibrates a *verbalised* number, and no temperature-scaling baseline is run.
- **Rewarding Doubt** ([2503.02623](https://arxiv.org/abs/2503.02623)); **DCPO** ([2603.09117](https://arxiv.org/abs/2603.09117)) — gradient conflict between accuracy and calibration, decoupling fixes it; **CAPO** ([2604.12632](https://arxiv.org/abs/2604.12632)); reward-function study ([2607.04332](https://arxiv.org/abs/2607.04332)) — "confidence reward hacking"; forecasting ([2607.00164](https://arxiv.org/abs/2607.00164)) — naive Brier-RL *degrades* calibration in practice.
- **Most on-point** ([2601.13284](https://arxiv.org/abs/2601.13284), AWS, Jan 2026): Qwen3 1.7–8B, MCQ + moderation, calibration measured on the *decision-token probability*. RLVR → overconfident (CSQA ECE 24.4); SFT 7.4; calibration-aware GRPO 16.0. **GRPO + isotonic → 4.8, theirs + isotonic → 3.4: post-hoc closes most of the gap.**
- Older: PPO-M/CDPO ([2410.09724](https://arxiv.org/abs/2410.09724)); focal-loss vs temperature ([2408.11598](https://arxiv.org/abs/2408.11598)).

**Verdict:** training-time calibration is real but gains are mostly vs *uncalibrated* RL. For a frozen harness, post-hoc captures most of the value.

### What "parallel sampler" plausibly is (inference only), ranked
1. **Decoder reading option-symbol logits, one prefill, no decode** — most plausible. JevBench v1.3 has SemIf (frozen Qwen3.5-4B) at 73.1 vs Jev 74.4; Jev's 70–500 ms P95 and "output tokens free" pricing match a 4–9B prefill with zero decode. "RLCD" = fine-tuning with a proper scoring rule on label-restricted logits (what reflex did: [github.com/kshetrajna12/reflex](https://github.com/kshetrajna12/reflex)). "Parallel" = many questions/options in one batched pass.
2. Bidirectional backbone with per-option marker heads (Laya: ModernBERT + [MASK] markers + scorer MLP) — consistent, but pure encoders score 40.6 / 23.1 on JevBench.
3. Masked-diffusion LM — feasible (djev), less likely for a startup.
4. Medusa/EAGLE/MTP heads — draft-quality, not calibrated. Low.
5. Cross-encoder reranker — N passes per question. Lowest.

### Discrete diffusion for decisions
DiffusionGemma: 26B MoE, 3.8B active, masked discrete diffusion on Gemma 4, 256 tokens/forward. A masked-diffusion model answers an MCQ in **one forward pass** (mask only the answer slot; LLaDA conditional likelihood [2502.09992](https://arxiv.org/abs/2502.09992)). djev does this via vLLM PR 57250; djev-dev reports 76.9 ms p50 quantised on B200. **Calibration is the weak point**: djev Calibration 65.4 vs Jev 82.7 vs SemIf 72.6. Real path for speed on 24 GB GPUs; must be post-hoc calibrated.

### Fine-tuning a frozen 12B
- Start from **base weights**: post-training wrecks MCQ logit calibration (GPT-4 report Fig. 8; [2311.13240](https://arxiv.org/abs/2311.13240)).
- Symbol binding / order: PIF ([2406.01026](https://arxiv.org/abs/2406.01026)); label-free LoRA on a Permutation Bias Metric cuts bias ~58 % ([2511.21709](https://arxiv.org/abs/2511.21709)); PA-GRPO ([2603.21016](https://arxiv.org/abs/2603.21016)).
- Calibration of logits: calibration-tuning ([2406.08391](https://arxiv.org/abs/2406.08391), ~1k graded examples, LoRA r=8); Thermometer ([2403.08819](https://arxiv.org/abs/2403.08819)).
- **Field evidence (reflex):** LoRA + log-loss on label logits, ~800 examples; in-distribution acc 62.7→76.8, ECE 0.120→0.051; **but on held-out JevBench the frozen model beat the LoRA** (0.658 vs 0.604). The author rejected the adapter and serves frozen + two-permutation averaging.

**Recommendation:** post-hoc calibration + debiasing is sufficient for v1. Training only as an optional stage: LoRA on Gemma-4-12B base, proper scoring rule on label-restricted logits, permutation-augmented, ~1–10k examples, always followed by temperature/isotonic, gated on an OOD hold-out.

---

## 3. Voice → instant action

### Prior art (speech-to-intent)
| System | Accuracy | Latency / hardware | Licence |
|---|---|---|---|
| [Picovoice Rhino](https://picovoice.ai/docs/faq/rhino/) (audio→intent) | >99 % clean, 97 % at 9 dB; [97.3 % vs Lex 84.3 % / Dialogflow 77.3 %](https://github.com/Picovoice/speech-to-intent-benchmark) | Cortex-M4 to desktop; built-in endpointing | Proprietary; [free tier 3 users; commercial from $6k](https://picovoice.ai/pricing/) |
| [Snips NLU](https://ar5iv.labs.arxiv.org/html/1805.10190) | F1 0.877 | 60 ms on Raspberry Pi 3 | Apache-2.0, archived |
| Rasa Open Source (DIET) | comparable | CPU, tens of ms | Apache-2.0 |
| E2E SLU, [Fluent Speech Commands](https://ar5iv.labs.arxiv.org/html/1904.03670) | 98.8 % audio→intent | small CNN/RNN | public |
| [Streaming E2E SLU (IJCAI'21)](https://arxiv.org/html/2105.10042) | 98.9 % | online | research |
| Keyword spotting ([BC-ResNet](https://arxiv.org/pdf/2106.04140)) | 96.6–98.7 % on 12 commands | 12 ms on Cortex-M7 | Apache/MIT |

For 10–100 fixed intents, 97–99 % at < 100 ms decision time has been solved since ~2019.

### Where the time goes
- **Endpointing**: silence timeouts [300–800 ms, "often the single largest contributor"](https://soniox.com/wiki/endpoint-detection); [Pipecat Smart Turn v3 ~65 ms, BSD-2](https://huggingface.co/pipecat-ai/smart-turn-v2); [LiveKit needs a 0.25 s silence floor](https://docs.livekit.io/agents/build/turns/turn-detector/).
- VAD: [Silero < 1 ms per 30 ms chunk, MIT](https://github.com/snakers4/silero-vad).
- ASR: [Moonshine v2 fixed latency Tiny 50 ms / Small 148 ms / Medium 258 ms](https://arxiv.org/abs/2602.12241v1), [MIT](https://github.com/moonshine-ai/moonshine); Kyutai STT 0.5 s delay; Parakeet TDT 0.6b-v3 batch-oriented.
- Intent: embedding classifier [< 14 ms](https://arxiv.org/abs/2608.30738).
- Perception: [Nielsen 0.1 s feels instantaneous](https://www.nngroup.com/articles/response-times-3-important-limits/); human turn gaps median +100 ms ([PNAS](https://www.pnas.org/doi/10.1073/pnas.0903616106)).

Realistic silence-gated floor on consumer hardware: 250–300 ms endpoint + 50–150 ms ASR + 15 ms decision + 5 ms grounding ≈ **350–500 ms**. Only early commit gets under 200 ms.

### Early commit
Streaming SLU spots > 30 % of intents before the utterance ends with no accuracy loss; Google's streaming intended-query detector ([2208.13322](https://arxiv.org/abs/2208.13322)) saves 600 ms; Alexa's speculative endpointer ~200 ms. Framework: early classification of time series — probabilistic classifier + confidence-threshold trigger ([2406.18332](https://arxiv.org/html/2406.18332v3)). **Calibration is the enabling ingredient.**

### Cascade and grounding
[semantic-router](https://github.com/aurelio-labs/semantic-router) ~100 ms vs ~5 s LLM; proactive-agent trigger 14 ms, 12–83× faster than LLM triggers ([2605.30152](https://arxiv.org/abs/2605.30152)); watch-assistant fallback study — 70.6 % of fallbacks unintended, filtered by a < 14 ms classifier ([2608.30738](https://arxiv.org/abs/2608.30738)); vLLM Semantic Router −47 % latency ([2510.08731](https://arxiv.org/html/2510.08731v1)). Grounding without an LLM: Windows Voice Access and Talon resolve app names against the running-app registry with grammar matching; CUA literature decouples planning from grounding (Agent S2, OS-Atlas a11y tree, UGround).

---

## 4. Universal interface and ecosystem

### Taxonomy
See report §4.1. Key sources: [typed-decisions dataset](https://huggingface.co/datasets/LocalLLaMA/typed-decisions); [TypeSafe confidence doc](https://docs.typesafe.ai/confidence.md); [probe-cascade abort](https://arxiv.org/abs/2607.06503); [LangChain harness with Jev](https://www.langchain.com/blog/building-a-harness-with-jev); [AutoRelAnnotator](https://arxiv.org/abs/2606.25871); [beri audit — decomposition 62.6 % → 95 %](https://www.beri.net/article/typesafe-jev-typed-decision-model-calibration-decomposition-shadow-eval); [jev-ultrafast](https://github.com/browser-use/jev-ultrafast); [Langfuse evals](https://langfuse.com/blog/2026-09-18-using-typesafes-jev-for-evals).

### API comparison
- **Jev**: `POST /v1/systemone {state, model, questions:{id:{type: noul|choice|score, instructions, criteria}}}` → answers with probabilities + confidence; 255 choices, 10 levels, 32k state, text-only ([api.md](https://docs.typesafe.ai/api.md)). Confidence = concentration statistic, not P(correct).
- **SemIf**: `{id, state, question, options:[{id, description}]}` → per-option probs.
- **GLiClass**: label list, single/multi-label, threshold → `[{label, score}]`; GLiNER2 composable schema ([2507.18546](https://arxiv.org/abs/2507.18546)).
- Structured outputs / Outlines / Instructor: shape guarantee, still autoregressive, no calibrated distribution.
- Extensions worth adopting: `depends_on`/`ask_if`, `alone`, `think:N` (djev); `images`, `samples`, `sequential` (razorback16/openjev); act/escalate head (Laya); separate confidence head (Verdict 2.0).

### Integration patterns
Cascade with calibrated threshold ([AutoRelAnnotator](https://arxiv.org/abs/2606.25871); [UCCI](https://arxiv.org/abs/2605.18796)); early abstention −13 % cost, −5 % error ([Zellinger et al.](https://arxiv.org/abs/2502.09054)); recall-controlled probes 55–60 % token savings ([Doomed from the Start](https://arxiv.org/abs/2607.06503)); shadow eval on 1–2k historical decisions (beri); caching 1.2–4.2× at 1–24 accuracy points cost ([decider](https://github.com/Mapika/decider)). Failure stories: version-alias drift, prompt injection via state, double negatives, wrong criteria < 25 %, "context rot", no abstention (beri, Langfuse, HN).

### Ecosystem snapshot (22 Sept 2026)
| Project | Base | Licence | Claimed | Does NOT do |
|---|---|---|---|---|
| [SemIf](https://github.com/TheoLeeCJ/SemIf) | Qwen3.5-4B frozen | MIT | bal-acc 0.813; JevBench 73.1 vs Jev 74.4 | multi-label, abstain, batch API, built-in calibration |
| [Laya](https://huggingface.co/convaiinnovations/laya) | ModernBERT-L 421M | Apache-2.0 | 0.766 typed-decisions, 33 ms | zero-shot ≈ random; > 50 options; 512 ctx |
| [Verdict 2.0](https://github.com/Heman10x-NGU/openJev-verdict-2.0) | ModernBERT 150M | NOASSERTION (GitHub) | 77.1 %, ECE 0.014 (head) | 512 ctx; 48 % on TypeSafe evals |
| [djev](https://github.com/mmastrac/djev) | DiffusionGemma 26B-A4B | Apache-2.0 | JevBench 73.0; 94 ms p50 | 24 GB GPU; calibration 65.4 |
| [decider](https://github.com/Mapika/decider) | Qwen3.5-2B SFT+RL | Apache-2.0 | 4 ms/3q | hard 0.459 |
| [kotoba DeBERTa](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large) | DeBERTa-v3-L | Apache-2.0 | 0.854 in-domain | 0.69 OOD |

Jev: $0.042/MTok in, output free; early access; **MCA forbids distillation, imitation training, competing products** ([MCA](https://typesafe.ai/legal/mca)).

---

## 5. Evaluation bar

| Benchmark | Owner / licence | Size | Ground truth | Tests |
|---|---|---|---|---|
| [JevBench v1.2/1.3](https://github.com/fstandhartinger/jevbench) | one-person project, MIT | 534 (72 easy, 96 standard, 146 judge, 220 hard; 133 held-out by hash) | LLM-written, frozen and hashed | ECE+Brier yes; order no; abstention no; latency ×2 *assumption* for self-hosted |
| [evals.typesafe.ai](https://evals.typesafe.ai/) | vendor | undisclosed | agreement with GPT-6 Astra / Fable 5.1 | accuracy only |
| SemIf frozen matrix | MIT | 706 rows + 108 perturbations | human-authored + upstream | balanced acc, macro-F1, NLL/Brier, TV distance, bootstrap CIs |
| [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | Apache-2.0 | 1,600 synthetic | LLM teacher | contaminated if trained on |
| WANLI, BFCL, MMLU/ARC, RouterBench, guardrail sets, Mind2Web/ScreenSpot | various | — | — | see report §5 |

Scorecard, contamination plan and current-numbers table: report §5. Key precedents: [Smooth ECE](https://arxiv.org/abs/2309.12236); [HELM](https://arxiv.org/abs/2211.09110); [AUGRC](https://arxiv.org/abs/2407.01032); [LiveBench rotation](https://arxiv.org/abs/2406.19314); [exchangeability test](https://arxiv.org/abs/2310.17623); [Kapoor & Narayanan](https://arxiv.org/abs/2207.07048).

---

## 6. Gemma 4 12B specifics

- Family launch 31 Mar / 2 Apr 2026; **12B "Unified" added 3 Jun 2026** ([releases](https://ai.google.dev/gemma/docs/releases)). Tech report [arXiv:2607.02770](https://arxiv.org/abs/2607.02770). HF ids `google/gemma-4-12B-it` / `google/gemma-4-12B`. 11.95B params, 48 layers, 5:1 local:global attention, 262k SentencePiece vocab, encoder-free multimodal projections. **Apache-2.0.**
- **Template** is `<|turn>role\n…<turn|>\n`; with thinking off the generation prompt ends with an **empty thought channel** `<|channel>thought\n<channel|>` — read the next position ([prompt format](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)).
- Whitespace is inside tokens: `"A"` ≠ `" A"`. "Mind the Gap" ([2509.15020](https://arxiv.org/html/2509.15020)): folding the space into the letter token cut Gemma 3 12B ECE 2.18→0.91.
- Instruct Gemmas open with `**`/prose often — "Look at the Text" measured 56.8 % first-token/text mismatch for Gemma-7b-it ([2404.08382](https://arxiv.org/html/2404.08382)). *(Not observed for Gemma 4 12B in our runs with the letter-only instruction: 0/144 leakage.)*
- Gemma 3 calibration under letter-logit readout: 27B base ECE 0.10 → instruct+template 0.245; 4B 0.06 → 0.42 ([2606.03437](https://arxiv.org/html/2606.03437v1)). Expect overconfidence.
- Ollama: `/api/generate` has `logprobs`/`top_logprobs` (max 20) since v0.12.11; `/v1` drops logprobs. llama-cpp-python: `llm.eval` then `llm.scores[n-1]` for full logits; Gemma 4 support from 0.3.25; 12B Unified conversion fixed in llama.cpp PR #24118 (4 Jun 2026). Ollama blobs: `~/.ollama/models/blobs/sha256-…`.
- Tags: `gemma4:12b` = `12b-it-q4_K_M` 7.6 GB; `12b-it-q8_0` 13 GB; `12b-it-bf16` 24 GB. Q8_0 fully offloaded ≈ 14 GB; BF16 does not fit 16 GB.

---

## 7. Local model landscape (pre-experiment survey)

- **GLi\* encoders (Knowledgator / Fastino):** GLiNER2.5 (74M / 0.2B / 0.3B, 22 Sep 2026), GLiFormer base/large (264M / 576M, 16 Sep 2026), GLiClass v2 modern-base/large (151M / 399M, ModernBERT, 8k ctx). Apache-2.0 on the Knowledgator/Fastino checkpoints; **older `urchade/gliner_*` are CC-BY-NC** ([verified](https://huggingface.co/urchade/gliner_base)). Independent bench (umstek/zero-shot-ie-bench, CPU): GLiNER2.5-base 83.3 % @ 111 ms; GLiClass-large 81.2 % @ 411 ms — vs Jev cloud 93.8 % @ 37 ms.
- **Direct clones:** Laya (322M ModernBERT, prompt-format-sensitive: 58.3 → 95.8 % on sentiment by changing input format); openJev-verdict-2.0 (149.6M ModernBERT + GLiClass heads, dual distribution/confidence heads, 20–25 ms, 77.1 %, Brier 0.064, ECE 0.014 — reference design, licence unclear); djev (DiffusionGemma).
- **Constrained decoding:** XGrammar default in vLLM/SGLang/TensorRT-LLM, < 40 µs/token ([2411.15100](https://arxiv.org/abs/2411.15100)); schema validity 78.6–92.9 % → 100 % ([2609.23742](https://arxiv.org/html/2609.23742)); does not remove token count.
- SetFit works (97 % with ~20 examples/class, 25 ms) but is effectively unmaintained. NeoBERT/mmBERT as backbones if training a head.
- **Benchmark warning:** option-order sensitivity is severe — one open model went 72 % → 21 % on reversed options. Test permutation robustness on your own data before believing any leaderboard.
- Systems: SGLang RadixAttention ([2312.07104](https://arxiv.org/pdf/2312.07104)) up to 6.4× on branching programs; vLLM APC hashes 16-token blocks; tree attention (Medusa [2401.10774](https://arxiv.org/pdf/2401.10774)); llama.cpp `--slot-save-path`; LMCache. Quantisation: FP8 KV logprob mismatch up to |Δ| ≈ 0.56 (vLLM #54035); cached ≠ fresh (vLLM #33123; [Thinking Machines batch invariance](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)). Early exit (CALM [2207.07061](https://arxiv.org/pdf/2207.07061), LayerSkip [2404.16710](https://arxiv.org/html/2404.16710v1)) corrupts confidence — do not use.
