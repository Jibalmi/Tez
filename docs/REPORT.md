# Tez — an open, local "System One" decision layer

**Research, experiments and design for a self-hosted alternative to TypeSafe AI's Jev**
Miguel Villax · September 2026 · github.com/Jibalmi/Tez

> **Revision note (22 Sept, afternoon).** The first version of this report was audited before
> release and two things were wrong. (1) `metrics.py` keyed balanced accuracy on the gold *position*
> instead of the option *id*, which inflated every headline number by ~1–3 points; all tables below
> are the corrected values (Gemma 4 12B Q8_0 is **0.943**, not 0.953). (2) The "~110 ms fixed floor /
> no partial-prefix reuse" conclusion in §3.4 was an artefact of Gemma's sliding-window attention
> cache, fixed with one server flag; the true per-word cost is **~30 ms**. Section 3.7 lists every
> audit finding. Nothing that was right has been re-stated more strongly than the data allows.

---

## 0. TL;DR

A "System One" model answers a typed question — *which option, yes/no, what level* — by reading a
probability distribution straight out of one forward pass, decoding nothing. Jev (TypeSafe AI,
launched 19 Sept 2026) commercialised this. It is hosted-only, closed-weight, single US region, and
its terms forbid using its outputs to build a competitor. The technique itself is public, older than
Jev, and works locally today.

**Decision quality**, laptop RTX 5080 (16 GB), zero training, SemIf's exact prompt and public
fixtures (authored144: 144 three-option decisions in three families; 95 % CIs by source-group
bootstrap):

| Model (frozen) | Authored mean-family bal. acc | Perturbation set | NLL raw → best\* | Brier raw → best\* | Reversal flip (36 pairs) | p50 fresh |
|---|---:|---:|---:|---:|---:|---:|
| **Gemma 4 12B Q8_0** | **0.943** (0.897–0.981) | **0.992** | 0.478 → **0.158** | 0.094 → **0.075** | **0 / 36** (≤ 9.6 %) | 110 ms |
| Gemma 4 12B Q4_K_M | 0.918 (0.871–0.958) | 0.981 | 0.549 → 0.184 | 0.130 → 0.092 | 1 / 36 | 100 ms |
| Qwen3.5-4B BF16, in-process (SemIf's own model, reproduced) | 0.813 (0.755–0.865) | — | 0.427 | 0.248 | — | 224 ms† |
| Qwen3.5-4B BF16 + 6-permutation average | **0.912** | — | — | — | — | 321 ms† |
| Gemma 3 4B Q4_K_M | 0.646 (0.570–0.728) | 0.729 | 3.433 → 0.696 | 0.661 → 0.397 | 13 / 36 (36 %) | 40 ms |
| Llama 3.2 3B Q4_K_M | 0.358 (0.319–0.399) — chance | 0.496 | 1.673 → 1.043 | 0.827 → 0.627 | 10 / 36 (28 %) | 25 ms |
| *SemIf published: Qwen3.5-4B / EXL3 27B* | *0.813 / **0.958*** | *0.766 / —* | | | | |

\* best = 6-permutation average + out-of-fold temperature (§3.2). † transformers/PyTorch, not llama.cpp; same 144 rows.

**Voice → action** (the use case built and measured in §4.4): 220 spoken commands over 16 actions,
Gemma 4 12B Q8_0, transcript placed *after* the action list so it is the only uncached suffix:

| | value |
|---|---:|
| Intent accuracy on the full utterance (strict / accepting the `then`/`alt` label of compound and ambiguous rows) | **0.905 / 0.982** |
| Out-of-scope detection (recall / precision) | 1.00 / 0.92 |
| Decision cost per streamed word (engine compute / HTTP round-trip, p50) | **30 ms / 45 ms** (was 205 / 230 before the cache fix) |
| Class-aware early-commit policy, 198 actionable utterances: harmful actions | **1** (an ASR-error row) |
| … final action consistent with the label | 0.985 |
| … first action fires at mean word 3.6 of 5.4; 53 % at or before the human-annotated commit word | |
| … false actions on 22 out-of-scope utterances | **2** ("youtube …", "my notes …" prefixes) |
| Compound commands ("open notes and type X") as two decisions on the residual | 22 / 22 route to the second action |
| ASR (faster-whisper base.en, GPU): WER on synthetic speech / intent agreement ASR-vs-text | 7.8 % / 97.3 % |
| Live demo: "open notes and type hello world" (wav) → Notepad launched at **+652 ms**, text typed at end of speech | |

Six findings that should shape the build:

1. **The technique works and a frozen 12B is at 0.943 on SemIf's benchmark** — clearly above their
   4B (0.813, p < 10⁻⁴ paired) but **not** a new best: SemIf's README lists an EXL3 27B at 0.958,
   and the 12B's edge over the 4B shrinks to *not significant* once the 4B gets permutation averaging
   (0.912, p = 0.18). A large share of the "scale" gap is option-order bias, which is cheap to remove.
2. **Symbol-logit decisions are scale-gated.** Llama 3.2 3B is at chance with 28 % order sensitivity;
   Gemma 3 4B is at 0.646 with a third of decisions flipping under reversal. Calibration cannot
   create knowledge. Screen the backbone for symbol binding (§2.3) before anything else.
3. **Raw confidence is overconfident but informative**; 6-permutation averaging + temperature gives
   NLL 0.158 / Brier 0.075 / ECE 0.037 and the best error detection (AUROC 0.906). One published
   method, Zhao et al.'s contextual calibration, is *harmful* here (0.951 → 0.778) for a reason worth
   understanding (§3.2).
4. **Prompt order is a systems decision.** Put the constant part (instructions, options) first and the
   variable part (transcript, evidence) last; with a sliding-window model use `--swa-full`. That took
   the per-word voice decision from 205 ms to 30 ms *and* raised accuracy (0.868 → 0.905), because the
   options now precede the evidence the model must bind them to.
5. **Early commit works only with one rule:** a confident `none` on a partial transcript means *keep
   listening*, never *do nothing*. With that, plus action classes (open = cheap and refinable; media
   and play = need two consistent partials; type/close = end of utterance only), the assistant opens
   Notepad while the user is still saying "and type …" with one harmful action in 198.
6. **The residual risk is end-pointing, not the decision model.** The two false actions on
   out-of-scope speech are prefixes that *are* commands until the sentence continues ("youtube is
   down again"). That is a property of language, and the fix is a cheap-action-first policy, not a
   better classifier.

The defensible open project is therefore **not another model — it is a scoring harness**: frozen
local backbone, symbol-logit readout, permutation debiasing, per-schema calibration, a conformal
abstention contract, a prefix-ordered prompt with real KV reuse, and a decision log. Nobody in the
current ecosystem ships that stack (§5.4).

---

## 1. What Jev is — documented vs inferred

**Documented** (docs.typesafe.ai, marketing blog; no paper, no model card, no technical report):

- API: `POST /v1/systemone {state, questions: {id: {type: noul | choice | score, instructions, criteria}}}`
  → per question a yes/no, a choice with per-option probabilities, or a score on up to 10 levels,
  plus a `confidence`. Up to 255 choices, 32k-token state, text only.
- **`confidence` is "a statistic computed from the distribution" — a concentration measure, not
  P(correct).** This matters: Jev does not claim what a calibrated P(correct) would claim.
- Claims: 70–500 ms, "20–200× faster / 40–400× cheaper than an LLM", output tokens free,
  $0.042 per million input tokens. Trained with "RLCD" (Reinforcement Learning for Calibrated
  Decisions). Uses a "parallel sampler".
- Hosted only, early access, single region (US West Coast), no self-host or open weights.
- The Master Customer Agreement **prohibits using outputs for distillation, imitation training or
  building a competing product.** Jev can never be a teacher for an open project.
- Independent numbers (JevBench, held-out tier): Intelligence 85.7, Calibration 82.7, score 74.4.
  A third-party audit found decomposition matters more than the model: a single "is this
  phishing?" question scored 62.6 %, five narrow sub-questions plus a rule scored 95 %.

**Inferred.** Everything about the architecture. The most plausible reading, by elimination
(§2 and the RL-training survey): a decoder LLM, one prefill, softmax over the logits of the option
symbols, no decode — i.e. what SemIf does explicitly — with a fine-tune on a proper scoring rule
over label-restricted logits, and "parallel" meaning many questions scored against one cached state.
SemIf, frozen and untrained, sits 1.3 points behind Jev on JevBench (73.1 vs 74.4). Diffusion-LM
readout (djev) is a real alternative but calibrates badly (65.4). Encoder-only clones score 23–41.

---

## 2. How the technique works, and what the literature says to do

### 2.1 Mechanism

Build a prompt: evidence, a criterion, options labelled `A, B, C…`, and an instruction to answer with
one letter. Run one forward pass. Take the logits at the first answer position, keep only the
declared letter tokens, softmax over them. That is the whole thing. SemIf's `direct.py` also checks
that every letter is a single round-trip token and that `prompt + letter` tokenises to `prompt` + one
token, so the read position is unambiguous. We reproduced that discipline in both harnesses (§3.1).

### 2.2 The reading list — what each paper changes in the build

| Paper | Finding | Consequence for us |
|---|---|---|
| Schick & Schütze, PET ([2001.07676](https://arxiv.org/abs/2001.07676)) | Pattern + *verbalizer*; models score meaningful label words better than bare symbols | `A/B/C` is the weakest verbalizer; meaningful labels are an ablation to run |
| Gao et al., LM-BFF ([2012.15723](https://arxiv.org/abs/2012.15723)) | Label-word choice swings accuracy by many points | Search label words on a dev set; do not hand-pick |
| Holtzman et al., Surface Form Competition ([2104.08315](https://arxiv.org/abs/2104.08315)) | Scoring option *text* splits mass across synonyms; PMI-DC fixes it at 1 extra pass/option | Symbol scoring *sidesteps* this — a genuine argument for the design |
| Zhao et al., Calibrate Before Use ([2102.09690](https://arxiv.org/abs/2102.09690)) | Content-free input exposes label bias; affine correction | Cheap — and **harmful here** (§3.2). Only apply when no option is a valid answer to empty input |
| Robinson & Wingate, MCSB ([2210.12353](https://arxiv.org/abs/2210.12353)) | Multiple-choice symbol binding is an ability models have to varying degrees | **Load-bearing.** Explains the scale cliff. Measure it before choosing a backbone |
| Zheng et al., PriDe ([2309.03882](https://arxiv.org/abs/2309.03882)) | Selection bias is mostly *token* bias on option IDs; estimate the prior on ~5 % of traffic and subtract | Near-free; on Gemma 4 the letter prior is already flat (§3.2), so it is a no-op there |
| Wong, Nouwen & Gatt ([2601.03914](https://arxiv.org/abs/2601.03914)) — verified directly | **Two-stage computation**: decide the winner in content space, then bind it to a symbol | Symbol readout taps only stage two; binding failure is its own error mode |
| Lee & Son ([2604.14634](https://arxiv.org/abs/2604.14634)) | Position bias grows sharply with option count | Short lists hide it; stress-test with more options than production uses |
| "Mind the Gap" ([2509.15020](https://arxiv.org/abs/2509.15020)) | Space-folded vs bare letter tokens differ; wrong choice ruins calibration | Read the exact token id the template produces (§3.6) |
| Tsvilodub et al. ([2403.00998](https://arxiv.org/abs/2403.00998)) | MCQ answers are not robust across scoring rules | Fix one rule; single-token options avoid length normalisation entirely |

### 2.3 Why small models fail: the binding cliff

SemIf's own ladder: Qwen3-0.6B **0.440**, MiniCPM5-2B 0.686, Qwen3.5-4B 0.813. Ours: Llama 3.2 3B
**0.358**, Gemma 3 4B 0.646, Gemma 4 12B 0.943. The two-stage account explains it: a small model may
still *know* the answer in content space and fail to route it to the letter. Reading logits only sees
the routing. Two implications: (a) the backbone must be MCSB-screened, not chosen by parameter count
alone; (b) a *trained* head on a small encoder (GLiClass-class) side-steps the emergence requirement,
which is why 150–400 M encoders compete at all — at the price of needing training data.

### 2.4 Calibration — what "calibrated" has to mean

From the calibration survey (full report in the research notes):

- Temperature scaling per **schema/question type**, fitted out-of-fold with folds split by source
  group, is the baseline. It cannot move accuracy; it fixes confidence only.
- **Conformal prediction** maps directly onto an orchestrator: prediction set of size 1 → act;
  size > 1 → escalate; empty → fall back to the full LLM. Mondrian (class-conditional) so rare,
  costly classes get their own quantile. ~1–2k labelled rows per group.
- Report **log-loss and Brier** as primary, reliability diagrams with CIs, risk–coverage and AURC/AUGRC
  for abstention. Binned ECE is biased and binning-sensitive — secondary only.
- Calibration dies silently under distribution shift. Monitor the *score distribution*, and route a
  sampled slice of auto-acted cases to humans too, or you only ever relabel the hard tail.
- RL-for-calibration (RLCR, DCPO, CAPO) is real but reported gains are mostly versus *uncalibrated*
  RL; the AWS decision-token study ([2601.13284](https://arxiv.org/abs/2601.13284)) shows post-hoc
  isotonic closes most of the gap (GRPO + isotonic ECE 4.8 vs calibration-aware GRPO + isotonic 3.4).
  **Post-hoc is sufficient for v1.** If fine-tuning later: base weights not instruct, proper scoring
  rule on label-restricted logits, permutation-augmented, gated on an OOD hold-out — one public
  attempt (reflex) shipped a LoRA that lost to its own frozen model out of distribution.

---

## 3. Experiments

### 3.1 Setup

- **Hardware.** NVIDIA RTX 5080 Laptop GPU (16 GB), Intel Core Ultra 9 275HX (24 cores, AVX2,
  **no AVX-512**), driver 592.01, Windows 11.
- **Engines.** (a) llama.cpp build **b11100** (22 Sept 2026), `win-cuda-13.4-x64`, `llama-server`
  `-ngl 99`, one slot, `cache_prompt=false` for every accuracy and calibration run (fresh scoring);
  `-c 2048 -b 512` for Q8_0 (14.2 GB VRAM). (b) `transformers` 5.8 / torch 2.10 in-process, BF16,
  `attn_implementation="sdpa"`, `logits_to_keep=1`, CUDA-synchronised timing — used to reproduce
  SemIf's own number on SemIf's own model and for batched permutations.
- **Models.** Gemma 4 12B: Ollama's Q4_K_M blob and a Q8_0 GGUF (both served directly by llama-server;
  Apache-2.0; thinking disabled). Gemma 3 4B: fresh `ggml-org` Q4_K_M GGUF (Ollama's blob is an old
  conversion llama.cpp refuses). Llama 3.2 3B Q4_K_M. Qwen3.5-4B BF16 from the Hub. Every server run
  asserts the loaded `model_path` from `/props` and records it in the manifest.
- **Prompt.** Byte-identical to SemIf's `direct-options-v1`: their system instruction and JSON
  payload (`evidence`, `criterion`, `options[{letter, description}]`), built by importing SemIf's own
  `core.py`, rendered into each model's chat template with the generation prompt appended. For
  Gemma 4 the system text is folded into the user turn (Gemma 3 convention); a native system-turn
  rendering was also run (`gemma4sys`) and changed one row of 144 (0.938 vs 0.943).
- **Readout.** llama-server: `/completion` with `n_predict=1, temperature=0, n_probs=200`; take
  `completion_probabilities[0].top_logprobs`, keep the declared letters, softmax. Because logprobs
  are already log-softmax over the vocabulary this equals SemIf's softmax over raw slot logits.
  transformers: last-position logits restricted to the slot ids, softmax. Both log top-1 token and
  any letter missing from the top-k; on the benchmark rows no letter was ever missing and top-1 was
  always a declared letter for every model in the tables.
- **Data.** SemIf's committed, MIT-licensed fixtures: `authored144.jsonl` (144 rows, 3 options each,
  three families × 48: evidence interpretation, rule application, candidate selection) and
  `perturbations108.jsonl` (36 originals × option reversal, criterion re-wrap, irrelevant context;
  frozen before outputs). Gold labels index each row's own option order. **Label provenance:** SemIf
  describes these as model-reviewed, not human-adjudicated; the data are public, so this is a
  reproduction against a published baseline, not a held-out claim.
- **Contamination.** SemIf's data was authored in September 2026; Gemma 4 12B was released 3 June
  2026. The evaluation post-dates the model.
- **Metrics.** SemIf's headline, *mean-family balanced accuracy* (mean over families of mean
  per-class recall, **classes keyed by option id**), with a source-group bootstrap 95 % CI; NLL and
  Brier primary; ECE-15 secondary; option-id-matched flip rate with Wilson 95 % CI and
  total-variation distance for perturbations; risk–coverage, AURC, AUROC of max-probability as an
  error detector; exact McNemar for paired model comparisons on the same rows.
- **Determinism.** Three independent Q8_0 runs of authored144 were byte-identical in every
  probability (`run2`, `run3` in `results/`).

Everything is reproducible from `experiments/run_direct.py`, `run_hf.py`, `metrics.py`,
`calibrate.py`, `perm_analysis.py`; every run writes a manifest with the data SHA-256, model path,
settings and latency summary.

### 3.2 Gemma 4 12B — headline, calibration, debiasing

**Accuracy** (authored144, fresh):

| | Q8_0 | Q4_K_M |
|---|---:|---:|
| Mean-family balanced accuracy | **0.943** (CI 0.897–0.981) | 0.918 (CI 0.871–0.958) |
| Plain accuracy | 0.951 | 0.931 |
| evidence_interpretation / rule_application / candidate_selection | 0.956 / 0.929 / 0.944 | 0.956 / 0.929 / 0.869 |

Q4 → Q8 on the same rows: 3 rows fixed, 0 broken, **McNemar p = 0.25** — the difference is in the
expected direction but not significant at n = 144. Candidate selection, the family where option
*ids* are literally "A" and "B" (the candidates' names) and can sit at the other slot letter after
shuffling, is where Q4 loses.

**Perturbations** (36 originals; Q8_0; flip = option-id argmax change; Wilson 95 % CI):

| Variant | Flips | Flip rate (CI) | TV distance | Acc. base → perturbed |
|---|---:|---:|---:|---:|
| option_reversal | **0 / 36** | 0 % (0–9.6 %) | 0.002 | 1.000 → 1.000 |
| criterion_wrapper | 1 / 36 | 2.8 % (0.5–14 %) | 0.025 | 1.000 → 0.972 |
| irrelevant_context | 0 / 36 | 0 % (0–9.6 %) | 0.006 | 1.000 → 1.000 |

Mean-family balanced accuracy on the perturbation set: **0.992** (CI 0.972–1.0); Q4: 0.981; SemIf's
4B: 0.766. Note the small n: "0 / 36" bounds the true reversal-flip rate at ≤ 9.6 %, not at zero.

**Order bias, measured properly.** All six orderings of every row were scored (`perm_analysis.py`):

| | value |
|---|---:|
| Mean letter mass over rows × orderings, A / B / C | 0.331 / 0.336 / 0.333 |
| Rows where all six orderings agree on the argmax | **95.8 %** |
| PriDe with the full-permutation prior | identical to raw (prior is flat) |

Gemma 4 12B has essentially no letter prior on this prompt; PriDe is a no-op, and permutation
averaging's value is in the 4 % of rows where orderings disagree — the borderline decisions.

**Calibration and debiasing** (same 144 rows; temperature fitted 5-fold out-of-fold by `group_id`;
plain accuracy):

| Method | Acc | NLL | Brier | ECE-15 | AUROC (err.) | AURC |
|---|---:|---:|---:|---:|---:|---:|
| raw | 0.951 | 0.478 | 0.094 | 0.047 | 0.834 | 0.013 |
| temperature (OOF) | 0.951 | 0.193 | 0.083 | 0.027 | 0.807 | 0.016 |
| 2-permutation average (orig + reversed) | 0.958 | 0.393 | 0.089 | 0.047 | 0.873 | 0.007 |
| 6-permutation average (mean) | 0.951 | 0.281 | 0.083 | 0.038 | **0.933** | **0.005** |
| **6-permutation average + temperature** | 0.951 | **0.158** | **0.075** | 0.037 | 0.906 | 0.006 |
| contextual calibration, "N/A" (Zhao) | **0.778** | 1.707 | 0.355 | 0.162 | 0.928 | 0.047 |
| contextual calibration, 3-placeholder average (Zhao's recipe) | 0.847 | 1.299 | 0.272 | 0.121 | 0.861 | 0.036 |
| … damped, α = 0.25 | 0.951 | 0.564 | 0.092 | 0.046 | 0.800 | 0.014 |

Reading it:

- **Raw is overconfident**: mean confidence 0.998 against accuracy 0.951. NLL 0.478 is driven by
  seven wrong answers held at p ≈ 10⁻⁴ for the gold option.
- **Temperature scaling fixes confidence for free** (NLL −60 %); accuracy untouched by construction.
- **Six orderings beat two** on every proper score and on error detection (AUROC 0.933 vs 0.873):
  the disagreement between orderings is itself the best uncertainty signal the frozen model has.
  It costs one batched forward pass of six prompts (§3.4).
- **Contextual calibration is a negative result on this workload, and averaging the three
  placeholders as Zhao et al. prescribe does not rescue it.** With the evidence replaced by "N/A",
  "[MASK]" or "", Gemma 4 chose `insufficient` in 144 / 144 / 143 rows at mean max-probability
  0.999. That is the *right* answer to an evidence-free question, not a label bias. Dividing by it
  penalises the 36 rows whose gold answer genuinely is `insufficient`. Damping the correction
  (α = 0.25) gets accuracy back and buys nothing. Rule: never apply content-free correction when the
  option set contains an "insufficient / unknown / none" answer that empty input should
  legitimately select — which, for a decision model built around abstention, is most option sets.

**Error analysis.** 7 misses in 144 at Q8; **six have gold = `insufficient`**: the model committed
to a candidate or a verdict when the evidence did not support one. All at p > 0.97. This is the
classic instruct-model failure — over-commitment — and it is precisely the failure that a calibrated
abstention layer exists to catch. Six-permutation disagreement flags it better than raw confidence.

### 3.3 Scale, the SemIf reproduction, and what permutation averaging is worth

Same 144 rows for every model; llama-server rows verified via `/props`; transformers rows via the
Hub model id:

| Model | Bal. acc (CI) | Reversal flip | Mean conf. | ECE raw | Perm-avg gain | p50 |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 12B Q8_0 | **0.943** (0.897–0.981) | 0 / 36 | 0.998 | 0.047 | +0.7 pt (2-perm) | 110 ms |
| Gemma 4 12B Q4_K_M | 0.918 (0.871–0.958) | 1 / 36 | 0.997 | 0.066 | +0.7 pt | 100 ms |
| Qwen3.5-4B BF16 (SemIf's model, in-process) | 0.813 (0.755–0.865) | — | 0.864 | — | **+9.9 pt (6-perm → 0.912)** | 224 ms† |
| Qwen3-4B BF16 (pure attention, in-process) | 0.740 (0.676–0.809) | — | — | — | **+9.2 pt (6-perm → 0.832)** | 75 ms† |
| Gemma 3 4B Q4_K_M | 0.646 (0.570–0.728) | 13 / 36 (36 %) | 0.990 | 0.336 | **+7.1 pt** | 40 ms |
| Llama 3.2 3B Q4_K_M | 0.358 (0.319–0.399) | 10 / 36 (28 %) | 0.827 | 0.390 | 0 | 25 ms |
| *SemIf: Qwen3-0.6B / MiniCPM5-2B / Qwen3.5-4B / EXL3 27B* | *0.440 / 0.686 / 0.813 / 0.958* | | | | | |

† transformers, BF16, `logits_to_keep=1`; not comparable with the llama.cpp timings.

Three things this table settles:

- **SemIf's published number reproduces exactly.** Their Qwen3.5-4B BF16 with their prompt through
  `apply_chat_template(enable_thinking=False)` gives 0.8132 in our harness against their 0.813.
  This anchors every other number in the report to a published baseline.
- **Paired significance** (exact McNemar on the 144 rows): 12B Q8 vs Qwen3.5-4B, 24 rows only the
  12B gets right vs 3 only the 4B — **p = 5 × 10⁻⁵**. 12B Q8 vs Gemma 3 4B: 44 vs 3, p = 2 × 10⁻¹⁰.
  12B Q8 vs Q4: 3 vs 0, p = 0.25 (not significant).
- **Permutation averaging closes most of the 4B → 12B gap.** Qwen3.5-4B with all six orderings
  batched into one forward pass goes 0.813 → **0.912** (18 rows fixed, 2 broken, p = 4 × 10⁻⁴), and
  the 12B's advantage over *that* is 7 vs 2 rows, **p = 0.18**. Gemma 3 4B gains 7 points the same
  way. In other words, roughly half of what looked like a scale effect is order bias that the small
  models have and the 12B does not, and it is removable for the cost of a wider batch. This is the
  cheapest accuracy available anywhere in the stack.

Llama 3.2 3B has no such reserve: nothing moves its accuracy. It reproduces SemIf's 0.6B result on a
different family and is the practical form of the MCSB finding — a sub-4B decoder is not a viable
backbone for symbol-logit decisions, however fast — while the 4B points show the cliff is not a step.

### 3.4 Latency and systems — what was wrong and what is true

The first version of this section reported a "~110 ms fixed floor" and "no partial-prefix reuse in
llama-server". Both were an artefact of one flag. Gemma 3/4 use sliding-window attention on most
layers; llama.cpp's SWA cache cannot be rolled back to an earlier position, so when a new prompt
shares a prefix with the cached one the server *silently re-evaluates the whole prompt*
(`timings.prompt_n` never drops). `--swa-full` allocates a full cache for those layers and prefix
reuse works as documented. Measured on Gemma 4 12B Q8_0, transcript-last voice prompt (§4.4):

| Condition | Tokens evaluated | prompt_ms p50 / p95 | HTTP round-trip p50 / p95 |
|---|---:|---:|---:|
| Default server, every streamed word | 394–397 (all) | 205 / — | 230 / — |
| `--swa-full`, every streamed word | **11** (word + `"` + template tail) | **30 / 36** | **45 / 56** |
| `--swa-full`, first word of a new utterance (shared header cached) | 11 | 31 | 45 |
| `--swa-full`, after an action fires (33-token context line invalidates one suffix) | 33 | ~55 | ~70 |
| `n_probs` 32 instead of 200 | 11 | 32 | 49 (no gain; 40 / 40 rows then lose letters — keep 200) |
| Same prompt, transcript *first* (SemIf order), cached | 358 | 148 / 179 | 221 / 252 |
| Fresh 142-token benchmark rows (accuracy runs) | 142 | 110 / 122 | 178 / 203 |

Conclusions that replace the old ones:

1. **Per-decision cost is proportional to the uncached suffix.** ~2.7 ms per token on this GPU for the
   12B at Q8, plus ~15 ms of HTTP/JSON. Sub-50 ms per streamed word is the measured state, not a
   projection. The remaining floor is the server round-trip; an in-process loop removes it.
2. **Prompt layout is the lever.** Constant material (instructions, option list, few-shot) goes first
   and is cached across *all* decisions; the variable material (evidence, transcript) goes last. For
   one-state-many-questions the state is the constant and the questions are the suffixes — the
   opposite layout — and SemIf's payload already does that. Both are the same rule.
3. **Batched permutations are cheap, and in-process prefix reuse is measurable.** Qwen3-4B
   (pure attention, transformers BF16, CUDA-synchronised): one fresh 148-token decision 75 ms p50;
   with the state's 63-token KV cached and only the 83-token suffix evaluated 54 ms; all six
   orderings in one batch 204 ms (2.7× the cost for +9 points). Cached vs fresh differed in argmax
   on 2 / 144 rows (mean TV 0.008) — SemIf's documented "cached ≠ fresh" regime, reproduced. On
   SemIf's evidence-first layout the shared prefix is short; the gain grows with the state length.
   On llama-server the six prompts can likewise share the cached header and differ only in the
   option block.
4. The 262k-entry Gemma vocabulary makes the full-vocabulary logits buffer 2 GB at batch 2048; a
   decision service only needs one position (`logits_to_keep=1`).

Quantisation (§3.2): Q4_K_M → Q8_0 moved probability mass on ~3 % of decisions and every argmax that
moved was a Q4 error being corrected, for ~10 ms more per decision. Weights at Q8 are the floor for
calibration work; KV cache and `lm_head` stay unquantised (FP8 KV caches distort logprobs by up to
~0.56 in published reports); cached vs fresh computation differs numerically, so SemIf's 0.7 % argmax
flip rate under prefix reuse is the expected regime, not a bug. Any change to quantisation or caching
is a new model: re-measure flip rate and calibration against a fresh reference.

### 3.5 An accidental ablation: wrong chat template

A scripting failure (§3.6) caused two runs labelled "Gemma 3" and "Llama 3.2" to actually hit the
still-running Gemma 4 server with the *other models' templates*. They are quarantined as invalid
scale-curve points, but they are an honest robustness observation about Gemma 4:

| Template fed to Gemma 4 12B Q4 | Top-1 token | Slot-restricted bal. acc |
|---|---|---:|
| Correct (`<\|turn>` + empty thought channel) | a letter, 144/144 | 0.918 |
| Gemma 3 (`<start_of_turn>`) | `<\|channel>` 144/144 | ~0.90 |
| Llama 3 headers | `<\|channel>` 142/144 | ~0.88 |

With the wrong template the model wanted to open a thinking channel, so the letter logits were read
*behind* a control token — and the restricted argmax was still right ~90 % of the time. Accuracy is
robust; the confidences from those runs are meaningless and were not used.

### 3.6 Things that went wrong and were caught

These cost hours and each one is a trap for anyone reproducing this:

- **`llama-cpp-python` prebuilt wheel crashed with `0xc000001d` (illegal instruction)** on both
  CPU and GPU: built with AVX-512, this CPU has none. The official llama.cpp release binaries
  dispatch per CPU at runtime and were used instead (CUDA 13.4 build for Blackwell).
- **Ollama caps `top_logprobs` at 20** and its OpenAI-compatible `/v1` endpoint drops logprobs
  entirely. With six options a confident model already pushed two letters off the list. Fine for
  argmax, not for calibration — hence llama-server with `n_probs=200`. With 16 options even 200 is
  not always enough (23 / 220 voice rows had one low-probability letter floored; harmless for the
  argmax, visible in calibration).
- **Gemma 4 is a thinking model.** Without the template's empty `<|channel>thought\n<channel|>`
  suffix, the first token is `<|channel>` at p ≈ 1 and a naive softmax over letters is
  meaningless. Ollama's `think:false` injects the suffix; a raw prompt must include it.
- **Bare vs space-prefixed letters are different tokens** (`'B'` −0.0 vs `' B'` −19.9 in one probe).
  Read the token the template actually produces.
- **Stale server processes.** `taskkill /IM …` under Git Bash silently fails (MSYS rewrites `/IM`
  as a path). Four `llama-server` processes were running; two "different model" runs were actually
  Gemma 4. Fixed by killing via PowerShell and **asserting the loaded model path from `/props`
  before every run** — now part of the harness discipline.
- **Ollama's Gemma 3 blob is an old conversion** missing `gemma3.attention.layer_norm_rms_epsilon`;
  today's llama.cpp refuses it. Fresh GGUFs from `ggml-org` are the safe source for calibration work.
- **Sliding-window cache defeats prefix reuse** unless `--swa-full` is set (§3.4). Always check
  `timings.prompt_n` in the response before believing any cache claim.
- **Qwen3.5-4B in transformers** falls back to an unoptimised path without `flash-linear-attention`
  and its hybrid cache cannot be deep-copied for prefix reuse; its timings are not representative and
  are not quoted as such. Qwen3-4B (pure attention) is the right small model for in-process work.
- **Python 3.14 + `importlib` module loading**: a module loaded from a path without being registered
  in `sys.modules` cannot define `@dataclass` (dataclasses resolve `cls.__module__`). Register it.
- **CTranslate2 (faster-whisper) needs CUDA 12 cuBLAS/cuDNN** next to a CUDA 13 torch; the pip
  `nvidia-*-cu12` wheels have the DLLs but Windows needs `os.add_dll_directory`. It also aborts at
  interpreter teardown alongside torch — flush and `os._exit`.

### 3.7 Audit findings on the first version of this report

| Finding | Severity | Resolution |
|---|---|---|
| Balanced accuracy keyed on gold *position* not option *id* | **High** — inflated every headline | Fixed in `metrics.py`; all tables regenerated (Q8 0.953 → 0.943, Q4 0.932 → 0.918, Gemma 3 0.665 → 0.646, Llama 0.458 → 0.358) |
| "12B beats the published baseline by 14 points" framed as method effect | Medium | SemIf also lists a 27B at 0.958; 12B is mid-scale. 4B + permutation averaging is within noise of the 12B (§3.3) |
| Q4 vs Q8 difference stated as a result | Medium | McNemar p = 0.25; reported as not significant |
| No CIs on flip rates | Medium | Wilson intervals added; "0 / 36" bounds ≤ 9.6 % |
| Labels described as authored; they are model-reviewed | Low | Stated in §3.1 |
| System text folded into user turn (not Gemma 4's native template) | Low | `gemma4sys` re-scored: 1 row of 144 changes |
| "~110 ms floor / no prefix reuse" | **High** for the voice claim | SWA cache; `--swa-full`; 30 ms per word measured (§3.4) |
| Determinism unverified | Low | 3 runs byte-identical |

---

## 4. Where it applies, and the interface

### 4.1 Decision types (from the universality survey)

| Decision | Context → decisions | Option set | Typical N | Confidence behaviour | If wrong |
|---|---|---|---:|---|---|
| Agent loop control (continue / review / stop) | one → many | fixed | 3–5 | act / escalate / don't | runaway or premature stop |
| Tool / agent / model selection | one → one | fixed | 3–20 | route; fall back to big model | wasted step |
| Risky-action gate | one → one | 2 | 2 | asymmetric: block on low P(safe) | destructive action |
| RAG relevance / chunk filtering | one → many | fixed levels | 2–5 | per-class cut-offs | silent context loss |
| Email / ticket triage | one → many | fixed | 3–50 | act / review / discard bands | mis-queue |
| Moderation / policy checks | one → many | 2 each | 2 | recall-first, human on doubt | harm or over-blocking |
| Form / intent routing | one → one | fixed | 20–80 | small encoders break here | mis-route |
| Guardrail pre-checks | one → many | 2 each | 2 | state is *untrusted input* | bypass |
| UI / computer-use action | one → two | **runtime** | ~10 ops × N targets | DONE needs independent check | wrong click, loops |
| Voice / home automation | one → one, **streamed** | fixed | 5–30 | low → keep listening | wrong device |
| Document classification | one → one/many | fixed | 5–255 | per-class calibration | mis-filing |
| Evidence-supports-X | one → one | 2–3 | 2–3 | NLI-style | false verification |
| A/B judge | one → one | 2–10 | 2–10 | position-flip rate matters | biased eval |

The single most important architectural finding across these: **decomposition wins**
(62.6 % → 95 % on phishing when one question became five). So the primitive is *one state → many
independent typed questions*, which is exactly the shape that needs prefix reuse — and which
llama-server does deliver once the cache is configured (§3.4).

### 4.2 The "one context → many" vs "one → one" rule, restated as a prompt-layout rule

An encoder cannot cache context: every option changes the whole representation. A decoder can cache
whatever is *constant across the decisions you are about to make* and score short suffixes.
Order-of-magnitude arithmetic: 400 M encoder × 4k context ≈ 1.6e12 param·tokens per decision; 4 B
decoder with a warm prefix scoring 8 options ≈ 3.2e11. **One context → many decisions favours the
cached decoder (~5×); one → one favours the encoder (~10×).** The measured voice case is the
degenerate version: the *option list* is the constant, the transcript is the suffix, and the decoder
wins by 7× against its own unordered prompt.

### 4.3 Recommended interface

Keep Jev's wire shape — the ecosystem (OpenRouter, Vercel AI Gateway, Pydantic AI) already speaks
it — and add what nobody ships:

- `choice` with `multi: true` and independent per-option probabilities;
- an implicit `__none__` / abstain option on every question — the audits name forced choice as the
  top failure, and our own errors were 6 of 7 "should have said insufficient";
- **two numbers per answer**: Jev-compatible distribution `confidence`, and a separately fitted
  `p_correct` with the id of the calibration set it came from;
- `depends_on` for conditional fan-out; `calibration_id` per workload; `prior` (what already
  happened in this episode) for multi-step decisions — measured to matter in §4.4;
- a **decision-log record** — inputs hash, options, distribution, chosen action, outcome — as a
  first-class output, because own-traffic data is the only thing that ever beat frontier models at
  routing (a 0.6 B router trained on 56k of its own queries: NDCG@10 0.771 vs 0.594 for Nova Lite,
  at 120 ms vs 684 ms).

### 4.4 Voice → instant action — built and measured

**The pipeline.** `data/voice/intents.json` defines 16 actions (open_notes, type_text, open_youtube,
play_youtube, open_spotify, play_liked_songs, play_spotify, pause/resume/next, volume up/down,
open_browser, search_web, close_app, none). `data/voice/commands.jsonl` has 220 labelled utterances
in six styles — canonical (77), paraphrase (55), noisy with fillers and ASR-like errors (33),
compound "open X and do Y" (22, labelled with the first action as `intent` and the second as `then`),
ambiguous (11, with an accepted `alt`), out-of-scope (22) — each with a human-annotated *commit
word* (the earliest word at which a listener knows the action). Every utterance was also synthesised
to 16 kHz WAV with Windows SAPI so the ASR stage could be timed on audio.

```
 mic / wav ──► 200 ms chunks ──► faster-whisper partial (base.en, GPU, ~40 ms)
                                        │ transcript so far
                                        ▼
        Gemma 4 12B Q8_0 · prompt = [instructions][16 actions][transcript]  (only the transcript is uncached)
                                        │ 16-way distribution, ~30 ms compute / ~45 ms round trip
                                        ▼
        commit policy ── 'none' on a prefix = keep listening
                      ── OPEN actions: fire at p ≥ 0.9 on one partial (cheap, refinable)
                      ── MEDIA / play_liked_songs: two consistent partials
                      ── play_youtube / play_spotify / search_web: open the app after two consistent partials, search at end
                      ── type_text / close_app: end of utterance only
                      ── after an action fires, later words are scored as a NEW decision with `prior` = that action
                                        │
                                        ▼
        deterministic slot extraction (regex on the final transcript) ──► OS action
```

**Decision quality on the full utterance** (`run_voice_fast.py`, 220 rows; "final" wording,
transcript last):

| | strict | accepting `then` / `alt` |
|---|---:|---:|
| All 220 | **0.905** | **0.982** |
| canonical / paraphrase / noisy | 1.000 / 0.964 / 0.970 | |
| compound (model routes to the *last* verb; both actions are right) | 0.273 | 1.000 |
| ambiguous | 0.818 | 1.000 |
| out-of-scope → `none` (recall / precision) | 1.000 / 0.917 | |

Two controls, same model, same rows: transcript *first* (SemIf's layout): 0.877, `none` precision
0.769, 148 ms per word. A "the user is still speaking, the transcript may be cut off" wording:
0.868, worse on every streaming metric too. The plain wording with the transcript last is the
configuration to use; the order change improved accuracy *and* removed 85 % of the latency.

**Streaming.** Every word prefix of every utterance was scored (1,158 decisions). What the model says
after one word is `none` at p ≈ 1 for 83 % of actionable utterances — correctly: "open" is not yet a
command. A naive "commit when p ≥ 0.9" therefore fires `none` at word 1 and is 17 % accurate. The
single rule that makes early commit work is *a confident `none` on a prefix never commits*.
With it (198 actionable utterances):

| Policy | Commits | Acc. at commit (strict / accepting) | Mean commit word (of 5.4) | At or before human word |
|---|---:|---:|---:|---:|
| non-none, p ≥ 0.9, stable 1 | 98.5 % | 0.918 / 0.928 | 2.5 | 93 % |
| non-none, p ≥ 0.99, stable 1 | 97.0 % | 0.932 / 0.938 | 2.7 | 89 % |
| non-none, p ≥ 0.9, stable 2 | 76.3 % | 0.960 / 0.967 | 3.5 | 21 % |

The wrong commits at stability 1 are almost all of one kind: "youtube …" → open_youtube before "play
lofi" arrives, "spotify" → open_spotify before "my liked songs". Those are not wrong actions in a
live assistant — opening YouTube while the user is still talking is the point — they are *refinable*
ones. That motivates the class-aware policy (`stream_policy.py`): actions are graded by what a wrong
early fire costs, and a later action that refines an earlier open (open_youtube → play_youtube) is
consistent, not contradictory.

| Class-aware policy, 198 actionable utterances | transcript last | transcript first |
|---|---:|---:|
| Utterances with a **harmful** action (executed action not in the accepted set nor an ancestor of it) | **1** (0.5 %) — "clothes the notepad", an ASR-error row | 2 |
| Final action consistent with the label | 0.985 | 0.965 |
| Mean word of first action | 3.6 | 3.6 |
| First action at or before the human commit word | 53 % | 53 % |
| Actions per utterance | 1.30 | 1.28 |
| Out-of-scope utterances (22) with a false action | **2** | 4 |

The two false actions on out-of-scope speech are "youtube is down again" → YouTube opens at word 1,
and "my notes from the meeting were …" → Notepad opens at word 2. Both prefixes *are* commands until
the sentence continues. This is end-pointing, not classification: the fix is that these are the
cheapest, most reversible actions in the set — which is exactly why the policy allows them early —
plus, in a product, a short undo window. It is the residual risk to state, and it will not go away
with a better model.

**Compound commands as two decisions.** After an action fires, the live loop scores only the words
that followed, with a `prior` line naming the action already taken. On the 22 compound rows the
residual routes to the second action 20 / 22 without the prior (it loses "youtube" from "open
youtube and play never gonna give you up") and **22 / 22 with it** (`residual_check.py`).

**ASR stage** (`asr_bench.py`, faster-whisper base.en FP16 on the same GPU, 200 ms chunks, greedy,
no VAD, re-decoding the growing buffer every chunk as the live loop does; synthetic speech, so a
lower bound on error):

| | value |
|---|---:|
| Partial decode latency p50 / p95 | 41 / 131 ms (35 ms at 1 s of audio, 60 ms at 4 s) |
| WER vs the command text / exact-match rate | 7.8 % / 75 % (noisy style 24 %, the rest ≤ 10 %) |
| Fraction of the final transcript's words already stable at 50 % / 75 % of the utterance | 81 % / 97 % |
| Intent from ASR transcript agrees with intent from true text | **97.3 %** (6 / 220 differ: "quit chrome" → "quick crow", "pause" → "paw") |
| Intent accuracy from ASR / from text | 0.900 / 0.909 |
| small.en for comparison: WER / agreement / partial p50 | 7.3 % / 97.7 % / **120 ms** — 3× the latency for < 1 point; base.en is the pick |

**End to end, live** (`voice_demo.py --wav`, timings from the recording's own clock; contended GPU,
so pessimistic): "open notes and type hello world" → partial "open note" at +493 ms → **Notepad
launched at +652 ms**, ~1.3 s before the user stops speaking; "and type hello world" scored as the
residual; `type_text` executed at end of speech with the exact slot "hello world". "play lofi hip hop
radio on youtube" → YouTube opened at +1.7 s (the ASR partials flickered between "lofi"/"low-fee",
which the two-partial rule on slot-bearing play actions correctly waited out), search executed at end
of speech with the full query. "text sarah that i'm on my way" (out of scope) → no action.

**Budget, measured on this machine, per streamed word:**

| Stage | ms | Notes |
|---|---:|---|
| ASR partial | ~40 | base.en GPU; grows with buffer length; small.en is slower but not more accurate on this set |
| Decision compute | ~30 | 11 uncached tokens on the 12B Q8; 2.7 ms/token |
| HTTP + JSON | ~15 | removable with an in-process loop |
| Policy + slot extraction | < 1 | pure Python |
| **Word → action dispatched** | **~85** | before the user has finished the next word |

Humans read ~100 ms as instantaneous; the decision is inside that. What remains outside it is
end-pointing for the deferred actions (type_text, close_app: the loop waits 600 ms of silence) and OS
launch time. Rhino/Snips-class speech-to-intent has hit 12–60 ms on fixed intents since 2019; what
this adds is an *editable, zero-shot* action set (add "open Notion" by editing a JSON line), a
calibrated early-commit signal with a measured harm rate, and one primitive shared with the LLM
fallback. Slot extraction is deliberately not the model's job: it is regex over the final transcript
and scores 90 / 93 exact on the labelled slots (the three misses are an ASR error, an anaphora —
"that band from earlier" — and "the latest mkbhd video").

### 4.5 Computer use — the honest verdict

Action spaces are tiny (OS-Atlas collapses 17 types to 10; Claude/OpenAI expose ~9) and scoring a
fixed action set is established (SayCan; ActionRank on tool calling: 247 vs 632 ms, −2.6 pts,
0 % invalid). But per-step time on OSWorld-Human is **planning 53–75 %, reflection 22–34 %,
grounding + execution 2–4 %, screenshots ~1 %.** The action name is single-digit percent of the
tokens. Replacing it attacks noise. The version that works: **replace the whole reasoning-plus-action
emission with one scored pass, gated by confidence, falling back to full reasoning on doubt** —
pair with GUI-Actor-style attention grounding so coordinates need no tokens either. Ceiling is
vision prefill, which none of this removes. The voice loop above is the text-only special case of
that design, and the class-aware commit policy (cheap/refinable actions early, irreversible ones
gated) transfers directly.

### 4.6 What a small model can be trusted with

Routers plateau at 60–75 % accuracy regardless of architecture (95 % of routing pairs sit within a
0.05 margin), published routers collapse to always picking the strong model at higher budgets, and
guardrail classifiers top out around F1 0.73–0.76 with bigger not better. Therefore:

- **Trust:** agent/tool *shortlisting* (ranked top 3–7, never a single hard pick), cheap-vs-expensive
  routing with escalate-on-doubt, recall-first safety pre-filters feeding a deterministic rail,
  cheap reversible actions on a calibrated early signal.
- **Never:** auto-approving irreversible actions; escalation calls on novel cases. The model may
  *raise* scrutiny, never lower it. Auto-approve stays deterministic allow-lists.

---

## 4.7 Head-to-head with Laya, on Laya's own benchmarks

Laya (github.com/NandhaKishorM/laya, Apache-2.0) is an encoder-based "System One": ModernBERT-large
421M (English) or mmBERT-base 322M (multilingual), one `[MASK]` marker per option, a small head
scoring the markers, trained by RL against a proper scoring rule ("RLCD"); a script-detecting
`Router` picks the checkpoint. It claims to beat Jev (typed-decisions 0.766 vs 0.727, ECE 0.081 vs
0.246, 33 ms on a T4) and states its limits: base checkpoints near chance zero-shot, the 0.766 is
fine-tuned on that benchmark's train split, keep choice questions under ~20 options.

Laya never ran Jev. We ran **Laya's three checkpoints and Tez on byte-identical rows on the same
GPU** (`experiments/bench_h2h.py`), on Laya's datasets and protocol (MASSIVE: 20 sampled options,
seed 13). Laya's published numbers reproduce (base 0.362, fine-tuned 0.766, Khmer 0.000 at 0.95
confidence), which validates the harness. Full table in the README and the paper; figures in
`docs/figures/`. Summary:

- **Zero-shot Tez beats the trained encoders on 6 of 9 English tasks** (Banking77 0.713 vs 0.395 via
  a chunked tournament; SST-5 0.512 vs 0.362; BoolQ 0.850 vs 0.843; prompt-injections 0.759 vs 0.672;
  MASSIVE-en 0.900 vs 0.830; typed-decisions 0.704 vs 0.362) and ties Emotion (0.540 vs 0.550). It
  loses where Laya was trained: AG News (0.880 vs 0.932) and NLI (XNLI-en 0.730 vs 0.900).
- **typed-decisions, 2,000 decisions: 0.704 zero-shot** vs Jev's published 0.727 and Laya's
  fine-tuned 0.766; **soft accuracy 0.575 ≈ Jev 0.580 > Laya-td 0.471**. By workflow: customer
  service 0.768, invoice 0.768, security 0.688, agent-trace 0.592; by primitive: noul 0.81, choice
  0.68, score 0.64. ECE 0.26 → 0.05 after a per-task temperature.
- **One model for every language:** MASSIVE macro 0.885 over 11 languages / 6 scripts, all ≥ 0.79,
  vs 0.524 for laya-multilingual and 0.354 for laya. Khmer 0.790 vs 0.000 / 0.210.
- **Order flip at 20 options 0.07** vs Laya 0.15–0.20 (Jev 0.13).
- **Calibration as shipped: Tez least over-confident** (mean ECE 0.212 vs 0.323 / 0.253); after a
  per-task refit all models sit at 0.07–0.10 — Laya's calibration claim is a post-refit claim.
- **Laya wins on single-question latency** (25–50 ms vs 57–250 ms here, because on these tasks the
  state is the variable suffix and is evaluated every time) — the crossover rule of §4.2 — and on
  the tasks in its training mix. It has no streaming/early-commit contract and no zero-shot strength.
- Not attempted: fine-tuning a decoder with the same proper-scoring objective on own-traffic data.
  Every Tez number above is therefore a floor.

## 4.8 The probe lab: reading decisions from the middle of the network

![probe lab](figures/tez_lab.png)

A second round asked what makes the mid-depth probe usable on *any* schema, borrowing methods from other fields
(two commissioned notes: `docs/research/abstract-methods-survey.md`, `docs/research/novelty-check.md`).
Almost everything runs on one cache of every layer's hidden state (`experiments/probe_lab.py`); full tables in
`BENCHMARKS.md` §4c, discussion in the paper's probe-lab section.

What worked:

- **The best layer can be found with no labels.** The intrinsic dimension of unlabelled states (TwoNN) bottoms
  out at layers 24–28 of the 4B, exactly the probe's accuracy plateau (0.79); confidence, the obvious
  alternative, picks the wrong layer.
- **On typed-decisions, the 4B's top three layers cost ten points of zero-shot accuracy.** Its letter readout scores
  0.583 at layer 29 and 0.485 at the top. It does not generalise: on Laya's public suite the same cut lowers the
  letters by 3.7 points on average, and the 12B keeps improving to full depth.
- **Cut the served model where the decision is made.** `experiments/gguf_truncate.py` keeps the first 24 of 32 blocks: a probe on the served state scores 0.793 (full model 0.776) at 58 vs 84 ms, from a 3.53 GB file. The zero-shot letters gain 10 points on typed-decisions but lose 17.4 on average across Laya's public suite (3.7 at 29 blocks): cut for probes, validate per task for letters.
- **On the 4B the most accurate readout is also the fastest.** Same prompts, caching off: a probe on the 24-block model decides in 58 ms (0.793), the full model's letters take 136 ms (0.483), the 12B's letters 207 ms (0.705).
- **Decide first, bind later, measured.** A probe trained on the usual option order still works on reversed options
  up to layer 14 (no loss) and 18 (−2.4 points), but loses 20 points at layer 26: read at layer 18 if a schema
  may reorder its options.
- **A decision needs 64 numbers.** One PCA learned from unlabelled traffic keeps 0.790 of 0.793, and a PCA learned
  on other workflows keeps 0.782 on a new one.
- **Label the typical rows first, and keep the zero-shot prior.** 50 typical labels per question on top of the 12B's
  zero-shot answers reach 0.766, the level of Laya's checkpoint fine-tuned on the whole train split.
- **Calibrated as it comes.** A logistic probe is fitted with a proper scoring rule: ECE 0.023 as read, against 0.262
  for the 12B's raw letters and 0.246 published for Jev.
- **A guaranteed error rate.** Conformal selection acts on 40 % of decisions with at most 5 % wrong (realised 5.2 %).
- **Cold start works when the prior is strong.** Starting from the 12B zero-shot, escalating the unsure 5 % and
  learning from them automates 95 % at 0.707; with a 10 % random audit slice it automates 77 % at 0.777.
- **Many decisions from one pass.** Five questions after one state, each read at its own marker: 2.7× fewer
  tokens for 1.5 points; later questions are not hurt by earlier ones.
- **Train in English, use in eleven languages.** A MASSIVE probe trained on English rows only scores 0.803 macro over
  11 languages (Laya's multilingual model 0.524); within 1–5 points of the 12B zero-shot on European and CJK
  languages, 13–22 below on Arabic, Hindi, Thai and Khmer.
- **Deciding on a workflow it has never seen.** A single "is this answer right?" probe, read at a forced answer token
  and trained on three workflows, scores 0.622 on the fourth (the 4B's own letters: 0.490; the 12B: 0.705);
  a question-agnostic letter probe does not transfer (0.521).
- **The benchmark is now the limit.** Every supervised route lands at 0.79–0.80. Where the benchmark's own label
  distribution is decisive the probe is right 98 % of the time; where the benchmark's labels are split it is right 56 %.

What did not: probes trained on zero-shot answers only match their teacher (0.709 vs 0.705); a Dawid–Skene label
model over several readouts (0.617); uncertainty-only labelling from a weak prior (worse than random labels);
batch calibration of the teacher; DoLa; adaptive depth (ties a fixed cut); LDA with an unlabelled covariance;
select-and-copy attention heads (0.50–0.53, the letters' level); a collapsing commit bound for voice.

## 5. What "best open one" would have to mean

### 5.1 The bar

No credible, independent, contamination-resistant benchmark for this class exists yet. JevBench is
the closest — independent, hashed held-out tier — but it is a one-person project with LLM-written
ground truth and a ×2 latency *assumption* for self-hosted models. Every "beats Jev" claim from an
open clone so far is self-authored data, cross-benchmark, or trained on the benchmark's own split.
This report's own numbers are on SemIf's public fixtures and are a reproduction, not a leaderboard.

Current independent numbers: Jev Intelligence **85.7**, Calibration **82.7** (held-out); SemIf 79.0 /
72.6; djev 73.0 with calibration 65.4; encoder clones 23–41 zero-shot. Jev leads every open model
by 6–20 points on identical-input head-to-heads.

### 5.2 Scorecard to adopt

Balanced accuracy + macro-F1 with group-bootstrap CIs and **paired tests between models** · **Brier +
NLL** primary, reliability diagrams with CIs, smooth-ECE secondary · TV distance on soft-label tasks ·
flip-rate with Wilson CIs and accuracy delta under permutation, re-wrap, irrelevant-context · **AUGRC**
and acc@coverage for abstention · p50/p95 latency **measured** with `prompt_n` logged, fresh and
cached, GPU and CPU · decisions/s at fixed VRAM · same prompt, same rows, labels and metrics hashed
before outputs, group-disjoint folds for anything fitted · determinism across repeated runs. Our
harness emits all of this.

### 5.3 Contamination plan

Public dev set for reproduction + a **private human-authored hold-out of ≥ 500 decisions**, SHA-256
committed and timestamped before the first checkpoint exists; group-disjoint construction; retire
and publish 20 % per quarter; publish an exchangeability test p-value against the private set;
never train on any benchmark's public tier you report on; only third-party runs in the headline.

### 5.4 The gap nobody has filled

| Project | Base | Licence | What it does not do |
|---|---|---|---|
| SemIf | Qwen3.5-4B frozen, logit read | MIT | multi-label, abstain, batch API, debiasing, built-in calibration |
| Laya | ModernBERT-L 421M | Apache-2.0 | zero-shot ≈ random; > 50 options; 512 ctx |
| openJev-verdict-2.0 | ModernBERT 150M | *NOASSERTION* on GitHub | 512 ctx; 48 % on TypeSafe's public evals |
| djev | DiffusionGemma 26B-A4B | Apache-2.0 | needs 24 GB; calibration 65.4 |
| decider | Qwen3.5-2B SFT+RL | Apache-2.0 | hard tier 0.459; English only |

**Nobody ships:** an abstain option with its own calibration; multi-label; a published `p_correct`
head fitted on a published calibration set; long-context *and* small; order-robust options as a
feature (permutation averaging is worth +10 points on a 4B); a standard decision-log format; a
streaming/early-commit contract; a benchmark with runtime-option-set tasks; reproducible zero-shot
strength (only the 4–27 B logit readers work untrained). That list is the product.

---

## 6. Architecture and roadmap

**Two tiers, one contract.**

```
 state + questions ──► [Tier 1: encoder fast path]  one→one, fixed schema, CPU, ~15–100 ms
                       [Tier 2: cached-decoder logit readout] constant-first prompt, GPU, ~30 ms per suffix
                                     │
                       symbol-logit readout ── permutation debias (batched) ── per-schema calibration
                                     │
                       conformal set ──► size 1: ACT · size >1: ESCALATE · empty: FULL LLM
                       streaming: 'none' = keep listening · action classes gate early commit
                                     │
                       decision log (inputs hash, options, distribution, action, outcome)
```

**v1 is a harness around a frozen model** — the evidence says post-hoc is enough (§2.4) and that
the frozen 12B reproduces and exceeds the published open 4B baseline. Concretely:

1. Backbone selection by **MCSB screen** on a candidate model (the Llama 3.2 3B result is what
   failing looks like). Gemma 4 12B passes; Q8_0 for calibration-grade probabilities. A 4B with
   batched permutation averaging is the budget option and is within noise of the 12B on this data.
2. Readout with the template's exact answer position and single-token slots; log top-k always;
   `n_probs` large enough for every letter.
3. **Six-ordering permutation averaging in one batch**; PriDe prior on ~5 % of traffic once volume
   allows (a no-op on Gemma 4, not on smaller models). Do **not** apply content-free contextual
   calibration to option sets with an abstain answer.
4. **Per-schema temperature**, fitted out-of-fold by source group; then **Mondrian conformal** for
   the act / escalate / fallback contract; publish risk–coverage and AUGRC.
5. **Prompt layout as part of the schema**: constant material first, variable last; `prior` for
   multi-step episodes; `--swa-full` (or the engine's equivalent) verified by `prompt_n`.
6. The **decision log** and a shadow-eval loop: run on 1–2k historical decisions, fit on half, score
   the other half, pin versions. Sample auto-acted cases to humans, not only escalations.
7. Serving: llama-server with prefix reuse for the transcript/question-suffix case (measured here);
   an in-process loop with `logits_to_keep=1` when the 15 ms HTTP cost matters; quantise weights
   only, never the KV cache or `lm_head`; treat any quantisation or caching change as a new model.
8. Only then, optionally: LoRA on Gemma 4 12B **base** with a proper scoring rule on label-restricted
   logits, permutation-augmented, gated on an OOD hold-out. Never on Jev outputs (terms), never on a
   benchmark's public tier you report.

**Integration point that already exists.** In Theioma, `cost-cascade-confidence.ts` states that
"R1.1 will replace `confidenceFromSignals` with an isotonic-regression-calibrated learned function"
behind a stable API, `classifyAuto` is a documented swap surface, `RouterTrainingRow` already stores
decision and outcome in one row, and `ApprovalRequest` carries a `confidence` field. The first
deliverable there is calibration, not a model: it costs no latency (a lookup) and it is what makes
the existing `DEFERRAL_THRESHOLD` mean something. The model sidecar goes *behind* that threshold —
`classifyAuto` is pure and p95 < 2 ms; putting a 30–100 ms model in front of it is a regression.

---

## 7. Licences and terms (verified)

- **Gemma 4 12B: Apache-2.0** (the old Gemma Terms do not apply to this family). Clean.
- **SemIf: MIT.** Fixtures, prompt and method are reusable with attribution.
- **`urchade/gliner_*` checkpoints: cc-by-nc-4.0 — non-commercial.** Only the Apache-2.0
  Knowledgator/Fastino checkpoints (GLiNER2.5, GLiClass-modern) are safe commercially.
- **openJev-verdict-2.0: licence NOASSERTION** on GitHub. Learn from the dual-head design; do not depend on it.
- **Jev MCA: no distillation, no imitation training, no competing product from its outputs.**
- faster-whisper / CTranslate2: MIT. Whisper weights: MIT. SAPI voices: Windows-bundled, used only
  to generate test audio.

---

## 8. Reproduce

```
# engine: llama.cpp b11100 win-cuda-13.4 (+ cudart zip) in tools/llamacpp/
llama-server.exe -m <gguf> -ngl 99 -c 2048 -b 512 --port 8091 --host 127.0.0.1 -np 1 --no-webui --swa-full
curl http://127.0.0.1:8091/props     # assert model_path before every run

# benchmark (SemIf fixtures, fresh scoring)
py experiments/run_direct.py --data data/semif/authored144.jsonl      --out results/authored144_<tag>.jsonl --template gemma4
py experiments/run_direct.py --data data/semif/authored144.jsonl      --out results/authored144_rev_<tag>.jsonl --reverse
py experiments/run_direct.py --data data/semif/authored144.jsonl      --out results/authored144_perm021_<tag>.jsonl --perm 0,2,1   # and 1,0,2 / 1,2,0 / 2,0,1
py experiments/run_direct.py --data data/semif/authored144.jsonl      --out results/authored144_cf_<tag>.jsonl --content-free      # --cf-text "[MASK]" / ""
py experiments/run_direct.py --data data/semif/perturbations108.jsonl --out results/perturb108_<tag>.jsonl
py experiments/metrics.py   --pred results/authored144_<tag>.jsonl
py experiments/metrics.py   --pred results/perturb108_<tag>.jsonl --base results/authored144_<tag>.jsonl
py experiments/calibrate.py --pred results/authored144_<tag>.jsonl --cf results/authored144_cf_<tag>.jsonl --rev results/authored144_rev_<tag>.jsonl
py experiments/perm_analysis.py --tag <tag> --out results/perm_analysis_<tag>.json

# in-process (SemIf-faithful) — fresh / prefix reuse / six orderings in one batch
py experiments/run_hf.py --model Qwen/Qwen3.5-4B --data data/semif/authored144.jsonl --out results/authored144_qwen35-4b-bf16-hf.jsonl --mode fresh
py experiments/run_hf.py --model Qwen/Qwen3.5-4B --data data/semif/authored144.jsonl --out results/authored144_qwen35-4b-bf16-hf-batchperm.jsonl --mode batchperm

# voice
py experiments/tts_wavs.py                                   # 220 SAPI wavs -> data/voice/wav
py experiments/run_voice_fast.py --variant final --order last --out results/voicefast_final_last_<tag>.jsonl
py experiments/stream_policy.py  --stream results/voicefast_final_last_<tag>_stream.jsonl --full results/voicefast_final_last_<tag>.jsonl
py experiments/residual_check.py --prior --out results/residual_check_prior_final_<tag>.json
py experiments/asr_bench.py      --model base.en --server http://127.0.0.1:8091 --out results/asr_bench_base-en_<tag>.jsonl
py experiments/voice_actions.py  --selftest data/voice/commands.jsonl
py experiments/voice_demo.py     --text "open notes and type hello world"      # dry run; --execute to really do it
py experiments/voice_demo.py     --wav data/voice/wav/v013.wav                  # or --mic
```

Every run writes `<out>.manifest.json` (data SHA-256, model path, settings, latency percentiles,
leakage counts). Result files for runs later found to have hit the wrong model are kept under
`results/INVALID_wrongmodel_*` and excluded from every table above.

---

## Appendix — research notes

Seven parallel literature surveys were commissioned for this report (mechanism papers; RL/training
and Jev's plausible architecture; voice-to-action; universal interface and ecosystem; evaluation
bar; Gemma 4 specifics; local model landscape). Their full text with links is in
[`docs/research/RESEARCH-NOTES.md`](research/RESEARCH-NOTES.md); the citations in §2, §4 and §5
are drawn from them. arXiv identifiers dated 2026 were surfaced by those surveys; 2601.03914 was
opened and verified directly, the rest should be clicked before any specific number is relied on.
