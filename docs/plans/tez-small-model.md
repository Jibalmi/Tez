# A small, fast Tez model: research and build plan

Plan, 2026-09-25. Nothing was trained and no GPU job was run for it. A number with a section mark comes from
`BENCHMARKS.md`, and a path means a file in `results/`. Numbers labelled *estimate* are planning guesses that the
pilot replaces with measurements. Laya's and dev-0.4b's figures are their own or come from the earlier study, and are
marked as such.

Contents: Summary · 1 Targets · 2 Where the others stand · 3 Options · 4 The student · 5 Data · 6 Evaluation ·
7 Serving · 8 Milestones, gates, pilot, budget · 9 Risks · 10 Sources checked

## Summary

- **Build option A.** Use a multilingual encoder, **mmBERT-base**: 307M parameters, 110M of them outside the
  embeddings, MIT licence, 8,192-token context. Put an option-span head on it and pack several questions after one
  state. Distil it from Gemma 4 12B letter distributions over a broad pool of schemas and states. Build an
  **mmBERT-small** version (140M parameters) for CPU. Ship it as ONNX, as a new `student` readout that runs in front
  of the 12B. Whatever the gate does not certify goes to the 12B, or back to the caller when no 12B is loaded.
- **Why A:**
  - It is the only option whose model family has been measured near the speed target on this GPU. laya-multilingual
    takes 30 ms for 1 question and 2.8 ms per question at 50 questions (§1b).
  - It is the only option with a CPU figure. Laya's mmBERT checkpoint takes 193 ms per question on 4 CPU cores
    (Laya's repo).
  - It has no symbol-binding cliff (§2).
  - Where the encoder is trained on a task, it holds up across languages. On XNLI, laya-multilingual scores 0.794
    against the 12B's 0.699 (§1).
- **The open question is the one Laya failed: answering zero-shot on schemas it has never seen.** Laya wins the four
  workflows in its training mix and loses all three that were held out (§1b). No small student, trained or probed,
  has yet been shown to transfer to held-out schemas. The general heads on the frozen 4B reach 0.52–0.62 on an unseen
  workflow, against 0.705 for the 12B (§4c). The week-1 pilot tests this before anything else. It includes a
  Qwen3.5-0.8B LoRA arm (option C) as a control.
- **Gates** (full criteria in §8):
  - **G1 (pilot):** the student keeps at least 80 % of the 12B's accuracy on held-out typed-decisions and beats Laya
    on at least 2 of the 3 held-out workflows.
  - **G2 (v1):** it keeps at least 90 %, beats every Laya checkpoint on each held-out workflow, clears 3× random in at
    least 48 of the 51 MASSIVE languages, and has ECE of at most 0.10.
  - **G3 (export):** at most 10 ms on the GPU and at most 60 ms on the CPU (small model, int8).
  - **G4 (cascade):** the cascade is within 1 point of the 12B alone, with the student answering at least 50 % of
    decisions.
- **Compute:** about 7 GPU-hours for the pilot and about 60–80 for a shipped v1 (*estimate*), over five weeks on the
  shared RTX 5080 Laptop, run under the GPU lock.

## 1. Targets

| target | v1 value | today | source |
|---|---|---|---|
| GPU, one question, state of up to 512 tokens | ≤ 10 ms p50 (ONNX) | 12B: 95–120 ms in process, 140–155 ms over HTTP; laya-multilingual: 30 ms; dev-0.4b: 27.6 ms on MPS as claimed, 43–91 ms in its own eval files | §5b, §1b, earlier study |
| GPU, many questions about one state | ≤ 2 ms per question at 50 questions | 12B in process: 27.1 ms; 4B state-only probes: 41 ms per call (but they need labels for every question); laya-multilingual: 2.8 ms | §5b, §1b |
| CPU (Core Ultra 9 275HX, 24 cores, AVX2) | ≤ 60 ms p50 for one question (small, int8); ≤ 150 ms (base) | Laya's CPU run (4-core EPYC, fp32 PyTorch): laya-multilingual 193 ms, laya 580 ms; dev-0.4b 210–310 ms (reviewer) | Laya `research/results/latency_cpu_m7a_xlarge_20260924.json`; earlier study |
| Zero-shot on held-out workflows | ≥ 90 % of the 12B's accuracy **and** above the best Laya checkpoint on each | 12B: typed-decisions 0.704, routing 0.970, jailbreak 0.865, toxicity 0.7125. Best Laya: 0.362 (typed-decisions, checkpoints not trained on it), 0.659, 0.805, 0.535 | §1, §1b |
| Languages | ≥ 48 of 51 MASSIVE languages above 3× random, mean ≥ 0.73; languages held out of training within 5 points of the trained ones | 12B: 51 of 51, mean 0.816; Laya routed: 48, mean 0.403 | §1b |
| Calibration and order robustness | mean ECE-15 as shipped ≤ 0.10; order flip at 20 options ≤ 0.10 | Tez: 0.140 as shipped, 0.078 after a per-task refit, flip 0.07. Laya: 0.226–0.323 as shipped, flip 0.15–0.20 | §1, §5c |
| Size | base ≤ 0.6 GB, small ≤ 0.3 GB (int8 weights, fp16 embeddings; *estimate* from parameter counts) | cut 4B GGUF: 3.53 GB; 12B Q8: 14.2 GB of VRAM at `-c 2048` | §4c, REPORT §3.1 |
| Contract | the same wire format and answer shapes, `__none__`, `p_correct` and the gate | — | `docs/API.md` |

The forward pass of a 110M-parameter encoder over 512 tokens is about 0.1 TFLOP (*estimate*). The 10 ms budget
therefore goes mostly on runtime overhead, not on compute.

## 2. Where the others stand (checked 2026-09-25 with `gh` and the Hub API)

| project | state on 2026-09-25 | change since the earlier study |
|---|---|---|
| **dev-0.4b** (GitHub `mpnikhil/dev-0.4b`, Hub `mpnikhil/dev-0.4b`) | **GitHub:** one commit, `2ce2563d` (authored 2026-09-21, committed 2026-09-22). No `LICENSE` file, although the README says Apache 2.0. **Hub:** repo created 2026-09-20, tagged apache-2.0, 396,880,897 parameters, last changed 2026-09-22. | None. Issue #1 (2026-09-22), "Model card's choice example does not reproduce on the published weights; run.json names a different checkpoint", is open with no reply. It is the same doubt about which checkpoint was benchmarked. |
| **Laya** (`NandhaKishorM/laya`, Apache-2.0) | v0.3.18, v0.3.19 and v0.3.20 were all released on 2026-09-24; `main` is 70 commits past v0.3.20. The Hub checkpoint `convaiinnovations/laya` has 421M parameters and was last changed 2026-09-24. | Per-language temperatures (`lang_temperatures`), in Python and TS. A tuned ONNX path that tokenises the state once. `answer_confidence`. Docs that stop presenting a confidence threshold as permission to act. **A CPU latency table:** on 4 cores, 5 questions cost about 5× one question ("batching questions saves little on CPU"). |
| **llama.cpp #29363** (Clauszy), "laya: Add support for the laya multilingual decision model" | Open, not a draft, review required, no reviews. The bot flagged it as a large PR that ignores the PR template. 3 commits on 2026-09-24; 32 files, +4,565 lines. | The earlier study called it a runtime for "Laya-style encoder-plus-head models". **It is specific to Laya.** It adds `LLM_ARCH_LAYA` (built on the modern-bert class), a converter for the laya-multilingual checkpoint only, and a separate runtime in `tools/laya/` with `llama-laya-cli`: no llama-server, no core changes. Its own port of mmBERT's pre-tokeniser (Metaspace plus byte-level BPE) is the useful part, because mainline llama.cpp cannot tokenise mmBERT. Scorer logits in F16 are within 0.13 of PyTorch, and the argmax agrees at Q4_K_M, Q5_K_M and Q8_0 on its golden cases. |
| **llama.cpp #29321** (hans00), "system-one : typed decision readout for models that answer instead of writing" | Open **draft**, no reviews. The bot flagged it: a new contributor with 3 open PRs, an AI-written description, a large PR, and maintainers cannot push to it. 2 commits, the last on 2026-09-23. Refers to issue #29022 (open). | **Correction: it does not add a llama-server task.** It adds a `tools/system-one` library and a `llama-system-one` CLI with three readouts: `letter_slot` (causal models), `masked_slot` (bidirectional models) and `rank_head` (K passes). It also adds two GGUF keys (`system_one.labels`, `system_one.segment_separator`) and a `system_one` prompt template stored in the GGUF. `POST /v1/systemone` on llama-server is listed as a follow-up that is deliberately not in the PR. |

What to take from dev-0.4b, and which of its faults this plan rules out:

- **Taken:**
  - per-type default temperatures (shipped, §5c);
  - the ordinal loss for `score` questions;
  - the option-span head;
  - the MCP proxy pattern, as the student's first consumer (tool choice over a runtime option set).
- **Ruled out:**
  - Calibration rows come only from training data, never from test data.
  - Train and validation sets are deduplicated.
  - The SHA-256 of the benchmarked weights goes into every results manifest.
  - Every dataset has a licence entry.
  - No Gemini-labelled data is used.

## 3. Options

### What each option has going for it

| option | zero-shot on unseen schemas | languages | speed on this GPU | many questions about one state |
|---|---|---|---|---|
| **A**: mmBERT encoder with an option-span head, distilled from the 12B | Not yet measured. Laya's encoders, trained on a narrow mix, lose all three held-out workflows and score 0.362 on typed-decisions (§1b, §1). Probes trained on the 12B's answers match it in-distribution: 0.709 against 0.705 (§4c). | laya-multilingual scores 0.794 on XNLI, where it was trained (the 12B: 0.699), but 0.524 on MASSIVE (§1). The capacity is there; the training data is not. | laya-multilingual: 30 ms for one question, 2.8 ms per question at 50 (PyTorch, §1b) | Only by packing several questions into one row. Laya and dev-0.4b encode one question per row. |
| **B**: cut Qwen3.5-4B, state-only probes plus a learned general head | General heads trained on 3 workflows reach 0.492–0.521 (letter-position head) and 0.622 ("is this right?") on the unseen fourth, against 0.705 for the 12B (§4c). State-only probes need labels for every question. | An English-trained probe scores 0.803 across 11 languages, 13–22 points below the 12B on ar, hi, th and km (§4c). | 58 ms per decision for a probe, 41 ms per call for state-only probes (§4c, §5b); voice 19.6 ms per word (§5b) | Yes: one state vector serves any number of probes (§5b). |
| **C**: Qwen3.5-0.8B (873M parameters, 18 of its 24 layers linear attention) LoRA-trained on the teacher's letters | Frozen small decoders sit at chance: Qwen3-0.6B 0.440 on SemIf, Llama 3.2 3B 0.358, Gemma 3 4B 0.646 with 36 % reversal flips (§2, REPORT §2.3). Trained small decoders: "decider" (Qwen3.5-2B, SFT and RL) scores 0.459 on JevBench hard (REPORT §5.4); reflex's LoRA lost to its own frozen model out of distribution (REPORT §2.4). | Pretrained on many languages; not measured here at 0.8B. | Not measured; the 12B takes 27.1 ms per question in process (§5b). | Yes, by caching the state. Qwen3.5's hybrid cache crashed on partial prefix reuse in one b11100 setup, and the runtime still turns caching off for it (§4c, §5b). |
| **D1**: GLiClass-x-base (mdeberta-v3-base, Apache-2.0, a zero-shot classifier) | Off the shelf; not measured here. | Multilingual; context of 512 tokens. | Not measured. | One row per question. |
| **D2**: EuroBERT-210m (310M parameters, Apache-2.0) | As for A. | **15 languages.** | Not measured. | As for A. |
| **D3**: per-schema students (distilled from the 12B plus labels) | Not zero-shot, by construction. | — | — | — |

### How each option ships, and the verdict

| option | ships as | verdict |
|---|---|---|
| **A** | ONNX today. GGUF has the ModernBERT architecture in mainline, but not mmBERT's tokeniser. | **Build it.** |
| **B** | GGUF, 3.53 GB | Keep it as the tier for schemas fitted with labels (0.793 on typed-decisions, §4c). It cannot reach the size or 10 ms targets. |
| **C** | GGUF, with the existing letters readout | Control arm in the pilot only. |
| **D1** | ONNX or PyTorch | Baseline in the evaluation only. The provenance of its training data ("synthetic and licensed") is not verified. |
| **D2** | GGUF (`eurobert` is in mainline) | Fallback only if GGUF becomes mandatory and the language target is dropped. |
| **D3** | — | Fallback if G1 fails. |

### Why A, and what would change the decision

1. **Speed and CPU.**
   - A is the only option whose family has been measured near the targets. B is 41–58 ms on the GPU and 3.5 GB.
     C has no measurement.
   - On CPU, an encoder's cost grows with total tokens (Laya's CPU table). Packing questions after one state is
     therefore the lever. It is the same arithmetic as §4c's single pass with 5 questions, which read 2.7× fewer
     tokens for 1.9 points.
2. **No binding cliff.**
   - The head reads each option's own tokens, so no letter has to be bound to an answer. Sub-4B decoders fail at
     exactly that binding (§2).
   - One pass handles 2–255 options. The 12B needs a 5-pass tournament beyond 26 options (§1).
3. **Languages and context.**
   - mmBERT was pretrained on 1,833 languages, including FineWeb-2. The training data has to cover them.
   - Its 8,192-token context covers JevBench hard's states of up to 3.9k tokens (§5).
   - XLM-R-large (561M parameters) and mDeBERTa-v3 stop at 512 tokens.
   - ModernBERT-large (used by dev-0.4b and Laya's English checkpoint) is English only. EuroBERT covers 15 languages.
4. **Licence.** mmBERT is MIT. Laya already ships a checkpoint of the same family, so the runtime path is known.

What would change the decision:

- If, in the pilot, C holds at least 5 points more of the 12B's accuracy than A on held-out rows **and** runs at
  ≤ 10 ms per question at 50 questions, build C.
- If neither reaches 70 % of the 12B's accuracy, stop the zero-shot student and build D3.

| backbone for A | parameters (total / outside embeddings) | languages | context | licence | llama.cpp mainline |
|---|---|---|---|---|---|
| **mmBERT-base** | 307M / 110M | 1,833 (pretraining) | 8,192 | MIT | architecture yes, tokeniser no |
| **mmBERT-small** | 140M / 42M | 1,833 | 8,192 | MIT | architecture yes, tokeniser no |
| ModernBERT-large | 396M | English | 8,192 | Apache-2.0 | yes |
| EuroBERT-210m | 310M | 15 | 8,192 | Apache-2.0 | yes |
| XLM-R-large | 561M | 100 | 512 | MIT | — |

Parameter counts and licences are from the Hub cards and APIs (2026-09-25).

## 4. The student

- **Input.** One row reads: state, then 1–8 questions. Each question gives its type, its instructions, and each
  option as a marker token followed by "label: description".
  - The order of the questions and of the options is shuffled at every epoch, and the teacher's distribution is
    permuted to match.
  - `noul` is two options (no, yes) under the statement.
  - `score` presents its levels as ordered options.
  - `__none__` is added to 20 % of `choice` questions, with the probability the teacher gives it.
- **Head.** dev-0.4b's head: `Linear(2H, H) - GELU - Linear(H, 1)` over [the question-marker state; the mean of the
  option's tokens]. Softmax runs over the options within each question. The anchor is the question marker, not
  `[CLS]`, because a packed row holds several questions.
- **Loss.**
  - KL divergence from the teacher over each question's options.
  - Plus 0.5 × cross-entropy where a licensed dataset has gold labels (weight *estimate*, tuned in the pilot).
  - Plus, for `score` questions, the squared distance between the teacher's and the student's cumulative
    distributions (dev-0.4b's ordinal loss).
  - A temperature per question type is fitted afterwards, on a held-out split labelled by the teacher.
- **Teacher targets.**
  - Gemma 4 12B Q8_0 letters, `gemma4` template. Raw logits at T = 1 are stored for each of two option orders, the
    original and the reversed.
  - The target is the average of the two orders at §5c's default temperatures: 6.01 for yes/no, 4.71 for choices
    with up to 10 options, 2.66 for 11–26, and 5.18 for score. Those temperatures were fitted leaving each task
    out, bring ECE-15 to 0.132 against 0.218 at T = 1, and never change the argmax.
  - Six orders give the best error detection (REPORT §3.2) but cost 3× as much, so two it is.
- **Training.** Full fine-tune in bf16, sequence length 2,048, then a short 8,192-token phase. Learning rate and
  epochs are set in the pilot.
- **Packing check.** On the frozen 4B, five questions in one pass lost 1.9 points (0.774 against 0.793, §4c).
  - For the student, packed rows must stay within 1.5 points of one-question rows.
  - If they do not, add a block attention mask so that each question sees only the state and itself.

## 5. Data

### Training sources (licence as tagged on the Hub, 2026-09-25)

| source | licence | used for | relation to evaluation |
|---|---|---|---|
| `HuggingFaceFW/fineweb-2` | ODC-By 1.0 (plus Common Crawl's terms of use) | states in 43 languages | none |
| `HuggingFaceFW/fineweb` | ODC-By 1.0 | English states | none |
| `CohereLabs/aya_dataset` | Apache-2.0 | human-written conversations in 65 languages | none |
| `wikimedia/wikipedia` | CC-BY-SA-3.0 / GFDL | encyclopedic states in every language | none |
| `clinc/clinc_oos` | CC-BY-3.0 | gold intents, 150 options plus out-of-scope | same family as Banking77 and MASSIVE (not their data) |
| `google-research-datasets/go_emotions` | Apache-2.0 | gold emotions, 28 labels | same family as DAIR Emotion |
| `fancyzhx/dbpedia_14` | CC-BY-SA-3.0 | gold topics | same family as AG News |
| `tau/commonsense_qa` | MIT | gold 5-way choices | none |
| `allenai/ai2_arc` | CC-BY-SA-4.0 | gold 4-way choices | none |
| `CohereLabs/Global-MMLU` | Apache-2.0 | gold 4-way choices in 42 languages | none |
| `cambridgeltl/xcopa` | CC-BY-4.0 | gold 2-way choices in 11 languages | none |
| `facebook/belebele` | CC-BY-SA-4.0 | gold 4-way reading comprehension, 122 language variants (rows in the 8 held-out languages dropped) | none |
| `Salesforce/xlam-function-calling-60k` | CC-BY-4.0, gated. Generated by DeepSeek-V2-Chat and Mixtral-8x22B, both open-weight. | tool choice over runtime option sets | none |
| `bitext/Bitext-customer-support-llm-chatbot-training-dataset` | CDLA-Sharing-1.0 | support intents (27) | **S-ship only**: same domain as typed-decisions customer service and Laya's support triage |
| Synthetic states and schemas written by Gemma 4 12B | Apache-2.0 model | agent traces, tool logs, forms, alerts | domain filter below |

The share-alike licences (CC-BY-SA, CDLA-Sharing) bind a redistributed dataset, not the model. The training set is
not redistributed; the model card carries the attributions and the licence ledger.

### Excluded

| source | reason |
|---|---|
| Jev or TypeSafe outputs of any kind | their terms, and this plan's rule |
| Anything labelled or generated by Gemini, GPT or Claude (for example `HuggingFaceH4/ultrachat_200k`, MIT but written by ChatGPT), and `glaiveai/glaive-function-calling-v2` (generator unverified) | the closed models' terms |
| `urchade/gliner_*` | CC-BY-NC-4.0 |
| `lmsys/toxic-chat`, `Tobi-Bueck/customer-support-tickets` | CC-BY-NC-4.0; used for evaluation only |
| Every dataset used in evaluation, all of its splits (typed-decisions train included, which laya-td trained on) | the zero-shot claim |
| SLURP and HWU64 (MASSIVE's sources); MNLI, SNLI and ANLI (XNLI's family) | leakage by source, not only by row |
| Jailbreak and toxicity sets (for example `allenai/wildjailbreak`) | these families stay held out, as in Laya |
| Sets with no licence tag or an "unknown" one on the Hub (`Rowan/hellaswag`, `allenai/winogrande`, `ybisk/piqa`, `allenai/openbookqa`) | not verified |

### Schemas, states and languages

- **Schemas.**
  - v1 has 20,000 question schemas. Gemma 4 12B writes them for clusters of states grouped by domain, and each
    schema is reused on about 50 states.
  - Type mix: 35 % choice, 30 % yes/no, 25 % score, 10 % large choice. Large choices (27–255 options) come only from
    gold datasets. Synthetic schemas have 2–20 options.
  - Schemas are deduplicated by MiniLM cosine (≥ 0.9) and by option set.
  - A schema whose option labels overlap an evaluation question's by 50 % or more is dropped.
- **States.**
  - v1 has 200,000 states: 60 % FineWeb-2 or FineWeb passages of up to 512 tokens, 15 % Aya, 10 % Wikipedia, 10 %
    synthetic, 5 % from gold-labelled sets.
  - Questions are 75 % in English and 25 % written by the 12B in the state's own language.
- **Languages.**
  - 43 of MASSIVE's 51 locales are in training, with English at 40 % of states.
  - **8 are held out entirely:** am, cy, is, jv, km, mn, my, sw (five scripts). The 12B scores 0.79, 0.33, 0.62,
    0.72, 0.79, 0.80, 0.82 and 0.70 on them (`results/vs_laya/massive51.json`).
  - The pilot holds out km.

### Size and teacher time

The rates are measured in §5b on ticket-sized states: 187 ms per state for 5 question readings, 289 ms for 10, and
1,357 ms for 50, in process with the state read once and every question's suffix in one decode. Over HTTP, reading
the state first costs 495 ms for 5 readings and 752 ms for 10. Each state here gets 5 questions × 2 orders = 10
readings.

| | states | decisions | teacher, in process (the GPU to itself) | teacher, over the production llama-server (shared) |
|---|---:|---:|---:|---:|
| pilot | 6,000 | 30,000 | 29 min (6,000 × 289 ms) | 75 min (6,000 × 752 ms) |
| v1 | 200,000 | 1,000,000 | 16.1 h | 41.8 h |

- **Longer states.** Passages of up to 512 tokens will cost more than tickets. Allow up to +50 % (*estimate*).
- **Generating schemas and synthetic states is the uncertain part.**
  - The 12B decodes at about 40 tokens per second on a single stream: about 24 ms per token, derived from §4b's
    thinking-budget timings (2.0 s at 32 tokens, 4.3 s at 128). That is not a throughput benchmark; measure it on
    day 1.
  - The pilot needs 1,200 schemas × about 150 tokens, roughly 75 min.
  - v1 needs 3M tokens of schemas and 5M of synthetic states, about 55 h on a single stream. It must therefore run
    with parallel slots; a 3–4× gain is an *estimate*. If that fails, cut synthetic states to 10,000 or write them
    with Qwen3.5-4B (Apache-2.0).

## 6. Evaluation

**Two students.**

- **S-clean** carries the zero-shot claim. It is trained with every evaluation family, domain and language held out:
  - typed-decisions' four workflows (agent traces, customer service, invoices, security alerts);
  - Laya's seven application workflows;
  - the NLI family;
  - the 8 held-out languages.

  The domain filter embeds each state and drops those nearer to an evaluation workflow's centroid than to any
  training cluster.
- **S-ship** is what gets deployed. It is trained on everything licensed except the evaluation rows themselves, and
  its results on those families are reported as in-domain.

| suite (rows) | status for S-clean | 12B today | best other today | source |
|---|---|---|---|---|
| typed-decisions test (2,000) | held out | 0.704, soft 0.575 | laya-td 0.766 (trained on its train split), Jev 0.727 (published), laya 0.362 | §1 |
| Laya's held-out workflows: routing (399), jailbreak (400), toxicity (400) | held out for everyone | 0.970 / 0.865 / 0.7125 | 0.659 / 0.805 / 0.535 | §1b |
| Laya's workflows in Laya's own mix: spam, phishing, RAG, triage (400 each) | held out | 0.9675 / 0.8975 / 0.610 / 0.4075 | 0.9925 / 0.9925 / 0.6725 / 0.540 | §1b |
| Laya's public suite: AG News, Emotion, Banking77, SST-5, BoolQ, prompt-injections, XNLI in 10 languages | related families for topic and emotion; NLI held out | §1 table | §1 table | §1 |
| MASSIVE, 51 languages × 100 rows, 20 options | 8 languages held out; the intent schema held out | 51 of 51 languages, mean 0.816 | Laya routed: 48, 0.403 | §1b |
| SemIf authored144 and perturbations108 | held out | 0.943 / 0.992 | Qwen3.5-4B: 0.813 | §2 |
| Voice, 220 commands, scored word by word | held out | 0.909, 1 harmful action in 198, 33.5 ms per word | 4B probe: 0.882, 3 harmful, 19.6 ms | §5b |
| JevBench public tiers (72 / 48 / 111) | held out (indicative) | 0.958 / 1.000 / 0.703 through the runtime | — | §5 |
| dev-0.4b's tasks: Banking77, BoolQ, Yelp | held out for S-clean; inside dev-0.4b's training | 0.713 (tournament), 0.850; Yelp not run | dev-0.4b published: 0.913 / 0.852 / 0.627 | §1, earlier study |

**Compared on the same rows:**

- the 12B's letters (existing rows in `results/h2h/rows_<task>_tez.jsonl` and `results/vs_laya/`);
- the 4B probes, marked as supervised (§4b, §4c);
- Laya's three checkpoints (existing rows);
- dev-0.4b (new rows; English, up to 1,024 tokens);
- GLiClass-x-base (new rows);
- the pilot's arms, S-clean, S-ship and the small student.

New rows are written as `results/h2h/rows_<task>_tez-s-<variant>.jsonl`, each with a manifest that includes the
weights' SHA-256.

**Metrics:**

- accuracy, with SemIf's balanced accuracy keyed by option id;
- soft accuracy and KL divergence against typed-decisions' soft gold;
- NLL and Brier as the primary calibration scores;
- ECE-15 as shipped and after one out-of-fold temperature per task;
- order flip at 20 options;
- conformal act rate at α = 0.05 and 0.10 (`tez/gate.py`);
- retention: the student's accuracy divided by the 12B's, per task;
- exact McNemar against the 12B on the same rows;
- accuracy split by how decided the benchmark's own soft label is (§4c): the benchmark's split labels cap every
  model.

**Latency.**

- Laya's protocol: 1, 5, 10 and 50 distinct questions, a new state for each call, p50 and p95.
- On the GPU and on the CPU. CPU runs use all 24 cores and, for comparison with Laya's EPYC table, 4 threads.
- Also cold load and peak memory.
- GPU timings go through `experiments/run_when_idle.py`, under the GPU lock and with VRAM snapshots (§5b's rules;
  runs affected by another process go to `contaminated/`).

**Leakage guard.**

1. Before the first teacher call, commit `data/small_model/eval_hashes.json`. It holds the SHA-256 of
   `tez.schema.state_key` for every evaluation state (every split of typed-decisions, every row under
   `results/h2h/` and `results/vs_laya/`, all 51 MASSIVE test sets, SemIf, voice, JevBench), plus 5-gram MinHash
   signatures.
2. The training pool is filtered:
   - drop on an exact hash match;
   - drop at MinHash Jaccard ≥ 0.8;
   - drop at MiniLM cosine ≥ 0.95;
   - drop a schema when its option set overlaps an evaluation question's by 50 % or more.

   The manifest records how many rows each step dropped, and the hash of the final training set.
3. `tez eval` already skips rows a probe was trained on (`train_hashes`); the student's manifest uses the same field.
4. An evaluation row found in training after the fact quarantines the result, as with `results/INVALID_*`.

## 7. Serving and integration

- **Artifact.** `tez-s-base` and `tez-s-small`: an ONNX graph (encoder and head), `tokenizer.json`, and a manifest
  (training-set hash, teacher model and template, calibration, evaluation results).
- **Runtime.** A new `tez/student.py` on onnxruntime or onnxruntime-gpu, behind an optional `student` extra, so
  `import tez` stays base-only.
- **Engine.**
  - `tez.readout` gains `student`, and `--student PATH` turns it on.
  - Under `auto`, a fitted probe answers if the schema has one; otherwise the student does.
  - A question the student does not `act` on at the schema's alpha is read again by the 12B's letters in the same
    request (`--escalate letters`). With no 12B loaded (CPU only), it comes back as `escalate`.
  - The wire change moves as one unit: `docs/API.md`, the wire and server tests, the TypeScript types and
    `tez/integrations/remote.py`.
- **The gate.**
  - With labels, `tez fit` fits the student's temperature and conformal cuts exactly as it does for letters.
  - Without labels, a default cut per question type is fitted on a held-out split labelled by the teacher. That cut
    bounds disagreement with the 12B, not error, and the answer says so in its `calibration_id`.
- **Why a cascade can pay this time.**
  - The 4B → 12B cascade lost: 0.854 at 117 ms against the 12B alone at 0.951 and 110 ms (§4). The small model cost
    40 of those 110 ms, and its only uncertainty signal was disagreement between option orders.
  - Here the student has to cost no more than 10 % of a 12B call and act on at least half the decisions at the
    12B's accuracy (G4).
- **Voice.** An encoder cannot cache the action list, so each word re-reads the whole prompt: the action list plus
  the transcript. At ≤ 10 ms a pass, that still beats the 4B probe's 19.6 ms per word. G4 checks it.
- **The fitted tier stays.** Once a schema has labels, the cut 4B with per-schema probes is still the most accurate
  route (0.793, §4c). The student's question-anchor vector can also act as `tez fit`'s embedder; that is an M4
  option.
- **GGUF, later.**
  1. Convert the encoder to a `modern-bert` GGUF, serve `/embedding --pooling none`, and run the head in numpy in
     Tez, as probes run today. This is blocked on mmBERT's tokeniser in mainline.
  2. Take #29363's tokeniser port if it lands. Its Laya-specific architecture and CLI are not reusable as they
     stand.
  3. #29321's `masked_slot` readout would need a masked-letter head instead of the span head.

  v1 ships ONNX.

## 8. Milestones, gates, pilot, budget

| milestone | when | work | gate to pass |
|---|---|---|---|
| M0 | days 1–2 | licence ledger; eval hashes committed; data plumbing; measure the decoding rate | the ledger covers every source; 0 exact hash matches in the pilot pool |
| M1: pilot | week 1 | the arms below on 30,000 decisions | **G1** |
| M2: v1 data and S-clean / S-ship | weeks 2–3 | 20,000 schemas, 200,000 states, 1M decisions; train both | **G2** |
| M3: small student and export | week 4 | distil mmBERT-small from S-ship; ONNX fp16 and int8; parity and latency | **G3** |
| M4: integration | week 5 | `student` readout, cascade, gate, voice loop, `tez doctor` check, a BENCHMARKS section | **G4** |
| M5 (optional) | later | GGUF by route 1 or 2 of §7 | argmax agreement with ONNX ≥ 99.5 % |

**G1 (the pilot, arm A-base).** A-base must pass all of these:

- On held-out typed-decisions: accuracy ≥ 0.563 and soft accuracy ≥ 0.46 (80 % of the 12B's 0.704 and 0.575).
- On Laya's held-out workflows: above the best Laya checkpoint on at least 2 of 3 (0.659, 0.805, 0.535).
- On MASSIVE's 11 languages: macro ≥ 0.70 (12B 0.885, laya-multilingual 0.524), with km (unseen) above 3× random.
- Retention at 30,000 decisions at least 3 points above retention at 10,000. Otherwise more data is not the lever.
- On §1b's protocol, as fast as laya-multilingual: ≤ 30 ms for one question and ≤ 2.8 ms per question at 50 (PyTorch).
- ECE-15 ≤ 0.15 as read.

Decision:

- If A passes, go to M2 with A.
- If C beats A by at least 5 points of retention and meets the latency line, go with C.
- If neither keeps 70 % on typed-decisions (0.493), stop the zero-shot student and go to D3.

**G2 (S-clean).** S-clean must pass all of these:

- typed-decisions ≥ 0.634.
- On the held-out workflows, the higher of 90 % of the 12B and the best Laya checkpoint: routing ≥ 0.873,
  jailbreak ≥ 0.805, toxicity ≥ 0.641.
- MASSIVE: at least 48 of 51 languages above 3× random and a mean ≥ 0.73; the 8 held-out languages retain within 5
  points of the trained ones.
- SemIf ≥ 0.85.
- Order flip ≤ 0.10.
- Mean ECE-15 as shipped ≤ 0.10.
- Packed rows within 1.5 points of one-question rows.

**G3 (export).**

- ONNX against PyTorch: argmax agreement ≥ 99.5 % and maximum |Δp| ≤ 0.02.
- GPU (ONNX): p50 ≤ 10 ms for one question, and ≤ 2 ms per question at 50.
- CPU: p50 ≤ 60 ms (small, int8) and ≤ 150 ms (base, int8).
- The small student keeps at least 95 % of the base student's held-out accuracy.

**G4 (the cascade).**

- On typed-decisions: accuracy ≥ 0.694 (12B alone minus 1 point), with the student answering at least 50 % and a
  lower mean time than the 12B alone.
- Voice: intent accuracy ≥ 0.88, at most 2 harmful actions in 198, and ≤ 15 ms per word.

### The week-1 pilot

| day | GPU? | work |
|---|---|---|
| 1 | no | Licence ledger. Commit `eval_hashes.json`. Sample 6,000 states: 3,000 from FineWeb in English, 2,000 from FineWeb-2 in de, es, fr, ja, zh, ar, hi, th and ko (km held out), 500 from Aya, and 500 gold rows (CLINC with 20 sampled options, CommonsenseQA, Global-MMLU). Filter them. Time the 12B's decoding for 10 minutes under the lock. |
| 2 | yes, 12B | Generate 1,200 schemas (about 75 min, derived) and assign 5 to each state. Label with the teacher in process, in two orders (29 min at the §5b rate, up to 45 min for longer states). Keep a 10,000-decision subset for the learning curve. |
| 3 | yes | Train A-base on 10,000 and on 30,000 decisions, A-small on 30,000, and C (Qwen3.5-0.8B, LoRA r = 16, question-first letters so it drops into the existing readout) on 30,000. Log training tokens per second, which sets the M2 budget. |
| 4 | yes | Run every arm on typed-decisions test, Laya's three held-out workflows, MASSIVE in 11 languages and SemIf. Add new baseline rows for dev-0.4b and GLiClass-x-base. Measure latency on GPU and CPU. |
| 5 | no | Write `results/small_model/pilot.json` and a draft BENCHMARKS section; take the G1 decision. |

### Compute budget (GPU-hours on the RTX 5080 Laptop, all *estimates* unless the rate is cited)

| stage | GPU-h | basis |
|---|---:|---|
| pilot: schemas | 1.5 | 180,000 tokens at about 40 tokens/s (derived from §4b) |
| pilot: teacher | 0.75 | 6,000 × 289 ms (§5b), plus 50 % for longer states |
| pilot: training, 4 arms | 2 | *estimate*; measured on day 3 |
| pilot: baselines, evaluation, latency | 2 | about 4,400 rows each for dev-0.4b and GLiClass; the students take minutes |
| **pilot total** | **about 6–7** | |
| M2: schemas and synthetic states | 14–20 | 8M tokens; needs parallel slots (least certain line) |
| M2: teacher | 20–25 | 200,000 × 289 ms = 16.1 h (§5b), plus longer states |
| M2: training S-clean and S-ship | 8–15 | from the pilot's measured throughput |
| M2: evaluation | 3 | |
| M3: small student, export, latency | 4–6 | |
| M4: cascade and voice | 3 | |
| **total** | **about 60–80** | two or three nights a week under the lock |

The 12B needs about 14 GB of VRAM, so teacher runs and training runs take turns and never share the card. Teacher
runs in process stop and restore the production llama-server through the idle runner. The shared HTTP path (752 ms
per state) avoids that, at 2.6× the time.

## 9. Risks

| risk | evidence that it is real | mitigation | how we see it |
|---|---|---|---|
| Overfitting to its own training mix (Laya's pattern) | Laya wins the 4 workflows in its mix and loses the 3 held out; its checkpoints not trained on typed-decisions score 0.352–0.362 on it (§1b, §1) | S-clean holds out families, domains and languages; diversity quotas on schemas; gates only on held-out rows | the gap between S-ship in-family and S-clean held-out |
| No small student transfers zero-shot | B's general heads reach 0.52–0.62 on an unseen workflow (§4c); "decider" scores 0.459 on JevBench hard (REPORT §5.4) | G1's stop rule, with D3 as the fallback | retention and its learning curve in the pilot |
| The student copies the teacher's errors | a probe trained on the 12B's answers scores 0.709 against the teacher's 0.705 (§4c); 6 of the 12B's 7 SemIf misses are over-commitment where the answer was "insufficient" (REPORT §3.2) | teacher read in two orders; gold labels mixed in; `__none__` in training; the gate escalates | student accuracy on the rows where the teacher is wrong |
| Loss in other languages | the English-trained 4B probe is 13–22 points below the 12B on ar, hi, th and km (§4c); laya-multilingual scores 0.524 on MASSIVE (§1) | 60 % of states not in English; 8 languages held out; results reported per language | G2's language lines |
| Licences | NC sets in the evaluation suite; closed-model-generated data in the wild; Gemini labels in dev-0.4b | a ledger for every source (Hub licence, provenance); the training set is built from ledger entries only; datasets are never redistributed | review at M0 and before release |
| Order sensitivity | Gemma 3 4B flips 36 % of decisions when the options are reversed (§2) | options shuffled in training; the span head reads options, not letters | order flip ≤ 0.10 |
| Long states | JevBench hard has states of up to 3.9k tokens (§5) | an 8,192-token training phase | accuracy by state length |
| A noisy benchmark ceiling | typed-decisions' gold is the mean of 3 samples from a "roughly 4B-class" teacher endpoint (dataset card); split labels cap every model (§4c: 56 %) | report by how decided the gold is; no gate on the ambiguous bucket alone | §4c's bucket table |
| Timings contaminated by other work on the shared GPU | §5b's contaminated runs | the idle runner, the lock and VRAM snapshots | the manifest's `guard` field |
| C's speed rests on the hybrid cache | b11100 crashed on partial prefix reuse in one setup (§4c) | Qwen3-0.6B (pure attention, Apache-2.0) as C's alternative | `tez doctor`; `prompt_n` |
| GGUF blocked | mainline cannot tokenise mmBERT (#29363 ported its own tokeniser) | ship ONNX; watch #29363 | M5 |

## 10. Sources checked

- In this repo:
  - `README.md` and `BENCHMARKS.md` §1, §1b, §2, §3, §4, §4b, §4c, §5, §5b, §5c.
  - `docs/REPORT.md` §2.3, §2.4, §3.1, §3.2, §5.4.
  - `tez/fit.py`, `tez/readout.py`, `tez/engine.py` (`decide_question`), `tez/gate.py`, `tez/backends.py`,
    `experiments/run_when_idle.py`.
  - `results/vs_laya/massive51.json` and `results/teacher_labels_gemma4-12b-q8_0.json`.
- GitHub (with `gh`, 2026-09-25):
  - `ggml-org/llama.cpp` pull requests #29363 and #29321 (body, files, commits, comments), issue #29022, and
    mainline `src/llama-arch.cpp`.
  - `mybigday/system-one-llama.cpp` at `6db5037`: `tools/system-one/README.md`.
  - `mpnikhil/dev-0.4b`: commits, README and issue #1.
  - `NandhaKishorM/laya`: releases, the comparison from v0.3.20 to main, `BENCHMARKS.md`, and
    `research/results/latency_cpu_m7a_xlarge_20260924.json`.
- Hugging Face Hub API and cards (2026-09-25):
  - Models: `jhu-clsp/mmBERT-base`, `jhu-clsp/mmBERT-small`, `Qwen/Qwen3.5-0.8B`, `EuroBERT/EuroBERT-210m`,
    `answerdotai/ModernBERT-large`, `FacebookAI/xlm-roberta-large`, `microsoft/mdeberta-v3-base`,
    `knowledgator/gliclass-x-base`, `mpnikhil/dev-0.4b`, `convaiinnovations/laya`.
  - Datasets: the licence tags of every dataset in §5, plus the `LocalLLaMA/typed-decisions` and
    `Salesforce/xlam-function-calling-60k` cards for how their labels were made.
