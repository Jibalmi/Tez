# Novelty check: eight planned probe experiments (23 Sept 2026)

Closest prior work for each planned experiment, and what would be new. All arXiv ids were opened and checked on 23 Sept 2026; non-arXiv sources are marked.

**The headline result is already published.** A probe on a hidden state beating the same model's label-token readout is known: Hidden Calibration (Cho et al., 2024, arXiv:2406.16535, NAACL 2025), Prompt-Augmented Linear Probing (Cho et al., 2022, arXiv:2212.10873, AAAI 2023), the MCQ knowledge-prediction gap (Park et al., 2025, arXiv:2509.23782, ICML 2026), and the decide-then-bind account of letter answers (Wong, Nouwen & Gatt, 2026, arXiv:2601.03914). Tez's 0.793 vs 0.49 is a strong instance of that finding, not a new one. Any novelty has to come from the experiments below.

## Summary

| # | Experiment | Closest prior work | Verdict | What would be new |
|---|---|---|---|---|
| 1 | Probe trained on its own (or a bigger model's) zero-shot answers | Buckmann 2025 (2505.08662); Burns 2023, W2S App. D.2/E.1 (2312.09390); Charikar 2024 (2405.15116) | Partly new | The teacher is a linear readout of the student's own final layer, so W2S theory predicts gain only at mid depth: a layer-by-layer test |
| 2 | Anytime depth with SPRT or max-prob stopping | Jazbec 2023 (2306.02652) and 2024 (2311.05931); AdaInfer (2403.02181); Hůla 2026 (2604.18592) | Already done (mechanism) | Only a calibrated likelihood-ratio stop on per-question probes for 2–77 options, judged against Tez's static layer-20 cut |
| 3 | Many decisions from one pass, read at markers | Multi-Task Inference (2402.11597); multi-problem prompting (2406.10786); ArmoRM (2406.12845); Hydragen (2402.05099) | Partly new | No-generation readout at several markers in one sequence, and its cross-question interference vs isolated cached suffixes |
| 4 | English-only probe tested on 10 languages | Shared Doubt (2605.31220); MASSIVE (2204.08582); Wendler 2024 (2402.10588) | Partly new | Intent-probe transfer on a decoder LLM vs the zero-shot readout per language, with per-layer curves (incl. Khmer) |
| 5 | Task-agnostic heads on unseen workflows | Chen 2026 HF card (bilinear heads, negative result); UniMC (2210.08590); truth probes (2407.08582 vs 2410.02707) | Partly new; expect a negative | Controlled leave-one-workflow-out comparison of the three heads, with the letter readout as the floor |
| 6 | Human-in-the-loop cold start | Cache & Distil (2310.13561); CoAnnotating (2310.15638); Rauch 2025 (2506.01992); Réveillard 2025 (2510.23557) | Partly new (combines known parts) | Human labels needed to reach the offline probe, starting from the zero-shot readout, with valid coverage under online updates |
| 7 | Dimensions per decision | Hernandez & Andreas 2021 (2105.07109); Tigges 2023 (2310.15154); Zhang 2024 (2403.14001); Njaradi 2026 (2605.20105) | Already done (in general) | Only the count for typed decisions, and whether unlabeled PCA helps at 10–50 labels per question |
| 8 | Logit lens, tuned lens and DoLa on letters | Wiegreffe 2024 (2407.15018); Park 2025 (2509.23782); Belrose 2023 (2303.08112); Chuang 2023 (2309.03883) | Already done | Diagnostic only: where letters become decodable vs where the probe peaks; DoLa expected null |

## 1. Weak-to-strong and label-free probes

- *Revealing economic facts: LLMs know more than they say*. Buckmann et al., 2025, arXiv:2505.08662. Hidden-state probes beat the model's text answers. A transfer variant used the LLM's own text outputs as noisy labels, plus gold labels for other variables, and still beat the text output by 7.2 points on average. This is "a probe beats its own teacher", shown for regression and with some gold labels.
- *Weak-to-Strong Generalization*. Burns et al., 2023, arXiv:2312.09390. App. D.2 trains new linear heads on frozen models using weak labels and finds the same pattern as fine-tuning. App. E.1 shows that when the student can exactly reproduce the supervisor's errors, the gain disappears. Errors the student cannot predict can be beaten.
- *Quantifying the Gain in Weak-to-Strong Generalization*. Charikar et al., 2024, arXiv:2405.15116. With a fixed strong representation and a linear head fitted to weak labels, the gain over the teacher is measured by the student's misfit to those labels.
- *Zero-Shot Text Classification with Self-Training*. Gera et al., 2022, arXiv:2210.17541 (EMNLP 2022). Fine-tuning a zero-shot classifier on its own most confident predictions gives consistent gains across text-classification tasks.
- *Discovering Latent Knowledge in LMs Without Supervision (CCS)*. Burns et al., 2022, arXiv:2212.03827. A probe found without labels, using logical consistency, beats zero-shot accuracy by about 4% on average.

**What would be new.** Here the teacher is not a separate model. Up to the final RMSNorm scale, the letter softmax is a multinomial logistic regression on the student's own final-layer state, so it sits inside the probe's hypothesis class. Burns et al.'s E.1 then predicts no gain at the final layer. Any gain must come from mid layers, where late symbol-binding errors cannot be linearly reproduced: a layer-by-layer test of W2S's claim about reproducible errors. The 12B-teacher variant is ordinary W2S (D.2), with the twist that the teacher is the larger model. Report the performance gap recovered against the gold-label probe (0.793), and report the probe's misfit to its teacher, which Charikar says predicts the gain.

**Caveats.** Soft labels make the probe copy the teacher more exactly, which E.1 suggests is the wrong direction. Confidence filtering (Gera) and W2S's confidence loss are the published counter-measures. First beat label-free fixes to the teacher itself: Batch Calibration (arXiv:2309.17249). Cluster-then-label on the output distribution already exists as Prototypical Calibration (arXiv:2205.10183).

## 2. Anytime and adaptive-depth decisions

- *The Right Tool for the Job*. Schwartz et al., 2020, arXiv:2004.07453 (ACL 2020). Per-layer classifiers on BERT exit at a calibrated maximum probability, trading speed for accuracy per instance.
- *Towards Anytime Classification in Early-Exit Architectures by Enforcing Conditional Monotonicity*. Jazbec et al., 2023, arXiv:2306.02652 (NeurIPS 2023). A post-hoc product of experts over exits makes accuracy rise monotonically with depth. In log space that product is the running sum an SPRT uses.
- *Early-Exit Neural Networks with Nested Prediction Sets*. Jazbec et al., 2023, arXiv:2311.05931 (UAI 2024). Anytime-valid confidence sequences give prediction sets that stay nested across exits. This is the sequential-testing version of "stop when the evidence is enough".
- *Not All Layers of LLMs Are Necessary During Inference (AdaInfer)*. Fan et al., 2024, arXiv:2403.02181 (IJCAI 2025). On frozen Llama 2 and OPT, an SVM on per-layer statistics stops early. It prunes 17.8% of layers on average and up to 43% on sentiment, with under 1% loss.
- *Two-dimensional early exit optimisation of LLM inference*. Hůla et al., 2026, arXiv:2604.18592. Uses per-layer adapters on frozen 3–8B LLMs. The stop rule accumulates the top-2 margin as sentences and layers are added, and gains a further 1.4–2.3× over layer-wise exit on sentiment tasks.

Also relevant: the patience rule in PABEE (arXiv:2006.04152), and CATs' conformal consistency with the full model (arXiv:2104.08803). The classic origin is Wald's SPRT (Annals of Mathematical Statistics, 1945).

**What would be new.** Very little on method. Per-layer probes, max-probability gates and accumulated-evidence stopping on frozen LLMs are all published. The only new part is a Wald-style log-likelihood-ratio stop on calibrated per-question probes for 2–77-option decisions. The honest baseline is Tez's static cut at layer 20 (1.54× prefill, 0.790). The layer curve drops to 0.737 at layer 16 and 0.661 at layer 14, so an adaptive rule probably has only about 1.1–1.3× left to gain.

**Caveat.** Residual-stream layers are highly correlated, so summing per-layer log-probabilities overcounts the evidence. Thresholds need calibration. Confidence is also actively recalibrated in the upper layers (arXiv:2511.00280), so mid-layer confidence cannot be taken at face value.

## 3. Many decisions from one forward pass

- *Multi-Task Inference: Can LLMs Follow Multiple Instructions at Once?* Son et al., 2024, arXiv:2402.11597 (ACL 2024). Putting several instructions in one call cut inference time 1.46×. It improved strong models: up to +7.3% for Llama-2-70B-Chat and +12.4% for GPT-4. Answers were generated.
- *Evaluating LLMs with Multiple Problems at once*. Z. Wang et al., 2024, arXiv:2406.10786. Across 53,100 zero-shot multi-problem prompts and 13 LLMs, models handled several classification problems from one source about as well as separately, with identified failure conditions.
- *Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts (ArmoRM)*. H. Wang et al., 2024, arXiv:2406.12845. One linear regression layer on frozen backbone features predicts 19 objectives. This is the "one pass over the state, one probe per question" pattern; SentEval (arXiv:1803.05449) is the classic version.
- *One Embedder, Any Task (INSTRUCTOR)*. Su et al., 2022, arXiv:2212.09741 (Findings ACL 2023). Conditioning the embedding on the task instruction beats generic embeddings across 70 tasks, by 3.4% on average. This is why a state-only pass should lose to question-conditioned states.
- *Hydragen*. Juravsky et al., 2024, arXiv:2402.05099. Shared-prefix attention batching gives up to 32× throughput. It is the exact, interference-free way to put the state in once and ask many questions.

**What would be new.** We found no paper that reads several typed decisions at markers inside one causal sequence without generating, using letter logits or per-marker probes. None measures what a later question loses or gains from seeing earlier, unanswered questions. That interference measurement is the new result. The baseline is prefix-cached isolated suffixes, which cost almost the same. The state-only variant with per-question probes is established; only the size of its conditioning gap on typed decisions is new.

**Caveat.** Speed is not the novelty, because prefix caching already gives the saving (see `speed-methods-survey.md`). Marker k also sees questions 1 to k−1 with no answers, which is a distribution shift that isolated suffixes avoid.

## 4. Zero-shot cross-lingual transfer of a probe

- *MASSIVE*. FitzGerald et al., 2022, arXiv:2204.08582. It defines the protocol of training on en-US and testing on every other locale. Under it, fine-tuned XLM-R Base drops from 85.1% to 70.6% average intent accuracy. ja-JP is worst, partly because of the character spacing used.
- *How multilingual is Multilingual BERT?* Pires et al., 2019, arXiv:1906.01502. This is the origin: English-fine-tuned mBERT transfers zero-shot, with systematic weaknesses for some language pairs.
- *Do Llamas Work in English?* Wendler et al., 2024, arXiv:2402.10588. The logit lens shows middle-layer states decoding to the English form of the answer, which is why an English mid-layer probe might transfer. A caution: Bayazit et al., 2026, arXiv:2609.00155, find that representation-based and decoding-based probes disagree.
- *Shared Doubt: Zero-Shot Cross-Lingual Confidence Estimation*. Kyriakou et al., 2026, arXiv:2605.31220. A linear correctness probe trained in one language transfers zero-shot to typologically diverse languages. Its features concentrate in middle layers, and transfer depends on how similar the source language is.
- *Exploring Multilingual Probing in LLMs*. D. Li et al., 2024, arXiv:2409.14459. Probing accuracy is much lower for low-resource languages, and their layer-wise trends differ.

**What would be new.** Probe transfer across languages is established for truth, confidence and refusal directions (refusal: arXiv:2505.17306). We found no such test for a task-label probe on a decoder LLM on MASSIVE. The new data points: the per-language gap between an English-only probe and (a) the zero-shot letter readout in that language (12B: macro 0.885 over 11 languages; the 4B's figure is still unmeasured) and (b) MASSIVE's 70.6% XLM-R zero-shot. Add per-layer curves: does the best transfer layer come earlier than the best in-language layer?

**Caveat.** Khmer, Hindi and Thai break into many more tokens, and Li 2024 predicts a penalty there. Score the probe on the same 20-option protocol as the readout; the 0.850 English probe was over all 60 intents.

## 5. Task-agnostic heads for unseen questions

- *Benchmarking Zero-shot Text Classification (entailment approach)*. Yin et al., 2019, arXiv:1909.00161 (EMNLP 2019). The origin of one universal yes/no head applied to each candidate label, which is the shape of variant (iii).
- *Zero-Shot Learners for NLU via a Unified Multiple Choice Perspective (UniMC)*. Yang et al., 2022, arXiv:2210.08590 (EMNLP 2022). A shared option-slot scorer trained across tasks answers unseen tasks zero-shot. It is a fine-tuned 235M model, and the fine-tuned counterpart of variant (i).
- *On the Universal Truthfulness Hyperplane Inside LLMs*. J. Liu et al., 2024, arXiv:2407.08582 (EMNLP 2024). A truth probe trained on more than 40 datasets generalises across tasks. Dataset diversity matters more than volume.
- *LLMs Know More Than They Show*. Orgad et al., 2024, arXiv:2410.02707. The truthfulness signal sits in the exact-answer tokens, but error detectors fail to generalise across datasets.
- Not a paper: *jev-local-lab-decision-heads*. M. Chen, Hugging Face model card, Sept 2026, not peer-reviewed. Cosine-bilinear heads (hidden state vs mean-pooled label embeddings) on a frozen 4-bit Qwen2.5-1.5B gained 14.8 points on their training families. They scored 3.25 points below zero-shot on 6 unseen families and 6.83 below on an external workflow suite. The author's summary: the head "learned task families it was trained on, not how to read decisions".

**What would be new.** All three head types have been tried, and the closest direct test of (ii) is negative. What is missing is a controlled leave-one-workflow-out comparison of (i), (ii) and (iii) on the same frozen features, with the zero-shot letter readout as the floor.

**Caveats.** typed-decisions has four workflows, so leave-one-workflow-out has only four folds. Probes readily learn task format rather than the intended property (Sahoo et al., 2026, arXiv:2606.02907). In our harness, Laya's base checkpoints, a shared option scorer trained on other data, scored 0.362 on typed-decisions. Expect all three heads to trail the letter readout unless they are trained on many diverse workflows (Liu 2024).

## 6. Human-in-the-loop cold start

- *Cache & Distil*. Ramírez et al., 2023, arXiv:2310.13561 (Findings ACL 2024). A student is trained online on an expensive LLM's answers, and active-learning policies decide which requests to send to the LLM. It is our loop with an LLM in place of the human.
- *CoAnnotating*. M. Li et al., 2023, arXiv:2310.15638 (EMNLP 2023). Items are split between an LLM and human annotators by uncertainty, doing up to 21% better than random allocation.
- *No Free Lunch in Active Learning*. Rauch et al., 2025, arXiv:2506.01992. Active learning with linear probes on frozen LLM embeddings, across ten tasks. A diversity-based cold start helps, and the best query strategy depends on embedding quality.
- *Minimizing Human Intervention in Online Classification*. Réveillard et al., 2025, arXiv:2510.23557 (AISTATS 2026). An online classifier on embeddings decides for each query whether to pay a human for the label. It comes with regret bounds in terms of labels queried.
- *Conformal Prediction with LLMs for MCQA*. Kumar et al., 2023, arXiv:2305.18404. Conformal sets on letter-choice MCQ track accuracy and support answering selectively. This is the escalation gate.

**What would be new.** Every part exists; the combination appears unpublished. That is the zero-shot letter readout as the prior, conformal escalation to a human, and a per-question probe on the same pass updated from escalated labels. The new result is its curve of human labels needed to match the offline probe (10 labels per question → 0.669, 50 → 0.741, 300 → 0.793).

**Caveats.** Escalated labels are a biased sample, because only uncertain items get labelled. Updating the probe online breaks exchangeability, so coverage needs adaptive conformal inference (Gibbs & Candès, 2021, arXiv:2106.00170).

## 7. How many dimensions a decision needs

- *The Low-Dimensional Linear Geometry of Contextualized Word Representations*. Hernandez & Andreas, 2021, arXiv:2105.07109 (CoNLL 2021). Linguistic features live in low-dimensional linear subspaces. The paper measures accuracy against probe rank.
- *Linear Representations of Sentiment in LLMs*. Tigges et al., 2023, arXiv:2310.15154. A single direction captures sentiment. Ablating it removes 76% of above-chance zero-shot accuracy on SST.
- *Evaluating Unsupervised Dimensionality Reduction Methods for Pretrained Sentence Embeddings*. G. Zhang et al., 2024, arXiv:2403.14001. PCA cuts embedding size by almost 50% with no significant downstream loss.
- *Emergence of a High-Dimensional Abstraction Phase in Language Transformers*. Cheng et al., 2024, arXiv:2405.15471 (ICLR 2025). Intrinsic dimension peaks at an intermediate layer, and that peak predicts transfer. This could explain why layer 26 is best.
- *Optimal Representation Size*. Njaradi et al., 2026, arXiv:2605.20105. Theory: when downstream labels are scarce, compressed representations are optimal.

**What would be new.** Low dimensionality of linear concepts is well established. The only new results are the dimension count for each typed-decision question type (yes/no, choice, score), and whether PCA fitted on unlabeled traffic lifts the 10–50-label regime, as Njaradi predicts.

**Caveat.** PCA on raw states is dominated by 1–3 rogue dimensions (Timkey & van Schijndel, 2021, arXiv:2109.04404). Yet z-scoring cost the 4B about 2 points (0.793 → 0.772). So compare raw PCA, standardised PCA and random projection (Johnson & Lindenstrauss, 1984, Contemporary Mathematics 26; no arXiv) rather than assume one.

## 8. Logit lens, tuned lens and DoLa on the letter readout

- *Eliciting Latent Predictions from Transformers with the Tuned Lens*. Belrose et al., 2023, arXiv:2303.08112. Per-layer affine probes decode intermediate states more reliably than the logit lens. The logit lens itself comes from nostalgebraist's 2020 LessWrong post, not a paper.
- *DoLa: Decoding by Contrasting Layers*. Chuang et al., 2023, arXiv:2309.03883 (ICLR 2024). Contrasting late-layer and early-layer logits raised LLaMA's TruthfulQA scores, including the multiple-choice tasks, by 12–17 points.
- *Overthinking the Truth*. Halawi et al., 2023, arXiv:2307.09476. Decoding label tokens from intermediate layers on classification tasks reveals a "critical layer". After it, false demonstrations steadily degrade accuracy, so an intermediate readout can beat the final layer.
- *Answer, Assemble, Ace*. Wiegreffe et al., 2024, arXiv:2407.15018 (ICLR 2025). Projecting A/B/C/D onto the vocabulary layer by layer shows that middle-layer attention drives answer-symbol prediction and later heads amplify it.
- *Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions (KAPPA)*. Park et al., 2025, arXiv:2509.23782 (ICML 2026). The subspaces that hold the knowledge and the prediction diverge. DoLa, run as a baseline, lost ground on all five MCQ sets for both Qwen2.5-7B and Llama-3.1-8B; the Llama losses were 0.3–3.0 points. Aligning the two subspaces gained up to about 17 points.

**What would be new.** Logit lens on MC letters and DoLa on MCQ are both published, and DoLa did not help. The only new use is diagnostic: locate the layer where the letter becomes decodable, and compare it with where the probe peaks (layers 20–28). Run it on 2–77-option decisions, where the 4B's letters score 0.49.

**What to expect.** DoLa should be null. A tuned lens is a probe restricted to letter tokens, so at best it approaches the probe. Wong et al. (arXiv:2601.03914) and the select-and-copy heads of Tulchinskii et al., 2024 (arXiv:2410.02343, up to +16% over logit selection) both predict that letters lag behind the content.
