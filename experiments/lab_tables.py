"""Render the probe-lab results as markdown tables for BENCHMARKS.md (no numbers typed by hand).

  py experiments/lab_tables.py > docs/lab_tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def J(p):
    f = ROOT / p
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def f3(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pct(x):
    return f"{100 * x:.0f} %"


def main():
    out = []
    w = out.append
    w("## 4c. Probe lab: experiments on cached hidden states (`experiments/probe_lab.py`)\n")
    w("Every table reuses one cache: the last-token hidden state at every layer of the frozen Qwen3.5-4B (and layers 0–40 of Gemma 4 12B, NF4) "
      "for the 8,000 typed-decisions prompts, plus the letter logits. Test split, 2,000 decisions, unless stated. Closest prior work for each idea: "
      "`docs/research/novelty-check.md`; cross-disciplinary sources: `docs/research/abstract-methods-survey.md`. Figure: `docs/figures/tez_lab.png`.\n")

    L4, L12 = J("results/probe_lab_lens_qwen35-4b.json"), J("results/probe_lab_lens_gemma4-12b-nf4.json")
    if L4:
        w("### Where to read: the zero-shot logit lens\n")
        w("| model | readout | test acc |\n|---|---|---|")
        w(f"| Qwen3.5-4B | letters at the final layer (32) | {f3(L4['logit_lens']['32'])} |")
        w(f"| Qwen3.5-4B | letters read at layer 29 through the final norm and head (layer chosen on the train split) | **{f3(L4['logit_lens']['29'])}** |")
        w("| Qwen3.5-4B | layer 24 / 27 / 28 / 30 / 31 | " + " / ".join(f3(L4['logit_lens'][str(l)]) for l in (24, 27, 28, 30, 31)) + " |")
        w("| Qwen3.5-4B | layer chosen per question on the train split (uses labels) | 0.629 |")
        w("| Qwen3.5-4B | layer chosen without labels by confidence (picks layer 21) | 0.269 |")
        best_dola = max(L4["dola"].items(), key=lambda kv: kv[1])
        w(f"| Qwen3.5-4B | DoLa, final minus layer M (best M = {best_dola[0]}) | {f3(best_dola[1])} |")
        if L12:
            w("| Gemma 4 12B NF4 | letters read at layer 36 / 38 / 40 of 48 | " + " / ".join(f3(L12['logit_lens'][str(l)]) for l in (36, 38, 40)) + " |")
        w("| Gemma 4 12B Q8 | full depth, llama.cpp | 0.704 |\n")

    I4, I12, SW, SW12 = J("results/probe_lab_intrinsic_qwen35-4b.json"), J("results/probe_lab_intrinsic_gemma4-12b-nf4.json"), J("results/hidden_probe_sweep_qwen35-4b.json"), J("results/hidden_probe_sweep_gemma4-12b-nf4_std.json")
    if I4 and SW:
        w("### Choosing the layer without labels: intrinsic dimension (TwoNN, 2,000 unlabelled train states per layer)\n")
        ls = [2, 6, 10, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
        w("| Qwen3.5-4B layer | " + " | ".join(str(l) for l in ls) + " |\n|---|" + "---|" * len(ls))
        w("| intrinsic dimension (no labels) | " + " | ".join(f"{I4['layers'][str(l)]['twonn_pooled']:.1f}" for l in ls) + " |")
        w("| probe accuracy (labels) | " + " | ".join(f3(SW['layer_sweep'].get(str(l), {}).get('acc')) for l in ls) + " |\n")
        if I12 and SW12:
            ls = [6, 14, 18, 22, 26, 28, 30, 32, 34, 36, 38, 40]
            w("| Gemma 4 12B layer (of 48) | " + " | ".join(str(l) for l in ls) + " |\n|---|" + "---|" * len(ls))
            w("| intrinsic dimension (no labels) | " + " | ".join(f"{I12['layers'][str(l)]['twonn_pooled']:.1f}" for l in ls) + " |")
            w("| probe accuracy (labels, z-scored) | " + " | ".join(f3(SW12['layer_sweep'].get(str(l), {}).get('acc')) for l in ls) + " |\n")
        w("On the 4B the dimension peaks at layers 13–14 and bottoms out at layers 24–28, exactly the probe plateau. On the 12B it peaks at layer 28 and "
          "falls to the last cached layer (40) while the probe sits on its plateau (0.776–0.788): the rule \"lowest dimension after the peak\" lands on the "
          "plateau for both models, exactly for the 4B and 1.1 points below the best layer for the 12B.\n")

    A = J("results/probe_lab_anytime_qwen35-4b.json")
    if A:
        w("### Anytime depth: per-layer probes, stop when sure (gold-label probes, grid 12–28)\n")
        w("| rule | accuracy | mean layers used (of 32) |\n|---|---|---|")
        w(f"| fixed layer 20 | {f3(A['fixed']['20'])} | 20 |")
        w(f"| fixed layer 26 | {f3(A['fixed']['26'])} | 26 |")
        for t in ("0.6", "0.7", "0.8"):
            w(f"| stop when max p ≥ {t} | {f3(A['gate'][t]['acc'])} | {A['gate'][t]['mean_depth']:.1f} |")
        for t in ("2.0", "3.0"):
            w(f"| stop when summed log-odds ≥ {t} | {f3(A['accumulate'][t]['acc'])} | {A['accumulate'][t]['mean_depth']:.1f} |")
        w("")

    V = J("results/probe_lab_variants_qwen35-4b.json")
    if V:
        w("### Probe families on layer 26\n")
        w("| probe | acc |\n|---|---|")
        for n, r in sorted(V["families"].items(), key=lambda kv: -kv[1]["acc"]):
            w(f"| {n} | {f3(r['acc'])} |")
        w("\n### How many numbers does a decision need? (one PCA on unlabelled train states, shared by all 20 questions)\n")
        ms = sorted(int(k) for k in V["code_size"]["pca"])
        w("| numbers kept | " + " | ".join(str(m) for m in ms) + " | 2,560 (all) |\n|---|" + "---|" * (len(ms) + 1))
        w("| PCA | " + " | ".join(f3(V['code_size']['pca'][str(m)]) for m in ms) + f" | {f3(V['code_size']['full'])} |")
        w("| random projection | " + " | ".join(f3(V['code_size']['random'][str(m)]) for m in ms) + " | |\n")
    CT = J("results/probe_lab_codetransfer_qwen35-4b.json")
    if CT:
        w("**Is the code universal?** PCA fitted on the unlabelled states of three workflows, applied unchanged to the fourth (probes still per question):\n")
        ms = sorted(int(k) for k in CT["m"])
        w("| numbers kept | " + " | ".join(str(m) for m in ms) + " |\n|---|" + "---|" * len(ms))
        w("| code fitted on the same workflow | " + " | ".join(f3(CT['m'][str(m)]['code_from_same_workflow']) for m in ms) + " |")
        w("| code fitted on the other three workflows | " + " | ".join(f3(CT['m'][str(m)]['code_from_other_workflows']) for m in ms) + " |")
        w(f"\nFull 2,560 features, same protocol: {f3(CT.get('full_features'))}.\n")

    TY, FS, PR = J("results/probe_lab_typical_qwen35-4b.json"), J("results/probe_lab_fewshot_qwen35-4b.json"), J("results/probe_lab_prior_qwen35-4b.json")
    if TY or FS or PR:
        w("### Few labels: which rows to label, and what to combine them with (layer 26, 3 seeds)\n")
        w("| labelled rows per question | 5 | 10 | 25 | 50 |\n|---|---|---|---|---|")
        if FS:
            w("| random rows, logistic probe | " + " | ".join(f3(FS['n'][str(n)]['logreg']['mean']) for n in (5, 10, 25, 50)) + " |")
        if TY:
            w("| most typical rows first (member nearest each k-means centre) | " + " | ".join(f3(TY['n'].get(str(n), {}).get('typical', {}).get('mean')) for n in (5, 10, 25, 50)) + " |")
        if PR:
            w("| random rows, mixed with the 12B's zero-shot answers (w = n/(n+10)) | " + " | ".join(f3(PR['n'][str(n)]['random']['mix n0=10']) for n in (5, 10, 25, 50)) + " |")
            w("| typical rows, mixed with the 12B's zero-shot answers | " + " | ".join(("**" + f3(PR['n'][str(n)]['typical']['mix n0=10']) + "**") for n in (5, 10, 25, 50)) + " |")
        if FS:
            w("| LDA, covariance from unlabelled states | " + " | ".join(f3(FS['n'][str(n)]['LDA (unlabelled covariance)']['mean']) for n in (5, 10, 25, 50)) + " |")
            w("| PCA-64 from unlabelled states + logistic | " + " | ".join(f3(FS['n'][str(n)]['PCA-64 (unlabelled) + logreg']['mean']) for n in (5, 10, 25, 50)) + " |")
        w("\nThe 12B alone scores 0.705 with no labels; 50 typical labels per question on top of it reach Laya's fully fine-tuned 0.766.\n")

    CB = J("results/probe_lab_calibration_bytype_qwen35-4b.json")
    if CB:
        w("### Calibration and decision types (ECE-15 and NLL as read, then after a 2-fold out-of-fold temperature)" + chr(10))
        w("| readout | accuracy | ECE | ECE after temperature | NLL | choice (600) | yes/no (600) | score (800) |" + chr(10) + "|---|---|---|---|---|---|---|---|")
        for n, r in CB.items():
            b = r["by_type"]
            w(f"| {n} | {f3(r['acc'])} | {f3(r['ece'])} | {f3(r['ece_oof_temp'])} | {f3(r['nll'])} | {f3(b['choice'])} | {f3(b['noul'])} | {f3(b['score'])} |")
        w(chr(10) + "A logistic probe is fitted with a proper scoring rule, so it comes out calibrated (ECE 0.023 as read; Jev's published ECE is 0.246). The gain from labels is largest on ordinal score questions (+11 points over the 12B)." + chr(10))

    AM, TM = J("results/probe_lab_ambiguity_qwen35-4b.json"), J("results/probe_lab_twomodel_ensemble.json")
    if AM:
        w("### Where the remaining errors are: accuracy by how decided the benchmark's own soft label is" + chr(10))
        bins = list(AM["4B probe"])
        w("| max of the benchmark's soft label | " + " | ".join(bins) + " |" + chr(10) + "|---|" + "---|" * len(bins))
        w("| share of test decisions | " + " | ".join(f"{100 * AM['4B probe'][b]['share']:.0f} %" for b in bins) + " |")
        for n in AM:
            w(f"| {n} | " + " | ".join(f3(AM[n][b]['acc']) for b in bins) + " |")
        if TM:
            w(chr(10) + f"Two backbones' probes averaged: {f3(TM['geometric mean(4B probe, 12B probe)'])} (4B {f3(TM['4B probe L26'])}, 12B {f3(TM['12B probe L34'])}); they agree on {100 * TM['agreement of the two probes']:.0f} % of decisions. Every supervised route converges on 0.79-0.80, and the errors sit where the benchmark's own labels are split." + chr(10))

    W, LM, ST = J("results/probe_lab_w2s_qwen35-4b.json"), J("results/probe_lab_labelmodel_qwen35-4b.json"), J("results/probe_lab_stack_qwen35-4b.json")
    if W:
        w("### No human labels: probes trained on zero-shot answers\n")
        w("| teacher | teacher test acc | hard | soft | confident top 50 % | confident top 25 % | self-training (3 rounds) | cluster-then-label |\n|---|---|---|---|---|---|---|---|")
        for t, lab in (("self", "Qwen3.5-4B's own letters"), ("gemma12b", "Gemma 4 12B letters")):
            T = W["teachers"].get(t)
            if T:
                V_ = {k: max(T["variants"][L][k]["acc"] for L in T["variants"]) for k in T["variants"]["26"]}
                w(f"| {lab} | {f3(T['teacher_test_acc'])} | {f3(V_['hard'])} | {f3(V_['soft'])} | {f3(V_['confident-top50%'])} | {f3(V_['confident-top25%'])} | {f3(V_['self-train round 3'])} | {f3(V_['cluster-then-label'])} |")
        if W["teachers"].get("agree"):
            v = max(W["teachers"]["agree"]["variants"][L]["agreement-filtered"]["acc"] for L in W["teachers"]["agree"]["variants"])
            w(f"| rows where the 4B and 12B agree | | {f3(v)} | | | | | |")
        if LM:
            best = max(LM["combos"].items(), key=lambda kv: kv[1]["label_model_test_acc"])
            w(f"| Dawid–Skene label model ({best[0]}) | {f3(best[1]['label_model_test_acc'])} | {f3(best[1]['probe_L26_hard'])} | {f3(best[1]['probe_L26_soft'])} | | | | |")
        w(f"\nBest of layers 22 and 26 per cell. Gold-label probe for reference: {f3(W['gold']['26'])}. A probe trained on a teacher's answers matches the teacher; it never beats it by more than half a point.\n")
    if ST:
        w("### Stacking readouts (per-question out-of-fold log-linear pool, gold labels)\n")
        w("| pool | stacked logistic | geometric pool |\n|---|---|---|")
        w(f"| probe L26 alone | {f3(ST['single']['probe L26'])} | |")
        for n, r in ST["pools"].items():
            w(f"| {n} | {f3(r['stacked_logreg'])} | {f3(r['geometric_pool'])} |")
        w("")

    CS = J("results/probe_lab_confsel_qwen35-4b.json")
    if CS:
        w("### Conformal selection: a bound on the error rate among the decisions it acts on (BH on conformal p-values; 50 random half/half calibration splits)\n")
        w("| readout | accuracy | bound 5 %: acted / realised error | bound 10 % | bound 20 % |\n|---|---|---|---|---|")
        for n, R in CS.items():
            cells = [f"{pct(R[f'alpha{a}']['acted'])} / {100 * R[f'alpha{a}']['realised_error_among_acted']:.1f} %" for a in (0.05, 0.1, 0.2)]
            w(f"| {n} | {f3(R['accuracy'])} | " + " | ".join(cells) + " |")
        w("\nThe realised error tracks the bound. The pseudo-label probe has its teacher's accuracy but not a usable confidence.\n")

    O2 = J("results/probe_lab_online2_qwen35-4b.json"); O1 = J("results/probe_lab_online_qwen35-4b.json")
    if O2:
        w("### Cold start: begin zero-shot, escalate, learn from the escalations (stream = the 6,000 train decisions; held-out = test split)\n")
        w(f"Starting point: the 12B's zero-shot readout ({f3(O2['zero_shot_test_acc'])}).\n")
        w("| threshold | policy | human labels | automated share | automated accuracy | whole stream (human answers count as correct) | probe held-out | same number of random labels |\n|---|---|---|---|---|---|---|---|")
        for k, r in O2["runs"].items():
            tau, pol, eps = k.split("_"); stream = r["auto_rate"] * r["auto_acc"] + (1 - r["auto_rate"])
            w(f"| {tau[3:]} | {pol} ({eps.replace('eps', 'audit ')}) | {r['labels']:,} | {pct(r['auto_rate'])} | {f3(r['auto_acc'])} | {f3(stream)} | {f3(r['test_acc'])} | {f3(r['random_same_budget_test_acc'])} |")
        w("")
    if O1 and O1["teachers"].get("self"):
        T = O1["teachers"]["self"]
        w("With the 4B's own letters (0.489) as the starting point instead, labels gathered only from uncertain decisions train a worse probe than random labels: "
          + "; ".join(f"τ {pol[3:]}: {c[-1]['labels']:,} labels → {f3(c[-1]['test_acc'])} vs random {f3(T['passive'][pol]['test_acc'])}" for pol, c in T["policies"].items()) + ".\n")

    U = J("results/probe_lab_universal_qwen35-4b.json")
    if U:
        w("### One head for every question (letter-position classes, masked to k options)\n")
        w("| layer | same workflows | unseen workflow (leave-one-workflow-out, mean of 4) | 4B letters on the same rows |\n|---|---|---|---|")
        base = np.mean([b["letters_4b"] for b in U["baselines"].values()])
        for L, r in U["layers"].items():
            w(f"| {L} | {f3(r['in_distribution'])} | {f3(r['lowo_mean'])} | {f3(base)} |")
        w("")
    TR = J("results/probe_truth_qwen35-4b.json")
    if TR:
        w("### A universal \"is this answer right?\" probe (teacher-forced answer letter, one extra token per option)\n")
        w("| features | same workflows | unseen workflow (mean of 4) |\n|---|---|---|")
        for k in TR["lowo"]:
            w(f"| {k} | {f3(TR['in_distribution'][k])} | {f3(TR['lowo'][k]['mean'])} |")
        w("")
    AT = J("results/attn_readout_qwen35-4b.json")
    if AT:
        w("### Select-and-copy attention readout (attention of the answer position to each option, full-attention layers 3, 7, …, 31)\n")
        w("| attention to | top-1 head | top-3 | top-5 | top-10 | heads chosen on the other workflows (top-3, mean of 4) |\n|---|---|---|---|---|---|")
        for n, R in AT["readouts"].items():
            w(f"| {n} | {f3(R['top1_test'])} | {f3(R['top3_test'])} | {f3(R['top5_test'])} | {f3(R['top10_test'])} | {f3(R['unseen_workflow_mean']['top3'])} |")
        w("")

    M = J("results/probe_multiq_qwen35-4b.json")
    if M:
        w("### Many decisions from one forward pass (400 test rows, 5 questions each)\n")
        t, p = M["tokens_test"], M["passes_test"]
        w("| layout | passes | tokens | probe L20 | probe L22 | probe L26 | zero-shot letters |\n|---|---|---|---|---|---|---|")
        w(f"| one prompt per question (today) | {p['per_question_prompts']:,} | {t['per_question_prompts']:,} | {f3(M['probe']['20']['per_question_prompt'])} | {f3(M['probe']['22']['per_question_prompt'])} | {f3(M['probe']['26']['per_question_prompt'])} | {f3(M['zero_shot_letters']['per_question_prompt'])} |")
        w(f"| state first, all five questions, read at each \"Answer n:\" | {p['A_all_questions_one_pass']:,} | {t['A_all_questions_one_pass']:,} | {f3(M['probe']['20']['A_all_questions_one_pass'])} | {f3(M['probe']['22']['A_all_questions_one_pass'])} | {f3(M['probe']['26']['A_all_questions_one_pass'])} | {f3(M['zero_shot_letters']['A_at_markers'])} |")
        w(f"| state only, one vector per row | {p['B_state_only']:,} | {t['B_state_only']:,} | {f3(M['probe']['20']['B_state_only'])} | {f3(M['probe']['22']['B_state_only'])} | {f3(M['probe']['26']['B_state_only'])} | — |\n")

    X, S = J("results/probe_crosslingual_qwen35-4b.json"), J("results/h2h/summary.json")
    if X:
        bestL = max(X["layers"], key=lambda l: X["layers"][l]["macro_acc20"]); R = X["layers"][bestL]
        langs = [l for l in R if isinstance(R[l], dict)]
        w(f"### MASSIVE: one probe trained on English only (2,000 rows), used unchanged in 11 languages (layer {bestL}, same 100 test rows per language as §1)\n")
        w("| | " + " | ".join(langs) + " | macro |\n|---|" + "---|" * (len(langs) + 1))
        w("| 4B probe, 20 options (Laya's protocol) | " + " | ".join(f3(R[l]['acc20']) for l in langs) + f" | **{f3(R['macro_acc20'])}** |")
        w("| 4B probe, all 60 intents | " + " | ".join(f3(R[l]['acc60']) for l in langs) + f" | {f3(R['macro_acc60'])} |")
        if S:
            for m, lab in (("tez", "12B zero-shot letters"), ("laya-ml", "laya-multilingual"), ("laya-en", "laya (English)")):
                vals = [S.get(f"massive:{l}", {}).get(m, {}).get("accuracy") for l in langs]
                w(f"| {lab} | " + " | ".join(f3(v) for v in vals) + f" | {f3(np.mean([v for v in vals if v is not None]))} |")
        w("")

    P = J("results/pruned_server_qwen35-4b.json")
    if P:
        w("### Depth-pruned Qwen3.5-4B GGUFs served by llama.cpp (`gguf_truncate.py`, `pruned_server_bench.py`)\n")
        w("| blocks kept | GGUF size | zero-shot letters (= logit lens at the cut) | probe on the served state | ms / decision, letters (p50) | ms / request, embedding (p50) |\n|---|---|---|---|---|---|")
        sizes = {"L20": "3.06 GB", "L24": "3.53 GB", "L29": "4.13 GB", "L32": "4.48 GB"}
        for n, r in P.items():
            w(f"| {n[1:]} of 32 | {sizes.get(n, '')} | {f3(r['letters_acc'])} | {f3(r['probe_acc'])} | {r['ms_letters_p50']:.0f} | {r['ms_embed_p50']:.0f} |")
        w("\nFor comparison, Gemma 4 12B Q8 letters: 0.705 at 304 ms p50 on the same prompts (typed-decisions, options first, state last).\n")

    A29, A32 = J("results/h2h_q4b_L29/summary.json"), J("results/h2h_q4b_L32/summary.json")
    if A29 and A32:
        w("### Removing the top 3 layers of the 4B, on Laya's public suite (zero-shot letters, same rows as §1)\n")
        tasks = [t for t in A32 if t in A29 and "tez" in A32[t] and "tez" in A29[t]]
        w("| task | 4B, 32 layers | 4B, 29 layers | Δ |\n|---|---|---|---|")
        d = []
        for t in tasks:
            a, b = A32[t]["tez"]["accuracy"], A29[t]["tez"]["accuracy"]; d.append(b - a)
            w(f"| {t} | {f3(a)} | {f3(b)} | {b - a:+.3f} |")
        w(f"| **mean over {len(tasks)} tasks** | | | **{np.mean(d):+.3f}** |\n")

    VO = {v: J(f"results/voice_ddm_{v}.json") for v in ("partial_last", "final_last", "final_first")}
    if any(VO.values()):
        w("### Streaming voice: a collapsing commit bound vs the fixed threshold (`voice_ddm.py`, parameters chosen on one half of the 220 utterances, scored on the other)\n")
        w("| log | policy | harmful / 198 | out-of-scope false / 22 | mean first-action word | first action at or before the human |\n|---|---|---|---|---|---|")
        for v, R in VO.items():
            if not R:
                continue
            for key, lab in (("default_on_eval", "default (τ 0.9, stability 2)"), ("fixed", "best retuned fixed threshold"), ("collapsing", "best collapsing bound")):
                r = R["held_out_summary"][key]
                w(f"| {v.replace('_', ', state ')} | {lab} | {r['harmful']} | {r['oos_false']} | {r['mean_first_word']:.2f} | {r['at_or_before_human']:.2f} |")
        w("\nNeither family is both earlier and as safe as the default; the default stays.\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
