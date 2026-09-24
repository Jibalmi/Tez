"""Aggregate results/speed/*.json into results/speed/summary.json and results/speed/tables.md (the tables of REPORT.md).

Reads whatever exists (every experiment script writes its own JSON); missing experiments are skipped.
  python experiments/speed_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import speed_common as sc

OUT = sc.OUT


def load(name):
    p = OUT / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f(x, nd=1):
    if x is None:
        return "-"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return f"{x:,}"
    return f"{x:,.{nd}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- Exp 1
LAYA_RUNS = [  # (file, label of the server/engine); *_unique = every question of a call distinct (realistic)
    ("laya_prod_unique.json", "12B, production llama-server (-np 1, --cache-ram default), distinct questions"),
    ("laya_tuned_unique.json", "12B, llama-server --cache-ram 0, distinct questions"),
    ("laya_np8_unique.json", "12B, llama-server -np 8 -kvu --cache-ram 0, distinct questions"),
    ("laya_12b_inproc_unique.json", "12B, in-process llama.dll, distinct questions"),
    ("laya_prod.json", "12B, production llama-server, Laya's verbatim protocol"),
    ("laya_tuned.json", "12B, llama-server --cache-ram 0, Laya's verbatim protocol"),
    ("laya_12b_inproc.json", "12B, in-process llama.dll, Laya's verbatim protocol"),
    ("laya_q4bL24_inproc_unique.json", "Qwen3.5-4B 24 blocks, in-process, distinct questions"),
]
ARM_NAMES = {
    "tez:today": "tez serve, today's layout (runtime as deployed)",
    "http:today": "direct /completion per question, today's layout",
    "http:statefirst": "direct /completion per question, state first (state cached once per call)",
    "http-par:today": "concurrent /completion (one per question), today's layout",
    "http-par:statefirst": "concurrent /completion, state first",
    "http-multi:statefirst": "one /completion with all prompts, state first",
    "seq_today": "sequential, today's layout (KV rollback reuse)",
    "seq_statefirst": "sequential, state first (KV rollback reuse)",
    "batch_today": "one batch, today's layout (no sharing)",
    "batch_statefirst": "state once + all question suffixes in ONE decode (seq_cp)",
    "onepass_markers": "one sequence, all questions, read at markers",
}


def guard_note(d):
    g = (d or {}).get("gpu_guard")
    if not g:
        return "no snapshot"
    if not g["contaminated"]:
        return "clean"
    reasons = g["reasons_before"] + g["reasons_after"]
    used = max(g["before"].get("used_mb", 0), g["after"].get("used_mb", 0))
    if all(r.startswith("our process") and "shared" in r for r in reasons) and used < 12000:
        mb = max(int(r.split(" has ")[1].split(" MB")[0]) for r in reasons)
        # /embedding makes libllama output every prompt token, so llama-server holds a pinned host buffer for all of
        # their logits: shared GPU memory by Windows' accounting, not a spill (the GPU is far from full)
        return (f"flagged by the 200 MB rule: own pinned host output buffer ({mb} MB) of the /embedding path; "
                f"{used / 1024:.1f} of 15.9 GB VRAM in use, no spill")
    return "CONTAMINATED: " + "; ".join(reasons)


def sm_note(d):
    g = (d or {}).get("gpu") or {}
    vals = [v.get("during", {}).get("sm_mhz_busy_p50") for v in g.values() if isinstance(v, dict)]
    vals = [v for v in vals if v]
    return f"{min(vals):.0f}-{max(vals):.0f} MHz" if vals else "-"


def exp1():
    rows, summ = [], {}
    for fname, label in LAYA_RUNS:
        d = load(fname)
        if not d:
            continue
        for arm, sizes in d["arms"].items():
            cells = []
            for n in ("1", "5", "10", "50"):
                s = sizes.get(n)
                cells.append(f"{f(s['p50'], 0)} / {f(s['p95'], 0)}" if s else "-")
            s50 = sizes.get("50") or {}
            tok = s50.get("prompt_tokens_evaluated_per_call_p50", s50.get("tokens_evaluated_per_call_p50"))
            rows.append([label, ARM_NAMES.get(arm, arm)] + cells + [f(s50.get("ms_per_question_p50"), 1), f(tok, 0), sm_note(d), guard_note(d)])
            summ[f"{fname}:{arm}"] = {n: {k: sizes[n][k] for k in ("p50", "p95", "ms_per_question_p50", "ms_per_question_p95")
                                          if k in sizes[n]} for n in sizes}
        g = d.get("gpu", {})
        summ[f"{fname}:gpu"] = {n: g[n].get("during") for n in g}
        summ[f"{fname}:guard"] = guard_note(d)
    t = table(["model, server / engine", "arm", "1 q p50 / p95 (ms)", "5 q", "10 q", "50 q", "ms per q at 50 (p50)",
               "tokens evaluated per 50-q call", "GPU SM clock (busy p50 per size)", "VRAM guard"], rows)
    acc_rows, acc = [], {}
    for fname, label in (("td_prod_today_http.json", "today's layout (options first, state last), HTTP"),
                         ("td_prod_statefirst_http.json", "state first, HTTP, prompt cache")):
        d = load(fname)
        if d:
            acc_rows.append([label, f(d["accuracy"], 4), d["by_type"].get("choice"), d["by_type"].get("noul"), d["by_type"].get("score"),
                             f(d.get("prompt_tokens_evaluated_total"), 0), "-"])
            acc[fname] = {k: d[k] for k in ("accuracy", "by_type", "prompt_tokens_evaluated_total") if k in d}
    for fname in ("td_12b_inproc.json", "td_q4bL24_inproc.json"):
        d = load(fname)
        if not d:
            continue
        for arm, s in d.items():
            if arm == "meta" or not isinstance(s, dict) or "accuracy" not in s:
                continue
            comp = "; ".join(f"{k[3:]}: agree {v['agreement']}, McNemar p {v['mcnemar']['p']}" for k, v in s.items() if k.startswith("vs_"))
            acc_rows.append([f"{fname.replace('td_', '').replace('.json', '')} {ARM_NAMES.get(arm, arm)}", f(s["accuracy"], 4),
                             s["by_type"].get("choice"), s["by_type"].get("noul"), s["by_type"].get("score"), "-", comp or "-"])
            acc[f"{fname}:{arm}"] = {k: s[k] for k in s if k not in ("gpu",)}
    ta = table(["layout / engine", "accuracy (2,000)", "choice", "noul", "score", "prompt tokens evaluated", "paired vs HTTP today's layout"], acc_rows)
    return t, ta, {"laya_protocol": summ, "typed_decisions": acc}


# ---------------------------------------------------------------------------------------------- Exp 2
def exp2():
    acc = load("probe_accuracy.json") or {}
    names = {"L24_stateonly": "state only, one vector per state (llama-server /embedding)",
             "L24_stateonly_inproc": "state only, one vector per state (in-process)",
             "L24_sfq": "state first, then the question: one vector per question, state shared (in-process)",
             "qprompt_served_L24 (results/probe_cache_served_L24.npz)": "question first, state last: one prompt per question (served, BENCHMARKS)"}
    rows = [[names.get(k, k), f(v["accuracy"], 4), v["by_type"].get("choice"), v["by_type"].get("noul"), v["by_type"].get("score"), v["C"]]
            for k, v in acc.items()]
    ta = table(["features (Qwen3.5-4B, 24 blocks)", "probe accuracy (2,000)", "choice", "noul", "score", "C"], rows)
    lat_rows, lat = [], {}
    for fname, label in (("probe_latency_L24_http.json", "llama-server /embedding (-np 1)"),
                         ("probe_latency_L24_httpnp8.json", "llama-server -np 8, one /embedding per question, concurrent"),
                         ("probe_latency_L24_httpnp8multi.json", "llama-server -np 8, one /embedding with all prompts"),
                         ("probe_latency_L24_inproc.json", "in-process llama.dll")):
        d = load(fname)
        if not d:
            continue
        for layout, sizes in d["layouts"].items():
            cells = [f"{f(sizes[n]['p50'], 1)} / {f(sizes[n]['p95'], 1)}" if n in sizes else "-" for n in ("1", "2", "5", "10", "20", "50")]
            s50 = sizes.get("50", {})
            lat_rows.append([label, layout] + cells + [f(s50.get("ms_per_question_p50"), 2), guard_note(d)])
            lat[f"{fname}:{layout}"] = {n: {k: sizes[n][k] for k in ("p50", "p95", "ms_per_question_p50", "feature_ms_p50", "probe_ms_p50")} for n in sizes}
        lat[f"{fname}:guard"] = guard_note(d)
    tl = table(["engine", "layout", "1 q p50 / p95 (ms)", "2 q", "5 q", "10 q", "20 q", "50 q", "ms per q at 50", "VRAM guard"], lat_rows)
    return ta, tl, {"accuracy": acc, "latency": lat}


# ---------------------------------------------------------------------------------------------- Exp 3
VOICE_RUNS = [
    ("voice_12b_http_prod.json", "Gemma 4 12B letters, production llama-server (cache on, the published baseline)"),
    ("voice_12b_inproc.json", "Gemma 4 12B letters, in-process"),
    ("voice_q4b_inproc.json", "Qwen3.5-4B (32 blocks) letters, in-process"),
    ("voice_q4bL24_http_nocache.json", "Qwen3.5-4B 24 blocks letters, llama-server, cache off (the runtime's Qwen setting)"),
    ("voice_q4bL24_inproc.json", "Qwen3.5-4B 24 blocks letters, in-process"),
    ("voice_q4bL24_probe_http.json", "Qwen3.5-4B 24 blocks probe at the last word (2-fold CV), llama-server /embedding, cache on (exact extension)"),
    ("voice_q4bL24_probe.json", "Qwen3.5-4B 24 blocks probe at the last word (2-fold CV), in-process, exact extension"),
    ("voice_q4bL24_probetail.json", "Qwen3.5-4B 24 blocks probe at the answer position (2-fold CV), in-process"),
    ("voice_q4b_probetail.json", "Qwen3.5-4B (32 blocks) probe at the answer position (2-fold CV), in-process"),
    ("voice_q4bL24_probetail_defer.json", "Qwen3.5-4B 24 blocks probe at the answer position (2-fold CV), in-process, deferred commit"),
    ("voice_q4bL24_probeend_defer.json", "Qwen3.5-4B 24 blocks probe after '\"<|im_end|>' (2-fold CV), in-process, deferred commit"),
    ("voice_12b_inproc_defer.json", "Gemma 4 12B letters, in-process, deferred commit"),
]


def exp3():
    rows, summ = [], {}
    for fname, label in VOICE_RUNS:
        d = load(fname)
        if not d:
            continue
        lat = d["latency_per_word"]["incremental_words"]
        variants = [("", d)] if "accuracy" in d else [(f" [{v}]", d[f"probe_{v}"]) for v in ("full", "prefix") if f"probe_{v}" in d]
        for suffix, blk in variants:
            a, p = blk["accuracy"], blk["streaming_policy"]
            g = d.get("gpu", {}).get("during", {}) or {}
            rows.append([label + suffix, f"{f(lat['compute_ms'].get('p50'))} / {f(lat['compute_ms'].get('p95'))}",
                         f"{f(lat['round_trip_ms'].get('p50'))} / {f(lat['round_trip_ms'].get('p95'))}", f(lat["tokens_evaluated"].get("p50"), 0),
                         f(a["intent_accuracy"], 3), f(a["intent_accuracy_alt_or_then_ok"], 3), f"{f(a['none_recall'], 2)} / {f(a['none_precision'], 2)}",
                         f"{p['harmful']} / {p['actionable']}", f"{p['oos_false_actions']} / {p['oos_rows']}", f(p["mean_first_action_word"], 2),
                         f(p["first_action_at_or_before_human"], 2), f"{g.get('thermal_throttle_frac', '-')}", guard_note(d)])
            summ[fname + suffix] = {"latency_incremental_words": lat, "first_word": d["latency_per_word"]["first_word"],
                                    "accuracy": {k: v for k, v in a.items() if k != "accuracy_by_style"},
                                    "policy": {k: v for k, v in p.items() if not k.endswith("detail")}, "gpu": g,
                                    "guard": guard_note(d)}
    t = table(["model / engine", "compute ms per word p50 / p95", "round trip ms per word p50 / p95", "tokens per word",
               "intent acc.", "acc. accepting then/alt", "none recall / precision", "harmful", "OOS false actions",
               "first action word", "first action <= human word", "thermal-throttled share", "VRAM guard"], rows)
    return t, summ


# ---------------------------------------------------------------------------------------------- Exp 4
def exp4():
    rows, summ = [], {}
    for fname, label in (("overhead_prod.json", "production llama-server"), ("overhead_tuned.json", "llama-server --cache-ram 0")):
        d = load(fname)
        if not d:
            continue
        m = d["modes"]
        for mode, name in (("tez_serve", "client -> tez serve -> llama-server (runtime)"), ("engine", "Tez engine in-process -> llama-server"),
                           ("direct_np200", "direct /completion, n_probs 200"), ("direct_np20", "direct /completion, n_probs 20"),
                           ("direct_np0", "direct /completion, n_probs 0"), ("health_llama", "GET /health (HTTP floor)")):
            s = m.get(mode)
            if not s:
                continue
            pm = s.get("prompt_ms", {})
            rows.append([label, name, f"{f(s['round_trip_ms']['p50'])} / {f(s['round_trip_ms']['p95'])}", f(pm.get("p50")),
                         f(s.get("prompt_n", {}).get("p50"), 0), f(s.get("bytes", {}).get("p50"), 0), guard_note(d)])
        summ[fname] = dict(d["decomposition_p50"], guard=guard_note(d), gpu=d.get("gpu"))
    t = table(["server", "path", "round trip p50 / p95 (ms)", "llama prompt_ms p50", "tokens evaluated", "response bytes", "VRAM guard"], rows)
    return t, summ


def qwen_cache():
    rows, summ = [], {}
    for fname in sorted(p.name for p in OUT.glob("qwen_cache_check_*.json")):
        d = load(fname)
        for name, s in d["scenarios"].items():
            steps = "; ".join(f"{st.get('path')} n={st.get('prompt_n')} cached={st.get('cache_n')} {st.get('ms')} ms" for st in s["steps"])
            rows.append([fname.replace("qwen_cache_check_", "").replace(".json", ""), name, "CRASH" if s["crashed"] else "ok", steps])
            summ[f"{fname}:{name}"] = {"crashed": s["crashed"], "steps": s["steps"]}
    return table(["model", "scenario", "server", "steps (evaluated / reused tokens, ms)"], rows), summ


def headline() -> dict:
    """The fastest measured configuration for each target, with the same-session baseline next to it."""
    h = {}
    base = load("laya_prod_unique.json")
    inp = load("laya_12b_inproc_unique.json")
    np8 = load("laya_np8_unique.json")
    td = load("td_12b_inproc.json") or {}
    if base and inp:
        b50 = base["arms"]["tez:today"]["50"]
        i50 = inp["arms"]["batch_statefirst"]["50"]
        h["i_50_questions_one_state_letters_12b"] = {
            "config": "in-process llama.dll, Gemma 4 12B Q8_0: state evaluated once, all 50 question suffixes in one "
                      "llama_decode (llama_memory_seq_cp shares the state), exact letter logits",
            "p50_ms_per_call": i50["p50"], "p95_ms_per_call": i50["p95"], "ms_per_question_p50": i50["ms_per_question_p50"],
            "accuracy_typed_decisions": (td.get("batch_statefirst") or {}).get("accuracy"),
            "accuracy_todays_layout_http": (load("td_prod_today_http.json") or {}).get("accuracy"),
            "paired_vs_todays_layout": ((td.get("batch_statefirst") or {}).get("vs_td_rows_prod_today_http.jsonl")),
            "baseline_tez_serve_today_p50_ms": b50["p50"], "baseline_tez_serve_today_p95_ms": b50["p95"],
            "speedup_vs_tez_serve_p50": round(b50["p50"] / i50["p50"], 2),
            "same_run_reference_seq_today_p50_ms": inp["arms"]["seq_today"]["50"]["p50"],
            "best_over_llama_server_http": (np8 or {}).get("arms", {}).get("http-multi:statefirst", {}).get("50"),
            "protocol": "Laya's latency protocol with 50 distinct questions per call, new ticket every call",
        }
    ph, pi, pn = load("probe_latency_L24_http.json"), load("probe_latency_L24_inproc.json"), load("probe_latency_L24_httpnp8.json")
    pa = load("probe_accuracy.json") or {}
    if ph and pi:
        s_h, s_i = ph["layouts"]["stateonly"], pi["layouts"]["stateonly"]
        h["ii_many_questions_probes"] = {
            "config": "Qwen3.5-4B cut to 24 blocks: ONE last-token state of the state-only prompt per call, then one "
                      "logistic probe per question on that vector (probe cost: microseconds)",
            "http_llama_server_ms_by_n": {n: [s_h[n]["p50"], s_h[n]["p95"]] for n in s_h},
            "inprocess_ms_by_n": {n: [s_i[n]["p50"], s_i[n]["p95"]] for n in s_i},
            "ms_per_question_at_50_http": s_h["50"]["ms_per_question_p50"],
            "ms_per_question_at_50_inprocess": s_i["50"]["ms_per_question_p50"],
            "accuracy_typed_decisions": (pa.get("L24_stateonly") or {}).get("accuracy"),
            "alternatives": {
                "question_in_prompt (today's layout)": {"accuracy": (pa.get("qprompt_served_L24 (results/probe_cache_served_L24.npz)") or {}).get("accuracy"),
                                                        "ms_50q_http_np8": (pn or {}).get("layouts", {}).get("qprompt", {}).get("50", {}).get("p50"),
                                                        "ms_50q_inprocess": pi["layouts"].get("qprompt", {}).get("50", {}).get("p50")},
                "state_first_then_question (sfq)": {"accuracy": (pa.get("L24_sfq") or {}).get("accuracy"),
                                                     "ms_50q_inprocess": pi["layouts"].get("sfq", {}).get("50", {}).get("p50")},
            },
        }
    v12 = load("voice_12b_inproc.json")
    vb = load("voice_12b_http_prod.json")
    if v12 and vb:
        L = v12["latency_per_word"]["incremental_words"]
        Lb = vb["latency_per_word"]["incremental_words"]
        h["iii_voice_word_to_action"] = {
            "config": "in-process llama.dll, Gemma 4 12B Q8_0 letters: constant prefix cached on a sequence, per word "
                      "one decode (new word on the committed sequence + closing quote and tail on a copy)",
            "compute_ms_p50": L["compute_ms"]["p50"], "compute_ms_p95": L["compute_ms"]["p95"],
            "round_trip_ms_p50": L["round_trip_ms"]["p50"], "round_trip_ms_p95": L["round_trip_ms"]["p95"],
            "intent_accuracy": v12["accuracy"]["intent_accuracy"], "harmful": v12["streaming_policy"]["harmful"],
            "baseline_http_round_trip_ms_p50": Lb["round_trip_ms"]["p50"], "baseline_http_round_trip_ms_p95": Lb["round_trip_ms"]["p95"],
            "baseline_http_compute_ms_p50": Lb["compute_ms"]["p50"],
        }
    vp = load("voice_q4bL24_probetail_defer.json")
    if vp:
        L = vp["latency_per_word"]
        blk = vp["probe_prefix"]
        h["iii_voice_word_to_action_with_labels"] = {
            "config": "in-process llama.dll, Qwen3.5-4B cut to 24 blocks, logistic probe on the state at the answer "
                      "position (after the closing quote and template tail), deferred commit; probe trained on every "
                      "prefix of ~110 labelled utterances (2-fold CV over the 220 commands)",
            "compute_ms_p50": L["incremental_words"]["compute_ms"]["p50"], "compute_ms_p95": L["incremental_words"]["compute_ms"]["p95"],
            "round_trip_ms_p50": L["incremental_words"]["round_trip_ms"]["p50"], "round_trip_ms_p95": L["incremental_words"]["round_trip_ms"]["p95"],
            "deferred_commit_ms_p50": L["deferred_commit_ms"]["p50"],
            "intent_accuracy": blk["accuracy"]["intent_accuracy"], "intent_accuracy_accepting_then_alt": blk["accuracy"]["intent_accuracy_alt_or_then_ok"],
            "harmful": blk["streaming_policy"]["harmful"], "oos_false_actions": blk["streaming_policy"]["oos_false_actions"],
            "full_utterance_training": {"intent_accuracy": vp["probe_full"]["accuracy"]["intent_accuracy"],
                                        "harmful": vp["probe_full"]["streaming_policy"]["harmful"]},
        }
    return h


def main():
    t1, t1a, s1 = exp1()
    t2a, t2l, s2 = exp2()
    t3, s3 = exp3()
    t4, s4 = exp4()
    tq, sq = qwen_cache()
    md = ["## Exp 1: many questions about one state (Gemma 4 12B Q8_0 letters unless stated; Laya's protocol, new ticket per call)",
          t1, "", "### Typed-decisions accuracy by layout (400 rows x 5 questions)", t1a, "",
          "## Exp 2: probes (Qwen3.5-4B cut to 24 blocks)", "### Accuracy", t2a, "", "### Latency per call (Laya's protocol)", t2l, "",
          "## Exp 3: streamed voice word -> action (220 commands, 1,158 words; incremental words k >= 2)", t3, "",
          "## Exp 4: single-question overhead", t4, "", "## Qwen3.5 prompt-cache safety on llama-server b11100", tq, ""]
    (OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
    old = json.loads((OUT / "summary.json").read_text(encoding="utf-8")) if (OUT / "summary.json").exists() else {}
    old.update({"headline": headline(), "exp1_many_questions": s1, "exp2_probes": s2, "exp3_voice": s3, "exp4_overhead": s4,
                "qwen_cache_check": sq})
    (OUT / "summary.json").write_text(json.dumps(old, indent=1, default=str), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
