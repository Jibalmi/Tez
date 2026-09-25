"""Write BENCHMARKS.md section 1b (Laya's own benchmarks, Tez in Laya's place) from results/vs_laya/*.json.

  py experiments/vs_laya_tables.py            # rewrites the section between its markers in BENCHMARKS.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "vs_laya"
START = "<!-- vs_laya:start -->"
END = "<!-- vs_laya:end -->"
SYS = ["tez", "laya", "laya-multilingual", "laya-typed-decisions"]
NAME = {"tez": "**Tez**", "laya": "laya", "laya-multilingual": "laya-multilingual", "laya-typed-decisions": "laya-typed-decisions"}


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def f4(x):
    """As the result file gives it: up to four decimals, at least three."""
    if x is None:
        return "—"
    s = f"{x:.4f}"
    return s[:-1] if s.endswith("0") else s


def main():
    apps = json.loads((R / "apps.json").read_text(encoding="utf-8"))
    m51 = json.loads((R / "massive51.json").read_text(encoding="utf-8"))
    sp = json.loads((R / "speed_per_call.json").read_text(encoding="utf-8"))
    sel = json.loads((R / "selective.json").read_text(encoding="utf-8"))
    L = []
    L.append(START)
    L.append("## 1b. On Laya's own benchmarks: Laya's charts, redrawn with Tez")
    L.append("")
    L.append("Laya publishes its comparison with Jev as a set of charts. `experiments/make_vs_figures.py` redraws each one "
             "with Tez in Laya's place (`docs/figures/vs/`, listed in its README); the measurements below fill the panels "
             "Tez had never been run on. Tez and every Laya checkpoint answered byte-identical rows; Laya's workflow "
             "scores come from Laya's own harness (CPU, fp32, as in its published run), Tez from the letter readout on "
             "the RTX 5080 laptop. Jev was never run here.")
    L.append("")
    L.append("![Tez vs Jev, with Laya on the same rows](docs/figures/vs/tez_vs_jev.png)")
    L.append("")
    L.append("### Laya's seven application workflows (400 rows each, Laya's sampling, seed 13; `experiments/vs_laya_apps.py`)")
    L.append("")
    L.append("| task | " + " | ".join(NAME[s] for s in SYS) + " | Laya published (laya / multilingual / typed) |")
    L.append("|---|" + "---:|" * len(SYS) + "---|")
    for t in apps["tasks"]:
        vals = [t["systems"].get(s, {}).get("accuracy") for s in SYS]
        best = max(v for v in vals if v is not None)
        cells = [("**" + f4(v) + "**") if v == best else f4(v) for v in vals]
        pub = t.get("laya_published", {})
        pubs = " / ".join(f4(pub.get(k)) for k in ("laya", "laya-multilingual", "laya-typed-decisions"))
        tag = " (held out)" if t.get("held_out") else " (in Laya's training)"
        n = "" if t.get("n") == 400 else f", n = {t['n']}"
        L.append(f"| {t['label']}{tag}{n} | " + " | ".join(cells) + f" | {pubs} |")
    L.append("")
    L.append("Tez wins the three held-out workflows; Laya wins the four in its training mix. Our reruns of laya and "
             "laya-typed-decisions match Laya's published numbers to four decimals; the released laya-multilingual "
             "checkpoint does not reproduce its published row (higher here on five tasks, most on model routing), cause "
             "not found (`results/vs_laya/apps.json`, `meta` and `reproduction`).")
    L.append("")
    L.append("### MASSIVE intent in all 51 languages (20 options, 100 rows each; `experiments/vs_laya_massive51.py`)")
    L.append("")
    u = m51["usable_3x_random"]
    mac = m51.get("macro_accuracy", {})
    L.append("| system | languages above 3× random (of 51) | mean accuracy |")
    L.append("|---|---:|---:|")
    for s, lab in (("tez", "**Tez** zero-shot"), ("laya-routed", "laya routed (English checkpoint for en, multilingual otherwise)"),
                   ("laya-multilingual", "laya-multilingual"), ("laya", "laya (English)")):
        L.append(f"| {lab} | {u.get(s)} | {f3(mac.get(s)) if isinstance(mac.get(s), (int, float)) else '—'} |")
    lp = m51.get("laya_published_usable_3x_random")
    if lp:
        L.append("")
        if isinstance(lp, dict):
            L.append(f"Laya's own published run counts {lp.get('laya-routed')} usable languages with routing "
                     f"(laya-multilingual {lp.get('laya-multilingual')}, laya {lp.get('laya')}); our rerun of the released "
                     "multilingual checkpoint scores higher in some languages (see the file's `reproduction`).")
        else:
            L.append(f"Laya's own published run counts {lp} usable languages with routing.")
    tez = m51["systems"]["tez"]
    low = sorted(tez.items(), key=lambda kv: kv[1])[:5]
    L.append("")
    L.append("Tez beats Laya's router in every language (the smallest margin is English, 0.90 vs 0.82); its weakest are "
             + ", ".join(f"{k} {v:.2f}" for k, v in low) + ".")
    L.append("")
    L.append("### Speed per call (one state, p50; `experiments/vs_laya_speed.py`)")
    L.append("")
    q = sp["questions_per_call"]
    L.append("| system | " + " | ".join(f"{n} q / call" for n in q) + " | ms per question at 50 |")
    L.append("|---|" + "---:|" * (len(q) + 1))
    for s, lab in (("tez", "**Tez** (tez serve, question first, new ticket each call)"), ("laya", "laya (same GPU)"), ("laya-multilingual", "laya-multilingual (same GPU)")):
        r = sp["systems"].get(s)
        if not r:
            continue
        L.append(f"| {lab} | " + " | ".join(f"{v:,.0f} ms" for v in r["p50_ms_per_call"]) + f" | {r['p50_ms_per_question'][-1]:.1f} ms |")
    L.append("")
    L.append("Laya batches: its cost per question falls to 2.8–7.4 ms at 50 questions per call. Tez runs one forward pass "
             "per question. In these runs the runtime read every question first and the state after it, so the state was "
             "evaluated again for each question and the cost per question stayed near 90–140 ms. The runtime now reads two "
             "or more questions state first (`--layout auto`), so llama-server's prompt cache can keep the state for the "
             "questions after the first. On an idle machine the runtime now answers 50 questions in 2,878 ms over llama-server and 1,147 ms in process (23 ms each; §5b). §5b "
             "measures the layout on direct `/completion` calls and in process.")
    L.append("")
    L.append("### Selective automation on typed-decisions (accuracy on the decisions acted on, most confident first; `experiments/vs_laya_selective.py`)")
    L.append("")
    cov = sel["coverage"]
    L.append("| system | " + " | ".join(f"{int(c * 100)} %" for c in cov) + " |")
    L.append("|---|" + "---:|" * len(cov))
    for s, lab in (("tez letters", "**Tez** letters (zero-shot)"), ("laya-typed-decisions", "laya-typed-decisions (fine-tuned on it)"),
                   ("laya", "laya"), ("laya-multilingual", "laya-multilingual")):
        v = sel["systems"].get(s)
        if v:
            L.append(f"| {lab} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
    su = sel.get("summary", {}).get("tez letters", {})
    if su:
        L.append("")
        L.append(f"Tez's letters read at T = 1 are over-confident (ECE-15 {su.get('ece'):.3f}; most answers sit in the top "
                 f"confidence bin); one out-of-fold temperature brings it to {su.get('ece_after_temperature'):.3f} on these rows "
                 "(2-fold split of `bench_h2h.py`; §4c's 0.067 used the probe lab's split), and the default temperature Tez "
                 "now applies to unfitted questions, fitted without typed-decisions, to 0.051 (§5c). laya-typed-decisions, "
                 "fine-tuned on this benchmark, ranks its own errors better at every coverage.")
    L.append("")
    L.append(END)
    block = "\n".join(L)
    b = ROOT / "BENCHMARKS.md"
    s = b.read_text(encoding="utf-8")
    if START in s:
        s = s[:s.index(START)] + block + s[s.index(END) + len(END):]
    else:
        anchor = "## 2. SemIf's benchmark"
        assert anchor in s
        s = s.replace(anchor, block + "\n\n---\n\n" + anchor, 1)
    b.write_text(s, encoding="utf-8")
    print("BENCHMARKS.md section 1b written")


if __name__ == "__main__":
    main()
