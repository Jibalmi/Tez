"""Publish Laya's charts, redrawn with Tez, on the site's benchmarks page.

Copies docs/figures/vs (composites and panels) into site/assets/figures/vs/ as full-size PNGs plus light WebP previews,
records each image's origin next to it, and writes the section between the vs-laya markers in site/benchmarks.html
(tables built from results/vs_laya/*.json, captions from the figure titles in docs/figures/vs/README.md).

  py scripts/build_vs_laya_section.py
"""
from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "figures" / "vs"
DST = ROOT / "site" / "assets" / "figures" / "vs"
RES = ROOT / "results" / "vs_laya"
PAGE = ROOT / "site" / "benchmarks.html"
START, END = "<!-- vs-laya:start -->", "<!-- vs-laya:end -->"
EMBED = Path.home() / ".claude" / "skills" / "impeccable" / "scripts" / "embed-prompt.mjs"
BLOB = "https://github.com/Jibalmi/Tez/blob/main/"
NBSP = chr(0xA0)
ORIGIN = ("Origin: data chart, not an AI-generated image. Rendered with matplotlib by experiments/make_vs_figures.py "
          "from BENCHMARKS.md, results/h2h and results/vs_laya (github.com/Jibalmi/Tez); {what}.")

COMPOSITES = [
    ("tez_vs_jev_full.png", "The full comparison", "Accuracy against Jev, where Tez leads and where others lead, every workflow on identical rows, English against the rest, speed, calibration, typed-decisions, all 51 languages and why --swa-full matters."),
    ("tez_benchmark.png", "Every language, speed, Jev and calibration", "Which system can read each language, beside speed per call, accuracy against Jev and calibration."),
    ("tez_benchmark_common.png", "Consolidated benchmark", "Every workflow and task, English against the rest, speed, batching and typed-decisions."),
    ("tez_selective.png", "Selective automation", "Accuracy on the typed-decisions answers each system acts on, most confident first."),
    ("tez_reliability.png", "Reliability and risk-coverage", "How well Tez's zero-shot confidence matches its accuracy on typed-decisions, as shipped and after one temperature."),
]


PANELS = [  # order and names for the gallery: most informative first
    ("workflows_every_checkpoint.png", "Every workflow, every checkpoint"),
    ("all_51_languages.png", "All 51 languages"),
    ("where_tez_leads.png", "Where Tez leads"),
    ("where_others_lead.png", "Where others lead"),
    ("accuracy_vs_jev.png", "Accuracy against Jev"),
    ("speed_per_call.png", "Speed per call"),
    ("batching.png", "Batching"),
    ("speed_per_decision.png", "Speed per decision"),
    ("english_vs_rest.png", "English against the rest"),
    ("every_language.png", "Every language"),
    ("language_coverage.png", "Language coverage"),
    ("xnli_languages.png", "XNLI languages"),
    ("calibration_vs_jev.png", "Calibration against Jev"),
    ("calibration_shipped_vs_temperature.png", "Calibration, as shipped and after one temperature"),
    ("reliability.png", "Reliability"),
    ("risk_coverage.png", "Risk and coverage"),
    ("typed_decisions_ladder.png", "typed-decisions ladder"),
    ("swa_full.png", "Why --swa-full matters"),
    ("and_facts.png", "Deployment facts"),
]


def f4(x):
    if x is None:
        return "–"
    s = f"{x:.4f}"
    return s[:-1] if s.endswith("0") else s


def f3(x):
    return "–" if x is None else f"{x:.3f}"


def esc(s):
    return html.escape(str(s), quote=True)


def panel_titles():
    """Map panel file name -> its title (the finding), from docs/figures/vs/README.md."""
    out = {}
    readme = (SRC / "README.md").read_text(encoding="utf-8")
    for m in re.finditer(r"\| \[`panels/([^`]+)`\]\([^)]*\) \| (.*?) \| (.*?) \|", readme):
        out[m.group(1)] = (m.group(2).strip(), m.group(3).strip())
    return out


def publish_images():
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "panels").mkdir(exist_ok=True)
    (DST / "web").mkdir(exist_ok=True)
    made = []
    for src in sorted(SRC.glob("*.png")) + sorted((SRC / "panels").glob("*.png")):
        rel = src.relative_to(SRC)
        full = DST / rel
        shutil.copyfile(src, full)
        im = Image.open(src).convert("RGB")
        w, h = im.size
        scale = min(1.0, 1600 / w)
        prev = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        web = DST / "web" / (rel.as_posix().replace("/", "__").replace(".png", ".webp"))
        prev.save(web, "WEBP", quality=82, method=6)
        what = f"copied from docs/figures/vs/{rel.as_posix()}"
        subprocess.run(["node", str(EMBED), str(full), "--prompt", ORIGIN.format(what=what)], check=True, capture_output=True)
        subprocess.run(["node", str(EMBED), str(web), "--prompt", ORIGIN.format(what=f"web preview of {rel.as_posix()}")],
                       check=True, capture_output=True)
        made.append((rel.as_posix(), (w, h), web.name, prev.size))
    return made


def card(rel, size, web, name, desc, tall=False):
    w, h = size
    return (f'<li><figure class="fig-card{" is-tall" if tall else ""}">'
            f'<a class="paper" href="./assets/figures/vs/{esc(rel)}"><img src="./assets/figures/vs/web/{esc(web)}" '
            f'width="{w}" height="{h}" loading="lazy" decoding="async" alt="{esc(name)}. Open full size."></a>'
            f'<figcaption><span class="fig-name">{esc(name)}</span><span class="fig-desc">{esc(desc)}</span>'
            f'<span class="fig-meta">{esc(rel.split("/")[-1])} · {w} × {h}</span></figcaption></figure></li>')


def tables():
    apps = json.loads((RES / "apps.json").read_text(encoding="utf-8"))
    m51 = json.loads((RES / "massive51.json").read_text(encoding="utf-8"))
    sp = json.loads((RES / "speed_per_call.json").read_text(encoding="utf-8"))
    sel = json.loads((RES / "selective.json").read_text(encoding="utf-8"))
    sysk = ["tez", "laya", "laya-multilingual", "laya-typed-decisions"]
    out = []
    # workflows
    rows = []
    for t in apps["tasks"]:
        vals = [t["systems"].get(k, {}).get("accuracy") for k in sysk]
        best = max(v for v in vals if v is not None)
        cells = "".join(f'<td class="num{" best" if v == best else ""}">{f4(v)}</td>' for v in vals)
        tag = "held out" if t.get("held_out") else "in Laya's training"
        n = "" if t.get("n") == 400 else f" ({t['n']} rows)"
        rows.append(f'<tr><th scope="row">{esc(t["label"])}{esc(n)}<span class="row-note">{tag}</span></th>{cells}</tr>')
    out.append('<h3 id="vs-laya-workflows">Laya\'s seven application workflows</h3>'
               '<p>400 rows each, built exactly as Laya\'s <code>bench_apps.py</code> builds them (seed 13). Tez zero-shot '
               'against Laya\'s three checkpoints, which we reran with Laya\'s own harness. Best in each row in bold.</p>'
               '<div class="table-wrap"><table class="table"><thead><tr><th scope="col">Workflow</th>'
               '<th scope="col" class="num">Tez</th><th scope="col" class="num">laya</th>'
               '<th scope="col" class="num">laya-multilingual</th><th scope="col" class="num">laya-typed-decisions</th>'
               '</tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>"
               '<p class="source">Tez wins the three workflows held out of Laya\'s training; Laya wins the four in it. '
               'Our reruns of laya and laya-typed-decisions match Laya\'s published numbers to four decimals; the released '
               'laya-multilingual checkpoint scores higher here than in Laya\'s own table. '
               f'<a href="{BLOB}results/vs_laya/apps.json">results/vs_laya/apps.json</a>, '
               f'<a href="{BLOB}experiments/vs_laya_apps.py">experiments/vs_laya_apps.py</a>.</p>')
    # 51 languages
    u, mac = m51["usable_3x_random"], m51.get("macro_accuracy", {})
    lang_rows = "".join(
        f'<tr><th scope="row">{esc(lab)}</th><td class="num{" best" if k == "tez" else ""}">{u.get(k)}</td>'
        f'<td class="num{" best" if k == "tez" else ""}">{f3(mac.get(k))}</td></tr>'
        for k, lab in (("tez", "Tez zero-shot"), ("laya-routed", "laya, routed"), ("laya-multilingual", "laya-multilingual"),
                       ("laya", "laya (English checkpoint)")))
    out.append('<h3 id="vs-laya-51">MASSIVE intent in all 51 languages</h3>'
               '<p>20 options, 100 rows per language, the same rows for every system. A language counts as usable above '
               'three times random (0.15).</p>'
               '<div class="table-wrap"><table class="table"><thead><tr><th scope="col">System</th>'
               '<th scope="col" class="num">Usable languages of 51</th><th scope="col" class="num">Mean accuracy</th>'
               '</tr></thead><tbody>' + lang_rows + "</tbody></table></div>"
               '<p class="source">Laya\'s own published run counts 45 usable with routing. '
               f'<a href="{BLOB}results/vs_laya/massive51.json">results/vs_laya/massive51.json</a>.</p>')
    # speed per call
    q = sp["questions_per_call"]
    sp_rows = []
    for k, lab in (("tez", "Tez, through tez serve"), ("laya", "laya, same GPU"), ("laya-multilingual", "laya-multilingual, same GPU")):
        r = sp["systems"].get(k)
        if r:
            cells = "".join(f'<td class="num">{v:,.0f}{NBSP}ms</td>' for v in r["p50_ms_per_call"])
            sp_rows.append(f'<tr><th scope="row">{esc(lab)}</th>{cells}<td class="num">{r["p50_ms_per_question"][-1]:.1f}{NBSP}ms</td></tr>')
    out.append('<h3 id="vs-laya-speed">Speed per call</h3>'
               '<p>p50 per call with 1, 5, 10 and 50 questions about one message, on the same laptop GPU. Laya batches '
               'its questions; Tez reads the message again for each question, so its cost per question barely falls.</p>'
               '<div class="table-wrap"><table class="table"><thead><tr><th scope="col">System</th>'
               + "".join(f'<th scope="col" class="num">{n} per call</th>' for n in q)
               + '<th scope="col" class="num">Per question at 50</th></tr></thead><tbody>' + "".join(sp_rows)
               + "</tbody></table></div>"
               f'<p class="source"><a href="{BLOB}results/vs_laya/speed_per_call.json">results/vs_laya/speed_per_call.json</a>, '
               f'<a href="{BLOB}experiments/vs_laya_speed.py">experiments/vs_laya_speed.py</a>.</p>')
    # selective automation
    cov = sel["coverage"]
    sel_rows = []
    for k, lab in (("tez letters", "Tez zero-shot"), ("laya-typed-decisions", "laya-typed-decisions, fine-tuned on it"),
                   ("laya", "laya"), ("laya-multilingual", "laya-multilingual")):
        v = sel["systems"].get(k)
        if v:
            sel_rows.append(f'<tr><th scope="row">{esc(lab)}</th>' + "".join(f'<td class="num">{x:.3f}</td>' for x in v) + "</tr>")
    out.append('<h3 id="vs-laya-selective">Selective automation on typed-decisions</h3>'
               '<p>Accuracy on the decisions a system acts on when it automates only its most confident share.</p>'
               '<div class="table-wrap"><table class="table"><thead><tr><th scope="col">System</th>'
               + "".join(f'<th scope="col" class="num">{int(c * 100)}{NBSP}%</th>' for c in cov)
               + "</tr></thead><tbody>" + "".join(sel_rows) + "</tbody></table></div>"
               '<p class="source">laya-typed-decisions, fine-tuned on this benchmark, ranks its own errors better at every '
               f'coverage. <a href="{BLOB}results/vs_laya/selective.json">results/vs_laya/selective.json</a>.</p>')
    return "".join(out)


def main():
    made = publish_images()
    titles = panel_titles()
    by = {rel: (size, web) for rel, size, web, _ in made}
    comps = [card(rel, by[rel][0], by[rel][1], name, desc, tall=(rel == "tez_vs_jev_full.png"))
             for rel, name, desc in COMPOSITES if rel in by]
    panels = []
    named = dict(PANELS)
    order = [f for f, _ in PANELS] + sorted(r.split("/")[1] for r in by if r.startswith("panels/") and r.split("/")[1] not in named)
    for fname in order:
        rel = "panels/" + fname
        if rel not in by:
            continue
        size, web = by[rel]
        title, _src = titles.get(fname, ("", ""))
        name = named.get(fname, fname.replace(".png", "").replace("_", " "))
        panels.append(card(rel, size, web, name, title, tall=size[1] > 1.6 * size[0]))
    head_size, head_web = by["tez_vs_jev.png"]
    section = (
        START
        + '<section class="doc-section" aria-labelledby="vs-laya">'
        '<h2 id="vs-laya">Laya\'s charts, redrawn with Tez</h2>'
        '<p>Laya publishes its comparison with TypeSafe Jev as a set of charts. Each one is redrawn here in the same '
        'layout with Tez in Laya\'s place: Tez in blue, Jev in grey as a published reference, and Laya\'s three '
        'checkpoints as comparators. We measured Tez on every benchmark behind those charts, including the ones it had '
        'never been run on: Laya\'s seven application workflows, all 51 languages, speed per call and selective '
        'automation. Tez and every Laya checkpoint answered byte-identical rows; Jev was never run here.</p>'
        f'<figure class="lab-figure"><a class="paper" href="./assets/figures/vs/tez_vs_jev.png"><img src="./assets/figures/vs/web/{esc(head_web)}" '
        f'width="{head_size[0]}" height="{head_size[1]}" loading="lazy" decoding="async" alt="Tez against TypeSafe Jev, '
        'with Laya on the same rows: accuracy on the public datasets, language coverage, speed, calibration and the '
        'deployment facts. Open full size."></a><figcaption>Tez against Jev\'s published figures, with Laya\'s '
        'checkpoints on the same rows. Select to open full size.</figcaption></figure>'
        + tables()
        + '<h3 id="vs-laya-gallery">Every chart, and every panel on its own</h3>'
        '<p>The composites first, then each panel cut out at a size that reads well alone. Select any image to open it '
        f'full size. The list with sources is in <a href="{BLOB}docs/figures/vs/README.md">docs/figures/vs/README.md</a>.</p>'
        '<ul class="figure-grid">' + "".join(comps) + "".join(panels) + "</ul>"
        f'<p class="source">Source: <a href="{BLOB}BENCHMARKS.md#1b-on-layas-own-benchmarks-layas-charts-redrawn-with-tez">BENCHMARKS.md §1b</a>, '
        f'<a href="{BLOB}experiments/make_vs_figures.py">experiments/make_vs_figures.py</a>.</p>'
        "</section>" + END)
    s = PAGE.read_text(encoding="utf-8")
    if START in s:
        s = s[:s.index(START)] + section + s[s.index(END) + len(END):]
    else:
        anchor = "      <!-- ============================================================== summary -->"
        i = s.index(anchor)
        j = s.index("</section>", i) + len("</section>")
        s = s[:j] + "\n\n      " + section + s[j:]
    if "./assets/css/research.css" not in s:
        s = s.replace('<link rel="stylesheet" href="./assets/css/charts.css">',
                      '<link rel="stylesheet" href="./assets/css/charts.css">\n<link rel="stylesheet" href="./assets/css/research.css">', 1)
    toc_old = '            <li><a href="#head-to-head">Head-to-head</a></li>'
    toc_new = '            <li><a href="#vs-laya">Laya\'s charts, redrawn</a></li>\n' + toc_old
    if 'href="#vs-laya"' not in s:
        s = s.replace(toc_old, toc_new, 1)
    PAGE.write_text(s, encoding="utf-8")
    print(f"published {len(made)} images; section written to {PAGE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
