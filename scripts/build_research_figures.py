"""Publish the one-chart-per-finding panels on the research page.

Copies docs/figures/panels/*.png into site/assets/figures/panels/ as full-size PNGs plus light WebP previews, and the
two grids (docs/figures/tez_dashboard.png, tez_lab.png) into site/assets/figures/; records each image's origin next to
it (a tEXt chunk in a PNG, a .json sidecar beside a WebP), as scripts/build_vs_laya_section.py does; and writes two
galleries into site/research.html between their markers: the probe lab's charts in the probe-lab section
(<!-- lab-panels:start/end -->) and every other chart in the figures section (<!-- result-panels:start/end -->).
Names, findings and groups come from docs/figures/panels/README.md, which the figure generators write.

  py scripts/build_research_figures.py
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "figures" / "panels"
DST = ROOT / "site" / "assets" / "figures" / "panels"
GRIDS = ("tez_dashboard.png", "tez_lab.png")
PAGE = ROOT / "site" / "research.html"
EMBED = Path.home() / ".claude" / "skills" / "impeccable" / "scripts" / "embed-prompt.mjs"
BLOB = "https://github.com/Jibalmi/Tez/blob/main/"
PREVIEW_W = 1000                       # a card is at most ~500 px wide; previews are 2x that
NBSP = chr(0xA0)
ORIGIN = ("Origin: data chart, not an AI-generated image. Rendered with matplotlib by experiments/{script} from {data} "
          "(github.com/Jibalmi/Tez); {what}.")


def origin(name, what):
    lab = name.startswith("lab_") or name == "tez_lab.png"
    script = "make_lab_figure.py" if lab else "make_dashboard.py"
    data = "the probe-lab results in results/" if lab else "BENCHMARKS.md and results/"
    return ORIGIN.format(script=script, data=data, what=what)


def esc(s):
    return html.escape(str(s), quote=True)


def md_text(s):
    """A cell of the index as plain text: backticks and link syntax dropped."""
    return re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s).replace("`", "").strip()


def read_index():
    """[(group, part, file, name, finding)] in the index's order; part is 'Results' or 'The probe lab'."""
    rows, part, group = [], None, None
    for line in (SRC / "README.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            part = line[3:].strip()
        elif line.startswith("### "):
            group = line[4:].strip()
        else:
            m = re.match(r"\| \[(.+?)\]\(([\w.-]+\.png)\) \| (.*?) \| (.*?) \|$", line)
            if m:
                rows.append((group, part, m.group(2), m.group(1), md_text(m.group(3))))
    if not rows:
        raise SystemExit(f"no charts listed in {SRC / 'README.md'}; run the figure generators first")
    return rows


def embed(path, prompt):
    if not EMBED.exists():
        raise SystemExit(f"{EMBED} not found: it records each image's origin (see scripts/build_vs_laya_section.py)")
    subprocess.run(["node", str(EMBED), str(path), "--prompt", prompt], check=True, capture_output=True)


def publish(names):
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "web").mkdir(exist_ok=True)
    keep = set()
    made = {}
    for name in names:
        src = SRC / name
        full = DST / name
        shutil.copyfile(src, full)
        embed(full, origin(name, f"copied from docs/figures/panels/{name}"))
        im = Image.open(src).convert("RGB")
        w, h = im.size
        scale = min(1.0, PREVIEW_W / w)
        prev = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        web = DST / "web" / name.replace(".png", ".webp")
        prev.save(web, "WEBP", quality=82, method=6)
        embed(web, origin(name, f"web preview of docs/figures/panels/{name}"))
        made[name] = ((w, h), web.name, prev.size)
        keep |= {full.name, web.name, web.name + ".json"}
    for old in list(DST.glob("*.png")) + list((DST / "web").iterdir()):
        if old.name not in keep:                   # a chart the generators no longer draw
            old.unlink()
    for grid in GRIDS:
        dst = ROOT / "site" / "assets" / "figures" / grid
        shutil.copyfile(ROOT / "docs" / "figures" / grid, dst)
        embed(dst, origin(grid, f"copied from docs/figures/{grid}"))
    return made


def card(name, title, finding, group, made):
    (w, h), web, (pw, ph) = made[name]
    return ("          <li>\n"
            '            <figure class="fig-card">\n'
            f'              <a class="paper" href="./assets/figures/panels/{esc(name)}"><img src="./assets/figures/panels/web/'
            f'{esc(web)}" width="{pw}" height="{ph}" loading="lazy" decoding="async" alt="{esc(finding)}. Open full size.">'
            "</a>\n"
            f'              <figcaption><span class="fig-name">{esc(title)}</span><span class="fig-desc">{esc(finding)}</span>'
            f'<span class="fig-meta">{esc(group)} · {esc(name)} · {w}{NBSP}×{NBSP}{h}</span></figcaption>\n'
            "            </figure>\n"
            "          </li>\n")


def grid_size(name):
    return Image.open(ROOT / "docs" / "figures" / name).size


def lab_block(rows, made):
    w, h = grid_size("tez_lab.png")
    cards = "".join(card(f, n, t, g, made) for g, part, f, n, t in rows if part == "The probe lab")
    return ("\n"
            "        <p>Each experiment as a chart of its own, titled with what it found; the numbers behind every one are in "
            '<a href="https://github.com/Jibalmi/Tez/blob/main/BENCHMARKS.md#4c-probe-lab-experiments-on-cached-hidden-states-'
            'experimentsprobe_labpy">BENCHMARKS.md §4c</a>. Select a chart to open it full size. The same charts on one page: '
            f'<a href="./assets/figures/tez_lab.png">tez_lab.png</a> ({w}{NBSP}×{NBSP}{h}).</p>\n'
            '        <ul class="figure-grid">\n' + cards + "        </ul>\n        ")


def result_block(rows, made):
    w, h = grid_size("tez_dashboard.png")
    cards = "".join(card(f, n, t, g, made) for g, part, f, n, t in rows if part == "Results")
    return ("\n"
            '        <h3 id="figures-results">Every measurement, one chart per finding</h3>\n'
            "        <p>Accuracy against Laya and Jev, languages, calibration, speed, voice, SemIf's benchmark and the probes, "
            "each chart titled with its finding and sourced in its footer; the list with sources is in "
            f'<a href="{BLOB}docs/figures/panels/README.md">docs/figures/panels/README.md</a>. The same charts on one page: '
            f'<a href="./assets/figures/tez_dashboard.png">tez_dashboard.png</a> ({w}{NBSP}×{NBSP}{h}).</p>\n'
            '        <ul class="figure-grid">\n' + cards + "        </ul>\n        ")


def fill(text, name, body):
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    if start not in text or end not in text:
        raise SystemExit(f"{PAGE.relative_to(ROOT)}: markers {start} ... {end} not found")
    i, j = text.index(start) + len(start), text.index(end)
    return text[:i] + body + text[j:]


def main():
    rows = read_index()
    made = publish([r[2] for r in rows])
    page = PAGE.read_text(encoding="utf-8")
    page = fill(page, "lab-panels", lab_block(rows, made))
    page = fill(page, "result-panels", result_block(rows, made))
    PAGE.write_text(page, encoding="utf-8")
    print(f"published {len(made)} charts and {len(GRIDS)} grids; galleries written to {PAGE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
