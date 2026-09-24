"""Keep the documentation pages' shared chrome in sync, and their links honest.

The docs are plain HTML files in site/docs/, one page per tab. Each page carries three marked regions that this
script owns and rewrites; everything else on a page is written by hand:

  <!-- docs-tabs:start --> ... <!-- docs-tabs:end -->     the section tab strip, with the current page marked
  <!-- docs-rail:start --> ... <!-- docs-rail:end -->     "On this page", built from the article's h2 and h3 ids
  <!-- docs-pager:start --> ... <!-- docs-pager:end -->   links to the previous and next page

It also
  - points each in-docs link (#id or ./page.html#id) at the page that now holds the id, so a section can move
    between pages without breaking links, and fails on a link whose target exists nowhere;
  - checks links from the docs to the rest of the site (../page.html#id);
  - rewrites links elsewhere on the site that still use the old single page (docs.html#id);
  - writes the id-to-page map that site/docs.html uses to forward old links.

  python scripts/build_docs.py           rewrite the files
  python scripts/build_docs.py --check   change nothing; exit 1 if a file is out of date or a link is broken
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
DOCS = SITE / "docs"
REPO = "https://github.com/Jibalmi/Tez"

# (file, tab label, one line for the pager). The order is the tab order.
PAGES = [
    ("index.html", "Start", "Install, run and send a first request"),
    ("guides.html", "Guides", "Schemas, readouts, fitting, the gate, many questions"),
    ("recipes.html", "Recipes", "Tool calling, routing, guardrails, triage and voice"),
    ("hooks.html", "Hooks", "Decision logs, tracing and your own callbacks"),
    ("integrations.html", "Integrations", "Python, TypeScript, LangChain, MCP and Docker"),
    ("api.html", "API", "The /v1/systemone wire format and every endpoint"),
    ("reference.html", "Reference", "CLI, Python API, troubleshooting and FAQ"),
]

LUCIDE = ('<svg{cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
          'stroke-linejoin="round" aria-hidden="true">{paths}</svg>')
ICON_LIST = '<path d="M3 12h.01"/><path d="M3 18h.01"/><path d="M3 6h.01"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M8 6h13"/>'
ICON_CHEVRON = '<path d="m6 9 6 6 6-6"/>'
ICON_LEFT = '<path d="m12 19-7-7 7-7"/><path d="M19 12H5"/>'
ICON_RIGHT = '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>'
ICON_PENCIL = ('<path d="M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 '
               '.623.622l4.353-1.32a2 2 0 0 0 .83-.497z"/><path d="m15 5 4 4"/>')

HEADING = re.compile(r"<h([23])\b([^>]*)>(.*?)</h\1>", re.S)
ID_ATTR = re.compile(r'\bid="([^"]+)"')
ARTICLE = re.compile(r"<article\b.*?</article>", re.S)
DOC_LINK = re.compile(r'href="(\./[a-z0-9-]+\.html)?#([^"]+)"')
SITE_LINK = re.compile(r'href="\.\./([a-z0-9-]+\.html)#([^"]+)"')
OLD_DOCS_LINK = re.compile(r'href="\./docs\.html(?:#([^"]*))?"')
NEW_DOCS_LINK = re.compile(r'href="\./docs/([a-z0-9-]+\.html)#([^"]+)"')
MAP_REGION = re.compile(r"(/\* docs-map:start \*/)(.*?)(/\* docs-map:end \*/)", re.S)


def icon(paths: str, cls: str = "") -> str:
    return LUCIDE.format(cls=f' class="{cls}"' if cls else "", paths=paths)


def text_of(fragment: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


def region(text: str, name: str, body: str, where: str) -> str:
    rx = re.compile(rf"^([ \t]*)<!-- {name}:start -->.*?<!-- {name}:end -->", re.S | re.M)
    m = rx.search(text)
    if not m:
        raise SystemExit(f"{where}: missing <!-- {name}:start --> / <!-- {name}:end --> markers")
    pad = m.group(1)
    lines = "\n".join(pad + line if line else line for line in body.splitlines())
    block = f"{pad}<!-- {name}:start -->\n{lines}\n{pad}<!-- {name}:end -->"
    return text[:m.start()] + block + text[m.end():]


def headings(page_text: str) -> list[tuple[int, str, str]]:
    """(level, id, label) for every h2/h3 with an id inside the article; data-rail="skip" leaves one out."""
    art = ARTICLE.search(page_text)
    out = []
    for level, attrs, inner in HEADING.findall(art.group(0) if art else ""):
        hid = ID_ATTR.search(attrs)
        if not hid or 'data-rail="skip"' in attrs:
            continue
        label = re.search(r'data-rail-label="([^"]+)"', attrs)
        out.append((int(level), hid.group(1), html.unescape(label.group(1)) if label else text_of(inner)))
    return out


def tabs_html(current: str) -> str:
    items = []
    for file, label, _ in PAGES:
        if file == current:
            items.append(f'<li><a href="./{file}" aria-current="page">{label}<span class="tab-mark" aria-hidden="true"></span></a></li>')
        else:
            items.append(f'<li><a href="./{file}">{label}</a></li>')
    return "\n".join([
        '<nav class="docs-tabs" aria-label="Documentation sections">',
        '  <div class="container">',
        '    <ul class="docs-tabs-list" role="list">',
        *("      " + i for i in items),
        "    </ul>",
        "  </div>",
        "</nav>",
    ])


def rail_html(file: str, heads: list[tuple[int, str, str]]) -> str:
    first = html.escape(heads[0][2]) if heads else "On this page"
    rows = []
    for level, hid, label in heads:
        cls = ' class="toc-sub"' if level == 3 else ""
        rows.append(f'<li{cls}><a href="#{hid}">{html.escape(label)}</a></li>')
    return "\n".join([
        '<nav class="doc-nav" data-toc aria-label="On this page">',
        f'  <button class="toc-toggle" type="button" data-toc-toggle aria-expanded="false" aria-controls="toc-panel">'
        f'{icon(ICON_LIST)}<span class="toc-current">{first}</span>{icon(ICON_CHEVRON, "toc-chevron")}</button>',
        '  <div class="toc-panel" id="toc-panel">',
        '    <p class="toc-heading">On this page</p>',
        '    <ul class="toc-list">',
        *("      " + r for r in rows),
        "    </ul>",
        f'    <a class="rail-edit" href="{REPO}/edit/main/site/docs/{file}">{icon(ICON_PENCIL)}<span>Edit this page</span></a>',
        "  </div>",
        "</nav>",
    ])


def pager_html(file: str) -> str:
    i = [p[0] for p in PAGES].index(file)
    parts = ['<nav class="pager" aria-label="Previous and next page">']
    if i > 0:
        f, label, desc = PAGES[i - 1]
        parts.append(f'  <a class="pager-link pager-prev" href="./{f}" rel="prev" aria-label="Previous: {label}">'
                     f'<span class="pager-title">{icon(ICON_LEFT)}{label}</span><span class="pager-desc">{desc}</span></a>')
    if i < len(PAGES) - 1:
        f, label, desc = PAGES[i + 1]
        parts.append(f'  <a class="pager-link pager-next" href="./{f}" rel="next" aria-label="Next: {label}">'
                     f'<span class="pager-title">{label}{icon(ICON_RIGHT)}</span><span class="pager-desc">{desc}</span></a>')
    parts.append("</nav>")
    return "\n".join(parts)


def ids_in(text: str) -> set[str]:
    return set(ID_ATTR.findall(text))


def build() -> tuple[dict[Path, str], list[str], dict[str, str]]:
    """Return (new file contents, errors, id -> docs page) without touching the disk."""
    errors: list[str] = []
    out: dict[Path, str] = {}

    # 1. chrome, from each page's own headings
    pages: dict[str, str] = {}
    for file, _, _ in PAGES:
        path = DOCS / file
        if not path.exists():
            errors.append(f"site/docs/{file} is missing")
            continue
        text = path.read_text(encoding="utf-8")
        heads = headings(text)
        if not heads:
            errors.append(f"site/docs/{file}: no h2 or h3 with an id in the article")
        text = region(text, "docs-tabs", tabs_html(file), file)
        text = region(text, "docs-rail", rail_html(file, heads), file)
        text = region(text, "docs-pager", pager_html(file), file)
        pages[file] = text

    # 2. which page holds which id (article ids only; they must be unique across the docs)
    owner: dict[str, str] = {}
    for file, text in pages.items():
        art = ARTICLE.search(text)
        for hid in ids_in(art.group(0) if art else ""):
            if hid in owner and owner[hid] != file:
                errors.append(f"id #{hid} is defined on both {owner[hid]} and {file}")
            owner[hid] = file
    all_ids = {file: ids_in(text) for file, text in pages.items()}

    site_ids: dict[str, set[str]] = {}

    def ids_of_site_page(name: str) -> set[str]:
        if name not in site_ids:
            p = SITE / name
            site_ids[name] = ids_in(p.read_text(encoding="utf-8")) if p.exists() else set()
        return site_ids[name]

    # 3. in-docs links follow their target; links out to the site must resolve
    docs_files = {f for f, _, _ in PAGES}
    for file, text in pages.items():
        def fix(m: re.Match, file: str = file) -> str:
            page, hid = m.group(1), m.group(2)
            target = page[2:] if page else file
            if target not in docs_files:
                # ./benchmarks.html#x from site/docs/ points into the docs folder: the site page is ../benchmarks.html
                errors.append(f"site/docs/{file}: link {page}#{hid} is not a docs page (use ../{target})")
                return m.group(0)
            if hid in all_ids.get(target, set()):
                return f'href="#{hid}"' if target == file else f'href="./{target}#{hid}"'
            if hid in owner:
                return f'href="#{hid}"' if owner[hid] == file else f'href="./{owner[hid]}#{hid}"'
            errors.append(f"site/docs/{file}: link to #{hid} ({page or 'same page'}) has no target")
            return m.group(0)

        text = DOC_LINK.sub(fix, text)
        for name, hid in SITE_LINK.findall(text):
            if name != "docs.html" and hid not in ids_of_site_page(name):
                errors.append(f"site/docs/{file}: link to ../{name}#{hid} has no target")
        out[DOCS / file] = text

    # 4. the rest of the site: old single-page links move to the page that holds the id
    for path in sorted(SITE.glob("*.html")):
        if path.name == "docs.html":
            continue
        text = path.read_text(encoding="utf-8")

        def fix_old(m: re.Match, path: Path = path) -> str:
            hid = m.group(1)
            if not hid:
                return 'href="./docs/index.html"'
            if hid in owner:
                return f'href="./docs/{owner[hid]}#{hid}"'
            errors.append(f"site/{path.name}: link to docs.html#{hid} has no target")
            return m.group(0)

        def fix_new(m: re.Match, path: Path = path) -> str:
            page, hid = m.group(1), m.group(2)
            if hid in all_ids.get(page, set()):
                return m.group(0)
            if hid in owner:
                return f'href="./docs/{owner[hid]}#{hid}"'
            errors.append(f"site/{path.name}: link to docs/{page}#{hid} has no target")
            return m.group(0)

        new = NEW_DOCS_LINK.sub(fix_new, OLD_DOCS_LINK.sub(fix_old, text))
        if new != text:
            out[path] = new

    # 5. the forwarding map on the old single page
    redirect = SITE / "docs.html"
    if redirect.exists():
        text = redirect.read_text(encoding="utf-8")
        body = json.dumps(dict(sorted(owner.items())), separators=(",", ":"))
        if MAP_REGION.search(text):
            out[redirect] = MAP_REGION.sub(lambda m: m.group(1) + body + m.group(3), text, count=1)
        else:
            errors.append("site/docs.html: missing /* docs-map:start */ ... /* docs-map:end */")
    return out, errors, owner


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="change nothing; exit 1 when out of date or broken")
    args = ap.parse_args()
    out, errors, owner = build()
    stale = [p for p, text in out.items() if p.read_text(encoding="utf-8") != text]
    for e in errors:
        print(f"error: {e}")
    if args.check:
        for p in stale:
            print(f"out of date: {p.relative_to(ROOT).as_posix()} (run python scripts/build_docs.py)")
        return 1 if errors or stale else 0
    for p in stale:
        p.write_text(out[p], encoding="utf-8", newline="\n")
        print(f"wrote {p.relative_to(ROOT).as_posix()}")
    print(f"{len(PAGES)} pages, {len(owner)} anchors, {len(stale)} files changed, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
