/* Reading pages: the page map with scroll-spy (a disclosure on narrow screens), the docs tab strip, heading anchor
   links, and a small highlighter for the code examples. Everything degrades to plain, working HTML without it. */
(function () {
  "use strict";

  const LINK_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>';

  /* ---------------------------------------------------------------- sidebar */
  function initToc(nav) {
    nav.classList.add("js");
    const links = Array.from(nav.querySelectorAll('a[href^="#"]'));
    const items = links
      .map((a) => {
        let el = null;
        try { el = document.getElementById(decodeURIComponent(a.getAttribute("href").slice(1))); } catch (e) { el = null; }
        return { a, el };
      })
      .filter((it) => it.el);
    if (!items.length) return;
    const toggle = nav.querySelector("[data-toc-toggle]");
    const current = nav.querySelector(".toc-current");
    let active = null;
    let ticking = false;

    function offset() {
      const pad = parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop);
      return (isFinite(pad) ? pad : 80) + 12;
    }
    function keepVisible(a) {
      if (nav.scrollHeight <= nav.clientHeight + 2) return;
      const nr = nav.getBoundingClientRect(), ar = a.getBoundingClientRect();
      if (ar.top < nr.top + 48) nav.scrollTop -= nr.top + 48 - ar.top;
      else if (ar.bottom > nr.bottom - 48) nav.scrollTop += ar.bottom - nr.bottom + 48;
    }
    function update() {
      ticking = false;
      const off = offset();
      let pick = items[0];
      for (const it of items) {
        if (it.el.getBoundingClientRect().top - off <= 1) pick = it;
        else break;
      }
      const doc = document.documentElement;
      if (window.innerHeight + window.scrollY >= doc.scrollHeight - 2) {
        // at the very bottom the last sections cannot reach the top: prefer the last one on screen
        const onScreen = items.filter((it) => it.el.getBoundingClientRect().top < window.innerHeight - 40);
        if (onScreen.length) pick = onScreen[onScreen.length - 1];
      }
      if (pick === active) return;
      if (active) active.a.removeAttribute("aria-current");
      active = pick;
      active.a.setAttribute("aria-current", "true");
      if (current) current.textContent = active.a.textContent.trim();
      keepVisible(active.a);
    }
    function schedule() {
      if (!ticking) { ticking = true; requestAnimationFrame(update); }
    }
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    window.addEventListener("hashchange", schedule);
    window.addEventListener("load", schedule);
    update();

    if (toggle) {
      const setOpen = (open) => {
        nav.setAttribute("data-open", open ? "true" : "false");
        toggle.setAttribute("aria-expanded", open ? "true" : "false");
      };
      setOpen(false);
      toggle.addEventListener("click", () => setOpen(nav.getAttribute("data-open") !== "true"));
      links.forEach((a) => a.addEventListener("click", () => setOpen(false)));
      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && nav.getAttribute("data-open") === "true") { setOpen(false); toggle.focus(); }
      });
      document.addEventListener("pointerdown", (e) => {
        if (nav.getAttribute("data-open") === "true" && !nav.contains(e.target)) setOpen(false);
      });
    }
  }

  /* ---------------------------------------------------------------- docs section tabs */
  // On narrow screens the strip scrolls sideways: bring the current tab into view and fade the side that has more.
  function initTabs(list) {
    function mark() {
      const max = list.scrollWidth - list.clientWidth;
      if (max <= 1) { list.removeAttribute("data-more"); return; }
      const atStart = list.scrollLeft <= 1;
      const atEnd = list.scrollLeft >= max - 1;
      list.setAttribute("data-more", atStart ? "end" : atEnd ? "start" : "both");
    }
    const current = list.querySelector('[aria-current="page"]');
    if (current && list.scrollWidth > list.clientWidth + 1) {
      const lr = list.getBoundingClientRect();
      const cr = current.getBoundingClientRect();
      list.scrollLeft += cr.left - lr.left - (lr.width - cr.width) / 2;
    }
    mark();
    list.addEventListener("scroll", mark, { passive: true });
    window.addEventListener("resize", mark);
  }

  /* ---------------------------------------------------------------- table labels */
  // On phones reading-page tables reflow into labelled cells (docs.css). The docs pages carry data-label from
  // build_docs.py and charts.js labels the tables it draws; this labels any other table from its header row.
  function labelTable(table) {
    const heads = Array.from(table.querySelectorAll("thead th")).map((th) => th.textContent.trim());
    if (!heads.length) return;
    table.querySelectorAll("tbody tr").forEach((tr) => {
      Array.from(tr.children).forEach((cell, i) => {
        if (!cell.hasAttribute("data-label") && heads[i]) cell.setAttribute("data-label", heads[i]);
      });
    });
  }

  /* ---------------------------------------------------------------- heading anchors */
  function initAnchors(root) {
    root.querySelectorAll("h2[id], h3[id]").forEach((h) => {
      if (h.querySelector(".heading-anchor") || h.hasAttribute("data-no-anchor")) return;
      const a = document.createElement("a");
      a.className = "heading-anchor";
      a.href = `#${h.id}`;
      a.setAttribute("aria-label", `Link to this section: ${h.textContent.trim()}`);
      a.innerHTML = LINK_ICON;
      h.appendChild(a);
    });
  }

  /* ---------------------------------------------------------------- highlighter */
  const KEYWORDS = /\b(?:import|from|as|def|return|for|in|if|elif|else|while|with|try|except|raise|class|and|or|not|is|None|True|False|lambda|yield|pass|break|continue)\b/y;
  const RULES = {
    json: [
      ["key", /"(?:[^"\\\n]|\\.)*"(?=\s*:)/y],
      ["str", /"(?:[^"\\\n]|\\.)*"/y],
      ["num", /-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/y],
      ["num", /\b(?:true|false|null)\b/y],
      ["punct", /[{}[\]:,]/y],
    ],
    bash: [
      ["prompt", /^\$ /my],
      ["com", /(?<=^|[ \t])#[^\n]*/my],
      ["str", /'[^']*'/y],
      ["str", /"(?:[^"\\]|\\.)*"/y],
      ["punct", /\\(?=\n)/y],
    ],
    python: [
      ["com", /#[^\n]*/y],
      ["str", /[rRbBfF]{0,2}(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')/y],
      ["num", KEYWORDS],
      ["num", /\b\d+(?:\.\d+)?\b/y],
    ],
    yaml: [
      ["com", /(?<=^|[ \t])#[^\n]*/my],
      ["key", /^[ \t]*(?:-[ \t]+)?[\w"'.-]+(?=:(?:[ \t]|$))/my],
      ["str", /"(?:[^"\\\n]|\\.)*"|'[^'\n]*'/y],
      ["num", /\b\d+(?:\.\d+)?\b/y],
      ["punct", /[[\]{},]/y],
    ],
    powershell: [
      ["com", /(?<=^|[ \t])#[^\n]*/my],
      ["str", /@'[\s\S]*?'@|@"[\s\S]*?"@|'[^'\n]*'|"(?:[^"`\n]|`.)*"/y],
      ["num", /\$[\w:]+/y],
    ],
    ts: [
      ["com", /\/\/[^\n]*|\/\*[\s\S]*?\*\//y],
      ["str", /`(?:[^`\\]|\\.)*`|"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'/y],
      ["num", /\b(?:import|from|export|const|let|var|function|return|await|async|new|if|else|for|of|in|type|interface|as|true|false|null|undefined)\b/y],
      ["num", /\b\d+(?:\.\d+)?\b/y],
    ],
  };

  function tokenize(lang, src) {
    const rules = RULES[lang];
    if (!rules) return [[null, src]];
    const out = [];
    let plain = "";
    let i = 0;
    while (i < src.length) {
      let hit = null;
      for (const [cls, re] of rules) {
        re.lastIndex = i;
        const m = re.exec(src);
        if (m && m.index === i && m[0].length) { hit = [cls, m[0]]; break; }
      }
      if (hit) {
        if (plain) { out.push([null, plain]); plain = ""; }
        out.push(hit);
        i += hit[1].length;
      } else {
        plain += src[i];
        i += 1;
      }
    }
    if (plain) out.push([null, plain]);
    return out;
  }

  function highlight(code) {
    const lang = code.getAttribute("data-lang");
    if (!RULES[lang] || code.dataset.hl) return;
    const src = code.textContent;
    const frag = document.createDocumentFragment();
    tokenize(lang, src).forEach(([cls, str]) => {
      if (!cls) { frag.appendChild(document.createTextNode(str)); return; }
      const span = document.createElement("span");
      span.className = `tok-${cls}`;
      span.textContent = str;
      frag.appendChild(span);
    });
    code.textContent = "";
    code.appendChild(frag);
    code.dataset.hl = "1";
  }

  function init() {
    document.querySelectorAll(".docs-tabs-list").forEach(initTabs);
    document.querySelectorAll(".doc-main .table-wrap table").forEach(labelTable);
    document.querySelectorAll("[data-toc]").forEach(initToc);
    document.querySelectorAll(".prose, [data-anchors]").forEach(initAnchors);
    document.querySelectorAll("pre > code[data-lang]").forEach(highlight);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
