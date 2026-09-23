/* Tez charts: small, dependency-free SVG charts for the benchmark report.
 *
 *   TezCharts.rows(el, spec)     categories as rows on a numeric x axis: bars, dodged dots, ranges,
 *                                confidence intervals and dumbbells (two values joined by a segment)
 *   TezCharts.xy(el, spec)       numeric x and y: lines with markers, connected scatter, reference lines, bands
 *   TezCharts.panels(el, spec)   small multiples of either, each panel its own chart on a shared scale
 *   TezCharts.table(el, spec)    the accessible data table that carries the same numbers
 *   TezCharts.legend(el, items)  HTML legend drawn with the marks' own shapes
 *
 * Colours come from CSS classes (charts.css), so a theme change restyles every chart without a redraw.
 * Charts re-layout when their container changes width. Every point is reachable by keyboard: Tab enters a
 * chart, the arrow keys move between points, and the tooltip shows on hover, focus and tap. Text that comes
 * from data is always inserted with textContent.
 */
(function () {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  let uid = 0;

  /* ------------------------------------------------------------------ helpers */

  function svg(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    if (attrs) {
      for (const k in attrs) {
        if (attrs[k] !== undefined && attrs[k] !== null && attrs[k] !== false) n.setAttribute(k, String(attrs[k]));
      }
    }
    if (parent) parent.appendChild(n);
    return n;
  }

  function text(parent, x, y, str, cls, attrs) {
    const t = svg("text", Object.assign({ x: round(x), y: round(y), class: cls }, attrs || {}), parent);
    t.textContent = str;
    return t;
  }

  function round(v) { return Math.round(v * 10) / 10; }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function isNum(v) { return typeof v === "number" && isFinite(v); }

  function linear(d0, d1, r0, r1) {
    const f = (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
    f.d = [d0, d1];
    return f;
  }
  function logScale(d0, d1, r0, r1) {
    const l0 = Math.log(d0), l1 = Math.log(d1);
    const f = (v) => r0 + ((Math.log(v) - l0) / (l1 - l0)) * (r1 - r0);
    f.d = [d0, d1];
    return f;
  }

  const fmt = {
    f0: (v) => (isNum(v) ? String(Math.round(v)) : "—"),
    f1: (v) => (isNum(v) ? v.toFixed(1) : "—"),
    f2: (v) => (isNum(v) ? v.toFixed(2) : "—"),
    f3: (v) => (isNum(v) ? v.toFixed(3) : "—"),
    pct0: (v) => (isNum(v) ? Math.round(v * 100) + " %" : "—"),
    pct1: (v) => (isNum(v) ? (v * 100).toFixed(1) + " %" : "—"),
    ms: (v) => (isNum(v) ? Math.round(v) + " ms" : "—"),
    int: (v) => (isNum(v) ? Math.round(v).toLocaleString("en-US") : "—"),
    tick: (v) => (isNum(v) ? String(Number(v.toFixed(3))) : ""),
  };

  /* text measurement with the chart's own CSS (so widths match the real font once it has loaded) */
  let measurer = null;
  function textWidth(str, cls) {
    if (!measurer) {
      measurer = svg("svg", { width: 1, height: 1, "aria-hidden": "true", focusable: "false" });
      measurer.setAttribute("style", "position:absolute;left:-9999px;top:0;visibility:hidden;font-family:var(--font-sans)");
      document.body.appendChild(measurer);
    }
    const t = text(measurer, 0, 0, str, cls);
    let w = 0;
    try { w = t.getComputedTextLength(); } catch (e) { w = 0; }
    if (!w) w = String(str).length * 6.6;
    measurer.removeChild(t);
    return w;
  }

  /* shorten a label with an ellipsis when it cannot fit (the full text stays in the table and tooltip) */
  function fit(str, cls, maxW) {
    if (!str || maxW <= 0 || textWidth(str, cls) <= maxW) return str;
    let s = String(str);
    while (s.length > 1 && textWidth(`${s}…`, cls) > maxW) s = s.slice(0, -1);
    return `${s.trimEnd()}…`;
  }

  /* marker shapes, centred on (x, y), r is the nominal radius */
  function shapePath(shape, x, y, r) {
    switch (shape) {
      case "square": {
        const s = r * 0.9;
        return `M${round(x - s)},${round(y - s)}h${round(2 * s)}v${round(2 * s)}h${round(-2 * s)}Z`;
      }
      case "diamond": {
        const d = r * 1.3;
        return `M${round(x)},${round(y - d)}L${round(x + d)},${round(y)}L${round(x)},${round(y + d)}L${round(x - d)},${round(y)}Z`;
      }
      case "triangle": {
        const t = r * 1.25;
        return `M${round(x)},${round(y - t)}L${round(x + t)},${round(y + t * 0.8)}L${round(x - t)},${round(y + t * 0.8)}Z`;
      }
      case "triangle-down": {
        const t = r * 1.25;
        return `M${round(x)},${round(y + t)}L${round(x + t)},${round(y - t * 0.8)}L${round(x - t)},${round(y - t * 0.8)}Z`;
      }
      default:
        return `M${round(x - r)},${round(y)}a${r},${r} 0 1,0 ${2 * r},0a${r},${r} 0 1,0 ${-2 * r},0Z`;
    }
  }

  /* a bar that is square at the baseline and rounded (4px) at the data end */
  function barPath(x0, x1, y, h, r) {
    const w = Math.max(0, x1 - x0);
    r = Math.min(r, h / 2, w / 2);
    if (w <= 0) return "";
    return `M${round(x0)},${round(y)}H${round(x1 - r)}A${r},${r} 0 0 1 ${round(x1)},${round(y + r)}V${round(y + h - r)}A${r},${r} 0 0 1 ${round(x1 - r)},${round(y + h)}H${round(x0)}Z`;
  }

  function roleClass(role) { return role === "tez" ? "m-tez" : role === "pub" ? "m-pub" : "m-other"; }
  function lineClass(role) { return role === "tez" ? "l-tez" : role === "pub" ? "l-pub" : "l-other"; }

  /* a small swatch (legend and tooltip keys) */
  function swatch(item, w, h) {
    w = w || 16; h = h || 14;
    const s = svg("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h, "aria-hidden": "true", focusable: "false" });
    const cx = w / 2, cy = h / 2;
    const role = item.role || "other";
    if (item.kind === "bar" || item.kind === "range") {
      if (role === "pub") svg("rect", { x: 1.5, y: cy - 3.5, width: w - 3, height: 7, rx: 2, class: "m-bar m-pub" }, s);
      else svg("rect", { x: 1, y: cy - 4, width: w - 2, height: 8, rx: 2, class: roleClass(role) }, s);
    } else if (item.kind === "line") {
      svg("line", { x1: 0, y1: cy, x2: w, y2: cy, class: `m-line ${lineClass(role)}` }, s);
      if (item.shape) svg("path", { d: shapePath(item.shape, cx, cy, 3), class: `m-dot ${roleClass(role)}` }, s);
    } else if (item.kind === "ref") {
      svg("line", { x1: 0, y1: cy, x2: w, y2: cy, class: "c-ref" }, s);
    } else if (item.kind === "band") {
      svg("rect", { x: 0, y: 1, width: w, height: h - 2, rx: 2, class: "c-band" }, s);
    } else {
      svg("path", { d: shapePath(item.shape || "circle", cx, cy, 4), class: `m-dot ${roleClass(role)}` }, s);
    }
    return s;
  }

  function legend(ul, items) {
    ul.textContent = "";
    items.forEach((it) => {
      const li = document.createElement("li");
      li.appendChild(swatch(it));
      const span = document.createElement("span");
      span.textContent = it.label;
      li.appendChild(span);
      ul.appendChild(li);
    });
  }

  /* ------------------------------------------------------------------ mounting, tooltip, keyboard */

  const mounted = [];

  function mount(container, draw) {
    container.textContent = "";
    container.setAttribute("data-state", "ready");
    const tip = document.createElement("div");
    tip.className = "chart-tip";
    tip.setAttribute("aria-hidden", "true");
    container.appendChild(tip);
    const state = { container, tip, points: [], active: -1, width: 0, svg: null, cross: null };

    function render(force) {
      const w = Math.floor(container.clientWidth);
      if (!w || (!force && w === state.width)) return;
      state.width = w;
      const focusIndex = state.svg && state.svg.contains(document.activeElement) ? state.active : -1;
      if (state.svg) state.svg.remove();
      hideTip(state);
      state.points = [];
      state.active = -1;
      state.cross = null;
      state.svg = draw(w, state);
      container.insertBefore(state.svg, tip);
      wire(state);
      if (focusIndex >= 0 && state.points[focusIndex]) state.points[focusIndex].node.focus();
    }
    state.render = render;
    render(true);
    if ("ResizeObserver" in window) {
      let raf = 0;
      const ro = new ResizeObserver(() => {
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(() => render(false));
      });
      ro.observe(container);
    } else {
      window.addEventListener("resize", () => render(false));
    }
    mounted.push(state);
    return state;
  }

  /* redraw once the web fonts are in, so measured label widths use Geist, not the fallback */
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(() => mounted.forEach((s) => s.render(true)));
  }

  function addPoint(state, node, info) {
    node.setAttribute("class", "pt");
    node.setAttribute("tabindex", "-1");
    node.setAttribute("role", "img");
    node.setAttribute("aria-label", info.aria);
    node.dataset.i = String(state.points.length);
    state.points.push(Object.assign({ node }, info));
  }

  function wire(state) {
    const pts = state.points;
    if (!pts.length) return;
    pts[0].node.setAttribute("tabindex", "0");
    const root = state.svg;

    function activate(i, how) {
      if (i < 0 || i >= pts.length) return;
      if (state.active >= 0 && pts[state.active]) {
        pts[state.active].node.removeAttribute("data-active");
        pts[state.active].node.setAttribute("tabindex", "-1");
      }
      state.active = i;
      const p = pts[i];
      p.node.setAttribute("tabindex", "0");
      if (how !== "focus") p.node.setAttribute("data-active", "true");
      showTip(state, p);
    }

    root.addEventListener("keydown", (e) => {
      const node = e.target.closest ? e.target.closest(".pt") : null;
      if (!node) return;
      const i = Number(node.dataset.i);
      let j = null;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") j = Math.min(pts.length - 1, i + 1);
      else if (e.key === "ArrowLeft" || e.key === "ArrowUp") j = Math.max(0, i - 1);
      else if (e.key === "Home") j = 0;
      else if (e.key === "End") j = pts.length - 1;
      else if (e.key === "Escape") { hideTip(state); return; }
      if (j === null) return;
      e.preventDefault();
      pts[j].node.focus();
    });
    pts.forEach((p, i) => {
      p.node.addEventListener("focus", () => activate(i, "focus"));
      p.node.addEventListener("blur", () => {
        if (state.active === i) hideTip(state);
      });
      if (!state.hoverLayer) {
        p.node.addEventListener("pointerenter", (e) => { if (e.pointerType !== "touch") activate(i, "hover"); });
        p.node.addEventListener("pointerleave", (e) => { if (e.pointerType !== "touch" && document.activeElement !== p.node) hideTip(state); });
        p.node.addEventListener("click", (e) => { e.stopPropagation(); activate(i, "tap"); });
      }
    });

    if (state.hoverLayer) {
      const layer = state.hoverLayer;
      const pick = (e) => {
        const box = root.getBoundingClientRect();
        const x = e.clientX - box.left, y = e.clientY - box.top;
        let best = -1, bestD = Infinity;
        pts.forEach((p, k) => {
          const dx = p.anchor.x - x, dy = p.anchor.y - y;
          const d = dx * dx + dy * dy * 0.35;
          if (d < bestD) { bestD = d; best = k; }
        });
        return best;
      };
      layer.addEventListener("pointermove", (e) => { const k = pick(e); if (k >= 0) activate(k, "hover"); });
      layer.addEventListener("pointerleave", () => { if (!root.contains(document.activeElement)) hideTip(state); });
      layer.addEventListener("click", (e) => { e.stopPropagation(); const k = pick(e); if (k >= 0) activate(k, "tap"); });
    }
  }

  document.addEventListener("pointerdown", (e) => {
    mounted.forEach((s) => {
      if (s.tip.getAttribute("data-on") === "true" && !s.container.contains(e.target)) hideTip(s);
    });
  });

  function hideTip(state) {
    state.tip.removeAttribute("data-on");
    if (state.cross) state.cross.removeAttribute("data-on");
    if (state.active >= 0 && state.points[state.active]) state.points[state.active].node.removeAttribute("data-active");
  }

  function showTip(state, p) {
    const tip = state.tip;
    tip.textContent = "";
    if (p.tip.title) {
      const t = document.createElement("p");
      t.className = "tip-title";
      t.textContent = p.tip.title;
      tip.appendChild(t);
    }
    (p.tip.rows || []).forEach((r) => {
      const row = document.createElement("div");
      row.className = "tip-row";
      row.appendChild(swatch({ kind: r.kind === "bar" || r.kind === "range" ? "bar" : r.kind === "line" ? "line" : "dot", role: r.role, shape: r.shape }, 14, 10));
      const v = document.createElement("span");
      v.className = "tip-value";
      v.textContent = r.value;
      const n = document.createElement("span");
      n.className = "tip-name";
      n.textContent = r.name;
      row.appendChild(v);
      row.appendChild(n);
      tip.appendChild(row);
    });
    if (p.tip.note) {
      const n = document.createElement("p");
      n.className = "tip-note";
      n.textContent = p.tip.note;
      tip.appendChild(n);
    }
    if (state.cross && isNum(p.crossX)) {
      state.cross.setAttribute("x1", p.crossX);
      state.cross.setAttribute("x2", p.crossX);
      state.cross.setAttribute("data-on", "true");
    }
    tip.setAttribute("data-on", "true");
    const cw = state.container.clientWidth;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let x = p.anchor.x + 14;
    if (x + tw > cw) x = p.anchor.x - tw - 14;
    x = clamp(x, 0, Math.max(0, cw - tw));
    let y = p.anchor.y - th - 10;
    if (y < 0) y = p.anchor.y + 14;
    tip.style.transform = "";
    tip.style.left = `${Math.round(x)}px`;
    tip.style.top = `${Math.round(y)}px`;
  }

  /* ------------------------------------------------------------------ axis helpers */

  /* a tick label centred on its tick, pulled inside the chart at either edge */
  function tickLabel(g, px, y, label, width) {
    const w = textWidth(label, "c-tick");
    let anchor = "middle", tx = px;
    if (px - w / 2 < 0) { anchor = "start"; tx = Math.max(0, px - 3); }
    else if (width && px + w / 2 > width) { anchor = "end"; tx = Math.min(width, px + 3); }
    text(g, tx, y, label, "c-tick", { "text-anchor": anchor });
  }

  function xAxis(g, x, spec, top, bottom, opts) {
    const ticks = spec.ticks || [];
    const tf = spec.tickFmt || fmt.tick;
    ticks.forEach((t) => {
      const px = round(x(t));
      svg("line", { x1: px, x2: px, y1: top, y2: bottom, class: "c-grid" }, g);
      tickLabel(g, px, bottom + 15, tf(t), opts && opts.width);
    });
    if (spec.label && !(opts && opts.noLabel)) {
      const mid = (x(x.d[0]) + x(x.d[1])) / 2;
      const room = Math.min(Math.abs(x(x.d[1]) - x(x.d[0])) + 40, 2 * Math.min(mid, (opts && opts.width ? opts.width : Infinity) - mid));
      text(g, mid, bottom + 32, fit(spec.label, "c-axis-label", room), "c-axis-label", { "text-anchor": "middle" });
    }
  }

  /* ------------------------------------------------------------------ rows chart */

  function valueOf(raw) {
    if (raw === null || raw === undefined) return null;
    if (Array.isArray(raw)) return { lo: raw[0], hi: raw[1], v: null };
    if (typeof raw === "object") return raw;
    return { v: raw };
  }

  function rows(container, spec) {
    const id = `c${++uid}`;
    return mount(container, (W, state) => {
      const rowsData = spec.rows;
      const series = spec.series;
      const rowH = spec.rowHeight || 30;
      const barH = spec.barHeight || 12;
      const showTips = spec.labels === "tips";
      const vcols = (spec.valueColumns || []).filter((c, i) => W >= 460 || i === 0);
      const colW = 60;
      const hasSub = rowsData.some((r) => r.sub && r.sub.trim());
      const labelMax = Math.max(...rowsData.map((r) => Math.max(textWidth(r.label, "c-row-label"), r.sub ? textWidth(r.sub, "c-row-sub") : 0)));
      let labelW = spec.labelWidth || clamp(Math.ceil(labelMax) + 14, 36, Math.floor(W * (spec.labelShare || 0.4)));
      // narrow screens: when the labels do not fit beside the marks, put each label above its row
      const stacked = !spec.labelWidth && Math.ceil(labelMax) + 14 > labelW;
      if (stacked) labelW = 0;
      const labelBand = stacked ? (hasSub ? 32 : 18) : 0;
      const tipW = showTips ? Math.ceil(textWidth(spec.tipSample || "0.000", "c-value")) + 12 : (spec.ends ? Math.ceil(textWidth(spec.tipSample || "0.000", "c-value")) + 10 : 10);
      const padTop = vcols.length || (spec.ref && spec.ref.length) ? 24 : 4;
      const dividers = rowsData.filter((r) => r.divider).length;
      const rowsH = rowsData.length * (rowH + labelBand) + dividers * 10;
      const axisH = spec.x.label ? 42 : 24;
      const H = padTop + rowsH + axisH;
      const x0 = labelW, x1 = W - vcols.length * colW - tipW;
      const x = linear(spec.x.min, spec.x.max, x0, x1);

      const root = svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": spec.ariaLabel || "", id });
      const furniture = svg("g", { "aria-hidden": "true" }, root);
      xAxis(furniture, x, spec.x, padTop - 2, padTop + rowsH, { width: W });
      if (spec.baseline !== false && series.some((s) => s.kind === "bar")) {
        svg("line", { x1: round(x0), x2: round(x0), y1: padTop - 2, y2: padTop + rowsH, class: "c-baseline" }, furniture);
      }
      (spec.ref || []).forEach((r) => {
        const px = round(x(r.value));
        svg("line", { x1: px, x2: px, y1: padTop - 6, y2: padTop + rowsH, class: "c-ref" }, furniture);
        text(furniture, px + (r.anchor === "end" ? -4 : 4), padTop - 10, r.label, "c-ref-label", { "text-anchor": r.anchor === "end" ? "end" : "start" });
      });
      vcols.forEach((c, k) => {
        const cx = W - (vcols.length - 1 - k) * colW - 4;
        text(furniture, cx, padTop - 10, c.header, "c-col-head", { "text-anchor": "end" });
      });

      // strongest value label (for bar charts that bold the best result)
      let maxV = -Infinity;
      if (spec.boldMax) {
        rowsData.forEach((r) => series.forEach((s) => { const v = valueOf(s.values[r.key]); if (v && isNum(v.v)) maxV = Math.max(maxV, v.v); }));
      }

      const laneSeries = series.filter((s) => s.kind === "dot" || s.kind === "ci");
      const laneGap = spec.laneGap || 6;
      const laneOf = (s) => {
        if (!spec.dodge) return 0;
        const i = laneSeries.indexOf(s);
        return i < 0 ? 0 : (i - (laneSeries.length - 1) / 2) * laneGap;
      };

      const marks = svg("g", {}, root);
      let yCursor = padTop;
      rowsData.forEach((r) => {
        if (r.divider) {
          svg("line", { x1: 0, x2: W, y1: round(yCursor + 5), y2: round(yCursor + 5), class: "c-divider", "aria-hidden": "true" }, furniture);
          yCursor += 10;
        }
        const cy = yCursor + labelBand + rowH / 2;
        // row label: beside the row, or above it on narrow screens
        if (stacked) {
          text(furniture, 0, yCursor + 13, fit(r.label, "c-row-label", W - 4), "c-row-label");
          if (hasSub && r.sub) text(furniture, 0, yCursor + 27, fit(r.sub, "c-row-sub", W - 4), "c-row-sub");
        } else if (r.sub && hasSub) {
          text(furniture, 0, cy - 2, r.label, "c-row-label");
          text(furniture, 0, cy + 12, r.sub, "c-row-sub");
        } else {
          text(furniture, 0, cy + 4.5, r.label, "c-row-label");
        }

        // dumbbell connector
        if (spec.connect) {
          const a = valueOf(series.find((s) => s.key === spec.connect[0]).values[r.key]);
          const b = valueOf(series.find((s) => s.key === spec.connect[1]).values[r.key]);
          if (a && b && isNum(a.v) && isNum(b.v)) {
            svg("line", { x1: round(x(a.v)), x2: round(x(b.v)), y1: round(cy), y2: round(cy), class: `m-seg ${lineClass(r.role || "other")}`, "aria-hidden": "true" }, marks);
          }
        }

        const rowValues = series.map((s) => ({ s, val: valueOf(s.values[r.key]) }));
        const nameOf = (s) => (spec.tipName === "row" ? r.name || r.label : s.label);
        const tipRows = rowValues
          .filter((o) => o.val)
          .map((o) => ({
            name: nameOf(o.s),
            value: formatVal(o.s, o.val, spec),
            role: roleFor(o.s, r),
            shape: o.s.shape,
            kind: o.s.kind === "bar" || o.s.kind === "range" ? "bar" : "dot",
          }));

        rowValues.forEach(({ s, val }) => {
          const role = roleFor(s, r);
          const off = laneOf(s);
          if (!val) {
            if (s.kind === "bar" && spec.missingText) {
              text(furniture, round(x0 + 6), cy + 4, spec.missingText, "c-value-missing");
            }
            return;
          }
          const g = svg("g", {}, marks);
          const valueText = formatVal(s, val, spec);
          const context = spec.tipName === "row" ? spec.title || "" : r.aria || r.label;
          const aria = `${nameOf(s)}${context ? `, ${context}` : ""}: ${valueText}${r.ariaNote ? `. ${r.ariaNote}` : ""}`;
          let anchor;
          if (s.kind === "bar") {
            const y = cy + off - barH / 2;
            const xa = x(spec.x.min), xb = x(clamp(val.v, spec.x.min, spec.x.max));
            svg("rect", { x: round(x0), y: round(cy - rowH / 2), width: round(x1 - x0 + tipW), height: rowH, class: "pt-hit" }, g);
            svg("rect", { x: round(xa - 3), y: round(y - 3), width: round(Math.max(6, xb - xa + 6)), height: barH + 6, rx: 5, class: "pt-focus" }, g);
            if (role === "pub") {
              const d = barPath(xa + 0.75, xb - 0.75, y + 0.75, barH - 1.5, 3.25);
              if (d) svg("path", { d, class: "m-bar m-pub" }, g);
            } else {
              const d = barPath(xa, xb, y, barH, 4);
              if (d) svg("path", { d, class: `m-bar ${roleClass(role)}` }, g);
            }
            if (showTips) {
              const strong = spec.boldMax && val.v === maxV;
              text(g, round(xb + 6), cy + off + 4, valueText, strong ? "c-value-strong" : "c-value", { "aria-hidden": "true" });
            }
            anchor = { x: xb, y: y };
          } else if (s.kind === "range") {
            const y = cy + off - barH / 2;
            const xa = x(val.lo), xb = x(val.hi);
            svg("rect", { x: round(x0), y: round(cy - rowH / 2), width: round(x1 - x0 + tipW), height: rowH, class: "pt-hit" }, g);
            svg("rect", { x: round(xa - 3), y: round(y - 3), width: round(Math.max(6, xb - xa + 6)), height: barH + 6, rx: barH / 2 + 3, class: "pt-focus" }, g);
            if (role === "pub") {
              svg("rect", { x: round(xa + 0.75), y: round(y + 0.75), width: round(Math.max(1, xb - xa - 1.5)), height: barH - 1.5, rx: (barH - 1.5) / 2, class: "m-bar m-pub" }, g);
            } else {
              svg("rect", { x: round(xa), y: round(y), width: round(Math.max(2, xb - xa)), height: barH, rx: barH / 2, class: `m-bar ${roleClass(role)}` }, g);
            }
            if (showTips) text(g, round(xb + 6), cy + off + 4, valueText, role === "tez" ? "c-value-strong" : "c-value", { "aria-hidden": "true" });
            anchor = { x: xb, y };
          } else {
            // dot, optionally with a confidence interval
            const cx = x(clamp(val.v, spec.x.min, spec.x.max)), yy = cy + off;
            if (s.kind === "ci" && isNum(val.lo) && isNum(val.hi)) {
              const xl = x(val.lo), xh = x(val.hi);
              svg("line", { x1: round(xl), x2: round(xh), y1: round(yy), y2: round(yy), class: `m-ci ${lineClass(role)}` }, g);
              svg("line", { x1: round(xl), x2: round(xl), y1: round(yy - 4), y2: round(yy + 4), class: `m-ci ${lineClass(role)}` }, g);
              svg("line", { x1: round(xh), x2: round(xh), y1: round(yy - 4), y2: round(yy + 4), class: `m-ci ${lineClass(role)}` }, g);
            }
            svg("circle", { cx: round(cx), cy: round(yy), r: 12, class: "pt-hit" }, g);
            svg("circle", { cx: round(cx), cy: round(yy), r: 8.5, class: "pt-focus" }, g);
            svg("path", { d: shapePath(s.shape || "circle", cx, yy, s.r || 4.5), class: `m-dot ${roleClass(role)}` }, g);
            if (spec.ends && (s.endLabel !== false)) {
              const right = s.endSide === "left" ? false : true;
              const lx = s.kind === "ci" && isNum(val.hi) && right ? x(val.hi) : cx;
              text(g, round(right ? lx + 9 : cx - 9), yy + 4, valueText, role === "tez" ? "c-value-strong" : "c-value", { "text-anchor": right ? "start" : "end", "aria-hidden": "true" });
            }
            anchor = { x: cx, y: yy - 6 };
          }
          addPoint(state, g, {
            aria,
            anchor,
            tip: {
              title: r.tipTitle || (spec.tipName === "row" ? spec.title : r.label) || r.label,
              rows: spec.tipAll === false ? tipRows.filter((t) => t.name === nameOf(s)) : tipRows,
              note: r.tipNote || s.tipNote || null,
            },
          });
        });

        // value columns (direct labels for the series the story is about)
        vcols.forEach((c, k) => {
          const s = series.find((q) => q.key === c.series);
          const val = s ? valueOf(s.values[r.key]) : null;
          const cx = W - (vcols.length - 1 - k) * colW - 4;
          text(furniture, cx, cy + 4, val ? (c.fmt || s.fmt || spec.fmt || fmt.f3)(val.v) : "—", k === 0 ? "c-value-strong" : "c-value", { "text-anchor": "end" });
        });
        yCursor += rowH + labelBand;
      });
      return root;
    });

    function roleFor(s, r) {
      if (s.role === "byRow") return r.role || "other";
      return s.role || "other";
    }
    function formatVal(s, val, sp) {
      const f = s.fmt || sp.fmt || fmt.f3;
      if (isNum(val.v)) return f(val.v);
      if (isNum(val.lo) && isNum(val.hi)) return (s.rangeFmt || sp.rangeFmt || ((a, b) => `${a}–${b}`))(val.lo, val.hi);
      return "—";
    }
  }

  /* ------------------------------------------------------------------ xy chart */

  function xy(container, spec) {
    const id = `c${++uid}`;
    return mount(container, (W, state) => {
      const series = spec.series;
      const yTicks = spec.y.ticks || [];
      const ytf = spec.y.tickFmt || fmt.tick;
      const yTickW = Math.ceil(Math.max(...yTicks.map((t) => textWidth(ytf(t), "c-tick")), 10)) + 8;
      const endLabels = spec.endLabels !== false && series.some((s) => s.endLabel);
      let endW = 0;
      if (endLabels) {
        endW = Math.ceil(Math.max(...series.filter((s) => s.endLabel).map((s) => textWidth(`${s.endLabel} ${s.endValue || "0.000"}`, "c-end-label")))) + 14;
        if (endW > W * 0.42) endW = 0;
      }
      const left = yTickW + 2;
      const top = spec.y.label ? 26 : 10;
      const right = W - (endW || 14);
      const plotH = spec.height || 220;
      const bottom = top + plotH;
      const H = bottom + (spec.x.label ? 42 : 24);
      const x = spec.x.log ? logScale(spec.x.min, spec.x.max, left + 6, right - 6) : linear(spec.x.min, spec.x.max, left + (spec.x.pad || 0), right - (spec.x.pad || 0));
      const y = linear(spec.y.min, spec.y.max, bottom, top);

      const root = svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": spec.ariaLabel || "", id });
      const furniture = svg("g", { "aria-hidden": "true" }, root);

      (spec.bands || []).forEach((b) => {
        const xa = x(b.x0), xb = x(b.x1);
        svg("rect", { x: round(xa), y: top, width: round(Math.max(1, xb - xa)), height: plotH, class: "c-band" }, furniture);
        if (b.label) text(furniture, round((xa + xb) / 2), top + 13, b.label, "c-band-label", { "text-anchor": "middle" });
      });
      yTicks.forEach((t) => {
        const py = round(y(t));
        svg("line", { x1: left, x2: right, y1: py, y2: py, class: "c-grid" }, furniture);
        text(furniture, left - 8, py + 3.5, ytf(t), "c-tick", { "text-anchor": "end" });
      });
      const xtf = spec.x.tickFmt || fmt.tick;
      (spec.x.ticks || []).forEach((t) => {
        const px = round(x(t));
        svg("line", { x1: px, x2: px, y1: bottom, y2: bottom + 4, class: "c-baseline" }, furniture);
        tickLabel(furniture, px, bottom + 16, xtf(t), W);
      });
      svg("line", { x1: left, x2: right, y1: bottom, y2: bottom, class: "c-baseline" }, furniture);
      if (spec.y.label) text(furniture, 0, 12, fit(spec.y.label, "c-axis-label", W - 4), "c-axis-label");
      if (spec.x.label) {
        const mid = (left + right) / 2;
        text(furniture, round(mid), bottom + 34, fit(spec.x.label, "c-axis-label", 2 * Math.min(mid, W - mid) - 4), "c-axis-label", { "text-anchor": "middle" });
      }

      if (spec.diag) {
        const lo = Math.max(spec.x.min, spec.y.min), hi = Math.min(spec.x.max, spec.y.max);
        svg("line", { x1: round(x(lo)), y1: round(y(lo)), x2: round(x(hi)), y2: round(y(hi)), class: "c-ref" }, furniture);
        if (spec.diag.label) {
          const v = lo + (spec.diag.at || 0.7) * (hi - lo);
          text(furniture, round(x(v) + 6), round(y(v) + 14), spec.diag.label, "c-ref-label", { "text-anchor": "start" });
        }
      }
      (spec.ref || []).forEach((r) => {
        const py = round(y(r.y));
        svg("line", { x1: left, x2: right, y1: py, y2: py, class: "c-ref" }, furniture);
        const lx = r.align === "right" ? right - 4 : left + 6;
        text(furniture, lx, py + (r.below ? 13 : -5), r.label, "c-ref-label", { "text-anchor": r.align === "right" ? "end" : "start" });
      });

      const cross = svg("line", { x1: 0, x2: 0, y1: top, y2: bottom, class: "c-cross" }, furniture);
      state.cross = cross;

      const lines = svg("g", { "aria-hidden": "true" }, root);
      series.forEach((s) => {
        const pts = s.points.filter((p) => isNum(p[0]) && isNum(p[1]));
        if (s.line !== false && pts.length > 1) {
          const d = pts.map((p, i) => `${i ? "L" : "M"}${round(x(p[0]))},${round(y(p[1]))}`).join("");
          svg("path", { d, class: `m-line ${lineClass(s.role)}` }, lines);
        }
      });

      // points, ordered by x then series, so the arrow keys walk the chart left to right
      const all = [];
      series.forEach((s, si) => s.points.forEach((p) => { if (isNum(p[0]) && isNum(p[1])) all.push({ s, si, p }); }));
      all.sort((a, b) => a.p[0] - b.p[0] || a.si - b.si);
      const marks = svg("g", {}, root);
      const xfTitle = spec.tipTitle || ((v) => `${spec.x.name || "x"} ${xtf(v)}`);
      all.forEach(({ s, p }) => {
        const px = x(p[0]), py = y(p[1]);
        const g = svg("g", {}, marks);
        svg("circle", { cx: round(px), cy: round(py), r: 8, class: "pt-focus" }, g);
        if (s.markers !== false) svg("path", { d: shapePath(s.shape || "circle", px, py, s.r || 3.6), class: `m-dot ${roleClass(s.role)}` }, g);
        const label = p[2] && p[2].label;
        if (label) {
          const pos = (p[2] && p[2].pos) || "above";
          const dx = pos === "right" ? 9 : pos === "left" ? -9 : 0;
          const dy = pos === "below" ? 15 : pos === "right" || pos === "left" ? 4 : -9;
          text(furniture, round(px + dx), round(py + dy), label, "c-point-label", { "text-anchor": pos === "right" ? "start" : pos === "left" ? "end" : "middle" });
        }
        const f = s.fmt || spec.fmt || fmt.f3;
        const sameX = series
          .map((q) => ({ q, hit: q.points.find((pp) => pp[0] === p[0] && isNum(pp[1])) }))
          .filter((o) => o.hit);
        const tipRows = (p[2] && p[2].tipRows) || (spec.tipAll === false ? [{ q: s, hit: p }] : sameX).map((o) => ({
          name: o.q.label, value: (o.q.fmt || spec.fmt || fmt.f3)(o.hit[1]), role: o.q.role, shape: o.q.shape, kind: o.q.line === false ? "dot" : "line",
        }));
        const title = (p[2] && p[2].title) || xfTitle(p[0]);
        addPoint(state, g, {
          aria: `${s.label}, ${title}: ${f(p[1])}${p[2] && p[2].note ? `. ${p[2].note}` : ""}`,
          anchor: { x: px, y: py - 4 },
          crossX: round(px),
          tip: { title, rows: tipRows, note: (p[2] && p[2].note) || null },
        });
      });

      // end labels with a simple vertical de-collision and leader lines
      if (endLabels && endW) {
        const items = series
          .filter((s) => s.endLabel)
          .map((s) => {
            const pts = s.points.filter((p) => isNum(p[0]) && isNum(p[1]));
            const last = pts[pts.length - 1];
            return { s, px: x(last[0]), py: y(last[1]), ty: y(last[1]), v: (s.fmt || spec.fmt || fmt.f3)(last[1]) };
          })
          .sort((a, b) => a.ty - b.ty);
        const gap = 14;
        for (let i = 1; i < items.length; i++) if (items[i].ty - items[i - 1].ty < gap) items[i].ty = items[i - 1].ty + gap;
        for (let i = items.length - 2; i >= 0; i--) if (items[i + 1].ty - items[i].ty < gap) items[i].ty = items[i + 1].ty - gap;
        items.forEach((it) => {
          const lx = right + 8;
          if (Math.abs(it.ty - it.py) > 2 || it.px < right - 2) {
            svg("path", { d: `M${round(it.px + 5)},${round(it.py)}L${round(lx - 3)},${round(it.ty)}`, class: "c-leader" }, furniture);
          }
          const t = text(furniture, lx, round(it.ty + 4), "", "c-end-label");
          const a = svg("tspan", {}, t);
          a.textContent = `${it.s.endLabel} `;
          const b = svg("tspan", { class: "v" }, t);
          b.textContent = it.v;
        });
      }

      // hover layer on top: the pointer only has to be nearest to a point, not on it
      state.hoverLayer = svg("rect", { x: left, y: top, width: Math.max(1, right - left), height: plotH, fill: "transparent", "aria-hidden": "true" }, root);
      return root;
    });
  }

  /* ------------------------------------------------------------------ small multiples */

  function panels(container, spec) {
    container.textContent = "";
    container.setAttribute("data-state", "ready");
    const grid = document.createElement("div");
    grid.className = "panels";
    if (spec.min) grid.style.setProperty("--panel-min", `${spec.min}px`);
    container.appendChild(grid);
    return spec.panels.map((p) => {
      const cell = document.createElement("div");
      cell.className = "facet";
      const h = document.createElement(spec.headingTag || "h4");
      h.className = "panel-title";
      h.textContent = p.title;
      cell.appendChild(h);
      if (spec.withMeta !== false) {
        const m = document.createElement("p");
        m.className = "panel-meta";
        m.textContent = p.meta || "";
        cell.appendChild(m);
      }
      const c = document.createElement("div");
      c.className = "chart";
      cell.appendChild(c);
      grid.appendChild(cell);
      return p.kind === "xy" ? xy(c, p.spec) : rows(c, p.spec);
    });
  }

  /* ------------------------------------------------------------------ data table */

  function table(container, spec) {
    container.textContent = "";
    const wrap = document.createElement("div");
    wrap.className = "table-wrap";
    const t = document.createElement("table");
    t.className = "table data-table";
    if (spec.caption) {
      const c = document.createElement("caption");
      c.textContent = spec.caption;
      t.appendChild(c);
    }
    const thead = document.createElement("thead");
    const hr = document.createElement("tr");
    spec.columns.forEach((col) => {
      const th = document.createElement("th");
      th.scope = "col";
      th.textContent = col.label;
      if (col.num) th.className = "num";
      hr.appendChild(th);
    });
    thead.appendChild(hr);
    t.appendChild(thead);
    const tb = document.createElement("tbody");
    spec.rows.forEach((row) => {
      const tr = document.createElement("tr");
      spec.columns.forEach((col, ci) => {
        const cell = document.createElement(ci === 0 && spec.rowHeaders !== false ? "th" : "td");
        if (cell.tagName === "TH") cell.scope = "row";
        const v = col.get ? col.get(row) : row[col.key];
        cell.textContent = v === null || v === undefined || v === "" ? "—" : String(v);
        const cls = [];
        if (col.num) cls.push("num");
        if (v === null || v === undefined || v === "" || v === "—") cls.push("cell-muted");
        if (spec.cellClass) { const extra = spec.cellClass(row, col); if (extra) cls.push(extra); }
        if (cls.length) cell.className = cls.join(" ");
        tr.appendChild(cell);
      });
      if (spec.rowClass) { const rc = spec.rowClass(row); if (rc) tr.className = rc; }
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    wrap.appendChild(t);
    container.appendChild(wrap);
    return t;
  }

  /* ------------------------------------------------------------------ show-data toggles */

  function wireToggles(scope) {
    (scope || document).querySelectorAll("[data-show-data]").forEach((btn) => {
      if (btn.dataset.wired) return;
      btn.dataset.wired = "1";
      const target = document.getElementById(btn.getAttribute("aria-controls"));
      const label = btn.querySelector("[data-toggle-label]") || btn;
      btn.addEventListener("click", () => {
        const open = btn.getAttribute("aria-expanded") === "true";
        btn.setAttribute("aria-expanded", open ? "false" : "true");
        if (target) target.hidden = open;
        label.textContent = open ? "Show data" : "Hide data";
      });
    });
  }

  window.TezCharts = { rows, xy, panels, table, legend, swatch, fmt, wireToggles };
})();
