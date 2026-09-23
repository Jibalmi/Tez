/* Benchmarks page: every chart and table is drawn from data/benchmarks.json (values copied from BENCHMARKS.md,
   section named per block). Formatting keeps the precision BENCHMARKS.md prints; nothing is recomputed. */
(function () {
  "use strict";

  const C = window.TezCharts;
  if (!C) return;
  const F = C.fmt;
  const BENCH = "https://github.com/Jibalmi/Tez/blob/main/BENCHMARKS.md";

  const $ = (sel, root) => (root || document).querySelector(sel);
  let userMoved = false;
  ["wheel", "touchmove", "keydown", "pointerdown"].forEach((ev) =>
    window.addEventListener(ev, () => { userMoved = true; }, { passive: true, once: true }));

  /* bold the best value of a row (it may be Laya's or Jev's: losses are shown next to wins) */
  function bestOf(values, better) {
    const nums = values.filter((v) => typeof v === "number");
    if (!nums.length) return null;
    return better === "lower" ? Math.min(...nums) : Math.max(...nums);
  }

  function chart(id, fn) {
    const el = $(`[data-chart="${id}"]`);
    if (!el) return;
    try { fn(el); } catch (err) { fail(el, err); }
  }
  function table(id, spec) {
    const el = $(`[data-table="${id}"]`);
    if (!el) return;
    try { C.table(el, spec); } catch (err) { fail(el, err); }
  }
  function legend(id, items) {
    const el = $(`[data-legend="${id}"]`);
    if (el) C.legend(el, items);
  }
  function fail(el, err) {
    el.setAttribute("data-state", "error");
    el.textContent = "";
    const p = document.createElement("p");
    p.className = "chart-fallback";
    p.append("This view could not be drawn. The same numbers are in ");
    const a = document.createElement("a");
    a.href = BENCH;
    a.textContent = "BENCHMARKS.md";
    p.append(a, ".");
    el.appendChild(p);
    if (window.console && err) console.warn("benchmarks:", err);
  }

  const acc01 = { min: 0, max: 1, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1], tickFmt: (v) => (v === 0 ? "0" : v.toFixed(1)) };
  const langName = { en: "English", de: "German", fr: "French", es: "Spanish", ja: "Japanese", "zh-CN": "Chinese (Simplified)", zh: "Chinese", ar: "Arabic", hi: "Hindi", th: "Thai", ko: "Korean", km: "Khmer", ru: "Russian", tr: "Turkish" };
  const f3 = F.f3;
  const cell3 = (v) => (v === null || v === undefined ? "—" : f3(v));

  /* ---------------------------------------------------------------- head-to-head, small multiples */
  function h2h(d) {
    const H = d.h2h;
    const meta = {
      "AG News": "in Laya's training mix",
      "DAIR Emotion": "",
      "Banking77": "77 options, tournament for Tez",
      "SST-5": "ordinal score",
      "BoolQ": "yes/no · in Laya's training mix",
      "prompt-injections": "yes/no · held out",
      "MASSIVE intent, English": "20 options",
      "XNLI, English": "NLI is in Laya's training mix",
      "typed-decisions": "† fine-tuned on its train split",
    };
    const systems = [
      { label: "Tez", name: "Tez zero-shot", role: "tez" },
      { label: "laya", name: "laya", role: "other" },
      { label: "laya-ml", name: "laya-multilingual", role: "other" },
      { label: "laya-td", name: "laya-typed-decisions", role: "other" },
      { label: "Jev", name: "Jev (published)", role: "pub" },
    ];
    chart("h2h", (el) => {
      C.panels(el, {
        min: 232,
        panels: H.rows.map((row) => {
          const td = row.task === "typed-decisions";
          const unit = td ? `${row.n.toLocaleString("en-US")} decisions` : `n = ${row.n.toLocaleString("en-US")}`;
          const values = {};
          row.values.forEach((v, i) => { if (v !== null) values[String(i)] = v; });
          return {
            title: row.task,
            meta: [unit, meta[row.task] !== undefined ? meta[row.task] : row.note || ""].filter(Boolean).join(" · "),
            spec: {
              title: `${row.task}, ${unit}`,
              rows: systems.map((s, i) => ({ key: String(i), label: td && i === 3 ? "laya-td †" : s.label, name: s.name, role: s.role })),
              series: [{ key: "acc", label: "accuracy", kind: "bar", role: "byRow", values }],
              x: { min: 0, max: 1, ticks: [0, 0.5, 1], tickFmt: (v) => (v === 0 ? "0" : v === 1 ? "1.0" : v.toFixed(1)) },
              labels: "tips",
              boldMax: true,
              missingText: "not published",
              tipName: "row",
              rowHeight: 24,
              barHeight: 12,
              labelShare: 0.3,
              ariaLabel: `${row.task}: accuracy by system, ${unit}`,
            },
          };
        }),
      });
    });
    legend("h2h", [
      { label: "Tez, zero-shot (Gemma 4 12B Q8_0)", role: "tez", kind: "bar" },
      { label: "Laya checkpoints, same rows and GPU", role: "other", kind: "bar" },
      { label: "Jev, published figure (never run by us)", role: "pub", kind: "bar" },
    ]);
    table("h2h", {
      caption: `Accuracy per task. ${H.summary}`,
      columns: [
        { label: "task", get: (r) => r.task },
        { label: "n", num: true, get: (r) => r.n.toLocaleString("en-US") },
        ...H.systems.map((s, i) => ({ label: s, num: true, get: (r) => cell3(r.values[i]) })),
        { label: "note", get: (r) => r.note || "" },
      ],
      rows: H.rows,
      cellClass: (r, col) => {
        const i = H.systems.indexOf(col.label);
        return i >= 0 && r.values[i] !== null && r.values[i] === bestOf(r.values, "higher") ? "best" : "";
      },
    });
  }

  function workflow(d) {
    const W = d.typed_decisions_by_workflow;
    table("workflow", {
      caption: "typed-decisions accuracy by workflow",
      columns: [
        { label: "workflow", get: (r) => r.workflow },
        { label: W.systems[0], num: true, get: (r) => f3(r.values[0]) },
        { label: W.systems[1], num: true, get: (r) => f3(r.values[1]) },
      ],
      rows: W.rows,
      cellClass: (r, col) => {
        const i = W.systems.indexOf(col.label);
        return i >= 0 && r.values[i] === bestOf(r.values, "higher") ? "best" : "";
      },
    });
  }

  function tdDetail(d) {
    const T = d.typed_decisions_detail;
    const acc = d.h2h.rows.find((r) => r.task === "typed-decisions");
    const rows = [{ metric: "accuracy", values: acc.values, better: "higher" }].concat(T.rows);
    table("td-detail", {
      caption: `typed-decisions, 2,000 decisions. Tez with 4 worked examples in the cached prefix: accuracy ${f3(T.tez_4shot.accuracy)}, soft accuracy ${f3(T.tez_4shot.soft_accuracy)}.`,
      columns: [
        { label: "metric", get: (r) => `${r.metric} (${r.better} is better)` },
        ...T.systems.map((s, i) => ({ label: s, num: true, get: (r) => cell3(r.values[i]) })),
      ],
      rows,
      cellClass: (r, col) => {
        const i = T.systems.indexOf(col.label);
        if (i < 0) return "";
        const vals = r.values.filter((v) => v !== null);
        const best = r.better === "lower" ? Math.min(...vals) : Math.max(...vals);
        return r.values[i] === best ? "best" : "";
      },
    });
  }

  /* ---------------------------------------------------------------- languages and XNLI, dodged dot plots */
  function dotSeries(sys, values, shape, role) {
    return { key: sys, label: sys, kind: "dot", role, shape, values };
  }

  function languages(d) {
    const L = d.languages;
    const find = (name) => L.series.find((s) => s.system === name);
    const defs = [
      { src: "Tez zero-shot (12B letters)", label: "Tez zero-shot (Gemma 4 12B letters)", shape: "circle", role: "tez", key: "tez" },
      { src: "4B probe trained on English only", label: "Tez probe, trained on English rows only (Qwen3.5-4B, layer 20)", shape: "diamond", role: "tez", key: "probe" },
      { src: "laya-multilingual", label: "laya-multilingual", shape: "square", role: "other", key: "ml" },
      { src: "laya", label: "laya (English)", shape: "triangle", role: "other", key: "laya" },
      { src: "laya-typed-decisions", label: "laya-typed-decisions", shape: "triangle-down", role: "other", key: "td" },
    ];
    const series = defs.map((def) => {
      const s = find(def.src);
      const values = {};
      L.langs.forEach((lang, i) => { values[lang] = s.values[i]; });
      values.macro = s.macro;
      return Object.assign(dotSeries(def.label, values, def.shape, def.role), { key: def.key, label: def.label });
    });
    const rows = L.langs.map((lang) => ({ key: lang, label: lang, tipTitle: `${langName[lang] || lang} (${lang})`, aria: langName[lang] || lang }));
    rows.push({ key: "macro", label: "macro", divider: true, tipTitle: "Macro average over 11 languages", aria: "macro average" });
    chart("languages", (el) => C.rows(el, {
      rows, series,
      x: Object.assign({ label: "accuracy (MASSIVE intent, 20 options)" }, acc01),
      dodge: true, laneGap: 5, rowHeight: 32,
      valueColumns: [{ series: "tez", header: "Tez" }, { series: "ml", header: "laya-ml" }],
      ariaLabel: "MASSIVE intent accuracy by language for five systems",
    }));
    legend("languages", defs.map((def) => ({ label: def.label, role: def.role, shape: def.shape, kind: "dot" })));
    table("languages", {
      caption: "MASSIVE intent, 20 options, first 100 test rows per language, accuracy",
      columns: [{ label: "language", get: (r) => r.label }].concat(defs.map((def, k) => ({ label: def.label, num: true, get: (r) => cell3(series[k].values[r.key]) }))),
      rows: rows.map((r) => ({ key: r.key, label: r.key === "macro" ? "macro (11 languages)" : `${r.key} · ${langName[r.key] || r.key}` })),
      cellClass: (r, col) => {
        const k = defs.findIndex((def) => def.label === col.label);
        if (k < 0) return "";
        const vals = series.map((s) => s.values[r.key]);
        return vals[k] === bestOf(vals, "higher") ? "best" : "";
      },
    });
    table("languages-ece", {
      caption: "Macro ECE as shipped (lower is better)",
      columns: [{ label: "system", get: (r) => r[0] }, { label: "macro ECE", num: true, get: (r) => f3(r[1]) }],
      rows: Object.entries(L.macro_ece_shipped),
      cellClass: (r, col) => (col.label === "macro ECE" && r[1] === bestOf(Object.values(L.macro_ece_shipped), "lower") ? "best" : ""),
    });
  }

  function xnli(d) {
    const X = d.xnli_loss;
    const defs = [
      { src: "Tez", label: "Tez zero-shot", shape: "circle", role: "tez", key: "tez" },
      { src: "laya-multilingual", label: "laya-multilingual", shape: "square", role: "other", key: "ml" },
      { src: "laya", label: "laya (English)", shape: "triangle", role: "other", key: "laya" },
    ];
    const series = defs.map((def) => {
      const s = X.series.find((q) => q.system === def.src);
      const values = {};
      X.langs.forEach((lang, i) => { values[lang] = s.values[i]; });
      values.macro = s.macro;
      return { key: def.key, label: def.label, kind: "dot", role: def.role, shape: def.shape, values };
    });
    const rows = X.langs.map((lang) => ({ key: lang, label: lang, tipTitle: `${langName[lang] || lang} (${lang})`, aria: langName[lang] || lang }));
    rows.push({ key: "macro", label: "macro", divider: true, tipTitle: "Macro average over 10 languages", aria: "macro average" });
    chart("xnli", (el) => C.rows(el, {
      rows, series,
      x: Object.assign({ label: "accuracy (XNLI, 3 options)" }, acc01),
      dodge: true, laneGap: 6, rowHeight: 30,
      valueColumns: [{ series: "tez", header: "Tez" }, { series: "ml", header: "laya-ml" }],
      ariaLabel: "XNLI accuracy by language for Tez and two Laya checkpoints",
    }));
    legend("xnli", defs.map((def) => ({ label: def.label, role: def.role, shape: def.shape, kind: "dot" })));
    table("xnli", {
      caption: "XNLI, 100 rows per language, accuracy",
      columns: [{ label: "language", get: (r) => r.label }].concat(defs.map((def, k) => ({ label: def.label, num: true, get: (r) => cell3(series[k].values[r.key]) }))),
      rows: rows.map((r) => ({ key: r.key, label: r.key === "macro" ? "macro (10 languages)" : `${r.key} · ${langName[r.key] || r.key}` })),
      cellClass: (r, col) => {
        const k = defs.findIndex((def) => def.label === col.label);
        if (k < 0) return "";
        const vals = series.map((s) => s.values[r.key]);
        return vals[k] === bestOf(vals, "higher") ? "best" : "";
      },
    });
  }

  /* ---------------------------------------------------------------- calibration, order, latency */
  function calibration(d) {
    const K = d.calibration_order_latency;
    const names = ["Tez", "laya", "laya-multilingual", "laya-typed-decisions", "Jev (published)"];
    const roles = ["tez", "other", "other", "other", "pub"];
    const shipped = K.rows[0].values, refit = K.rows[1].values;
    const rows = names.map((n, i) => ({ key: String(i), label: n, role: roles[i] }));
    const vals = (arr) => { const o = {}; arr.forEach((v, i) => { if (v !== null) o[String(i)] = v; }); return o; };
    chart("calibration", (el) => C.rows(el, {
      rows,
      series: [
        { key: "shipped", label: "as shipped", kind: "dot", role: "byRow", shape: "square", values: vals(shipped) },
        { key: "refit", label: "after one temperature per task", kind: "dot", role: "byRow", shape: "circle", values: vals(refit), endSide: "left" },
      ],
      connect: ["shipped", "refit"],
      x: { min: 0, max: 0.35, ticks: [0, 0.1, 0.2, 0.3], tickFmt: (v) => (v === 0 ? "0" : v.toFixed(1)), label: "mean ECE-15 over tasks (lower is better)" },
      ends: true,
      rowHeight: 30,
      labelShare: 0.36,
      ariaLabel: "Calibration error as shipped and after one temperature per task, per system",
    }));
    legend("calibration", [
      { label: "as shipped", role: "other", shape: "square", kind: "dot" },
      { label: "after one temperature per task (2-fold out-of-fold)", role: "other", shape: "circle", kind: "dot" },
      { label: "Tez", role: "tez", shape: "circle", kind: "dot" },
      { label: "published (Jev)", role: "pub", shape: "square", kind: "dot" },
    ]);
    table("calibration", {
      caption: "Mean ECE-15 over tasks",
      columns: [
        { label: "system", get: (r) => r.label },
        { label: "as shipped", num: true, get: (r) => cell3(shipped[Number(r.key)]) },
        { label: "after one temperature per task", num: true, get: (r) => cell3(refit[Number(r.key)]) },
      ],
      rows,
      cellClass: (r, col) => {
        const arr = col.label === "as shipped" ? shipped : col.label === "after one temperature per task" ? refit : null;
        return arr && arr[Number(r.key)] !== null && arr[Number(r.key)] === bestOf(arr, "lower") ? "best" : "";
      },
    });

    // latency ranges: show where Laya wins
    const ms = K.rows[3].values.map((s) => s.split("-").map(Number));
    chart("latency-ranges", (el) => C.rows(el, {
      rows: names.map((n, i) => ({ key: String(i), label: n, name: n, role: roles[i] })),
      series: [{ key: "ms", label: "ms per decision, p50 range across tasks", kind: "range", role: "byRow", values: Object.fromEntries(ms.map((r, i) => [String(i), r])) }],
      x: { min: 0, max: 300, ticks: [0, 50, 100, 150, 200, 250, 300], label: "ms per decision, p50 (range across tasks)" },
      labels: "tips",
      tipSample: "236–276 ms",
      rangeFmt: (a, b) => `${a}–${b} ms`,
      tipName: "row",
      title: "ms per decision, p50 range across tasks",
      rowHeight: 28, barHeight: 10, labelShare: 0.36,
      ariaLabel: "Milliseconds per decision, p50 range across tasks, per system",
    }));
    table("latency-ranges", {
      caption: "ms per decision, p50 range across Laya's tasks (this GPU; Jev published)",
      columns: [{ label: "system", get: (r) => r.n }, { label: "ms per decision, p50", num: true, get: (r) => r.v.replace("-", "–") }],
      rows: names.map((n, i) => ({ n, v: K.rows[3].values[i] })),
    });

    const fmtRow = (r, v) => {
      if (v === null || v === undefined) return "—";
      if (typeof v === "string") return v.replace("-", "–");
      return /flip/.test(r.metric) ? F.f2(v) : f3(v);
    };
    table("cal-order-latency", {
      caption: `Calibration, option-order robustness and latency (${K.source})`,
      columns: [{ label: "", get: (r) => `${r.metric} (${r.better} is better)` }].concat(K.systems.map((s, i) => ({ label: s, num: true, get: (r) => fmtRow(r, r.values[i]) }))),
      rows: K.rows,
      cellClass: (r, col) => {
        const i = K.systems.indexOf(col.label);
        if (i < 0 || typeof r.values[0] === "string") return "";
        return r.values[i] === bestOf(r.values, "lower") ? "best" : "";
      },
    });
  }

  /* ---------------------------------------------------------------- readouts */
  function readouts(d) {
    const R = d.readouts_typed_decisions;
    const rows = R.rows.map((r) => Object.assign({}, r)).concat(R.references.map((r) => ({ readout: r.system, labels: "reference", accuracy: r.accuracy, ece: r.ece === undefined ? null : r.ece, ece_after_temperature: null, choice: null, noul: null, score: null, ref: true })));
    table("readouts", {
      caption: `${R.title}. Accuracy, calibration error (ECE-15) as read and after a 2-fold out-of-fold temperature, and accuracy per question type.`,
      columns: [
        { label: "readout", get: (r) => r.readout + (r.note ? " †" : "") },
        { label: "labels", get: (r) => r.labels },
        { label: "accuracy", num: true, get: (r) => cell3(r.accuracy) },
        { label: "ECE", num: true, get: (r) => cell3(r.ece) },
        { label: "ECE after temperature", num: true, get: (r) => cell3(r.ece_after_temperature) },
        { label: "choice (600)", num: true, get: (r) => cell3(r.choice) },
        { label: "yes/no (600)", num: true, get: (r) => cell3(r.noul) },
        { label: "score (800)", num: true, get: (r) => cell3(r.score) },
      ],
      rows,
      rowClass: (r) => (r.ref ? "is-ref" : ""),
    });
    const note = $("[data-text='readouts-note']");
    const withNote = R.rows.find((r) => r.note);
    if (note && withNote) note.textContent = `† ${withNote.readout}: ${withNote.note}.`;
  }

  /* ---------------------------------------------------------------- few labels, two panels */
  function fewLabels(d) {
    const FL = d.few_labels;
    const byName = (n) => FL.series.find((s) => s.name === n).values;
    const pts = (vals) => FL.n.map((n, i) => [n, vals[i]]).filter((p) => p[1] !== null);
    const laya = d.readouts_typed_decisions.references.find((r) => /laya/.test(r.system)).accuracy;
    const base = {
      x: { min: 4.5, max: 55, log: true, ticks: FL.n, tickFmt: (v) => String(v), label: "labelled rows per question (log scale)", name: "rows" },
      y: { min: 0.55, max: 0.8, ticks: [0.55, 0.6, 0.65, 0.7, 0.75, 0.8], tickFmt: F.f2, label: "accuracy" },
      height: 210,
      ref: [
        { y: FL.zero_label_reference, label: `12B zero-shot ${f3(FL.zero_label_reference)}`, align: "right", below: true },
        { y: laya, label: `laya-td fine-tuned ${f3(laya)}`, align: "left" },
      ],
      tipTitle: (v) => `${v} labelled rows per question`,
    };
    chart("few-labels", (el) => C.panels(el, {
      min: 300,
      withMeta: false,
      panels: [
        {
          title: "Labels alone",
          kind: "xy",
          spec: Object.assign({}, base, {
            ariaLabel: "Accuracy by labelled rows per question, labels alone",
            series: [
              { key: "typ", label: "most typical rows first", role: "tez", shape: "circle", points: pts(byName("most typical rows first")), endLabel: "typical" },
              { key: "rnd", label: "random rows", role: "other", shape: "square", points: pts(byName("random rows, logistic probe")), endLabel: "random" },
            ],
          }),
        },
        {
          title: "Labels plus the 12B's zero-shot answers",
          kind: "xy",
          spec: Object.assign({}, base, {
            ariaLabel: "Accuracy by labelled rows per question, with the zero-shot prior",
            series: [
              { key: "typ", label: "typical rows + 12B zero-shot prior", role: "tez", shape: "circle", points: pts(byName("typical rows + 12B zero-shot prior")), endLabel: "typical" },
              { key: "rnd", label: "random rows + 12B zero-shot prior", role: "other", shape: "square", points: pts(byName("random rows + 12B zero-shot prior")), endLabel: "random" },
            ],
          }),
        },
      ],
    }));
    legend("few-labels", [
      { label: "most typical rows labelled first", role: "tez", shape: "circle", kind: "line" },
      { label: "random rows", role: "other", shape: "square", kind: "line" },
      { label: "reference level", kind: "ref" },
    ]);
    table("few-labels", {
      caption: `${FL.title}, accuracy on typed-decisions`,
      columns: [{ label: "labelled rows per question", get: (r) => r.name }].concat(FL.n.map((n, i) => ({ label: String(n), num: true, get: (r) => cell3(r.values[i]) }))),
      rows: FL.series,
      cellClass: (r, col) => {
        const i = FL.n.map(String).indexOf(col.label);
        if (i < 0) return "";
        const vals = FL.series.map((s) => s.values[i]);
        return r.values[i] === bestOf(vals, "higher") ? "best" : "";
      },
    });
  }

  /* ---------------------------------------------------------------- where to read */
  function layers(d) {
    const P = d.probe_layers;
    const acc = P.layers.map((l, i) => [l, P.probe_accuracy[i]]);
    const idim = P.layers.map((l, i) => [l, P.intrinsic_dimension[i]]);
    const x = { min: 0, max: 32, ticks: [0, 4, 8, 12, 16, 20, 24, 28, 32], label: "layer of 32", name: "layer" };
    chart("layers", (el) => C.panels(el, {
      min: 560,
      withMeta: false,
      panels: [
        {
          title: "Probe accuracy (needs labels)",
          kind: "xy",
          spec: {
            x: Object.assign({}, x, { label: null }), y: { min: 0.4, max: 0.85, ticks: [0.4, 0.5, 0.6, 0.7, 0.8], tickFmt: F.f1, label: "probe accuracy" },
            height: 170, bands: [{ x0: 24, x1: 28, label: "24–28" }], endLabels: false,
            series: [{ key: "acc", label: "probe accuracy", role: "tez", shape: "circle", points: acc }],
            tipTitle: (v) => `layer ${v}`,
            ariaLabel: "Probe accuracy by layer",
          },
        },
        {
          title: "Intrinsic dimension of unlabelled states (no labels)",
          kind: "xy",
          spec: {
            x, y: { min: 8, max: 13, ticks: [8, 9, 10, 11, 12, 13], tickFmt: (v) => String(v), label: "intrinsic dimension (TwoNN)" },
            height: 150, bands: [{ x0: 24, x1: 28, label: "lowest" }], endLabels: false, fmt: F.f1,
            series: [{ key: "id", label: "intrinsic dimension", role: "other", shape: "square", points: idim, fmt: F.f1 }],
            tipTitle: (v) => `layer ${v}`,
            ariaLabel: "Intrinsic dimension of unlabelled states by layer",
          },
        },
      ],
    }));
    table("layers", {
      caption: "Frozen Qwen3.5-4B, typed-decisions: probe accuracy (labels) and intrinsic dimension of 2,000 unlabelled train states (TwoNN), by layer",
      columns: [
        { label: "layer", get: (r) => r[0] },
        { label: "probe accuracy", num: true, get: (r) => f3(r[1]) },
        { label: "intrinsic dimension", num: true, get: (r) => F.f1(r[2]) },
      ],
      rows: P.layers.map((l, i) => [l, P.probe_accuracy[i], P.intrinsic_dimension[i]]),
    });

    const R = P.reversed_options_same_probe;
    chart("reversed", (el) => C.xy(el, {
      x: { min: 7, max: 33, ticks: [8, 12, 16, 20, 24, 28, 32], label: "layer of 32", name: "layer" },
      y: { min: 0.5, max: 0.8, ticks: [0.5, 0.6, 0.7, 0.8], tickFmt: F.f1, label: "probe accuracy" },
      height: 220,
      bands: [{ x0: 8, x1: 14, label: "order-free" }],
      series: [
        { key: "trained", label: "options in the trained order", role: "other", shape: "square", points: R.layers.map((l, i) => [l, R.trained_order[i]]), endLabel: "trained order" },
        { key: "rev", label: "options reversed, same probe", role: "tez", shape: "circle", points: R.layers.map((l, i) => [l, R.reversed[i]]), endLabel: "reversed" },
      ],
      tipTitle: (v) => `layer ${v}`,
      ariaLabel: "Probe accuracy by layer with options in the trained order and reversed",
    }));
    legend("reversed", [
      { label: "options reversed, same probe", role: "tez", shape: "circle", kind: "line" },
      { label: "options in the trained order", role: "other", shape: "square", kind: "line" },
    ]);
    table("reversed", {
      caption: "Same per-question probes on prompts with the options reversed (Qwen3.5-4B, typed-decisions)",
      columns: [
        { label: "layer", get: (r) => r[0] },
        { label: "options in the trained order", num: true, get: (r) => f3(r[1]) },
        { label: "options reversed, same probe", num: true, get: (r) => f3(r[2]) },
      ],
      rows: R.layers.map((l, i) => [l, R.trained_order[i], R.reversed[i]]),
    });
  }

  /* ---------------------------------------------------------------- conformal selection */
  function conformal(d) {
    const K = d.conformal;
    const bounds = K.bounds.map((b) => Math.round(b * 100));
    const defs = [
      { src: K.rows[0], label: "4B probe, layer 26 (gold labels)", short: "probe", role: "tez", shape: "circle" },
      { src: K.rows[1], label: "12B letters (zero-shot)", short: "12B letters", role: "other", shape: "square" },
      { src: K.rows[2], label: "4B probe on 12B pseudo-labels (no human labels)", short: "pseudo-labels", role: "other", shape: "triangle" },
    ];
    const pctInt = (v) => `${v} %`;
    const pct1 = (v) => `${v.toFixed(1)} %`;
    const x = { min: 0, max: 21, ticks: [0, 5, 10, 15, 20], tickFmt: (v) => `${v} %`, label: "error bound chosen", name: "bound" };
    chart("conformal", (el) => C.panels(el, {
      min: 300,
      withMeta: false,
      panels: [
        {
          title: "Share of decisions acted on",
          kind: "xy",
          spec: {
            x, y: { min: 0, max: 100, ticks: [0, 25, 50, 75, 100], tickFmt: (v) => `${v} %`, label: "acted on" },
            height: 210, fmt: pctInt,
            series: defs.map((def) => ({
              key: def.short, label: def.label, role: def.role, shape: def.shape, endLabel: def.short, fmt: pctInt,
              points: bounds.map((b, i) => [b, Math.round(def.src.acted[i] * 100), { note: `Realised error among acted: ${pct1(def.src.realised_error[i] * 100)}` }]),
            })),
            tipTitle: (v) => `error bound ${v} %`,
            ariaLabel: "Share of decisions acted on by error bound, three readouts",
          },
        },
        {
          title: "Realised error among acted decisions",
          kind: "xy",
          spec: {
            x, y: { min: 0, max: 21, ticks: [0, 5, 10, 15, 20], tickFmt: (v) => `${v} %`, label: "realised error" },
            height: 210, endLabels: false, diag: { label: "realised = bound" }, fmt: pct1,
            series: defs.map((def) => ({
              key: def.short, label: def.label, role: def.role, shape: def.shape, line: false, fmt: pct1,
              points: bounds
                .map((b, i) => [b, def.src.realised_error[i] * 100, { note: `Acted on ${Math.round(def.src.acted[i] * 100)} % of decisions` }])
                .filter((p, i) => def.src.acted[i] > 0 || def.src.realised_error[i] > 0),
            })),
            tipTitle: (v) => `error bound ${v} %`,
            ariaLabel: "Realised error among acted decisions by error bound, three readouts",
          },
        },
      ],
    }));
    legend("conformal", defs.map((def) => ({ label: `${def.label}, accuracy ${f3(def.src.accuracy)}`, role: def.role, shape: def.shape, kind: "line" })).concat([{ label: "realised = bound", kind: "ref" }]));
    table("conformal", {
      caption: `${K.title}. ${K.note}`,
      columns: [
        { label: "readout", get: (r) => r.def.label },
        { label: "accuracy", num: true, get: (r) => f3(r.def.src.accuracy) },
      ].concat(bounds.map((b, i) => ({ label: `bound ${b} %: acted / realised error`, num: true, get: (r) => `${Math.round(r.def.src.acted[i] * 100)} % / ${pct1(r.def.src.realised_error[i] * 100)}` }))),
      rows: defs.map((def) => ({ def })),
    });
  }

  /* ---------------------------------------------------------------- cold start */
  function coldStart(d) {
    const S = d.cold_start_full;
    table("cold-start", {
      caption: `${S.title}. Starting point: the 12B's zero-shot readout (${f3(d.cold_start.starting_point)}).`,
      columns: [
        { label: "threshold", num: true, get: (r) => r.threshold.toFixed(1) },
        { label: "policy", get: (r) => r.policy },
        { label: "human labels", num: true, get: (r) => F.int(r.human_labels) },
        { label: "automated share", num: true, get: (r) => F.pct0(r.automated_share) },
        { label: "automated accuracy", num: true, get: (r) => f3(r.automated_accuracy) },
        { label: "whole stream", num: true, get: (r) => f3(r.whole_stream) },
        { label: "probe held-out", num: true, get: (r) => f3(r.probe_held_out) },
        { label: "same number of random labels", num: true, get: (r) => f3(r.random_labels) },
      ],
      rows: S.rows,
      rowHeaders: false,
    });
    const note = $("[data-text='cold-weak']");
    if (note) note.textContent = S.weak_prior_note;
  }

  /* ---------------------------------------------------------------- depth-pruned GGUFs and latency */
  function pruned(d) {
    const P = d.pruned;
    const blocks = P.rows.map((r) => r.blocks.split(" ")[0]);
    const tipFor = (r) => [
      { name: "probe on the served state", value: `${f3(r.probe)} · ${r.ms_probe} ms`, role: "tez", shape: "circle", kind: "line" },
      { name: "zero-shot letters", value: `${f3(r.letters)} · ${r.ms_letters} ms`, role: "other", shape: "square", kind: "line" },
    ];
    const gemma = d.latency_breakdown.rows.find((r) => /Gemma/.test(r.model));
    const gemmaMs = gemma.p50[d.latency_breakdown.columns.indexOf("completion, n_probs 200")];
    const gemmaAcc = d.readouts_typed_decisions.rows.find((r) => /^12B letters/.test(r.readout)).accuracy;
    chart("pruned", (el) => C.xy(el, {
      x: { min: 0, max: 225, ticks: [0, 50, 100, 150, 200], label: "ms per decision, p50", name: "ms" },
      y: Object.assign({ label: "accuracy (typed-decisions)" }, acc01),
      height: 240,
      endLabels: false,
      series: [
        {
          key: "probe", label: "probe on the served state", role: "tez", shape: "circle",
          points: P.rows.map((r, i) => [r.ms_probe, r.probe, { label: blocks[i], pos: ["left", "above", "above", "right"][i] || "above", title: `${r.blocks} blocks · ${r.size_gb.toFixed(2)} GB`, tipRows: tipFor(r) }]),
        },
        {
          key: "letters", label: "zero-shot letters of the cut model", role: "other", shape: "square",
          points: P.rows.map((r, i) => [r.ms_letters, r.letters, { label: blocks[i], pos: "below", title: `${r.blocks} blocks · ${r.size_gb.toFixed(2)} GB`, tipRows: tipFor(r) }]),
        },
        {
          key: "gemma", label: "Gemma 4 12B Q8_0 letters, full model", role: "other", shape: "diamond", line: false,
          points: [[gemmaMs, gemmaAcc, { label: "12B", pos: "left", title: "Gemma 4 12B Q8_0, zero-shot letters", tipRows: [{ name: "zero-shot letters (n_probs 200)", value: `${f3(gemmaAcc)} · ${gemmaMs} ms`, role: "other", shape: "diamond", kind: "dot" }] }]],
        },
      ],
      ariaLabel: "Accuracy against milliseconds per decision for the depth-pruned 4B GGUFs and the 12B",
    }));
    legend("pruned", [
      { label: "probe on the served state (4B cut)", role: "tez", shape: "circle", kind: "line" },
      { label: "zero-shot letters (4B cut)", role: "other", shape: "square", kind: "line" },
      { label: "Gemma 4 12B Q8_0 letters", role: "other", shape: "diamond", kind: "dot" },
    ]);
    table("pruned", {
      caption: `${P.title}. ${P.caveat}`,
      columns: [
        { label: "blocks kept", get: (r) => r.blocks },
        { label: "GGUF size", num: true, get: (r) => `${r.size_gb.toFixed(2)} GB` },
        { label: "zero-shot letters", num: true, get: (r) => f3(r.letters) },
        { label: "probe on the served state", num: true, get: (r) => f3(r.probe) },
        { label: "ms / decision, letters", num: true, get: (r) => r.ms_letters },
        { label: "ms / decision, probe", num: true, get: (r) => r.ms_probe },
      ],
      rows: P.rows,
      rowClass: (r) => (/^24/.test(r.blocks) ? "is-highlight" : ""),
    });
    const LB = d.latency_breakdown;
    table("latency-breakdown", {
      caption: LB.title,
      columns: [{ label: "model", get: (r) => r.model }].concat(LB.columns.map((c, i) => ({ label: c, num: true, get: (r) => `${r.p50[i]} / ${r.p90[i]}` }))),
      rows: LB.rows,
    });
  }

  /* ---------------------------------------------------------------- voice */
  function voice(d) {
    const V = d.voice_per_word;
    const rows = V.rows.map((r, i) => ({
      key: String(i), label: r.variant, name: r.variant, role: i === 0 ? "tez" : "other",
      sub: `${r.tokens_per_word} tokens evaluated per word`,
    }));
    chart("voice", (el) => C.rows(el, {
      rows,
      series: [{ key: "ms", label: "compute per streamed word, p50", kind: "bar", role: "byRow", values: Object.fromEntries(V.rows.map((r, i) => [String(i), r.compute_ms])) }],
      x: { min: 0, max: 220, ticks: [0, 50, 100, 150, 200], label: "ms of compute per streamed word, p50" },
      fmt: F.ms, tipSample: "205 ms", labels: "tips", tipName: "row", title: "compute per streamed word, p50",
      rowHeight: 40, barHeight: 14, labelShare: 0.44,
      ariaLabel: "Compute per streamed word for three prompt and cache set-ups",
    }));
    table("voice-per-word", {
      caption: `${V.title}. ${V.note}`,
      columns: [
        { label: "set-up", get: (r) => r.variant },
        { label: "compute ms", num: true, get: (r) => r.compute_ms },
        { label: "HTTP round trip ms", num: true, get: (r) => (r.round_trip_ms === null ? "—" : r.round_trip_ms) },
        { label: "tokens per word", num: true, get: (r) => r.tokens_per_word },
        { label: "intent accuracy", num: true, get: (r) => cell3(r.intent_accuracy) },
      ],
      rows: V.rows,
    });
    table("voice", {
      caption: d.voice.title,
      columns: [{ label: "measure", get: (r) => r.metric }, { label: "result", num: true, get: (r) => r.value }],
      rows: d.voice.rows,
    });
  }

  /* ---------------------------------------------------------------- SemIf */
  function semif(d) {
    const S = d.semif, X = d.semif_extra;
    const short = {
      "Gemma 4 12B Q8_0": ["Gemma 4 12B Q8_0", "the default backbone"],
      "Gemma 4 12B Q4_K_M": ["Gemma 4 12B Q4_K_M", ""],
      "Qwen3.5-9B Q8_0": ["Qwen3.5-9B Q8_0", ""],
      "Qwen3.5-4B BF16 (SemIf's model, reproduced)": ["Qwen3.5-4B BF16", "SemIf's model, reproduced in-process"],
      "Gemma 3 4B Q4_K_M": ["Gemma 3 4B Q4_K_M", ""],
      "Llama 3.2 3B Q4_K_M": ["Llama 3.2 3B Q4_K_M", ""],
    };
    const measured = S.rows.map((r) => {
      const ci = r.ci.split("-").map(Number);
      return { key: r.model, label: short[r.model][0], sub: short[r.model][1], name: r.model, role: r.model === "Gemma 4 12B Q8_0" ? "tez" : "other", v: { v: r.authored, lo: ci[0], hi: ci[1] }, tipNote: `95 % CI ${r.ci.replace("-", "–")}; p50 ${r.p50_ms} ms` };
    });
    const perm = X.permutation_row;
    measured.push({ key: perm.model, label: "Qwen3.5-4B + 6 orderings", sub: "permutation average, one batch", name: perm.model, role: "other", v: { v: perm.authored }, tipNote: `no CI reported; p50 ${perm.p50_ms} ms (in-process)` });
    measured.sort((a, b) => b.v.v - a.v.v);
    const published = X.published.map((p, i) => ({ key: p.model, label: p.model.replace("SemIf published: ", "SemIf: "), sub: "published", name: p.model, role: "pub", v: { v: p.authored }, divider: i === 0, tipNote: "SemIf's published figure" }));
    const all = measured.concat(published);
    chart("semif", (el) => C.rows(el, {
      rows: all.map((r) => ({ key: r.key, label: r.label, sub: r.sub || " ", name: r.name, role: r.role, divider: r.divider, tipTitle: r.name, tipNote: r.tipNote })),
      series: [{ key: "mfba", label: "mean-family balanced accuracy", kind: "ci", role: "byRow", shape: "circle", values: Object.fromEntries(all.map((r) => [r.key, r.v])) }],
      x: Object.assign({ label: "mean-family balanced accuracy" }, acc01),
      ref: [{ value: 1 / 3, label: "chance, 3 options" }],
      ends: true, rowHeight: 36, labelShare: 0.42, tipName: "row", title: "SemIf authored144",
      ariaLabel: "Mean-family balanced accuracy on SemIf's authored144 with 95 percent intervals",
    }));
    legend("semif", [
      { label: "Gemma 4 12B Q8_0 (the default)", role: "tez", shape: "circle", kind: "dot" },
      { label: "other frozen models, same harness", role: "other", shape: "circle", kind: "dot" },
      { label: "SemIf's published figures", role: "pub", shape: "circle", kind: "dot" },
    ]);
    table("semif", {
      caption: `${S.title}. ${X.paired_tests}`,
      columns: [
        { label: "model", get: (r) => r.model },
        { label: "authored144", num: true, get: (r) => cell3(r.authored) },
        { label: "95 % CI", num: true, get: (r) => (r.ci ? r.ci.replace("-", "–") : "—") },
        { label: "perturbations", num: true, get: (r) => cell3(r.perturbation) },
        { label: "NLL raw → best", num: true, get: (r) => (X.nll_brier.find((n) => r.model.indexOf(n.model.split(" (")[0]) === 0) || {}).nll || "—" },
        { label: "Brier raw → best", num: true, get: (r) => (X.nll_brier.find((n) => r.model.indexOf(n.model.split(" (")[0]) === 0) || {}).brier || "—" },
        { label: "reversal flips", num: true, get: (r) => r.reversal_flips || "—" },
        { label: "p50", num: true, get: (r) => (r.p50_ms ? `${r.p50_ms} ms` : "—") },
      ],
      rows: S.rows.concat([{ model: perm.model, authored: perm.authored, p50_ms: perm.p50_ms }], X.published.map((p) => ({ model: p.model, authored: p.authored, perturbation: p.perturbation }))),
    });
    const CR = X.calibration_recipes;
    table("semif-recipes", {
      caption: CR.title,
      columns: [
        { label: "recipe", get: (r) => r.recipe },
        { label: "NLL", num: true, get: (r) => f3(r.nll) },
        { label: "Brier", num: true, get: (r) => f3(r.brier) },
        { label: "ECE", num: true, get: (r) => f3(r.ece) },
      ],
      rows: CR.rows,
      cellClass: (r, col) => (r.recipe === "6-permutation mean + temperature" && col.label !== "ECE" ? "best" : ""),
    });
    const z = $("[data-text='zhao']");
    if (z) z.textContent = CR.zhao;
  }

  /* ---------------------------------------------------------------- JevBench public tiers */
  function jevbench(d) {
    const J = d.jevbench_public;
    table("jevbench", {
      caption: J.title,
      columns: [
        { label: "tier", get: (r) => `${r.tier} (${r.n})` },
        { label: "run", get: (r) => r.run || "experiment harness" },
        { label: "accuracy", num: true, get: (r) => f3(r.accuracy) },
        { label: "intelligence", num: true, get: (r) => (Number.isInteger(r.intelligence) ? String(r.intelligence) : r.intelligence.toFixed(1)) },
        { label: "ECE", num: true, get: (r) => f3(r.ece) },
        { label: "median", num: true, get: (r) => `${r.median_s.toFixed(2)} s` },
      ],
      rows: J.rows,
    });
    const w = $("[data-text='jev-weighted']");
    if (w) w.textContent = J.weighted_public_intelligence.toFixed(1);
    const ws = $("[data-text='jev-weighted-server']");
    if (ws && J.weighted_public_intelligence_server != null) ws.textContent = J.weighted_public_intelligence_server.toFixed(1);
  }

  /* ---------------------------------------------------------------- negative results */
  function negatives(d) {
    const ul = $("[data-list='negative']");
    if (!ul) return;
    ul.textContent = "";
    const items = d.negative_results.items.map((t) => ({ text: t })).concat(d.negative_results_extra.items);
    items.forEach((it) => {
      const li = document.createElement("li");
      li.textContent = it.text;
      ul.appendChild(li);
    });
  }

  function restoreHash() {
    if (!location.hash || location.hash.length < 2) return;
    let target = null;
    try { target = document.getElementById(decodeURIComponent(location.hash.slice(1))); } catch (e) { target = null; }
    if (!target) return;
    if (userMoved) return;
    target.scrollIntoView({ block: "start" });
  }

  function render(d) {
    [h2h, workflow, tdDetail, languages, xnli, calibration, readouts, fewLabels, layers, conformal, coldStart, pruned, voice, semif, jevbench, negatives]
      .forEach((fn) => { try { fn(d); } catch (err) { if (window.console) console.warn("benchmarks:", fn.name, err); } });
    C.wireToggles();
    requestAnimationFrame(restoreHash);
  }

  fetch("./data/benchmarks.json")
    .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
    .then(render)
    .catch((err) => {
      document.querySelectorAll("[data-chart], [data-table]").forEach((el) => fail(el, err));
      C.wireToggles();
    });
})();
