/* Tez UI: rendering shared by the landing page, the use-case gallery and the playground.
   - one answer (noul / choice / score) as probability rows with the chosen option marked
   - wire-format JSON with syntax classes; curl and Python snippets for a request
   - a replay engine that plays a recorded trace word by word
   Exposes window.TezUI. No dependencies; icons are Lucide paths (ISC licence). */
(function (root) {
  "use strict";

  var REPO = "https://github.com/Jibalmi/Tez";
  var BLOB = REPO + "/blob/main/";
  var DEFAULT_BASE = "http://127.0.0.1:8787";
  var MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

  /* ------------------------------------------------------------------ icons */
  var PATHS = {
    check: '<path d="M20 6 9 17l-5-5"/>',
    copy: '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>',
    replay: '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
    pause: '<rect x="14" y="4" width="4" height="16" rx="1"/><rect x="6" y="4" width="4" height="16" rx="1"/>',
    play: '<polygon points="6 3 20 12 6 21 6 3"/>',
    history: '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>',
    arrowRight: '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
    arrowLeft: '<path d="m12 19-7-7 7-7"/><path d="M19 12H5"/>',
    arrowUpRight: '<path d="M7 7h10v10"/><path d="M7 17 17 7"/>',
    alert: '<circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/>',
    circleCheck: '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    circleDashed: '<path d="M10.1 2.182a10 10 0 0 1 3.8 0"/><path d="M13.9 21.818a10 10 0 0 1-3.8 0"/><path d="M17.609 3.721a10 10 0 0 1 2.69 2.7"/><path d="M2.182 13.9a10 10 0 0 1 0-3.8"/><path d="M20.279 17.609a10 10 0 0 1-2.7 2.69"/><path d="M21.818 10.1a10 10 0 0 1 0 3.8"/><path d="M3.721 6.391a10 10 0 0 1 2.7-2.69"/><path d="M6.391 20.279a10 10 0 0 1-2.69-2.7"/>',
    loader: '<path d="M21 12a9 9 0 1 1-6.219-8.56"/>',
    search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    external: '<path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>',
    terminal: '<polyline points="4 17 10 11 4 5"/><line x1="12" x2="20" y1="19" y2="19"/>',
    plug: '<path d="M12 22v-5"/><path d="M9 8V2"/><path d="M15 8V2"/><path d="M18 8v5a4 4 0 0 1-4 4h-4a4 4 0 0 1-4-4V8Z"/>',
    send: '<path d="M14.536 21.686a.5.5 0 0 0 .937-.024l6.5-19a.496.496 0 0 0-.635-.635l-19 6.5a.5.5 0 0 0-.024.937l7.93 3.18a2 2 0 0 1 1.112 1.112z"/><path d="m21.854 2.147-10.94 10.939"/>',
    key: '<path d="m15.5 7.5 2.3 2.3a1 1 0 0 0 1.4 0l2.1-2.1a1 1 0 0 0 0-1.4L19 4"/><path d="m21 2-9.6 9.6"/><circle cx="7.5" cy="15.5" r="5.5"/>',
    x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    mic: '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/>',
    message: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    braces: '<path d="M8 3H7a2 2 0 0 0-2 2v5a2 2 0 0 1-2 2 2 2 0 0 1 2 2v5c0 1.1.9 2 2 2h1"/><path d="M16 21h1a2 2 0 0 0 2-2v-5c0-1.1.9-2 2-2a2 2 0 0 1-2-2V5a2 2 0 0 0-2-2h-1"/>'
  };

  function icon(name, cls) {
    return '<svg class="icon' + (cls ? " " + cls : "") + '" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" ' +
      'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">' + (PATHS[name] || "") + "</svg>";
  }

  /* ------------------------------------------------------------------ text and numbers */
  var ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) { return ESC[c]; }); }

  function fixed(x, digits) {
    if (x === null || x === undefined || typeof x !== "number" || !isFinite(x)) return "–";
    return x.toFixed(digits == null ? 3 : digits);
  }
  function fmtP(p) { return fixed(p, 3); }
  function fmtMs(ms) {
    if (typeof ms !== "number" || !isFinite(ms)) return "–";
    return (ms >= 100 ? String(Math.round(ms)) : ms.toFixed(1)) + " ms";
  }
  function formatDate(iso) {
    var m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(iso || ""));
    if (!m) return "";
    return Number(m[3]) + " " + MONTHS[Number(m[2]) - 1] + " " + m[1];
  }
  function plural(n, one, many) { return n + " " + (n === 1 ? one : (many || one + "s")); }

  /* An evidence value as the gallery shows it: accuracies with three decimals, latencies in ms. */
  function fmtEvidence(value, metric) {
    if (typeof value !== "number" || !isFinite(value)) return String(value);
    if (/\bms\b/.test(String(metric || "")) && value >= 1) return Math.round(value) + " ms";
    if (Math.round(value) === value && value > 1) return String(value);
    return value.toFixed(3);
  }

  /* "1 choice", "2 yes/no", "1 score" from a {choice, noul, score} count map. */
  function questionCounts(counts) {
    var out = [];
    [["choice", "choice"], ["noul", "yes/no"], ["score", "score"]].forEach(function (t) {
      var n = counts && counts[t[0]];
      if (n) out.push(n + " " + t[1]);
    });
    return out;
  }

  function proxyBadge() {
    return '<span class="badge-proxy">proxy<span class="visually-hidden">: measured on a related benchmark, not on this schema</span></span>';
  }

  function confidence(ps) {
    var k = ps.length;
    if (k < 2) return 1;
    var m = Math.max.apply(null, ps);
    return Math.min(1, Math.max(0, (k * m - 1) / (k - 1)));
  }

  function el(tag, cls, html) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html !== undefined) n.innerHTML = html;
    return n;
  }

  /* ------------------------------------------------------------------ syntax classes */
  function tokenize(text, rules) {
    var re = new RegExp(rules.map(function (r) { return "(" + r[0] + ")"; }).join("|"), "g");
    var out = "", last = 0, m;
    while ((m = re.exec(text)) !== null) {
      if (m[0] === "") { re.lastIndex += 1; continue; }
      out += esc(text.slice(last, m.index));
      for (var g = 1; g <= rules.length; g++) {
        if (m[g] !== undefined) {
          out += rules[g - 1][1] ? '<span class="' + rules[g - 1][1] + '">' + esc(m[g]) + "</span>" : esc(m[g]);
          break;
        }
      }
      last = re.lastIndex;
    }
    return out + esc(text.slice(last));
  }

  var STR_DQ = '"(?:\\\\.|[^"\\\\\\n])*"';
  var JSON_RULES = [
    [STR_DQ + "(?=\\s*:)", "tok-key"],
    [STR_DQ, "tok-str"],
    ["-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?", "tok-num"],
    ["\\b(?:true|false|null)\\b", "tok-num"],
    ["[{}\\[\\],:]", "tok-punct"]
  ];
  var PY_RULES = [
    ["#[^\\n]*", "tok-com"],
    ["f?" + STR_DQ, "tok-str"],
    ["f?'(?:\\\\.|[^'\\\\\\n])*'", "tok-str"],
    ["\\b(?:True|False|None)\\b", "tok-num"],
    ["\\b\\d+(?:\\.\\d+)?\\b", "tok-num"],
    ["[{}\\[\\]():,=]", "tok-punct"]
  ];
  var SH_RULES = [
    ["#[^\\n]*", "tok-com"],
    ["<<'JSON'", "tok-com"],
    [STR_DQ, "tok-str"],
    ["'[^'\\n]*'", "tok-str"],
    ["\\s--?[A-Za-z][\\w-]*", "tok-punct"],
    ["\\\\$", "tok-punct"]
  ];

  function jsonText(value, indent) { return typeof value === "string" ? value : JSON.stringify(value, null, indent == null ? 2 : indent); }
  function jsonHTML(value, indent) { return tokenize(jsonText(value, indent), JSON_RULES); }
  function pythonHTML(text) { return tokenize(text, PY_RULES); }
  function shellHTML(text) {
    // a command, optionally with a heredoc JSON body between <<'JSON' and a closing JSON line
    var lines = String(text).split("\n"), out = [], body = null;
    lines.forEach(function (line) {
      if (body) {
        if (line === "JSON") { out.push(jsonHTML(body.join("\n"))); out.push('<span class="tok-com">JSON</span>'); body = null; }
        else body.push(line);
        return;
      }
      var prompt = "";
      if (/^\$ /.test(line)) { prompt = '<span class="tok-prompt">$ </span>'; line = line.slice(2); }
      out.push(prompt + tokenize(line, SH_RULES));
      if (/<<'JSON'\s*$/.test(line)) body = [];
    });
    if (body) out.push(jsonHTML(body.join("\n")));
    return out.join("\n");
  }

  /* ------------------------------------------------------------------ Python literals */
  var PY_WIDTH = 88;
  function pyScalar(v) {
    if (v === null || v === undefined) return "None";
    if (v === true) return "True";
    if (v === false) return "False";
    if (typeof v === "number") return isFinite(v) ? String(v) : "None";
    return JSON.stringify(String(v));
  }
  function pyInline(v) {
    if (v === null || typeof v !== "object") return pyScalar(v);
    if (Array.isArray(v)) return "[" + v.map(pyInline).join(", ") + "]";
    return "{" + Object.keys(v).map(function (k) { return JSON.stringify(k) + ": " + pyInline(v[k]); }).join(", ") + "}";
  }
  function pyLiteral(v, level, used) {
    if (v === null || typeof v !== "object") return pyScalar(v);
    level = level || 0;
    var pad = "    ";
    var inner = new Array(level + 2).join(pad), outer = new Array(level + 1).join(pad);
    var inline = pyInline(v);
    if ((used || 0) + inline.length <= PY_WIDTH) return inline;
    if (Array.isArray(v)) {
      if (!v.length) return "[]";
      return "[\n" + v.map(function (x) { return inner + pyLiteral(x, level + 1, inner.length); }).join(",\n") + ",\n" + outer + "]";
    }
    var keys = Object.keys(v);
    if (!keys.length) return "{}";
    return "{\n" + keys.map(function (k) {
      var head = inner + JSON.stringify(k) + ": ";
      return head + pyLiteral(v[k], level + 1, head.length);
    }).join(",\n") + ",\n" + outer + "}";
  }

  /* ------------------------------------------------------------------ request snippets */
  function endpoint(base) { return String(base || DEFAULT_BASE).replace(/\/+$/, "") + "/v1/systemone"; }

  function curlSnippet(request, opts) {
    opts = opts || {};
    var lines = ["curl -sS " + endpoint(opts.base) + " \\", '  -H "Content-Type: application/json" \\'];
    if (opts.auth) lines.push('  -H "Authorization: Bearer $TEZ_API_KEY" \\');
    lines.push("  -d @- <<'JSON'");
    var text = (opts.auth ? "# the server was started with --api-key: export TEZ_API_KEY first\n" : "") +
      lines.join("\n") + "\n" + JSON.stringify(request, null, 2) + "\nJSON";
    return { text: text, html: shellHTML(text) };
  }

  function pythonSnippet(request, opts) {
    opts = opts || {};
    var out = [];
    if (opts.auth) out.push("import os", "");
    out.push("import requests", "");
    out.push("body = " + pyLiteral(request, 0, 7), "");
    if (opts.auth) {
      out.push('headers = {"Authorization": f"Bearer {os.environ[\'TEZ_API_KEY\']}"}');
      out.push('r = requests.post("' + endpoint(opts.base) + '", json=body, headers=headers, timeout=60)');
    } else {
      out.push('r = requests.post("' + endpoint(opts.base) + '", json=body, timeout=60)');
    }
    out.push("r.raise_for_status()");
    out.push('for qid, answer in r.json()["answers"].items():');
    out.push("    print(qid, answer)");
    var text = out.join("\n");
    return { text: text, html: pythonHTML(text) };
  }

  /* ------------------------------------------------------------------ answers */
  function splitHead(text) {
    // "Calm: neutral or positive tone" -> ["Calm", ": neutral or positive tone"]
    var s = String(text == null ? "" : text);
    var i = s.indexOf(": ");
    if (i > 0 && i <= 28) return [s.slice(0, i), s.slice(i)];
    return [s, ""];
  }

  /* Rows in display order: [{key, label, rest, title, p}] for one wire answer. */
  function answerRows(answer, question) {
    var crit = question && question.criteria;
    if (answer.type === "noul") {
      var yes = typeof answer.noul === "number" ? answer.noul : 0;
      return [
        { key: "true", label: "true", title: crit && crit["true"] || "", p: yes },
        { key: "false", label: "false", title: crit && crit["false"] || "", p: 1 - yes }
      ];
    }
    var probs = answer.probabilities || {};
    if (answer.type === "score") {
      var legend = answer.legend || {};
      return Object.keys(probs).map(function (k) {
        var text = legend[k] != null ? legend[k] : (Array.isArray(crit) ? crit[Number(k)] : "");
        var parts = splitHead(text);
        return { key: k, label: k + " " + parts[0], rest: parts[1], title: text || "", p: probs[k] };
      });
    }
    return Object.keys(probs).map(function (k) {
      var desc = crit && !Array.isArray(crit) && crit[k] != null ? crit[k] : "";
      return { key: k, label: k === "__none__" ? "none of these fits" : k, title: k === "__none__" ? "__none__ (abstain)" : desc, p: probs[k] };
    });
  }

  function chosenKey(answer, rows) {
    if (answer.type === "noul") return (answer.noul >= 0.5) ? "true" : "false";
    if (answer.type === "choice" && answer.choice != null) return String(answer.choice);
    var best = null;
    rows.forEach(function (r) { if (best === null || r.p > best.p) best = r; });
    return best ? best.key : null;
  }

  function rowHTML(r) {
    return '<span class="prob-label"' + (r.title ? ' title="' + esc(r.title) + '"' : "") + ">" +
      icon("check", "prob-check") + '<span class="prob-name">' + esc(r.label) + "</span>" +
      (r.rest ? '<span class="visually-hidden">' + esc(r.rest) + "</span>" : "") +
      '<span class="visually-hidden prob-sr"></span></span>' +
      '<span class="prob-bar" aria-hidden="true"><span class="prob-fill"></span></span>' +
      '<span class="prob-value">' + fmtP(r.p) + "</span>";
  }

  /* A list of probability rows that can be updated in place (the replay eases bar widths). */
  function ProbList(ul, rows) {
    this.ul = ul;
    this.rows = {};
    ul.innerHTML = "";
    if (!ul.hasAttribute("role")) ul.setAttribute("role", "list");
    ul.classList.add("prob-list");
    var self = this;
    rows.forEach(function (r) {
      var li = el("li", "prob-row", rowHTML(r));
      li.setAttribute("data-key", r.key);
      li.setAttribute("data-chosen", "false");
      ul.appendChild(li);
      self.rows[r.key] = { li: li, fill: li.querySelector(".prob-fill"), value: li.querySelector(".prob-value"), sr: li.querySelector(".prob-sr") };
    });
  }
  ProbList.prototype.update = function (probs, chosen) {
    var self = this;
    Object.keys(this.rows).forEach(function (k) {
      var row = self.rows[k], p = probs && typeof probs[k] === "number" ? probs[k] : 0;
      var on = chosen !== null && chosen !== undefined && k === String(chosen);
      row.fill.style.setProperty("--p", (Math.max(0, Math.min(1, p)) * 100).toFixed(2) + "%");
      row.value.textContent = probs ? fmtP(p) : "–";
      row.li.setAttribute("data-chosen", on ? "true" : "false");
      row.sr.textContent = on ? ", chosen" : "";
    });
  };

  function metaChips(meta) {
    if (!meta) return "";
    var out = "";
    if (meta.readout) out += '<span class="pill pill-outline" title="Readout used for this answer">' + esc(meta.readout) + "</span>";
    if (meta.decision === "act") out += '<span class="pill pill-act">' + icon("check") + "act</span>";
    else if (meta.decision === "escalate") out += '<span class="pill pill-escalate">' + icon("arrowUpRight") + "escalate</span>";
    return out;
  }

  function kv(label, value, mono) {
    return '<span class="kv"><span class="kv-k">' + esc(label) + '</span> <span class="kv-v' + (mono === false ? "" : " num") + '">' + esc(value) + "</span></span>";
  }

  /* One answer as a block: id and type, the question, probability rows, then the fields that matter. */
  function renderAnswer(qid, answer, opts) {
    opts = opts || {};
    var q = opts.question || null, meta = opts.meta || null;
    var maxRows = opts.maxRows || 6;
    var box = el("section", "answer");
    box.setAttribute("data-type", answer.type || "");
    var typeLabel = answer.type === "noul" ? "yes/no" : answer.type;
    var head = '<div class="answer-head"><code class="answer-id">' + esc(qid) + '</code><span class="pill">' + esc(typeLabel) + "</span>" +
      '<span class="answer-chips">' + metaChips(meta) + "</span></div>";
    var instr = q && q.instructions != null ? (typeof q.instructions === "string" ? q.instructions : JSON.stringify(q.instructions)) : "";
    box.innerHTML = head + (instr ? '<p class="answer-q">' + esc(instr) + "</p>" : "");

    var rows = answerRows(answer, q);
    var chosen = chosenKey(answer, rows);
    var probs = {};
    rows.forEach(function (r) { probs[r.key] = r.p; });
    var shown = rows, rest = [];
    if (answer.type === "choice" && rows.length > maxRows) {
      var sorted = rows.slice().sort(function (a, b) { return b.p - a.p; });
      shown = sorted.slice(0, maxRows - 1);
      rest = sorted.slice(maxRows - 1);
    }
    var ul = el("ul", "prob-list");
    box.appendChild(ul);
    new ProbList(ul, shown).update(probs, chosen);
    if (rest.length) {
      var sum = rest.reduce(function (s, r) { return s + (r.p || 0); }, 0);
      var det = el("details", "prob-more");
      det.innerHTML = "<summary>" + esc(plural(rest.length, "more option")) + ", together " + fmtP(sum) + "</summary>";
      var ul2 = el("ul", "prob-list");
      det.appendChild(ul2);
      new ProbList(ul2, rest).update(probs, chosen);
      box.appendChild(det);
    }

    var foot = [];
    if (answer.type === "noul") foot.push(kv("noul", fmtP(answer.noul)));
    if (answer.type === "choice") foot.push(kv("choice", answer.choice === "__none__" ? "__none__" : String(answer.choice), true));
    if (answer.type === "score") foot.push(kv("score", fixed(answer.score, 2)));
    if (answer.type !== "noul" && typeof answer.confidence === "number") foot.push(kv("confidence", fmtP(answer.confidence)));
    if (meta && typeof meta.p_correct === "number") foot.push(kv("p_correct", fmtP(meta.p_correct)));
    if (meta && meta.calibration_id) foot.push(kv("calibration", meta.calibration_id));
    box.appendChild(el("p", "answer-foot", foot.join("")));
    return box;
  }

  function renderAnswers(container, response, opts) {
    opts = opts || {};
    container.innerHTML = "";
    var wrap = el("div", "answers");
    var answers = (response && response.answers) || {};
    var metas = (response && response.tez && response.tez.questions) || {};
    Object.keys(answers).forEach(function (qid) {
      wrap.appendChild(renderAnswer(qid, answers[qid], {
        question: opts.questions ? opts.questions[qid] : null, meta: metas[qid], maxRows: opts.maxRows
      }));
    });
    container.appendChild(wrap);
    return wrap;
  }

  /* ------------------------------------------------------------------ recorded strips (site/data/strips.json) */
  function stripOptionText(strip, opt) {
    var d = opt.description || "";
    if (strip.question.type === "noul") d = d.replace(opt.key === "true" ? /^yes,\s*/i : /^no,\s*/i, "");
    return d;
  }

  function stripWireQuestion(strip) {
    var q = strip.question, out = { type: q.type, instructions: q.instructions };
    if (q.type === "noul") {
      out.criteria = {};
      ["true", "false"].forEach(function (k) {
        q.options.forEach(function (o) { if (o.key === k) out.criteria[k] = stripOptionText(strip, o) || null; });
      });
    } else if (q.type === "score") {
      out.criteria = q.options.map(function (o) { return o.description || ""; });
    } else {
      out.criteria = {};
      q.options.forEach(function (o) { out.criteria[o.key] = o.description || null; });
    }
    return out;
  }

  function stripRequest(strip) {
    var qs = {};
    qs[strip.question.id] = stripWireQuestion(strip);
    return { model: "tez-latest", state: strip.state, questions: qs };
  }

  function round(x, d) { var f = Math.pow(10, d); return Math.round(x * f) / f; }

  /* The wire answer for one trace point: exactly Jev's shapes. */
  function stripAnswer(strip, probs) {
    var q = strip.question, keys = q.options.map(function (o) { return o.key; });
    var ps = keys.map(function (k) { return typeof probs[k] === "number" ? probs[k] : 0; });
    if (q.type === "noul") return { type: "noul", noul: typeof probs["true"] === "number" ? probs["true"] : 0 };
    var p = {};
    keys.forEach(function (k, i) { p[k] = ps[i]; });
    var arg = 0;
    ps.forEach(function (v, i) { if (v > ps[arg]) arg = i; });
    if (q.type === "choice") return { type: "choice", choice: keys[arg], probabilities: p, confidence: round(confidence(ps), 4) };
    var legend = {};
    q.options.forEach(function (o, i) { legend[String(i)] = o.description || ""; });
    var score = ps.reduce(function (s, v, i) { return s + i * v; }, 0);
    return { type: "score", score: round(score, 3), legend: legend, probabilities: p, confidence: round(confidence(ps), 4) };
  }

  /* ------------------------------------------------------------------ replay engine */
  function noop() {}
  function reducedMotion() {
    return !!(root.matchMedia && root.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }

  /* Plays steps 0..n-1 at a reading pace. Callbacks: onStep(i), onReset(), onState(state, i), onDone().
     States: idle, playing, paused, done. Under reduced motion play() jumps to the final step. */
  function Replay(opts) {
    this.n = opts.steps;
    this.pace = opts.pace || 280;
    this.lead = opts.lead == null ? 400 : opts.lead;
    this.onStep = opts.onStep || noop;
    this.onReset = opts.onReset || noop;
    this.onState = opts.onState || noop;
    this.onDone = opts.onDone || noop;
    this.i = -1;
    this.state = "idle";
    this.timer = 0;
  }
  Replay.prototype._set = function (s) { this.state = s; this.onState(s, this.i); };
  Replay.prototype._clear = function () { if (this.timer) { clearTimeout(this.timer); this.timer = 0; } };
  Replay.prototype._schedule = function (ms) {
    var self = this;
    this._clear();
    this.timer = setTimeout(function () { self.timer = 0; self._tick(); }, ms);
  };
  Replay.prototype._tick = function () {
    if (this.state !== "playing") return;
    this.i += 1;
    this.onStep(this.i);
    if (this.i >= this.n - 1) { this._set("done"); this.onDone(); return; }
    this._schedule(this.pace);
  };
  Replay.prototype.play = function () {
    if (reducedMotion()) { this.finish(); return; }
    if (this.state === "playing") return;
    if (this.state === "done" || this.i >= this.n - 1) { this.i = -1; this.onReset(); }
    this._set("playing");
    this._schedule(this.i < 0 ? this.lead : this.pace);
  };
  Replay.prototype.pause = function () {
    if (this.state !== "playing") return;
    this._clear();
    this._set("paused");
  };
  Replay.prototype.toggle = function () { if (this.state === "playing") this.pause(); else this.play(); };
  Replay.prototype.restart = function () {
    this._clear();
    this.i = -1;
    this.state = "idle";
    if (reducedMotion()) { this.finish(); return; }
    this.onReset();
    this.play();
  };
  Replay.prototype.seek = function (i) {
    this._clear();
    this.i = Math.max(0, Math.min(this.n - 1, i));
    this.onStep(this.i);
    if (this.i >= this.n - 1) { this._set("done"); this.onDone(); }
    else this._set("paused");
  };
  Replay.prototype.finish = function () { this.seek(this.n - 1); };
  Replay.prototype.stop = function () { this._clear(); this.state = "idle"; };

  /* ------------------------------------------------------------------ misc */
  function getJSON(url) {
    if (!root.fetch) return Promise.resolve(null);
    return root.fetch(url, { cache: "no-cache" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; });
  }

  var SRC_RE = /\b((?:results|experiments|docs|data|examples|tez|scripts)\/[\w.\/-]*[\w\/]|BENCHMARKS\.md|README\.md)/g;
  function linkSources(text) {
    return esc(text).replace(SRC_RE, function (m) { return '<a href="' + BLOB + m + '">' + m + "</a>"; });
  }

  /* ARIA tabs for tablists created after site.js ran. Returns select(tab, focus). */
  function wireTabs(list, onSelect) {
    var tabs = Array.prototype.slice.call(list.querySelectorAll('[role="tab"]'));
    var vertical = list.getAttribute("aria-orientation") === "vertical";
    function select(tab, focus) {
      tabs.forEach(function (t) {
        var on = t === tab;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.setAttribute("tabindex", on ? "0" : "-1");
        var panel = document.getElementById(t.getAttribute("aria-controls"));
        if (panel) panel.hidden = !on;
      });
      if (focus) tab.focus();
      if (onSelect) onSelect(tab);
    }
    tabs.forEach(function (t, i) {
      t.addEventListener("click", function () { select(t, false); });
      t.addEventListener("keydown", function (e) {
        var next = vertical ? "ArrowDown" : "ArrowRight", prev = vertical ? "ArrowUp" : "ArrowLeft", j = null;
        if (e.key === next) j = (i + 1) % tabs.length;
        if (e.key === prev) j = (i - 1 + tabs.length) % tabs.length;
        if (e.key === "Home") j = 0;
        if (e.key === "End") j = tabs.length - 1;
        if (j !== null) { e.preventDefault(); select(tabs[j], true); }
      });
    });
    return select;
  }

  function stateText(state) { return typeof state === "string" ? state : JSON.stringify(state, null, 2); }

  root.TezUI = {
    REPO: REPO,
    BLOB: BLOB,
    DEFAULT_BASE: DEFAULT_BASE,
    icon: icon,
    esc: esc,
    el: el,
    fixed: fixed,
    fmtP: fmtP,
    fmtMs: fmtMs,
    formatDate: formatDate,
    plural: plural,
    fmtEvidence: fmtEvidence,
    questionCounts: questionCounts,
    proxyBadge: proxyBadge,
    confidence: confidence,
    tokenize: tokenize,
    jsonText: jsonText,
    jsonHTML: jsonHTML,
    pythonHTML: pythonHTML,
    shellHTML: shellHTML,
    pyLiteral: pyLiteral,
    endpoint: endpoint,
    curlSnippet: curlSnippet,
    pythonSnippet: pythonSnippet,
    answerRows: answerRows,
    chosenKey: chosenKey,
    ProbList: ProbList,
    renderAnswer: renderAnswer,
    renderAnswers: renderAnswers,
    stripWireQuestion: stripWireQuestion,
    stripRequest: stripRequest,
    stripAnswer: stripAnswer,
    Replay: Replay,
    reducedMotion: reducedMotion,
    getJSON: getJSON,
    linkSources: linkSources,
    wireTabs: wireTabs,
    stateText: stateText
  };
})(typeof window !== "undefined" ? window : globalThis);
