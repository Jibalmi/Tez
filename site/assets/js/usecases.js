/* Use-case gallery: search and filter over data/usecases.json, and a detail view routed by the hash
   (usecases.html#<id>) built from data/usecases/<id>.json, with recorded answers from data/replays.json when that
   file exists. Nothing here runs a model: answers shown are recorded tez serve responses. */
(function () {
  "use strict";

  var UI = window.TezUI;
  if (!UI) return;

  var TREE = UI.REPO + "/tree/main/examples/usecases/";
  var BACKEND = "http://127.0.0.1:8091";
  var PRIORITY = ["message", "text", "question", "task", "evidence"];

  var el = {
    index: document.querySelector("[data-uc-index]"),
    detail: document.querySelector("[data-uc-detail]"),
    list: document.querySelector("[data-uc-list]"),
    search: document.querySelector("[data-uc-search]"),
    modes: document.querySelector("[data-uc-modes]"),
    count: document.querySelector("[data-uc-count]"),
    empty: document.querySelector("[data-uc-empty]"),
    clear: document.querySelector("[data-uc-clear]")
  };
  if (!el.index || !el.detail) return;

  var items = [], byId = {}, filter = { q: "", mode: "" };
  var details = {}, replaysPromise = null, shownId = null, returnTo = null, baseTitle = document.title;

  /* ================================================================ data */
  function loadDetail(id) {
    if (!details[id]) details[id] = UI.getJSON("./data/usecases/" + encodeURIComponent(id) + ".json");
    return details[id];
  }

  function loadReplays() {
    if (!replaysPromise) {
      replaysPromise = UI.getJSON("./data/replays.json").then(function (data) {
        if (!data || !Array.isArray(data.replays)) return null;
        var index = {};
        data.replays.forEach(function (r) { index[r.usecase + ":" + r.sample_index] = r; });
        return { meta: data.meta || {}, index: index };
      });
    }
    return replaysPromise;
  }

  /* ================================================================ index */
  function cap(s) { s = String(s || ""); return s.charAt(0).toUpperCase() + s.slice(1); }

  function buildModes() {
    var order = [], counts = {};
    items.forEach(function (u) {
      var m = u.recommended_mode || "other";
      if (!(m in counts)) { order.push(m); counts[m] = 0; }
      counts[m] += 1;
    });
    var html = modeButton("", "All", items.length);
    order.forEach(function (m) { html += modeButton(m, cap(m), counts[m]); });
    el.modes.innerHTML = html;
  }

  function modeButton(value, label, count) {
    return '<button type="button" aria-pressed="' + (filter.mode === value ? "true" : "false") + '" data-mode="' + UI.esc(value) + '">' +
      UI.esc(label) + ' <span class="count">' + count + "</span></button>";
  }

  function matches(u) {
    if (filter.mode && u.recommended_mode !== filter.mode) return false;
    var terms = filter.q.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) return true;
    var hay = [u.id, u.title, u.job, u.recommended_mode, u.why_this_mode, u.limits].join(" ").toLowerCase();
    // each term must start a word, so "voice" does not match "invoice"
    return terms.every(function (t) {
      return new RegExp("(^|[^a-z0-9])" + t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).test(hay);
    });
  }

  function evidenceHTML(ev, cls) {
    if (!ev) return "";
    return '<span class="' + cls + '-value">' + UI.esc(UI.fmtEvidence(ev.value, ev.metric)) + "</span>" +
      '<span class="' + cls + '-body"><span class="' + cls + '-metric">' + UI.esc(ev.metric) + (ev.proxy ? " " + UI.proxyBadge() : "") + "</span>" +
      '<span class="source">' + UI.linkSources(ev.source || "") + "</span></span>";
  }

  function tagsHTML(u) {
    return '<span class="pill pill-accent">' + UI.esc(u.recommended_mode || "") + "</span>" +
      UI.questionCounts(u.questions).map(function (t) { return '<span class="pill">' + UI.esc(t) + "</span>"; }).join("");
  }

  function itemHTML(u) {
    return '<li class="uc-item" id="item-' + UI.esc(u.id) + '">' +
      '<div class="uc-item-main">' +
      '<h2 class="uc-item-title"><a href="#' + encodeURIComponent(u.id) + '">' + UI.esc(u.title) + "</a></h2>" +
      '<p class="uc-item-job">' + UI.esc(u.job || "") + "</p>" +
      '<div class="uc-tags">' + tagsHTML(u) + "</div>" +
      (u.limits ? '<p class="uc-item-limits"><strong>Limits.</strong> ' + UI.esc(u.limits) + "</p>" : "") +
      "</div>" +
      '<div class="uc-item-evidence">' + evidenceHTML((u.evidence || [])[0], "ev") + "</div></li>";
  }

  function renderList() {
    var shown = items.filter(matches);
    el.list.innerHTML = shown.map(itemHTML).join("");
    el.empty.hidden = shown.length > 0;
    el.list.hidden = shown.length === 0;
    el.count.textContent = shown.length === items.length ? UI.plural(items.length, "use case") : "Showing " + shown.length + " of " + UI.plural(items.length, "use case");
  }

  function bindIndex() {
    el.search.addEventListener("input", function () { filter.q = el.search.value; renderList(); });
    el.modes.addEventListener("click", function (e) {
      var b = e.target.closest("[data-mode]");
      if (!b) return;
      filter.mode = b.getAttribute("data-mode");
      Array.prototype.forEach.call(el.modes.querySelectorAll("[data-mode]"), function (x) {
        x.setAttribute("aria-pressed", x === b ? "true" : "false");
      });
      renderList();
    });
    el.clear.addEventListener("click", function () {
      filter.q = "";
      filter.mode = "";
      el.search.value = "";
      Array.prototype.forEach.call(el.modes.querySelectorAll("[data-mode]"), function (x) {
        x.setAttribute("aria-pressed", x.getAttribute("data-mode") === "" ? "true" : "false");
      });
      renderList();
      el.search.focus();
    });
  }

  /* ================================================================ routing */
  function hashId() {
    try { return decodeURIComponent(location.hash.replace(/^#/, "")); } catch (e) { return ""; }
  }

  function route() {
    var id = hashId();
    if (id && byId[id]) showDetail(id);
    else showIndex(id);
  }

  function showIndex(unknown) {
    var wasDetail = shownId !== null;
    shownId = null;
    el.detail.hidden = true;
    el.detail.innerHTML = "";
    el.index.hidden = false;
    document.title = baseTitle;
    if (unknown) el.count.textContent = "There is no use case called “" + unknown + "”. Showing all " + UI.plural(items.length, "use case") + ".";
    if (wasDetail && returnTo) {
      var link = document.querySelector("#item-" + cssId(returnTo) + " .uc-item-title a");
      if (link) {
        link.focus({ preventScroll: true });
        link.scrollIntoView({ block: "center" });
      }
    }
    returnTo = null;
  }

  function cssId(id) { return window.CSS && CSS.escape ? CSS.escape(id) : id.replace(/[^\w-]/g, "\\$&"); }

  function showDetail(id) {
    if (shownId === id) return;
    shownId = id;
    returnTo = id;
    var u = byId[id];
    el.index.hidden = true;
    el.detail.hidden = false;
    document.title = u.title + " · Tez use cases";
    el.detail.innerHTML = skeleton(u);
    window.scrollTo(0, 0);
    var h1 = el.detail.querySelector("h1");
    if (h1) h1.focus({ preventScroll: true });
    el.detail.querySelector("[data-uc-back]").addEventListener("click", function (e) {
      e.preventDefault();
      history.pushState(null, "", location.pathname + location.search);
      route();
    });
    Promise.all([loadDetail(id), loadReplays()]).then(function (res) {
      if (shownId !== id) return;
      fill(u, res[0], res[1]);
    });
  }

  /* ================================================================ detail */
  function skeleton(u) {
    var ev = (u.evidence || []).map(function (e) { return '<li class="ev">' + evidenceHTML(e, "ev") + "</li>"; }).join("");
    var anyProxy = (u.evidence || []).some(function (e) { return e.proxy; });
    return '<nav class="uc-back" aria-label="Breadcrumb"><a href="./usecases.html" data-uc-back>' + UI.icon("arrowLeft") + "All use cases</a></nav>" +
      '<header class="uc-d-head">' +
      '<h1 tabindex="-1">' + UI.esc(u.title) + "</h1>" +
      '<p class="lead">' + UI.esc(u.job || "") + "</p>" +
      '<div class="uc-tags">' + tagsHTML(u) + "</div>" +
      '<div class="uc-d-actions">' +
      '<a class="btn btn-primary" href="./playground.html?usecase=' + encodeURIComponent(u.id) + '&amp;sample=0">' + UI.icon("play") + "Open in playground</a>" +
      '<a class="btn btn-secondary" href="' + TREE + encodeURIComponent(u.id) + '">' + UI.icon("external") + "Files on GitHub</a>" +
      "</div></header>" +
      '<div class="uc-d-grid">' +
      '<div class="uc-d-main">' +
      '<section class="uc-d-section" aria-labelledby="d-decides"><h2 id="d-decides">What it decides</h2><div data-d-decides><p class="uc-status">Loading the schema.</p></div></section>' +
      '<section class="uc-d-section" aria-labelledby="d-samples"><h2 id="d-samples">Sample inputs</h2><div data-d-samples></div></section>' +
      '<section class="uc-d-section" aria-labelledby="d-schema"><h2 id="d-schema">Schema</h2><div data-d-schema></div></section>' +
      "</div>" +
      '<aside class="uc-d-side" aria-label="Recommendation, evidence and limits">' +
      '<section class="side-block"><h2 class="h3">Recommended mode</h2><p><span class="pill pill-accent">' + UI.esc(u.recommended_mode || "") + '</span></p><p class="side-text">' + UI.esc(u.why_this_mode || "") + "</p></section>" +
      '<section class="side-block"><h2 class="h3">Evidence</h2><ul class="ev-list" role="list">' + ev + "</ul>" +
      (anyProxy ? '<p class="side-note"><span class="badge-proxy">proxy</span> marks evidence measured on a related benchmark or protocol, not on labelled data for this schema.</p>' : "") +
      "</section>" +
      '<section class="side-block"><h2 class="h3">Limits</h2><p class="side-text">' + UI.esc(u.limits || "") + "</p></section>" +
      "</aside></div>";
  }

  function fill(u, d, rp) {
    var decides = el.detail.querySelector("[data-d-decides]");
    var samples = el.detail.querySelector("[data-d-samples]");
    var schema = el.detail.querySelector("[data-d-schema]");
    if (!d || !d.schema) {
      var link = '<a href="' + TREE + encodeURIComponent(u.id) + '">examples/usecases/' + UI.esc(u.id) + "</a>";
      decides.innerHTML = '<p class="uc-status">The schema did not load. It is in the repository under ' + link + ".</p>";
      samples.innerHTML = "";
      schema.innerHTML = "";
      return;
    }
    decides.innerHTML = decidesHTML(d.schema);
    schema.innerHTML = '<div class="code code-pane schema-code"><div class="code-head"><span class="code-title">examples/usecases/' + UI.esc(u.id) + '/schema.yaml</span>' +
      copyButton() + '</div><pre tabindex="0" aria-label="Schema YAML"><code>' + yamlHTML(d.yaml || "") + "</code></pre></div>";
    renderSamples(u, d, rp, samples);
  }

  function copyButton() {
    return '<button class="copy-btn" type="button" data-copy>' + UI.icon("copy") + "<span data-copy-label>Copy</span></button>";
  }

  function typeLine(q) {
    if (q.type === "noul") return "yes/no";
    if (q.type === "score") return "score · " + UI.plural((q.criteria || []).length, "level");
    return "choice · " + UI.plural(Object.keys(q.criteria || {}).length, "option");
  }

  function optionPairs(q) {
    if (q.type === "noul") {
      var c = q.criteria || {};
      return [["true", c["true"] || ""], ["false", c["false"] || ""]];
    }
    if (q.type === "score") return (q.criteria || []).map(function (t, i) { return [String(i), t]; });
    return Object.keys(q.criteria || {}).map(function (k) { return [k, q.criteria[k] || ""]; });
  }

  function decidesHTML(s) {
    var out = "";
    if (s.description) out += '<p class="d-desc">' + UI.esc(s.description) + "</p>";
    if (s.state) out += '<p class="d-state"><strong>Input.</strong> ' + UI.esc(s.state) + "</p>";
    out += '<div class="qdefs">';
    Object.keys(s.questions || {}).forEach(function (qid) {
      var q = s.questions[qid], pairs = optionPairs(q);
      var instr = typeof q.instructions === "string" ? q.instructions : JSON.stringify(q.instructions);
      var rows = function (list) {
        return list.map(function (p) { return "<div><dt>" + UI.esc(p[0]) + "</dt><dd>" + UI.esc(p[1]) + "</dd></div>"; }).join("");
      };
      out += '<div class="qdef"><div class="qdef-head"><code>' + UI.esc(qid) + '</code><span class="pill">' + UI.esc(typeLine(q)) + "</span></div>" +
        '<p class="qdef-instr">' + UI.esc(instr) + "</p>";
      if (pairs.length > 12) {
        out += '<dl class="qdef-opts">' + rows(pairs.slice(0, 8)) + "</dl>" +
          '<details class="qdef-more"><summary>Show all ' + pairs.length + " options</summary>" +
          '<dl class="qdef-opts">' + rows(pairs.slice(8)) + "</dl></details>";
      } else {
        out += '<dl class="qdef-opts">' + rows(pairs) + "</dl>";
      }
      out += "</div>";
    });
    out += "</div>";
    if (s.gate && typeof s.gate.alpha === "number") {
      out += '<p class="d-gate">Default gate: <code>alpha ' + s.gate.alpha + "</code>. It acts only on a schema fitted with <code>tez fit</code>; until then every answer is marked <code>escalate</code>.</p>";
    }
    return out;
  }

  /* ---------------------------------------------------------------- samples */
  function preview(state) {
    if (typeof state === "string") return state;
    if (state && typeof state === "object" && !Array.isArray(state)) {
      for (var i = 0; i < PRIORITY.length; i++) {
        if (typeof state[PRIORITY[i]] === "string") return state[PRIORITY[i]];
      }
    }
    return JSON.stringify(state);
  }

  function sameState(a, b) { return JSON.stringify(a) === JSON.stringify(b); }

  function sameQuestions(req, schema) {
    var a = Object.keys((req && req.questions) || {}).sort().join("|");
    var b = Object.keys((schema && schema.questions) || {}).sort().join("|");
    return a === b;
  }

  function renderSamples(u, d, rp, host) {
    var list = d.samples || [];
    if (!list.length) { host.innerHTML = '<p class="uc-status">This use case has no samples yet.</p>'; return; }
    var any = false;
    var tabs = "", panels = "";
    list.forEach(function (smp, i) {
      var rec = rp && rp.index[u.id + ":" + i];
      var fresh = rec && sameState(rec.state, smp.state) && sameQuestions(rec.request, d.schema);
      if (fresh) any = true;
      tabs += '<button type="button" role="tab" class="sample-tab" id="s-tab-' + i + '" aria-controls="s-panel-' + i + '" aria-selected="' + (i === 0) + '"' + (i ? ' tabindex="-1"' : "") + ">" +
        '<span class="t-num">Sample ' + (i + 1) + (typeof smp.state === "string" ? "" : ' <span class="t-kind">JSON</span>') + "</span>" +
        '<span class="t-prev">' + UI.esc(preview(smp.state)) + "</span></button>";
      panels += '<div class="sample-panel" id="s-panel-' + i + '" role="tabpanel" aria-labelledby="s-tab-' + i + '"' + (i ? " hidden" : "") + ">" +
        panelHTML(u, d, smp, i, fresh ? rec : null, rec && !fresh, rp) + "</div>";
    });
    var heading = el.detail.querySelector("#d-samples");
    if (heading) heading.textContent = any ? "Samples and recorded answers" : "Sample inputs";
    host.innerHTML = (any ? "" : '<p class="d-intro">No recorded responses are published for these samples yet. Each one shows the command that decides it on your machine.</p>') +
      '<div class="samples"><div class="sample-tabs" role="tablist" aria-orientation="vertical" aria-label="Samples">' + tabs + "</div>" +
      '<div class="sample-panels">' + panels + "</div></div>";
    UI.wireTabs(host.querySelector('[role="tablist"]'));
    host.querySelectorAll("[data-answers]").forEach(function (box) {
      var rec = rp.index[box.getAttribute("data-answers")];
      if (rec) UI.renderAnswers(box, rec.response, { questions: d.schema.questions, maxRows: 8 });
    });
  }

  function stateHTML(state) {
    if (typeof state === "string") return '<p class="state-text">' + UI.esc(state) + "</p>";
    return '<div class="code code-pane state-code"><pre tabindex="0" aria-label="Input JSON"><code>' + UI.jsonHTML(state) + "</code></pre></div>";
  }

  function panelHTML(u, d, smp, i, rec, stale, rp) {
    var out = '<div class="sample-block"><h3 class="h4">Input</h3>' + stateHTML(smp.state) +
      (smp.note ? '<p class="sample-note">' + UI.esc(cap(smp.note)) + ".</p>" : "") + "</div>";
    if (rec) {
      var m = rp.meta || {}, resp = rec.response || {}, tz = resp.tez || {}, usage = resp.usage || {};
      var bits = [];
      if (typeof rec.client_ms === "number") bits.push("client round trip " + UI.fmtMs(rec.client_ms));
      if (typeof tz.latency_ms === "number") bits.push("tez.latency_ms " + tz.latency_ms);
      if (typeof usage.input_tokens === "number") bits.push(usage.input_tokens + " input tokens");
      out += '<div class="sample-block"><div class="sample-answers-head"><h3 class="h4">Recorded answers</h3>' +
        '<span class="origin origin-recorded">' + UI.icon("history") + "Recorded response</span></div>" +
        '<p class="sample-meta">Recorded ' + UI.esc(UI.formatDate(m.recorded) || "") + " from <code>tez serve</code>" +
        (m.model ? ", " + UI.esc(m.model) : "") + (m.hardware ? ", " + UI.esc(m.hardware) : "") + ". " +
        (bits.length ? '<span class="num">' + UI.esc(bits.join(" · ")) + "</span>." : "") + "</p>" +
        '<div class="sample-answers" data-answers="' + UI.esc(u.id + ":" + i) + '"></div>' +
        '<details class="sample-json"><summary>Response JSON</summary><div class="code code-pane"><div class="code-head"><span class="code-title">' +
        UI.esc(resp.model || "response") + "</span>" + copyButton() + '</div><pre tabindex="0" aria-label="Response JSON"><code>' + UI.jsonHTML(resp) + "</code></pre></div></details></div>";
    } else {
      out += '<div class="sample-block">' + runLocally(u, smp, stale) + "</div>";
    }
    out += '<p class="sample-actions"><a class="btn btn-secondary btn-sm" href="./playground.html?usecase=' + encodeURIComponent(u.id) + "&amp;sample=" + i + '">' +
      UI.icon("play") + "Open this sample in the playground</a></p>";
    return out;
  }

  function shellQuote(s) { return "'" + String(s).replace(/'/g, "'\\''") + "'"; }

  function runLocally(u, smp, stale) {
    var schemaPath = "examples/usecases/" + u.id + "/schema.yaml";
    var lines;
    if (typeof smp.state === "string") {
      lines = ["$ tez decide --backend " + BACKEND + " --schema " + schemaPath + " --state " + shellQuote(smp.state)];
    } else {
      lines = ["$ cat > state.json <<'JSON'", JSON.stringify(smp.state, null, 2), "JSON",
        "$ tez decide --backend " + BACKEND + " --schema " + schemaPath + " --state-file state.json"];
    }
    return '<h3 class="h4">Run it locally</h3>' +
      '<p class="sample-meta">' + (stale ? "The published recording is for an earlier version of this sample, so it is not shown. " : "No recorded response for this sample yet. ") +
      "From a clone of the repository, with <code>llama-server</code> on port 8091:</p>" +
      '<div class="code code-pane run-code"><div class="code-head"><span class="code-title">POSIX shell</span>' + copyButton() +
      '</div><pre tabindex="0" aria-label="Command"><code>' + UI.shellHTML(lines.join("\n")) + "</code></pre></div>";
  }

  /* ---------------------------------------------------------------- YAML highlighting */
  function splitComment(line) {
    var quote = null;
    for (var i = 0; i < line.length; i++) {
      var c = line.charAt(i);
      if (quote) {
        if (c === "\\" && quote === '"') { i += 1; continue; }
        if (c === quote) quote = null;
        continue;
      }
      if (c === '"' || c === "'") { quote = c; continue; }
      if (c === "#" && (i === 0 || /\s/.test(line.charAt(i - 1)))) return [line.slice(0, i), line.slice(i)];
    }
    return [line, ""];
  }

  function yamlValue(v) {
    var m = /^(\s*)(.*?)(\s*)$/.exec(v);
    if (!m[2]) return UI.esc(v);
    var core = m[2], html;
    if (/^(-?\d+(\.\d+)?|true|false|null|~)$/.test(core)) html = '<span class="tok-num">' + UI.esc(core) + "</span>";
    else if (/^\[.*\]$/.test(core)) html = UI.tokenize(core, [['"(?:\\\\.|[^"\\\\])*"', "tok-str"], ["[\\[\\],]", "tok-punct"], ["[^\\[\\],\\s][^\\[\\],]*[^\\[\\],\\s]|[^\\[\\],\\s]", "tok-str"]]);
    else html = '<span class="tok-str">' + UI.esc(core) + "</span>";
    return UI.esc(m[1]) + html + UI.esc(m[3]);
  }

  function yamlLine(line) {
    var parts = splitComment(line), body = parts[0], com = parts[1], out;
    var m = /^(\s*)(- )?("(?:[^"\\]|\\.)*"|'[^']*'|[^\s:"'#-][^:]*?)(:)(\s.*|$)/.exec(body);
    if (m) {
      out = UI.esc(m[1]) + (m[2] ? '<span class="tok-punct">- </span>' : "") + '<span class="tok-key">' + UI.esc(m[3]) + '</span><span class="tok-punct">:</span>' + yamlValue(m[5]);
    } else {
      var li = /^(\s*)(- )(.*)$/.exec(body);
      out = li ? UI.esc(li[1]) + '<span class="tok-punct">- </span>' + yamlValue(li[3]) : UI.esc(body);
    }
    return out + (com ? '<span class="tok-com">' + UI.esc(com) + "</span>" : "");
  }

  function yamlHTML(text) { return String(text).replace(/\s+$/, "").split("\n").map(yamlLine).join("\n"); }

  /* ================================================================ start */
  UI.getJSON("./data/usecases.json").then(function (data) {
    if (!Array.isArray(data) || !data.length) {
      el.list.innerHTML = '<li class="uc-status">The use cases did not load. They are also in the repository under <a href="' + TREE + '">examples/usecases</a>.</li>';
      return;
    }
    items = data;
    items.forEach(function (u) { byId[u.id] = u; });
    buildModes();
    renderList();
    bindIndex();
    route();
    window.addEventListener("hashchange", route);
    window.addEventListener("popstate", route);
  });
})();
