/* Playground: an operate surface for a Tez server on the visitor's machine.
   Connect (GET /healthz, then /v1/models and /v1/schemas), build a POST /v1/systemone request (use case, sample,
   state, questions validated live against the rules in docs/API.md, Tez options), send it, and read the answers,
   the raw JSON and the same request as curl and Python. Without a connection, unedited site samples show the
   recorded tez serve responses from data/replays.json, labelled as recorded and never styled as live. */
(function () {
  "use strict";

  var UI = window.TezUI;
  if (!UI) return;

  var LLAMA_CMD = "llama-server -m gemma-4-12b-it-Q8_0.gguf -ngl 99 -c 4096 --port 8091 --embeddings --pooling last --swa-full";
  var SERVE_CMD = "tez serve --backend http://127.0.0.1:8091 --template gemma4";
  var STORE = { base: "tez-playground-base", key: "tez-playground-key", remember: "tez-playground-remember" };
  var SEND_TIMEOUT = 120000, CONNECT_TIMEOUT = 8000;
  var PRIORITY = ["message", "text", "question", "task", "evidence"];
  var DEFAULT_QUESTIONS = {
    is_urgent: { type: "noul", instructions: "Does this convey urgency?", criteria: { "true": "Explicitly time-sensitive", "false": "No urgency expressed" } },
    topic: { type: "choice", instructions: "What is the message about?", criteria: { billing: "Payments, payouts, invoices", technical: "Something is broken", sales: null } },
    anger: { type: "score", instructions: "How upset is the writer?", criteria: ["Calm", "Frustrated", "Very angry"] }
  };

  function $(sel) { return document.querySelector(sel); }
  var el = {
    form: $("[data-conn-form]"), base: $("[data-base]"), key: $("[data-key]"), connect: $("[data-connect]"),
    connectLabel: $("[data-connect-label]"), disconnect: $("[data-disconnect]"), remember: $("[data-remember]"),
    status: $("[data-status]"),
    usecase: $("[data-usecase]"), sample: $("[data-sample]"), state: $("[data-state]"), stateMode: $("[data-state-mode]"),
    stateMsg: $("[data-state-msg]"), questions: $("[data-questions]"), qMsg: $("[data-q-msg]"), format: $("[data-format]"),
    readout: $("[data-readout]"), alpha: $("[data-alpha]"), abstain: $("[data-abstain]"), tezMsg: $("[data-tez-msg]"),
    send: $("[data-send]"), sendLabel: $("[data-send-label]"), sendNote: $("[data-send-note]"),
    origin: $("[data-origin]"), banner: $("[data-banner]"), meta: $("[data-meta]"), out: $("[data-out]"), live: $("[data-live]"),
    json: $("[data-json]"), jsonTitle: $("[data-json-title]"), curl: $("[data-curl]"), python: $("[data-python]")
  };
  if (!el.form || !el.out) return;

  var S = {
    conn: { status: "idle", base: "", health: null, schemas: [], note: "" },
    site: [],
    siteDetail: {},
    detail: null,          // data/usecases/<id>.json of the selected site use case
    serverSchema: null,    // GET /v1/schemas/{name} of the selected server schema
    replays: null,         // {meta, index} or null when data/replays.json is not published
    source: { kind: "custom" },
    stateMode: "text",
    busy: false,
    shown: null,           // what the response panel shows: {origin: "live"|"replay", key}
    timer: 0
  };

  /* ================================================================ small helpers */
  var esc = UI.esc;
  function load(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function save(k, v) {
    try { if (v === null || v === undefined || v === "") localStorage.removeItem(k); else localStorage.setItem(k, v); } catch (e) { /* storage off: the page still works */ }
  }
  function isLocalPage() {
    return location.protocol === "file:" || ["localhost", "127.0.0.1", "[::1]", "::1"].indexOf(location.hostname) >= 0;
  }
  function isLoopback(url) { return /^https?:\/\/(localhost|127\.\d+\.\d+\.\d+|\[::1\])(:\d+)?(\/|$)/i.test(url); }
  function normBase(v) {
    v = String(v || "").trim().replace(/\/+$/, "");
    if (!v) return "";
    if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(v)) v = "http://" + v;
    return /^https?:\/\/[^\s/]+/i.test(v) ? v : "";
  }
  function host(url) { return String(url || "").replace(/^https?:\/\//i, ""); }
  function canon(v) {
    if (Array.isArray(v)) return "[" + v.map(canon).join(",") + "]";
    if (v && typeof v === "object") return "{" + Object.keys(v).sort().map(function (k) { return JSON.stringify(k) + ":" + canon(v[k]); }).join(",") + "}";
    return JSON.stringify(v);
  }
  function preview(state) {
    if (typeof state === "string") return state;
    if (state && typeof state === "object" && !Array.isArray(state)) {
      for (var i = 0; i < PRIORITY.length; i++) if (typeof state[PRIORITY[i]] === "string") return state[PRIORITY[i]];
    }
    return JSON.stringify(state);
  }
  function clip(s, n) { s = String(s); return s.length > n ? s.slice(0, n - 1).replace(/\s+\S*$/, "") + "…" : s; }
  function cmdBlock(title, lines) {
    return '<div class="code cmds"><div class="code-head"><span class="code-title">' + esc(title) + "</span>" + copyButton() + "</div><pre><code>" +
      lines.map(function (l) { return '<span class="cmd"><span class="tok-prompt">$ </span>' + esc(l) + "</span>"; }).join("") + "</code></pre></div>";
  }
  function copyButton() { return '<button class="copy-btn" type="button" data-copy>' + UI.icon("copy") + "<span data-copy-label>Copy</span></button>"; }
  function isConnected() { return S.conn.status === "ok" || S.conn.status === "degraded"; }

  /* ================================================================ HTTP */
  function call(method, url, body, opts) {
    opts = opts || {};
    var ctrl = window.AbortController ? new AbortController() : null;
    var timer = ctrl ? setTimeout(function () { ctrl.abort(); }, opts.timeout || SEND_TIMEOUT) : 0;
    var headers = {};
    if (body !== undefined) headers["Content-Type"] = "application/json";
    var key = el.key.value.trim();
    if (opts.auth !== false && key) headers.Authorization = "Bearer " + key;
    var t0 = window.performance ? performance.now() : Date.now();
    function now() { return (window.performance ? performance.now() : Date.now()) - t0; }
    return fetch(url, {
      method: method, headers: headers, cache: "no-store",
      body: body === undefined ? undefined : JSON.stringify(body), signal: ctrl ? ctrl.signal : undefined
    }).then(function (r) {
      var ms = now();
      return r.text().then(function (text) {
        clearTimeout(timer);
        var data = null;
        try { data = text ? JSON.parse(text) : null; } catch (e) { data = null; }
        return { kind: "http", ok: r.ok, status: r.status, data: data, text: text, ms: ms };
      });
    }, function (err) {
      clearTimeout(timer);
      return { kind: err && err.name === "AbortError" ? "timeout" : "network", ok: false, status: 0, ms: now() };
    });
  }

  function errorOf(r) {
    var e = (r.data && r.data.error) || {};
    return { type: e.type || "", message: e.message || (r.text ? String(r.text).slice(0, 400) : "HTTP " + r.status) };
  }

  /* ================================================================ connection */
  function connect(quiet) {
    var base = normBase(el.base.value);
    if (!base) {
      el.base.setAttribute("aria-invalid", "true");
      S.conn = { status: "invalid", base: "", health: null, schemas: [] };
      renderStatus();
      el.base.focus();
      return Promise.resolve(false);
    }
    el.base.removeAttribute("aria-invalid");
    el.base.value = base;
    save(STORE.base, base);
    persistKey();
    S.conn = { status: "connecting", base: base, health: null, schemas: [] };
    renderStatus();
    return call("GET", base + "/healthz", undefined, { auth: false, timeout: CONNECT_TIMEOUT }).then(function (h) {
      if (h.kind !== "http") {
        S.conn = { status: quiet ? "idle" : (h.kind === "timeout" ? "timeout" : "unreachable"), base: base, health: null, schemas: [],
          note: quiet ? "No Tez server answered at " + base + "." : "" };
        return false;
      }
      if (!h.ok || !h.data || typeof h.data !== "object" || !("status" in h.data)) {
        S.conn = { status: "error", base: base, health: null, schemas: [], http: h.status, message: errorOf(h).message };
        return false;
      }
      var health = h.data;
      return Promise.all([call("GET", base + "/v1/models", undefined, { timeout: CONNECT_TIMEOUT }),
                          call("GET", base + "/v1/schemas", undefined, { timeout: CONNECT_TIMEOUT })]).then(function (r) {
        if (r[0].status === 401 || r[1].status === 401) {
          S.conn = { status: "unauthorized", base: base, health: health, schemas: [], message: errorOf(r[0].status === 401 ? r[0] : r[1]).message };
          return false;
        }
        var schemas = r[1].ok && r[1].data && Array.isArray(r[1].data.schemas) ? r[1].data.schemas : [];
        S.conn = { status: health.status === "ok" ? "ok" : "degraded", base: base, health: health, schemas: schemas,
          models: r[0].ok ? r[0].data : null };
        return true;
      });
    }).then(function (ok) {
      buildUsecaseSelect();
      renderStatus();
      if (S.conn.status === "unauthorized") { el.key.setAttribute("aria-invalid", "true"); if (!quiet) el.key.focus(); }
      else el.key.removeAttribute("aria-invalid");
      refresh(true);
      return ok;
    });
  }

  function disconnect() {
    S.conn = { status: "idle", base: "", health: null, schemas: [], note: "Disconnected." };
    if (S.source.kind === "server") selectSource("custom");
    buildUsecaseSelect();
    renderStatus();
    refresh(true);
  }

  function persistKey() {
    save(STORE.remember, el.remember.checked ? "1" : null);
    save(STORE.key, el.remember.checked ? el.key.value.trim() : null);
  }

  function helpHTML(base) {
    var https = location.protocol === "https:" && /^http:/i.test(base || "");
    var list = [
      "Chrome may ask to allow this page to access devices on your local network. Allow it, then connect again.",
      "Safari may block <code>http://127.0.0.1</code> from an https page. Run the playground locally instead: in <code>site/</code> of the repository run <code>python -m http.server 8000</code> and open <code>http://localhost:8000/playground.html</code>."
    ];
    if (https && base && !isLoopback(base)) list.unshift("Browsers do not let an https page call <code>http://</code> on another machine. Use the local copy of the playground described below, or serve Tez on this machine.");
    return '<div class="pg-help"><p>Start the model server and Tez on this machine, one per terminal, then connect again:</p>' +
      cmdBlock("Two terminals", [LLAMA_CMD, SERVE_CMD]) +
      '<ul class="pg-help-list">' + list.map(function (t) { return "<li>" + t + "</li>"; }).join("") + "</ul></div>";
  }

  function renderStatus() {
    var c = S.conn, h = c.health || {}, html = "", icon = "circleDashed", tone = "neutral";
    switch (c.status) {
      case "connecting":
        icon = "loader"; html = "Connecting to <code>" + esc(c.base) + "</code>";
        break;
      case "ok":
      case "degraded": {
        var probes = 0;
        Object.keys(h.probes || {}).forEach(function (k) { probes += (h.probes[k] || []).length; });
        var names = (h.schemas || []).slice(0, 4).join(", ") + ((h.schemas || []).length > 4 ? ", …" : "");
        var chips = [
          ["backend", (h.backend || "?") + (h.backend_status ? " (" + h.backend_status + ")" : "")],
          ["model", h.model || "not reported"],
          ["template", h.template || "?"],
          ["schemas", String((h.schemas || []).length) + (names ? ": " + names : "")],
          ["probes", String(probes)]
        ];
        if (c.status === "ok") {
          icon = "circleCheck"; tone = "ok";
          html = "<strong>Connected</strong> to tez " + esc(h.version || "") + " at <code>" + esc(c.base) + "</code>";
        } else {
          icon = "alert"; tone = "warn";
          html = "<strong>Tez is running, but llama-server is not reachable</strong> at <code>" + esc(h.backend || "its backend") + "</code>. Requests fail with 503 until it answers.";
        }
        html += '<ul class="pg-chips" role="list">' + chips.map(function (x) {
          return '<li><span class="kv-k">' + esc(x[0]) + '</span> <span class="kv-v">' + esc(x[1]) + "</span></li>";
        }).join("") + "</ul>";
        if (c.status === "degraded") html += cmdBlock("Start llama-server", [LLAMA_CMD]);
        break;
      }
      case "unauthorized":
        icon = "key"; tone = "error";
        html = "<strong>This server needs an API key.</strong> It was started with <code>--api-key</code>. Enter that key above and connect again.";
        break;
      case "unreachable":
        icon = "alert"; tone = "error";
        html = "<strong>Could not reach <code>" + esc(c.base) + "</code>.</strong>" + helpHTML(c.base);
        break;
      case "timeout":
        icon = "alert"; tone = "error";
        html = "<strong><code>" + esc(c.base) + "</code> did not answer within " + (CONNECT_TIMEOUT / 1000) + " s.</strong>" + helpHTML(c.base);
        break;
      case "error":
        icon = "alert"; tone = "error";
        html = "<strong><code>" + esc(c.base) + "</code> answered HTTP " + esc(c.http) + ".</strong> " + esc(c.message || "") + " Is this address a Tez server?";
        break;
      case "invalid":
        icon = "alert"; tone = "error";
        html = "<strong>Enter the server address</strong>, for example <code>http://127.0.0.1:8787</code>.";
        break;
      default:
        html = "<strong>Not connected.</strong> " + (c.note ? esc(c.note) + " " : "") +
          (S.replays ? "Unedited site samples show recorded replays." : "Connect to a local server to send requests.");
    }
    el.status.setAttribute("data-tone", tone);
    el.status.innerHTML = '<div class="pg-status-line">' + UI.icon(icon, icon === "loader" ? "spin" : "") + '<div class="pg-status-text">' + html + "</div></div>";
    var connected = isConnected();
    el.connectLabel.textContent = c.status === "connecting" ? "Connecting" : (connected ? "Reconnect" : "Connect");
    el.connect.disabled = c.status === "connecting";
    el.connect.setAttribute("aria-busy", c.status === "connecting" ? "true" : "false");
    el.disconnect.hidden = !connected;
  }

  /* ================================================================ use cases and samples */
  function loadSite(id) {
    if (!S.siteDetail[id]) S.siteDetail[id] = UI.getJSON("./data/usecases/" + encodeURIComponent(id) + ".json");
    return S.siteDetail[id];
  }

  function loadReplays() {
    return UI.getJSON("./data/replays.json").then(function (data) {
      if (!data || !Array.isArray(data.replays) || !data.replays.length) return null;
      var index = {};
      data.replays.forEach(function (r) { index[r.usecase + ":" + r.sample_index] = r; });
      return { meta: data.meta || {}, index: index };
    });
  }

  function sourceValue() {
    if (S.source.kind === "site") return "site:" + S.source.id;
    if (S.source.kind === "server") return "server:" + S.source.name;
    return "custom";
  }

  function buildUsecaseSelect() {
    var html = "";
    if (S.site.length) {
      html += '<optgroup label="Examples on this site">' + S.site.map(function (u) {
        return '<option value="site:' + esc(u.id) + '">' + esc(u.title) + "</option>";
      }).join("") + "</optgroup>";
    }
    if (isConnected() && S.conn.schemas.length) {
      html += '<optgroup label="Loaded on your server">' + S.conn.schemas.map(function (s) {
        return '<option value="server:' + esc(s.name) + '">' + esc(s.name) + "</option>";
      }).join("") + "</optgroup>";
    }
    html += '<option value="custom">Custom questions</option>';
    el.usecase.innerHTML = html;
    el.usecase.value = sourceValue();
    if (el.usecase.value !== sourceValue()) el.usecase.value = "custom";
  }

  function selectSource(value, sampleIndex) {
    if (value.indexOf("site:") === 0) return selectSite(value.slice(5), sampleIndex || 0);
    if (value.indexOf("server:") === 0) return selectServer(value.slice(7));
    S.source = { kind: "custom" };
    S.detail = null;
    S.serverSchema = null;
    el.usecase.value = "custom";
    el.sample.innerHTML = '<option value="">No samples for custom questions</option>';
    el.sample.disabled = true;
    if (!el.questions.value.trim()) el.questions.value = JSON.stringify(DEFAULT_QUESTIONS, null, 2);
    refresh(true);
  }

  function selectSite(id, sampleIndex) {
    S.source = { kind: "site", id: id };
    S.serverSchema = null;
    el.usecase.value = "site:" + id;
    el.sample.disabled = true;
    return loadSite(id).then(function (d) {
      if (S.source.kind !== "site" || S.source.id !== id) return;
      S.detail = d;
      if (!d || !d.schema) {
        el.sample.innerHTML = '<option value="">Samples did not load</option>';
        setMsg(el.qMsg, "error", "The use case did not load from data/usecases/" + id + ".json.");
        return;
      }
      el.questions.value = JSON.stringify(d.schema.questions, null, 2);
      el.sample.innerHTML = (d.samples || []).map(function (s, i) {
        return '<option value="' + i + '">Sample ' + (i + 1) + ": " + esc(clip(preview(s.state), 56)) + "</option>";
      }).join("") + '<option value="custom">Your own input</option>';
      el.sample.disabled = false;
      var i = sampleIndex >= 0 && sampleIndex < (d.samples || []).length ? sampleIndex : 0;
      el.sample.value = String(i);
      applySample(i);
      refresh(true);
    });
  }

  function selectServer(name) {
    S.source = { kind: "server", name: name };
    S.detail = null;
    S.serverSchema = null;
    el.usecase.value = "server:" + name;
    el.sample.innerHTML = '<option value="">No samples for server schemas</option>';
    el.sample.disabled = true;
    setMsg(el.qMsg, "info", "Loading " + name + " from the server.");
    return call("GET", S.conn.base + "/v1/schemas/" + encodeURIComponent(name), undefined, { timeout: CONNECT_TIMEOUT }).then(function (r) {
      if (S.source.kind !== "server" || S.source.name !== name) return;
      if (r.kind === "http" && r.ok && r.data && r.data.questions) {
        S.serverSchema = r.data;
        el.questions.value = JSON.stringify(r.data.questions, null, 2);
        el.state.placeholder = r.data.state || "";
      } else {
        setMsg(el.qMsg, "error", "Could not load the schema: " + (r.kind === "http" ? "HTTP " + r.status + " " + errorOf(r).message : "no answer from the server") + ".");
      }
      refresh(true);
    });
  }

  function applySample(i) {
    var smp = S.detail && S.detail.samples && S.detail.samples[i];
    if (!smp) return;
    if (typeof smp.state === "string") { setStateMode("text"); el.state.value = smp.state; }
    else { setStateMode("json"); el.state.value = JSON.stringify(smp.state, null, 2); }
    el.state.placeholder = "";
  }

  function sampleText(i) {
    var smp = S.detail && S.detail.samples && S.detail.samples[i];
    if (!smp) return null;
    return typeof smp.state === "string" ? smp.state : JSON.stringify(smp.state, null, 2);
  }

  function setStateMode(mode) {
    S.stateMode = mode;
    Array.prototype.forEach.call(el.stateMode.querySelectorAll("[data-mode]"), function (b) {
      b.setAttribute("aria-pressed", b.getAttribute("data-mode") === mode ? "true" : "false");
    });
    el.state.classList.toggle("mono", mode === "json");
  }

  /* ================================================================ validation (docs/API.md rules) */
  function checkQuestion(qid, q) {
    var here = "questions." + qid;
    if (!qid.trim()) return "questions: question ids must be non-empty strings";
    if (!q || typeof q !== "object" || Array.isArray(q)) return here + " must be an object with type, instructions and criteria";
    if (["noul", "choice", "score"].indexOf(q.type) < 0) return here + ".type must be one of noul, choice, score; got " + JSON.stringify(q.type === undefined ? null : q.type);
    var ins = q.instructions;
    if (ins === undefined || ins === null || (typeof ins === "string" && !ins.trim())) return here + ".instructions is required";
    if (typeof ins !== "string" && typeof ins !== "object") return here + ".instructions must be a string, object or array";
    var c = q.criteria, keys, i;
    if (q.type === "noul") {
      if (c === undefined || c === null) return null;
      if (typeof c !== "object" || Array.isArray(c)) return here + '.criteria must be an object with optional "true" and "false" descriptions';
      keys = Object.keys(c);
      var extra = keys.filter(function (k) { return k !== "true" && k !== "false"; });
      if (extra.length) return here + '.criteria keys must be "true" and/or "false"; got ' + JSON.stringify(extra);
      for (i = 0; i < keys.length; i++) if (c[keys[i]] !== null && typeof c[keys[i]] !== "string") return here + ".criteria." + keys[i] + " must be a string or null";
      return null;
    }
    if (q.type === "choice") {
      if (!c || typeof c !== "object" || Array.isArray(c)) return here + ".criteria must be an object mapping each option label to a description or null";
      keys = Object.keys(c);
      if (keys.length < 2 || keys.length > 255) return here + ".criteria must have 2 to 255 options, got " + keys.length;
      for (i = 0; i < keys.length; i++) {
        if (!keys[i].trim()) return here + ".criteria labels must be non-empty strings";
        if (c[keys[i]] !== null && typeof c[keys[i]] !== "string") return here + ".criteria." + keys[i] + " must be a string or null";
      }
      return null;
    }
    if (!Array.isArray(c)) return here + ".criteria must be a list of level descriptions (level i = index i)";
    if (c.length < 2 || c.length > 10) return here + ".criteria must have 2 to 10 levels, got " + c.length;
    for (i = 0; i < c.length; i++) if (typeof c[i] !== "string") return here + ".criteria[" + i + "] must be a string";
    return null;
  }

  function summarize(qs) {
    var ids = Object.keys(qs);
    return UI.plural(ids.length, "question") + ": " + ids.map(function (id) {
      var q = qs[id];
      if (q.type === "noul") return id + " (yes/no)";
      if (q.type === "score") return id + " (score, " + q.criteria.length + " levels)";
      return id + " (choice, " + Object.keys(q.criteria).length + " options)";
    }).join(", ");
  }

  function readQuestions() {
    var text = el.questions.value;
    var optional = S.source.kind === "server";
    if (!text.trim()) {
      return optional ? { ok: true, value: null, info: "Empty: the server uses the schema's own questions." }
        : { ok: false, msg: "questions is required: an object mapping question ids to questions." };
    }
    var v;
    try { v = JSON.parse(text); } catch (e) { return { ok: false, msg: "Not valid JSON: " + e.message }; }
    if (!v || typeof v !== "object" || Array.isArray(v)) return { ok: false, msg: "questions must be an object mapping question ids to questions" };
    var ids = Object.keys(v);
    if (!ids.length) return { ok: false, msg: "questions must not be empty" };
    for (var i = 0; i < ids.length; i++) {
      var err = checkQuestion(ids[i], v[ids[i]]);
      if (err) return { ok: false, msg: err };
    }
    return { ok: true, value: v, info: summarize(v) };
  }

  function readState() {
    var t = el.state.value;
    if (!t.trim()) return { ok: false, msg: "state is required: the text or JSON to decide on." };
    if (S.stateMode === "text") return { ok: true, value: t };
    var v;
    try { v = JSON.parse(t); } catch (e) { return { ok: false, msg: "Not valid JSON: " + e.message + ". Choose Text to send it as a string." }; }
    if (v === null || (typeof v !== "string" && typeof v !== "object")) return { ok: false, msg: "state must be a string, object or array." };
    return { ok: true, value: v };
  }

  function readTez() {
    var out = {}, a = el.alpha.value.trim();
    if (el.readout.value !== "auto") out.readout = el.readout.value;
    if (el.abstain.checked) out.abstain = true;
    if (a) {
      var n = Number(a);
      if (!isFinite(n) || n <= 0 || n >= 1) return { ok: false, msg: "tez.gate.alpha must be a number between 0 and 1 (exclusive), for example 0.05." };
      out.gate = { alpha: n };
    }
    var info = "";
    if (out.gate) info = "The gate marks each answer act or escalate. Without a schema fitted with tez fit on the server, every answer escalates.";
    else if (out.readout === "probe") info = "probe is strict: the server answers 422 when a question has no trained probe.";
    return { ok: true, value: Object.keys(out).length ? out : null, info: info };
  }

  function build() {
    var st = readState(), qs = readQuestions(), tz = readTez(), req = null;
    if (st.ok && qs.ok && tz.ok) {
      req = { model: "tez-latest", state: st.value };
      if (S.source.kind === "server") req.schema = S.source.name;
      if (qs.value) req.questions = qs.value;
      if (tz.value) req.tez = tz.value;
    }
    return { st: st, qs: qs, tz: tz, req: req };
  }

  function setMsg(node, kind, text) {
    node.setAttribute("data-kind", kind || "");
    node.innerHTML = text ? (kind === "error" ? UI.icon("alert") : "") + "<span>" + esc(text) + "</span>" : "";
  }

  function mark(input, bad) { if (bad) input.setAttribute("aria-invalid", "true"); else input.removeAttribute("aria-invalid"); }

  /* ================================================================ refresh: fields, snippets, send state, replay */
  function schedule() {
    clearTimeout(S.timer);
    S.timer = setTimeout(function () { refresh(false); }, 140);
  }

  function refresh(resetShown) {
    var b = build();
    mark(el.state, !b.st.ok);
    setMsg(el.stateMsg, b.st.ok ? "" : "error", b.st.ok ? "" : b.st.msg);
    mark(el.questions, !b.qs.ok);
    setMsg(el.qMsg, b.qs.ok ? "ok" : "error", b.qs.ok ? b.qs.info : b.qs.msg);
    mark(el.alpha, !b.tz.ok);
    setMsg(el.tezMsg, b.tz.ok ? (b.tz.info ? "info" : "") : "error", b.tz.ok ? b.tz.info : b.tz.msg);

    var base = normBase(el.base.value) || UI.DEFAULT_BASE, auth = !!el.key.value.trim();
    if (b.req) {
      el.curl.innerHTML = UI.curlSnippet(b.req, { base: base, auth: auth }).html;
      el.python.innerHTML = UI.pythonSnippet(b.req, { base: base, auth: auth }).html;
    } else {
      el.curl.textContent = "Fix the request on the left to see it as curl.";
      el.python.textContent = "Fix the request on the left to see it as Python.";
    }

    var connected = isConnected();
    el.send.disabled = !b.req || S.busy;
    el.sendLabel.textContent = S.busy ? "Sending" : (connected ? "Send request" : "Connect and send");
    el.sendNote.textContent = !b.req ? "Fix the highlighted field to send." :
      (connected ? "Ctrl+Enter sends from any field." : "Connects to " + host(base) + " first. Ctrl+Enter sends from any field.");

    if (S.busy) return;
    var rec = matchReplay(b.req);
    var key = rec && rec !== "edited" ? "replay:" + rec.usecase + ":" + rec.sample_index : null;
    if (connected) {
      if (resetShown || !S.shown || S.shown.origin !== "live") showReady();
      return;
    }
    if (key) {
      if (!S.shown || S.shown.key !== key) showReplay(rec, b.req);
    } else {
      showOffline(offlineReason(rec));
    }
  }

  function offlineReason(rec) {
    if (!S.replays) return "none-published";
    if (rec === "edited") return "edited";
    if (S.source.kind === "site") return el.sample.value === "custom" || el.sample.value === "" ? "own-input" : "no-recording";
    return "not-site";
  }

  function matchReplay(req) {
    if (!S.replays || !req || S.source.kind !== "site") return null;
    var i = el.sample.value;
    if (i === "" || i === "custom") return null;
    var rec = S.replays.index[S.source.id + ":" + i];
    if (!rec) return null;
    return canon(rec.request) === canon(req) ? rec : "edited";
  }

  /* ================================================================ response panel */
  function setOrigin(kind) {
    if (kind === "live") el.origin.innerHTML = '<span class="origin origin-live">' + UI.icon("circleCheck") + "Live · " + esc(host(S.conn.base)) + "</span>";
    else if (kind === "replay") el.origin.innerHTML = '<span class="origin origin-recorded">' + UI.icon("history") + "Recorded replay, not live</span>";
    else el.origin.innerHTML = "";
  }

  function setJSON(title, value) {
    el.jsonTitle.textContent = title;
    if (value === null || value === undefined) el.json.textContent = "No response yet.";
    else if (typeof value === "string") el.json.textContent = value;
    else el.json.innerHTML = UI.jsonHTML(value);
  }

  function metaLine(parts) {
    parts = parts.filter(Boolean);
    el.meta.hidden = !parts.length;
    el.meta.innerHTML = parts.join('<span class="sep" aria-hidden="true"> · </span>');
  }

  function questionsFor(req) {
    if (req && req.questions) return req.questions;
    return S.serverSchema && S.serverSchema.questions ? S.serverSchema.questions : null;
  }

  function showReady() {
    S.shown = { origin: "ready" };
    setOrigin(null);
    el.banner.hidden = true;
    metaLine([]);
    el.out.innerHTML = '<div class="pg-empty">' + UI.icon("send") + "<p><strong>Ready.</strong> Send the request to see the answers from <code>" + esc(S.conn.base) + "</code>.</p></div>";
    setJSON("Response JSON", null);
  }

  function showOffline(reason) {
    var failed = S.conn.status === "unreachable" || S.conn.status === "timeout";
    var key = "offline:" + reason + ":" + failed + ":" + normBase(el.base.value);
    if (S.shown && S.shown.key === key) return;
    S.shown = { origin: "offline", key: key };
    setOrigin(null);
    el.banner.hidden = true;
    metaLine([]);
    var lead = {
      "none-published": "<strong>Not connected.</strong> Recorded replays are not published yet, so every request needs a local server.",
      "edited": "<strong>No recording for this request.</strong> Recorded replays cover the site samples exactly as published, without Tez options; this request differs. Connect to a local server to send it.",
      "own-input": "<strong>Your own input needs a live server.</strong> Connect to a local server to send it.",
      "no-recording": "<strong>No recording for this sample.</strong> Connect to a local server to send it.",
      "not-site": "<strong>Not connected.</strong> Recorded replays cover the site samples only. Connect to a local server to send this request."
    }[reason];
    el.out.innerHTML = '<div class="pg-empty pg-empty-offline">' + UI.icon("plug") + "<div><p>" + lead + "</p>" +
      (failed ? "<p>The connection panel above lists the commands that start the servers.</p>" : helpHTML(normBase(el.base.value))) + "</div></div>";
    setJSON("Response JSON", null);
  }

  function showReplay(rec, req) {
    S.shown = { origin: "replay", key: "replay:" + rec.usecase + ":" + rec.sample_index };
    var m = S.replays.meta || {}, resp = rec.response || {}, tz = resp.tez || {}, usage = resp.usage || {};
    setOrigin("replay");
    el.banner.hidden = false;
    el.banner.innerHTML = UI.icon("history") + "<p><strong>Recorded replay, not live.</strong> This response was recorded" +
      (m.recorded ? " on " + esc(UI.formatDate(m.recorded)) : "") + " from <code>tez serve</code>" + (m.model ? " with " + esc(m.model) : "") +
      (m.hardware ? " on an " + esc(m.hardware) : "") + ". Nothing runs in your browser. Connect to your own server to decide new inputs.</p>";
    metaLine([
      typeof rec.client_ms === "number" ? "recorded round trip <span class=\"num\">" + esc(UI.fmtMs(rec.client_ms)) + "</span>" : "",
      typeof tz.latency_ms === "number" ? "tez.latency_ms <span class=\"num\">" + esc(String(tz.latency_ms)) + "</span>" : "",
      typeof usage.input_tokens === "number" ? '<span class="num">' + usage.input_tokens + "</span> input tokens" : "",
      resp.model ? "<code>" + esc(resp.model) + "</code>" : ""
    ]);
    UI.renderAnswers(el.out, resp, { questions: questionsFor(req), maxRows: 8 });
    setJSON("Recorded response JSON", resp);
    announce("Recorded replay shown: " + UI.plural(Object.keys(resp.answers || {}).length, "answer") + ".");
  }

  function showLive(req, r) {
    var resp = r.data, tz = resp.tez || {}, usage = resp.usage || {};
    S.shown = { origin: "live", key: "live" };
    setOrigin("live");
    el.banner.hidden = true;
    metaLine([
      "client round trip <span class=\"num\">" + esc(UI.fmtMs(r.ms)) + "</span>",
      typeof tz.latency_ms === "number" ? "tez.latency_ms <span class=\"num\">" + esc(String(tz.latency_ms)) + "</span>" : "",
      typeof usage.input_tokens === "number" ? '<span class="num">' + usage.input_tokens + "</span> input tokens" : "",
      resp.model ? "<code>" + esc(resp.model) + "</code>" : ""
    ]);
    UI.renderAnswers(el.out, resp, { questions: questionsFor(req), maxRows: 8 });
    setJSON("Response JSON", resp);
    announce("Response received: " + UI.plural(Object.keys(resp.answers || {}).length, "answer") + " in " + UI.fmtMs(r.ms) + ".");
  }

  function showLoading() {
    S.shown = { origin: "loading" };
    setOrigin(null);
    el.banner.hidden = true;
    metaLine([]);
    el.out.innerHTML = '<div class="pg-empty pg-loading" aria-busy="true">' + UI.icon("loader", "spin") + "<p>Waiting for <code>tez serve</code> at <code>" + esc(S.conn.base) + "</code>.</p></div>";
    setJSON("Response JSON", "Waiting for the response.");
  }

  function showError(title, body, raw) {
    S.shown = { origin: "error" };
    setOrigin(null);
    el.banner.hidden = true;
    metaLine([]);
    el.out.innerHTML = '<div class="pg-error" role="alert">' + UI.icon("alert") + '<div class="pg-error-body"><p class="pg-error-title">' + title + "</p>" + (body || "") + "</div></div>";
    setJSON("Error response", raw === undefined ? null : raw);
  }

  function announce(text) {
    if (!el.live) return;
    el.live.textContent = "";
    setTimeout(function () { el.live.textContent = text; }, 30);
  }

  /* ================================================================ send */
  function send() {
    var b = build();
    if (!b.req) {
      refresh(false);
      var bad = document.querySelector('.pg-build [aria-invalid="true"]');
      if (bad) bad.focus();
      return;
    }
    if (S.busy) return;
    var ready = isConnected() ? Promise.resolve(true) : connect(false);
    ready.then(function (ok) {
      if (!ok) {
        var c = S.conn;
        if (c.status === "unauthorized") showError("401 unauthorized", "<p>The server needs an API key. Enter the key it was started with and send again.</p>");
        else if (c.status === "unreachable" || c.status === "timeout") showError("Could not reach <code>" + esc(c.base) + "</code>", "<p>The connection panel above lists the commands that start the servers and what to check in your browser.</p>");
        else if (c.status === "error") showError("<code>" + esc(c.base) + "</code> answered HTTP " + esc(c.http), "<p>" + esc(c.message || "") + " Is this address a Tez server?</p>");
        return;
      }
      S.busy = true;
      refresh(false);
      showLoading();
      el.send.setAttribute("aria-busy", "true");
      call("POST", S.conn.base + "/v1/systemone", b.req, { timeout: SEND_TIMEOUT }).then(function (r) {
        S.busy = false;
        el.send.removeAttribute("aria-busy");
        refresh(false);
        if (r.kind === "timeout") {
          showError("No answer within " + (SEND_TIMEOUT / 1000) + " s", "<p>llama-server may still be loading the model or busy with other requests, or the input is very long. Check its terminal and send again.</p>");
          return;
        }
        if (r.kind !== "http") {
          S.conn.status = "unreachable";
          renderStatus();
          showError("Could not reach <code>" + esc(S.conn.base) + "</code>", "<p>The server stopped answering. The connection panel above lists the commands that start it again.</p>");
          return;
        }
        if (r.ok && r.data && r.data.answers) { showLive(b.req, r); return; }
        httpError(r);
      });
    });
  }

  function fieldFor(msg) {
    if (/^questions|criteria|instructions/.test(msg)) return "questions";
    if (/^state|context|too long|prompt/.test(msg)) return "state";
    if (/tez\.|readout|probe|gate|alpha|abstain/.test(msg)) return "tez";
    if (/schema/.test(msg)) return "usecase";
    return null;
  }

  function httpError(r) {
    var e = errorOf(r), raw = r.data || r.text || null;
    var head = '<code class="pg-code-status">' + r.status + (e.type ? " " + esc(e.type) : "") + "</code>";
    if (r.status === 401) {
      S.conn.status = "unauthorized";
      renderStatus();
      el.key.setAttribute("aria-invalid", "true");
      el.key.focus();
      showError(head + " The server needs an API key", "<p>" + esc(e.message) + "</p><p>Enter the key <code>tez serve</code> was started with (<code>--api-key</code>) and send again.</p>", raw);
      return;
    }
    if (r.status === 422) {
      var f = fieldFor(e.message), where = "";
      if (f === "questions") { mark(el.questions, true); setMsg(el.qMsg, "error", "Server: " + e.message); where = "next to Questions"; }
      else if (f === "state") { mark(el.state, true); setMsg(el.stateMsg, "error", "Server: " + e.message); where = "next to State"; }
      else if (f === "tez") { setMsg(el.tezMsg, "error", "Server: " + e.message); where = "under Tez options"; }
      showError(head + " The server rejected the request", "<p>" + esc(e.message) + "</p>" + (where ? "<p>The message is also shown " + where + ".</p>" : ""), raw);
      return;
    }
    if (r.status === 503) {
      var backend = (S.conn.health && S.conn.health.backend) || "http://127.0.0.1:8091";
      showError(head + " llama-server is not reachable", "<p>" + esc(e.message) + "</p><p>Tez is running, but the model server at <code>" + esc(backend) + "</code> did not answer. Start it and send again:</p>" +
        cmdBlock("Start llama-server", [LLAMA_CMD]), raw);
      return;
    }
    showError(head + " Request failed", "<p>" + esc(e.message) + "</p>", raw);
  }

  /* ================================================================ events */
  function bind() {
    el.form.addEventListener("submit", function (e) { e.preventDefault(); connect(false); });
    el.disconnect.addEventListener("click", disconnect);
    el.base.addEventListener("input", function () { el.base.removeAttribute("aria-invalid"); schedule(); });
    el.base.addEventListener("change", function () {
      var b = normBase(el.base.value);
      if (S.conn.status !== "connecting" && S.conn.status !== "idle" && b !== S.conn.base) {
        if (S.source.kind === "server") selectSource("custom");
        S.conn = { status: "idle", base: "", health: null, schemas: [], note: "The server address changed; connect again." };
        buildUsecaseSelect();
        renderStatus();
        refresh(true);
      }
    });
    el.key.addEventListener("input", function () { el.key.removeAttribute("aria-invalid"); if (el.remember.checked) persistKey(); schedule(); });
    el.remember.addEventListener("change", persistKey);

    el.usecase.addEventListener("change", function () { selectSource(el.usecase.value, 0); });
    el.sample.addEventListener("change", function () {
      if (el.sample.value === "custom") {
        el.state.value = "";
        setStateMode("text");
        el.state.focus();
        refresh(false);
      } else {
        applySample(Number(el.sample.value));
        refresh(false);
      }
    });
    el.state.addEventListener("input", function () {
      var i = el.sample.value;
      if (S.source.kind === "site" && i !== "" && i !== "custom" && el.state.value !== sampleText(Number(i))) el.sample.value = "custom";
      schedule();
    });
    el.stateMode.addEventListener("click", function (e) {
      var b = e.target.closest("[data-mode]");
      if (!b) return;
      setStateMode(b.getAttribute("data-mode"));
      refresh(false);
    });
    el.questions.addEventListener("input", schedule);
    el.format.addEventListener("click", function () {
      try { el.questions.value = JSON.stringify(JSON.parse(el.questions.value), null, 2); } catch (e) { el.questions.focus(); }
      refresh(false);
    });
    [el.readout, el.abstain].forEach(function (n) { n.addEventListener("change", function () { refresh(false); }); });
    el.alpha.addEventListener("input", schedule);
    el.send.addEventListener("click", send);
    document.querySelector(".pg-build").addEventListener("keydown", function (e) {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); send(); }
    });
  }

  /* ================================================================ start */
  function start() {
    var storedBase = load(STORE.base);
    if (storedBase) el.base.value = storedBase;
    el.remember.checked = load(STORE.remember) === "1";
    if (el.remember.checked) { var k = load(STORE.key); if (k) el.key.value = k; }
    bind();
    renderStatus();
    Promise.all([UI.getJSON("./data/usecases.json"), loadReplays()]).then(function (res) {
      S.site = Array.isArray(res[0]) ? res[0] : [];
      S.replays = res[1];
      renderStatus();
      buildUsecaseSelect();
      var params = new URLSearchParams(location.search);
      var want = params.get("usecase"), n = parseInt(params.get("sample"), 10);
      var known = S.site.some(function (u) { return u.id === want; });
      var first = known ? want : (S.site[0] ? S.site[0].id : null);
      var go = first ? selectSite(first, isFinite(n) ? n : 0) : Promise.resolve(selectSource("custom"));
      Promise.resolve(go).then(function () { if (isLocalPage()) connect(true); });
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
})();
