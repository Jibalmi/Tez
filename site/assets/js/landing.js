/* Landing page: the recorded replay in the hero panel and the use-case preview.
   The hero's static HTML already shows the Support example's final state, so the page reads the same without
   this script; with it, the panel replays recorded runs from data/strips.json word by word. */
(function () {
  "use strict";

  var UI = window.TezUI;
  if (!UI) return;

  var KINDS = {
    "support-payouts": "Customer message",
    "support-refund-outage": "Customer message",
    "security-ssh": "Security alert",
    "voice-v062": "Spoken command, streamed transcript",
    "injection-cfo": "Text sent to an assistant"
  };
  var SHOW_AT = 0.01;      // with more than five options, options that never reach this are summarised in one line
  var PACE = 280;          // ms per word: reading speed, much slower than the measured steps
  var PREVIEW = ["support-triage", "security-triage", "prompt-injection-guard", "voice-commands", "invoice-routing", "multilingual-intent"];
  var VOICE_SOURCE = "results/voicefast_partial_last_gemma4-12b-q8_0_stream.jsonl";

  /* ================================================================ hero replay */
  function initDemo(demo) {
    function q(sel) { return demo.querySelector(sel); }
    var n = {
      buttons: Array.prototype.slice.call(demo.querySelectorAll("[data-example]")),
      toggle: q("[data-demo-toggle]"),
      restart: q("[data-demo-restart]"),
      kind: q("[data-demo-kind]"),
      qid: q("[data-demo-qid]"),
      qtype: q("[data-demo-qtype]"),
      msg: q("[data-demo-msg]"),
      full: q("[data-demo-fulltext]"),
      scrub: q("[data-demo-scrub]"),
      question: q("[data-demo-q]"),
      probs: q("[data-demo-probs]"),
      others: q("[data-demo-others]"),
      step: q("[data-demo-step]"),
      ms: q("[data-demo-ms]"),
      answer: q("[data-demo-answer]"),
      json: q("[data-demo-json]"),
      python: q("[data-demo-python]"),
      curl: q("[data-demo-curl]")
    };
    var note = document.querySelector("[data-demo-note]");
    var live = document.querySelector("[data-demo-live]");
    var meta = {}, strips = {}, strip = null, list = null, replay = null, spans = [], played = false, seen = false;

    UI.getJSON("./data/strips.json").then(function (data) {
      if (!data || !Array.isArray(data.strips)) return;     // keep the static final state
      meta = data.meta || {};
      data.strips.forEach(function (s) { strips[s.id] = s; });
      n.buttons = n.buttons.filter(function (b) {
        var ok = !!strips[b.getAttribute("data-example")];
        if (!ok) b.hidden = true;
        return ok;
      });
      if (!n.buttons.length) return;
      // under reduced motion the panel shows final states; the slider still steps through the words
      n.toggle.hidden = UI.reducedMotion();
      n.restart.hidden = UI.reducedMotion();
      n.scrub.disabled = false;
      bind();
      var first = n.buttons.filter(function (b) { return b.getAttribute("aria-pressed") === "true"; })[0] || n.buttons[0];
      select(first.getAttribute("data-example"), false);
      watch();
    });

    function bind() {
      n.buttons.forEach(function (b) {
        b.addEventListener("click", function () { select(b.getAttribute("data-example"), true); });
      });
      n.toggle.addEventListener("click", function () { if (replay) { seen = true; replay.toggle(); } });
      n.restart.addEventListener("click", function () { if (replay) { seen = true; replay.restart(); } });
      n.scrub.addEventListener("input", function () { if (replay) { seen = true; replay.seek(Number(n.scrub.value) - 1); } });
      document.addEventListener("visibilitychange", function () { if (document.hidden && replay) replay.pause(); });
    }

    /* autoplay once, the first time the panel is mostly on screen */
    function watch() {
      if (UI.reducedMotion()) return;
      if (!("IntersectionObserver" in window)) { seen = true; replay.restart(); return; }
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting && !seen) { seen = true; replay.restart(); io.disconnect(); }
        });
      }, { threshold: 0.45 });
      io.observe(demo);
    }

    function select(id, play) {
      var s = strips[id];
      if (!s) return;
      if (replay) replay.stop();
      strip = s;
      n.buttons.forEach(function (b) { b.setAttribute("aria-pressed", b.getAttribute("data-example") === id ? "true" : "false"); });
      render(s);
      replay = new UI.Replay({ steps: s.trace.length, pace: PACE, onStep: showStep, onReset: showEmpty, onState: onState, onDone: onDone });
      played = !!play;
      if (play) { seen = true; replay.restart(); } else replay.finish();
    }

    /* ---------------------------------------------------------------- per-example setup */
    function rowsFor(s) {
      var qn = s.question, rows = [], hidden = 0;
      if (qn.type === "noul") {
        var desc = {};
        qn.options.forEach(function (o) { desc[o.key] = o.description || ""; });
        return { rows: [{ key: "true", label: "true", title: desc["true"] }, { key: "false", label: "false", title: desc["false"] }], hidden: 0 };
      }
      var max = {};
      s.trace.forEach(function (t) {
        Object.keys(t.probabilities).forEach(function (k) { max[k] = Math.max(max[k] || 0, t.probabilities[k]); });
      });
      var many = qn.options.length > 5;
      qn.options.forEach(function (o, i) {
        if (many && (max[o.key] || 0) < SHOW_AT) { hidden += 1; return; }
        if (qn.type === "score") rows.push({ key: o.key, label: i + " · " + (o.description || ""), title: o.description || "" });
        else rows.push({ key: o.key, label: o.key, title: o.description || "" });
      });
      return { rows: rows, hidden: hidden };
    }

    function typeLabel(qn) {
      if (qn.type === "noul") return "yes/no";
      if (qn.type === "score") return "score · " + UI.plural(qn.options.length, "level");
      return "choice · " + UI.plural(qn.options.length, "option");
    }

    function render(s) {
      var qn = s.question;
      n.kind.textContent = KINDS[s.id] || (s.kind === "voice" ? "Spoken command" : "Message");
      n.qid.textContent = qn.id;
      n.qtype.textContent = typeLabel(qn);
      n.question.textContent = qn.instructions;
      n.full.textContent = (s.kind === "voice" ? "Transcript: " : "Message: ") + s.state;

      n.msg.innerHTML = "";
      spans = s.words.map(function (w, i) {
        var sp = document.createElement("span");
        sp.className = "w";
        sp.textContent = w;
        n.msg.appendChild(sp);
        if (i < s.words.length - 1) n.msg.appendChild(document.createTextNode(" "));
        return sp;
      });

      var r = rowsFor(s);
      list = new UI.ProbList(n.probs, r.rows);
      if (r.hidden) {
        n.others.hidden = false;
        n.others.textContent = r.hidden + " other " + (s.kind === "voice" ? "actions" : "options") + " stay below " + SHOW_AT.toFixed(2) + " at every step.";
      } else {
        n.others.hidden = true;
        n.others.textContent = "";
      }

      n.scrub.max = String(s.trace.length);

      var final = s.trace[s.trace.length - 1];
      var answers = {};
      answers[qn.id] = UI.stripAnswer(s, final.probabilities);
      n.json.innerHTML = UI.jsonHTML({ answers: answers });
      var req = UI.stripRequest(s);
      n.python.innerHTML = UI.pythonSnippet(req).html;
      n.curl.innerHTML = UI.curlSnippet(req).html;

      if (note) note.innerHTML = noteFor(s);
    }

    function noteFor(s) {
      if (s.kind === "voice") {
        var at = typeof s.human_commit_word === "number" && s.human_commit_word > 0 ? s.words[s.human_commit_word - 1] : null;
        return "Transcript replayed from the recorded voice-loop run, 16 actions in a cached prefix" +
          (at ? "; the dataset marks word " + s.human_commit_word + ", “" + UI.esc(at) + "”, as the point where the intent is clear" : "") +
          ' (<a href="' + UI.BLOB + VOICE_SOURCE + '">source</a>).';
      }
      var fresh = s.fresh_full_state || {}, tokens = fresh.runs && fresh.runs[0] ? fresh.runs[0].prompt_n : null;
      var when = UI.formatDate(meta.recorded);
      var out = "Synthetic message" + (when ? ", recorded " + UI.esc(when) : "");
      if (typeof fresh.prompt_ms_median === "number") {
        out += "; decided whole without the prompt cache it took " + fresh.prompt_ms_median.toFixed(1) + " ms of compute" + (tokens ? " (" + tokens + " tokens)" : "");
      }
      return out + ".";
    }

    /* ---------------------------------------------------------------- steps */
    function argmax(s, probs) {
      if (s.question.type === "noul") return (probs["true"] || 0) >= 0.5 ? "true" : "false";
      var best = null;
      s.question.options.forEach(function (o) { if (best === null || (probs[o.key] || 0) > (probs[best] || 0)) best = o.key; });
      return best;
    }

    function labelFor(s, key) {
      if (s.question.type !== "score") return key;
      var o = s.question.options[Number(key)];
      return "level " + key + (o && o.description ? ", " + o.description : "");
    }

    function setWords(i, done) {
      spans.forEach(function (sp, j) {
        sp.className = "w" + (j > i ? " is-pending" : "") + (j === i && !done ? " is-current" : "");
      });
    }

    function setScrub(i) {
      var total = strip.trace.length;
      n.scrub.value = String(i + 1);
      n.scrub.style.setProperty("--fill", ((i + 1) / total * 100).toFixed(2) + "%");
      n.scrub.setAttribute("aria-valuetext", i < 0 ? "Before the first word" : "Word " + (i + 1) + " of " + total + ", " + strip.words[i]);
    }

    function showEmpty() {
      played = true;
      setWords(-1, false);
      list.update(null, null);
      setScrub(-1);
      n.scrub.style.setProperty("--fill", "0%");
      n.step.innerHTML = "Waiting for the first word";
      n.ms.innerHTML = "";
      setAnswer(false, "Deciding as the words arrive");
    }

    function showStep(i) {
      var t = strip.trace[i];
      var done = i === strip.trace.length - 1 && replay && replay.state !== "playing";
      setWords(i, false);
      var top = argmax(strip, t.probabilities);
      list.update(t.probabilities, top);
      setScrub(i);
      n.step.innerHTML = 'Word <span class="num">' + (i + 1) + "</span> of " + strip.trace.length +
        ' <span class="demo-word">' + UI.esc(t.word) + "</span>";
      n.ms.innerHTML = '<span class="num">' + UI.fmtMs(t.prompt_ms) + "</span> compute" +
        (typeof t.http_ms === "number" ? ' <span class="sep" aria-hidden="true">·</span> <span class="num">' + UI.fmtMs(t.http_ms) + "</span> round trip" : "");
      if (!done) {
        var p = strip.question.type === "noul" ? (t.probabilities["true"] || 0) : (t.probabilities[top] || 0);
        setAnswer(false, "Leading <strong>" + UI.esc(labelFor(strip, top)) + "</strong> " +
          '<span class="sep" aria-hidden="true">·</span> ' + (strip.question.type === "noul" ? "P(true) " : "p ") + '<span class="num">' + UI.fmtP(p) + "</span>");
      }
    }

    function onDone() {
      var last = strip.trace.length - 1, t = strip.trace[last];
      setWords(last, true);
      var a = UI.stripAnswer(strip, t.probabilities), sep = ' <span class="sep" aria-hidden="true">·</span> ', text, spoken;
      if (a.type === "noul") {
        var yes = a.noul >= 0.5 ? "true" : "false";
        text = "Answer <strong>" + yes + "</strong>" + sep + 'P(true) <span class="num">' + UI.fmtP(a.noul) + "</span>";
        spoken = "Answer " + yes + ", probability of true " + UI.fmtP(a.noul);
      } else if (a.type === "score") {
        var top = argmax(strip, t.probabilities);
        text = 'Answer score <strong class="num">' + UI.fixed(a.score, 2) + "</strong>" + sep + UI.esc(labelFor(strip, top)) + sep + 'confidence <span class="num">' + UI.fmtP(a.confidence) + "</span>";
        spoken = "Answer score " + UI.fixed(a.score, 2) + ", " + labelFor(strip, top) + ", confidence " + UI.fmtP(a.confidence);
      } else {
        text = "Answer <strong>" + UI.esc(a.choice) + "</strong>" + sep + 'p <span class="num">' + UI.fmtP(a.probabilities[a.choice]) + "</span>" + sep + 'confidence <span class="num">' + UI.fmtP(a.confidence) + "</span>";
        spoken = "Answer " + a.choice + ", probability " + UI.fmtP(a.probabilities[a.choice]) + ", confidence " + UI.fmtP(a.confidence);
      }
      setAnswer(true, text);
      if (played && live) live.textContent = spoken;
    }

    function setAnswer(final, html) {
      n.answer.setAttribute("data-final", final ? "true" : "false");
      n.answer.innerHTML = UI.icon(final ? "check" : "circleDashed") + '<span class="demo-answer-text">' + html + "</span>";
    }

    function onState(state) {
      var playing = state === "playing";
      n.toggle.setAttribute("data-playing", playing ? "true" : "false");
      n.toggle.setAttribute("aria-label", playing ? "Pause the replay" : (state === "done" ? "Play the replay again" : "Play the replay"));
      n.toggle.setAttribute("title", playing ? "Pause" : "Play");
    }
  }

  /* ================================================================ use-case preview */
  function initUseCases(ul) {
    UI.getJSON("./data/usecases.json").then(function (items) {
      if (!Array.isArray(items) || !items.length) return;
      var byId = {};
      items.forEach(function (u) { byId[u.id] = u; });
      var pick = PREVIEW.filter(function (id) { return byId[id]; }).map(function (id) { return byId[id]; });
      items.forEach(function (u) { if (pick.length < 6 && pick.indexOf(u) < 0) pick.push(u); });
      ul.innerHTML = pick.map(rowHTML).join("");
    });
  }

  function rowHTML(u) {
    var ev = (u.evidence || [])[0];
    var tags = UI.questionCounts(u.questions).map(function (t) { return '<span class="pill">' + UI.esc(t) + "</span>"; }).join("");
    var evidence = ev ? '<div class="uc-row-evidence">' +
      '<span class="uc-row-value">' + UI.esc(UI.fmtEvidence(ev.value, ev.metric)) + "</span>" +
      '<span class="uc-row-metric">' + UI.esc(ev.metric) + (ev.proxy ? " " + UI.proxyBadge() : "") + "</span>" +
      '<span class="uc-row-src source">' + UI.linkSources(ev.source || "") + "</span></div>" : "";
    return '<li class="uc-row">' +
      '<div class="uc-row-head"><h3 class="uc-row-title"><a href="./usecases.html#' + encodeURIComponent(u.id) + '">' + UI.esc(u.title) + "</a></h3>" +
      '<span class="pill pill-accent">' + UI.esc(u.recommended_mode || "") + "</span></div>" +
      '<p class="uc-row-job">' + UI.esc(u.job || "") + "</p>" + evidence +
      '<div class="uc-row-tags">' + tags + "</div></li>";
  }

  function start() {
    var demo = document.querySelector("[data-demo]");
    if (demo) initDemo(demo);
    var ul = document.querySelector("[data-uc-preview]");
    if (ul) initUseCases(ul);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
})();
