/* Shared behaviour for every page: theme, mobile menu, copy buttons, tabs, current nav link. */
(function () {
  "use strict";

  var root = document.documentElement;
  var THEME_KEY = "tez-theme";

  function readTheme() {
    try { return localStorage.getItem(THEME_KEY); } catch (e) { return null; }
  }
  function writeTheme(value) {
    try {
      if (value) localStorage.setItem(THEME_KEY, value); else localStorage.removeItem(THEME_KEY);
    } catch (e) { /* storage unavailable: the choice lasts for this page view */ }
  }
  function systemDark() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function effectiveTheme() {
    var t = root.getAttribute("data-theme");
    if (t === "light" || t === "dark") return t;
    return systemDark() ? "dark" : "light";
  }
  function applyStored() {
    var t = readTheme();
    if (t === "light" || t === "dark") root.setAttribute("data-theme", t);
  }
  applyStored();

  function syncThemeButtons() {
    var dark = effectiveTheme() === "dark";
    document.querySelectorAll("[data-theme-toggle]").forEach(function (btn) {
      btn.setAttribute("aria-pressed", dark ? "true" : "false");
      btn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
      btn.setAttribute("title", dark ? "Light theme" : "Dark theme");
    });
  }

  function init() {
    syncThemeButtons();
    document.querySelectorAll("[data-theme-toggle]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var next = effectiveTheme() === "dark" ? "light" : "dark";
        root.setAttribute("data-theme", next);
        writeTheme(next);
        syncThemeButtons();
        document.dispatchEvent(new CustomEvent("tez:themechange", { detail: { theme: next } }));
      });
    });

    var header = document.querySelector(".site-header");
    var toggle = document.querySelector("[data-nav-toggle]");
    if (header && toggle) {
      toggle.addEventListener("click", function () {
        var open = header.getAttribute("data-open") === "true";
        header.setAttribute("data-open", open ? "false" : "true");
        toggle.setAttribute("aria-expanded", open ? "false" : "true");
      });
      document.addEventListener("keydown", function (e) {
        if (e.key === "Escape" && header.getAttribute("data-open") === "true") {
          header.setAttribute("data-open", "false");
          toggle.setAttribute("aria-expanded", "false");
          toggle.focus();
        }
      });
    }

    // mark the header link for this page by its full path (the docs have their own index.html); pages that mark
    // their link in the markup, like every docs page marking "Docs", keep it
    function pagePath(url) { return url.pathname.replace(/\/index\.html$/i, "/").toLowerCase(); }
    var here = pagePath(location);
    document.querySelectorAll(".nav a[href]").forEach(function (a) {
      if (a.hasAttribute("aria-current")) return;
      var target;
      try { target = new URL(a.getAttribute("href"), location.href); } catch (e) { return; }
      if (target.origin === location.origin && pagePath(target) === here) a.setAttribute("aria-current", "page");
    });

    document.addEventListener("click", function (e) {
      var btn = e.target.closest("[data-copy]");
      if (!btn) return;
      var sel = btn.getAttribute("data-copy");
      var src = sel ? document.querySelector(sel) : btn.closest(".code") && btn.closest(".code").querySelector("pre");
      if (!src) return;
      var text = src.innerText.replace(/^\$ /gm, "");
      var label = btn.querySelector("[data-copy-label]");
      function done(ok) {
        if (!label) return;
        var old = label.textContent;
        label.textContent = ok ? "Copied" : "Press Ctrl+C";
        setTimeout(function () { label.textContent = old; }, 1600);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () { done(true); }, function () { done(false); });
      } else {
        done(false);
      }
    });

    document.querySelectorAll("[role='tablist']").forEach(function (list) {
      var tabs = Array.prototype.slice.call(list.querySelectorAll("[role='tab']"));
      function select(tab, focus) {
        tabs.forEach(function (t) {
          var on = t === tab;
          t.setAttribute("aria-selected", on ? "true" : "false");
          t.setAttribute("tabindex", on ? "0" : "-1");
          var panel = document.getElementById(t.getAttribute("aria-controls"));
          if (panel) panel.hidden = !on;
        });
        if (focus) tab.focus();
        list.dispatchEvent(new CustomEvent("tez:tab", { detail: { id: tab.id } }));
      }
      tabs.forEach(function (t, i) {
        t.addEventListener("click", function () { select(t, false); });
        t.addEventListener("keydown", function (e) {
          var j = null;
          if (e.key === "ArrowRight") j = (i + 1) % tabs.length;
          if (e.key === "ArrowLeft") j = (i - 1 + tabs.length) % tabs.length;
          if (e.key === "Home") j = 0;
          if (e.key === "End") j = tabs.length - 1;
          if (j !== null) { e.preventDefault(); select(tabs[j], true); }
        });
      });
    });
  }

  /* sideways-scrolling code: mark overflow so CSS can fade the edge until the reader reaches the end */
  function markOverflow() {
    document.querySelectorAll("pre").forEach(function (pre) {
      var over = pre.scrollWidth > pre.clientWidth + 1;
      pre.setAttribute("data-overflow", over ? "true" : "false");
      var end = !over || pre.scrollLeft + pre.clientWidth >= pre.scrollWidth - 2;
      pre.setAttribute("data-at-end", end ? "true" : "false");
      if (!pre.__tezScroll) {
        pre.__tezScroll = true;
        pre.addEventListener("scroll", function () {
          var e = pre.scrollLeft + pre.clientWidth >= pre.scrollWidth - 2;
          pre.setAttribute("data-at-end", e ? "true" : "false");
        }, { passive: true });
      }
    });
  }
  var overflowTimer = null;
  function scheduleOverflow() { clearTimeout(overflowTimer); overflowTimer = setTimeout(markOverflow, 120); }
  window.addEventListener("resize", scheduleOverflow);
  window.addEventListener("load", markOverflow);
  document.addEventListener("tez:tab", scheduleOverflow);
  if ("MutationObserver" in window) {
    new MutationObserver(scheduleOverflow).observe(document.documentElement, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ["hidden"] });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
