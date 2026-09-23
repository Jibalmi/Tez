"""Capture the website for review: every page at desktop (1440) and mobile (390), full page, plus console errors.

Serves site/ on a local port, loads each page in headless Chromium with reduced motion (final states, no mid-animation
frames), waits for fonts and network idle, and writes PNGs plus a JSON log of console errors and failed requests.

  py scripts/capture_site.py --out .impeccable/review
  py scripts/capture_site.py --out .impeccable/review --pages index --theme dark
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["index", "usecases", "playground", "benchmarks", "docs", "research"]
VIEWPORTS = {"desktop": (1440, 900), "mobile": (390, 844)}


def serve(directory: Path, port: int):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    handler.log_message = lambda *a, **k: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / ".impeccable" / "review"))
    ap.add_argument("--pages", default=",".join(PAGES))
    ap.add_argument("--port", type=int, default=5507)
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    ap.add_argument("--viewports", default="desktop,mobile")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    httpd = serve(ROOT / "site", args.port)
    log = {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for vp in args.viewports.split(","):
                w, h = VIEWPORTS[vp]
                ctx = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=1,
                                          color_scheme=args.theme, reduced_motion="reduce")
                for name in args.pages.split(","):
                    page = ctx.new_page()
                    errors, failed = [], []
                    page.on("console", lambda m, e=errors: e.append(m.text) if m.type == "error" else None)
                    page.on("pageerror", lambda exc, e=errors: e.append(f"pageerror: {exc}"))
                    page.on("requestfailed", lambda r, f=failed: f.append(f"{r.url} {r.failure}"))
                    page.on("response", lambda r, f=failed: f.append(f"{r.status} {r.url}") if r.status >= 400 else None)
                    page.goto(f"http://127.0.0.1:{args.port}/{name}.html", wait_until="networkidle")
                    page.evaluate("document.fonts.ready")
                    page.wait_for_timeout(600)
                    suffix = "" if args.theme == "light" else "-dark"
                    fname = f"{vp}{suffix}.png" if name == "index" else f"{name}-{vp}{suffix}.png"
                    page.screenshot(path=str(out / fname), full_page=True)
                    dims = page.evaluate("[document.documentElement.scrollWidth, document.documentElement.scrollHeight, window.innerWidth]")
                    log[f"{name}:{vp}:{args.theme}"] = dict(file=fname, scroll_width=dims[0], height=dims[1], viewport=dims[2],
                                                           horizontal_overflow=dims[0] > dims[2], errors=errors, failed=failed)
                    print(f"{fname:32s} {dims[0]}x{dims[1]} overflow={dims[0] > dims[2]} errors={len(errors)} failed={len(failed)}", flush=True)
                    page.close()
                ctx.close()
            browser.close()
    finally:
        httpd.shutdown()
    (out / f"capture-log-{args.theme}.json").write_text(json.dumps(log, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
