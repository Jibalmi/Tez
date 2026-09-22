"""Deterministic slot extraction + action execution for the 16 voice intents.

The model's job is ONLY the typed decision (which intent). Everything after that is plain
code: slot extraction is regex over the transcript, and execution is OS calls. This is the
point of the architecture -- nothing is generated, so the action side is exact, testable
and adds ~0 ms.

Default is DRY RUN (prints the action). Pass execute=True / --execute to really do it.

Windows only for the executors (notepad, media keys via pynput, spotify: URIs, default
browser). Slot extraction is platform-independent.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------- slots
_FILLER_WORDS = ["um", "uh", "er", "erm", "hmm", "like", "please", "just", "now", "for me", "hey", "ok", "okay", "can you", "could you", "would you"]
_PLATFORMS = ["youtube", "you tube", "spotify", "google", "the web", "the internet", "online", "the browser", "chrome", "edge"]
_APPS = ["notes", "notepad", "spotify", "youtube", "browser", "chrome", "edge", "firefox", "calculator", "explorer", "word", "excel", "music", "the music"]


def _strip_fillers(s: str) -> str:
    for w in _FILLER_WORDS:
        s = re.sub(rf"(?<!\w){re.escape(w)}(?!\w)", " ", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip(" \"'.,!?")


def _after(text: str, verbs: list[str]) -> str | None:
    """Text after the FIRST occurrence of the longest-matching verb phrase; a repeated verb
    ('play play adele') is collapsed first."""
    t = re.sub(r"\b(\w+(?:\s+\w+)?)(\s+\1\b)+", r"\1", text, flags=re.I)
    best = None
    for v in sorted(verbs, key=len, reverse=True):
        m = re.search(rf"(?<!\w){re.escape(v)}(?!\w)", t, re.I)
        if m and (best is None or m.start() < best[0] or (m.start() == best[0] and m.end() > best[1])):
            best = (m.start(), m.end())
    return t[best[1]:] if best else None


def _drop_platform(s: str) -> str:
    for p in _PLATFORMS:
        s = re.sub(rf"\b(?:on|in|from|via|with)\s+{re.escape(p)}\b", " ", s, flags=re.I)
        s = re.sub(rf"^\s*{re.escape(p)}\s+", " ", s, flags=re.I)
    return s


def _first_app(text: str) -> str | None:
    for a in sorted(_APPS, key=len, reverse=True):
        if re.search(rf"(?<!\w){a}(?!\w)", text, re.I):
            return a.replace("the ", "")
    return None


def extract_slots(intent: str, text: str) -> dict:
    t = _strip_fillers(text)
    if intent == "type_text":
        body = _after(t, ["type in", "type out", "type", "write down", "write", "dictate", "enter", "put in", "put", "jot down", "note down", "add"])
        if body is not None:
            body = re.sub(r"^\s*(?:the text|the following|the words|this|that|these words|in it|in there|in notes|in notepad)[:,]?\s*", "", body, flags=re.I)
        return {"text": _strip_fillers(body or "")}
    if intent == "play_youtube":
        q = _after(t, ["search for", "search", "play", "watch", "put on", "stick on", "chuck on", "show me", "find", "look for", "pull up", "bring up"])
        if q is None and re.search(r"\byou ?tube\b", t, re.I):
            q = re.sub(r"^.*?\byou ?tube\b", "", t, flags=re.I)
        q = _drop_platform(q or "")
        q = re.sub(r"^\s*(?:for\s+)", "", q, flags=re.I)
        q = re.sub(r"^\s*(?:me\s+)?(?:a|the|some|that)?\s*(?:videos?|clips?)\s+(?:of|about|on|called)\s+", "", q, flags=re.I)
        q = re.sub(r"^\s*(?:that|some)\s+", "", q, flags=re.I)
        q = re.sub(r"(?<!latest)(?<!new)\s+videos?$", "", q, flags=re.I)
        return {"query": _strip_fillers(q)}
    if intent == "play_spotify":
        q = _after(t, ["play", "put on", "stick on", "chuck on", "listen to", "wanna hear", "want to hear", "hear", "queue up", "queue"])
        if q is None and re.search(r"\bspotify\b", t, re.I):
            q = re.sub(r"^.*?\bspotify\b", "", t, flags=re.I)
        q = _drop_platform(q or "")
        q = re.sub(r"^\s*(?:some|a bit of|a little|songs by|music by|something by|tracks by)\s+", "", q, flags=re.I)
        return {"query": _strip_fillers(q)}
    if intent == "search_web":
        q = _after(t, ["search the web for", "search for", "search", "look up", "look online for", "have a look online for", "have a look for", "google", "find out", "hunt down", "find", "check", "look for", "what is", "what's", "who is", "who's", "how do i", "how to", "i need", "show me", "pull up"])
        q = _drop_platform(q or "")
        q = re.sub(r"^\s*(?:for|about|on)\s+", "", q, flags=re.I)
        return {"query": _strip_fillers(q)}
    if intent == "close_app":
        app = _after(t, ["close", "quit", "exit", "kill", "shut down", "shut", "get rid of"])
        app = (app or "").strip()
        app = re.sub(r"^(?:down\s+|the\s+|this\s+|that\s+)*", "", app, flags=re.I)
        app = re.sub(r"\s+(?:down|app|application|window|program|for me)$", "", app, flags=re.I)
        app = _strip_fillers(app)
        if not app or app.lower() in ("it", "this", "that", "this one", "current", "the window"):
            app = _first_app(t) or "current"
        return {"app": app}
    return {}


# ----------------------------------------------------------------------------- actions
@dataclass
class Action:
    intent: str
    slots: dict
    steps: list[str] = field(default_factory=list)   # human-readable log of what was (or would be) done
    ok: bool = True
    ms: float = 0.0


_APP_EXE = {
    "notes": "notepad.exe", "notepad": "notepad.exe", "spotify": "Spotify.exe", "youtube": None,
    "browser": None, "chrome": "chrome.exe", "edge": "msedge.exe", "firefox": "firefox.exe",
    "calculator": "CalculatorApp.exe", "explorer": "explorer.exe", "word": "WINWORD.EXE", "excel": "EXCEL.EXE",
}


def _media_key(name: str, execute: bool, steps: list[str]):
    steps.append(f"press media key {name}")
    if execute:
        from pynput.keyboard import Controller, Key
        Controller().tap(getattr(Key, name))


def _start(cmd: str, execute: bool, steps: list[str]):
    steps.append(f"start {cmd}")
    if execute:
        os.startfile(cmd)  # noqa: S606 -- intended shell association launch


def run_action(intent: str, text: str, execute: bool = False, type_delay: float = 0.0) -> Action:
    t0 = time.perf_counter()
    slots = extract_slots(intent, text)
    a = Action(intent=intent, slots=slots)
    s = a.steps
    try:
        if intent == "open_notes":
            _start("notepad.exe", execute, s)
        elif intent == "type_text":
            body = slots.get("text", "")
            if not body:
                a.ok = False; s.append("nothing to type")
            else:
                s.append(f"type {body!r} into the focused window")
                if execute:
                    from pynput.keyboard import Controller
                    time.sleep(type_delay)
                    Controller().type(body)
        elif intent == "open_youtube":
            s.append("open https://www.youtube.com"); execute and webbrowser.open("https://www.youtube.com")
        elif intent == "play_youtube":
            q = slots["query"]
            if not q:
                a.ok = False; s.append("no query")
            else:
                url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(q)
                s.append(f"open {url}"); execute and webbrowser.open(url)
        elif intent == "open_spotify":
            _start("spotify:", execute, s)
        elif intent == "play_liked_songs":
            _start("spotify:collection:tracks:play", execute, s)   # liked songs; :play autoplays on desktop client
        elif intent == "play_spotify":
            q = slots["query"]
            if not q:
                a.ok = False; s.append("no query")
            else:
                _start("spotify:search:" + urllib.parse.quote(q), execute, s)
        elif intent in ("pause_media", "resume_media"):
            _media_key("media_play_pause", execute, s)
        elif intent == "next_track":
            _media_key("media_next", execute, s)
        elif intent == "volume_up":
            for _ in range(3):
                _media_key("media_volume_up", execute, s)
        elif intent == "volume_down":
            for _ in range(3):
                _media_key("media_volume_down", execute, s)
        elif intent == "open_browser":
            s.append("open default browser"); execute and webbrowser.open("about:blank")
        elif intent == "search_web":
            q = slots["query"]
            if not q:
                a.ok = False; s.append("no query")
            else:
                url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(q)
                s.append(f"open {url}"); execute and webbrowser.open(url)
        elif intent == "close_app":
            app = slots["app"]
            exe = _APP_EXE.get(app.lower())
            if app == "current" or exe is None:
                s.append("send Alt+F4 to the focused window")
                if execute:
                    from pynput.keyboard import Controller, Key
                    k = Controller()
                    with k.pressed(Key.alt):
                        k.tap(Key.f4)
            else:
                s.append(f"taskkill /IM {exe}")
                if execute:
                    subprocess.run(["taskkill", "/IM", exe, "/F"], capture_output=True, check=False)
        elif intent == "none":
            s.append("no action (out of scope)")
        else:
            a.ok = False; s.append(f"unknown intent {intent}")
    except Exception as exc:  # noqa: BLE001
        a.ok = False; s.append(f"error: {exc}")
    a.ms = (time.perf_counter() - t0) * 1000
    return a


# ----------------------------------------------------------------------------- self-test
def selftest(commands_path: str) -> dict:
    """Slot extraction against the labelled slots in data/voice/commands.jsonl. For compound
    rows the slot belongs to the terminal ('then') intent, which is what the executor would
    run after the first action."""
    rows = [json.loads(l) for l in open(commands_path, encoding="utf-8") if l.strip()]
    n = ok = 0
    misses = []
    for r in rows:
        if not r.get("slots"):
            continue
        intent = r.get("then") or r["intent"]
        got = extract_slots(intent, r["text"])
        for k, v in r["slots"].items():
            n += 1
            g = (got.get(k) or "").lower().strip()
            if g == str(v).lower().strip():
                ok += 1
            else:
                misses.append((r["id"], r["style"], r["text"], k, v, got.get(k)))
    return {"slots_checked": n, "exact": ok, "exact_rate": ok / n if n else None, "misses": misses}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--intent")
    ap.add_argument("--text", default="")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--selftest", default="")
    args = ap.parse_args()
    if args.selftest:
        r = selftest(args.selftest)
        print(json.dumps({k: v for k, v in r.items() if k != "misses"}))
        for m in r["misses"]:
            print("  MISS", m)
        sys.exit(0)
    a = run_action(args.intent, args.text, execute=args.execute)
    print(json.dumps(a.__dict__, ensure_ascii=False, indent=2))
