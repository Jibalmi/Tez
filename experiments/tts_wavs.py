"""Synthesise the voice commands to 16 kHz mono WAV with Windows SAPI so the ASR + decision
loop can be timed on real audio without recording sessions. Synthetic speech is cleaner than
a laptop mic, so treat ASR numbers as a lower bound on error, not a field result.

Usage: python experiments/tts_wavs.py --out data/voice/wav --limit 0 [--rate 0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "voice" / "wav"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rate", type=int, default=0, help="SAPI rate -10..10 (0 = normal, ~150 wpm)")
    ap.add_argument("--ids", default="", help="comma-separated ids to synthesise")
    args = ap.parse_args()

    import comtypes.client  # noqa: PLC0415
    from comtypes.gen import SpeechLib  # noqa: PLC0415  (generated on first import of SAPI below)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (ROOT / "data" / "voice" / "commands.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.ids:
        keep = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in keep]
    if args.limit:
        rows = rows[: args.limit]

    voice = comtypes.client.CreateObject("SAPI.SpVoice")
    voice.Rate = args.rate
    for r in rows:
        stream = comtypes.client.CreateObject("SAPI.SpFileStream")
        fmt = comtypes.client.CreateObject("SAPI.SpAudioFormat")
        fmt.Type = SpeechLib.SAFT16kHz16BitMono
        stream.Format = fmt
        stream.Open(str(out / f"{r['id']}.wav"), SpeechLib.SSFMCreateForWrite)
        voice.AudioOutputStream = stream
        voice.Speak(r["text"])
        stream.Close()
    print(f"wrote {len(rows)} wavs to {out}")


if __name__ == "__main__":
    # comtypes needs the SAPI typelib generated once; importing the object first does it.
    import comtypes.client
    comtypes.client.CreateObject("SAPI.SpVoice")
    main()
