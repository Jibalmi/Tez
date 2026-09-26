"""The shared-server arm of the voice replay: a llama-server of our own with the settings of Theioma's shared 12B, and
the replay's tez.stream session reading it over HTTP (the whole prompt per word, the server's prompt cache reusing the
prefix), instead of the model in process.

Theioma's coordinator runs Gemma 4 12B Q4_K_M on llama-server :8797 with `-ngl 99 -c 4096 -np 2 --swa-full -b 512`.
That port is in Theioma's fleet range, so this arm starts the same binary and GGUF with the same flags on a port of its
own (default 8873, outside 3000 and 8791-8818), waits for /health, runs experiments/voice_replay.py against it, and
stops the server. Any other arguments go to voice_replay.py.

  python experiments/voice_server_arm.py --gguf <Q4_K_M blob> -- --set theioma --asr oracle --tag th_oracle_12b_q4km_http
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--server", default="C:/temp/llamacpp/llama-server.exe")
    ap.add_argument("--port", type=int, default=8873)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if 8791 <= a.port <= 8818 or a.port == 3000:
        ap.error("that port belongs to Theioma's fleet")
    rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
    cmd = [a.server, "-m", a.gguf, "-ngl", "99", "-c", "4096", "-np", "2", "--swa-full", "-b", "512", "-t", str(a.threads),
           "--port", str(a.port), "--host", "127.0.0.1", "--no-webui"]
    log = open(ROOT / "results" / "voice" / f"llama_server_{a.port}.log", "w", encoding="utf-8")
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{a.port}"
    try:
        t0 = time.time()
        while True:
            if srv.poll() is not None:
                print("llama-server exited", file=sys.stderr)
                return 2
            try:
                if requests.get(f"{url}/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.time() - t0 > 300:
                print("llama-server did not come up", file=sys.stderr)
                return 2
            time.sleep(1)
        print(f"llama-server up on {url} in {time.time() - t0:.0f} s: {' '.join(cmd)}", flush=True)
        return subprocess.call([sys.executable, str(ROOT / "experiments" / "voice_replay.py"), "--backend", url,
                                "--server-cmd", " ".join(cmd), *rest])
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
