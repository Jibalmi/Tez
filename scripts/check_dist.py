"""Check what `python -m build` produced: the wheel holds the tez package and its licence files and nothing else, its
console scripts and extras, and its version is the one in tez/_version.py (which pyproject.toml reads). Used by CI and
the release workflow.

    python -m build && python scripts/check_dist.py dist
"""
from __future__ import annotations

import re
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("tez/__init__.py", "tez/cli.py", "tez/server.py", "tez/hooks.py", "tez/extract.py", "tez/mcp_server.py",
            "tez/integrations/__init__.py", "tez/integrations/langchain.py", "tez/integrations/remote.py",
            "tez/presets/__init__.py", "tez/presets/support-triage.yaml", "tez/presets/prompt-injection-guard.yaml")
EXTRAS = ("fit", "truncate", "langchain", "otel", "mcp", "all")
SCRIPTS = {"tez": "tez.cli:main", "tez-mcp": "tez.mcp_server:main"}
LICENSES = ("LICENSE", "LICENSES/Apache-2.0.txt", "THIRD_PARTY_NOTICES.md")     # tez/state.py is adapted from Laya


def source_versions() -> dict[str, str]:
    """{"tez/_version.py": version}: the one place the version lives (pyproject.toml reads it from there)."""
    code = (ROOT / "tez" / "_version.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', code, flags=re.M)
    return {"tez/_version.py": m.group(1) if m else "?"}


def check_wheel(path: Path) -> list[str]:
    problems = []
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        meta_name = next((n for n in names if n.endswith(".dist-info/METADATA")), None)
        meta = Parser().parsestr(z.read(meta_name).decode("utf-8")) if meta_name else None
        ep_name = next((n for n in names if n.endswith(".dist-info/entry_points.txt")), None)
        entry_points = z.read(ep_name).decode("utf-8") if ep_name else ""
    for script, target in SCRIPTS.items():
        if not re.search(rf"^{re.escape(script)}\s*=\s*{re.escape(target)}\s*$", entry_points, flags=re.M):
            problems.append(f"console script {script} = {target} is missing")
    for n in names:
        top = n.split("/", 1)[0]
        if not (top == "tez" or top.endswith(".dist-info")):
            problems.append(f"unexpected file outside tez/: {n}")
        if "__pycache__" in n or n.endswith((".pyc", ".pyo")):
            problems.append(f"compiled file in the wheel: {n}")
    for r in REQUIRED:
        if r not in names:
            problems.append(f"missing {r}")
    for lic in LICENSES:
        if not any(n.endswith(f".dist-info/licenses/{lic}") for n in names):
            problems.append(f"{lic} is not in the wheel's .dist-info/licenses/")
    if meta is None:
        problems.append("no METADATA")
        return problems
    versions = source_versions()
    for where, v in versions.items():
        if meta["Version"] != v:
            problems.append(f"wheel version {meta['Version']} != {v} in {where}")
    extras = set(meta.get_all("Provides-Extra") or [])
    missing = [e for e in EXTRAS if e not in extras]
    if missing:
        problems.append(f"extras missing from the metadata: {', '.join(missing)}")
    return problems


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    dist = Path(args[0] if args else ROOT / "dist")
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    problems = [] if wheels else [f"no wheel in {dist}"]
    if not sdists:
        problems.append(f"no sdist in {dist}")
    for w in wheels:
        problems += [f"{w.name}: {p}" for p in check_wheel(w)]
    for p in problems:
        print(p)
    if not problems:
        print(f"ok: {', '.join(w.name for w in wheels + sdists)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
