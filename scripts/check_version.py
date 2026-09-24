"""A release tag must name the version in tez/_version.py and pyproject.toml: tag v0.2.0 <-> version "0.2.0".
Used by .github/workflows/release.yml before anything is built or published.

    python scripts/check_version.py v0.2.0
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def versions(root: Path = ROOT) -> dict[str, str | None]:
    code = (root / "tez" / "_version.py").read_text(encoding="utf-8")
    toml = (root / "pyproject.toml").read_text(encoding="utf-8")
    m1 = re.search(r'^__version__\s*=\s*"([^"]+)"', code, flags=re.M)
    m2 = re.search(r'^version\s*=\s*"([^"]+)"', toml, flags=re.M)
    return {"tez/_version.py": m1.group(1) if m1 else None, "pyproject.toml": m2.group(1) if m2 else None}


def check(tag: str, root: Path = ROOT) -> list[str]:
    if not re.fullmatch(r"v\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?", tag):
        return [f"tag {tag!r} is not v<version> (for example v0.2.0 or v0.2.0rc1)"]
    wanted = tag[1:]
    return [f"{where} says {v!r}, the tag says {wanted!r}" for where, v in versions(root).items() if v != wanted]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: check_version.py <tag>", file=sys.stderr)
        return 2
    problems = check(args[0])
    for p in problems:
        print(p)
    if not problems:
        print(f"ok: {args[0]} matches tez/_version.py and pyproject.toml")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
