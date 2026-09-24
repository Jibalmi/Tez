"""A release tag must name the version in tez/_version.py, the one place the version lives: pyproject.toml reads it from
there ([project] dynamic = ["version"], [tool.setuptools.dynamic] version = {attr = "tez._version.__version__"}).
Tag v0.2.0 <-> __version__ = "0.2.0". Used by .github/workflows/release.yml before anything is built or published.

    python scripts/check_version.py v0.2.0
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTR = "tez._version.__version__"


def version(root: Path = ROOT) -> str | None:
    """__version__ in tez/_version.py."""
    code = (root / "tez" / "_version.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', code, flags=re.M)
    return m.group(1) if m else None


def _table(toml: str, name: str) -> str:
    """The body of one [table] of a TOML file (up to the next table header)."""
    m = re.search(rf"^\[{re.escape(name)}\]\s*$(.*?)(?=^\[|\Z)", toml, flags=re.M | re.S)
    return m.group(1) if m else ""


def pyproject_problems(root: Path = ROOT) -> list[str]:
    """pyproject.toml must take its version from tez/_version.py and not state one of its own."""
    toml = (root / "pyproject.toml").read_text(encoding="utf-8")
    project = _table(toml, "project")
    problems = []
    if re.search(r'^version\s*=', project, flags=re.M):
        problems.append("pyproject.toml states a version in [project]; it must read it from tez/_version.py "
                        '(dynamic = ["version"])')
    if not re.search(r'^dynamic\s*=\s*\[[^\]]*"version"', project, flags=re.M):
        problems.append('pyproject.toml: [project] must declare dynamic = ["version"]')
    dyn = _table(toml, "tool.setuptools.dynamic")
    if not re.search(rf'^version\s*=\s*\{{\s*attr\s*=\s*"{re.escape(ATTR)}"\s*\}}', dyn, flags=re.M):
        problems.append(f'pyproject.toml: [tool.setuptools.dynamic] must set version = {{attr = "{ATTR}"}}')
    return problems


def check(tag: str, root: Path = ROOT) -> list[str]:
    if not re.fullmatch(r"v\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?", tag):
        return [f"tag {tag!r} is not v<version> (for example v0.2.0 or v0.2.0rc1)"]
    wanted = tag[1:]
    problems = pyproject_problems(root)
    v = version(root)
    if v != wanted:
        problems.append(f"tez/_version.py says {v!r}, the tag says {wanted!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: check_version.py <tag>", file=sys.stderr)
        return 2
    problems = check(args[0])
    for p in problems:
        print(p)
    if not problems:
        print(f"ok: {args[0]} matches tez/_version.py (pyproject.toml reads its version from there)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
