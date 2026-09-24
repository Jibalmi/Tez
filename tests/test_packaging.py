"""Packaging around the runtime: CI and release workflows, compose files, the Docker entrypoint, the release and security
scripts, and the optional extras. Offline; skipped piece by piece when run outside a repository checkout."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
PINNED = re.compile(r"^[\w.-]+/[\w.-]+(/[\w.-]+)*@(v\d+|release/v\d+)$")


def load_script(relpath: str):
    path = ROOT / relpath
    if not path.exists():
        pytest.skip(f"{relpath} is not in this checkout")
    name = "tez_script_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module                    # registered first: dataclasses resolve cls.__module__
    spec.loader.exec_module(module)
    return module


def workflow(name: str) -> dict:
    path = WORKFLOWS / name
    if not path.exists():
        pytest.skip(f"{name} is not in this checkout")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    doc["on"] = doc.pop(True, doc.get("on"))      # YAML 1.1 reads the key `on` as a boolean
    return doc


def steps(job: dict) -> list[dict]:
    return job.get("steps", [])


# ---------------------------------------------------------------------------------------------- workflows
@pytest.mark.parametrize("name", ["ci.yml", "release.yml", "security.yml", "pages.yml"])
def test_every_action_is_pinned_to_a_major_version(name):
    doc = workflow(name)
    uses = [s["uses"] for job in doc["jobs"].values() for s in steps(job) if "uses" in s]
    uses += [job["uses"] for job in doc["jobs"].values() if "uses" in job]
    assert uses
    for ref in uses:
        assert ref.startswith("./.github/workflows/") or PINNED.match(ref), f"{name}: {ref} is not pinned to vN"


def test_ci_covers_the_promised_matrix():
    ci = workflow("ci.yml")
    assert {"push", "pull_request", "workflow_call"} <= set(ci["on"])
    test = ci["jobs"]["test"]
    assert test["strategy"]["matrix"]["os"] == ["ubuntu-latest", "windows-latest"]
    assert test["strategy"]["matrix"]["python"] == ["3.10", "3.11", "3.12", "3.13"]
    assert any('-m "not live"' in s.get("run", "") for s in steps(test))
    assert ci["jobs"]["ts-client"]["strategy"]["matrix"]["node"] == [20, 22]
    runs = " ".join(s.get("run", "") for job in ci["jobs"].values() for s in steps(job))
    for command in ("ruff check", "python -m build", "twine check", "scripts/check_dist.py", "npm test", "docker build"):
        assert command in runs


def test_release_tests_then_checks_the_tag_then_publishes_with_trusted_publishing():
    rel = workflow("release.yml")
    assert rel["on"]["push"]["tags"] == ["v*"]
    jobs = rel["jobs"]
    assert jobs["ci"]["uses"] == "./.github/workflows/ci.yml"
    assert jobs["build"]["needs"] == "ci" and jobs["publish"]["needs"] == "build"
    runs = [s.get("run", "") for s in steps(jobs["build"])]
    tag_check = next(i for i, r in enumerate(runs) if "check_version.py" in r)
    assert tag_check < next(i for i, r in enumerate(runs) if "python -m build" in r)
    publish = jobs["publish"]
    assert publish["permissions"] == {"id-token": "write"} and publish["environment"]["name"] == "pypi"
    assert any(s.get("uses", "").startswith("pypa/gh-action-pypi-publish@") for s in steps(publish))
    assert not any("password" in (s.get("with") or {}) for s in steps(publish))          # no stored token
    assert jobs["github-release"]["permissions"] == {"contents": "write"}
    assert rel["permissions"] == {"contents": "read"}


def test_security_workflow():
    sec = workflow("security.yml")
    runs = " ".join(s.get("run", "") for job in sec["jobs"].values() for s in steps(job))
    assert "scripts/check_banned_calls.py tez" in runs and "pip-audit" in runs and ".[all]" in runs
    assert "schedule" in sec["on"]


# ---------------------------------------------------------------------------------------------- compose and docker
def compose(name: str) -> dict:
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} is not in this checkout")
    text = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::?-([^}]*))?\}", lambda m: m.group(2) or "", path.read_text(encoding="utf-8"))
    return yaml.safe_load(text)


@pytest.mark.parametrize("name, gpu", [("compose.yaml", True), ("compose.cpu.yaml", False)])
def test_compose_files(name, gpu):
    doc = compose(name)
    llama, tez = doc["services"]["llama"], doc["services"]["tez"]
    image = llama["image"]
    assert image.startswith("ghcr.io/ggml-org/llama.cpp:server") and re.search(r"-b\d+$", image)   # pinned to a build
    assert ("cuda" in image) is gpu
    cmd = " ".join(llama["command"])
    assert cmd.startswith("-m /models/gemma-4-12b-it-Q8_0.gguf")
    for flag in ("-c 4096", "-b 512", "-np 1", "--swa-full", "--no-webui", "--host 0.0.0.0", "--port 8080"):
        assert flag in cmd
    assert ("-ngl 99" in cmd) is gpu
    assert ("deploy" in llama) is gpu
    if gpu:
        device = llama["deploy"]["resources"]["reservations"]["devices"][0]
        assert device["driver"] == "nvidia" and device["capabilities"] == ["gpu"]
    assert "/health" in " ".join(llama["healthcheck"]["test"]) and llama["volumes"] == ["./models:/models:ro"]
    assert tez["depends_on"] == {"llama": {"condition": "service_healthy"}}
    assert tez["ports"] == ["127.0.0.1:8787:8787"]                      # loopback only
    assert "/healthz" in " ".join(tez["healthcheck"]["test"])
    assert tez["environment"]["TEZ_BACKEND"] == "http://llama:8080"
    assert tez["build"] == {"context": ".", "dockerfile": "docker/Dockerfile"}


def test_dockerfile_runs_as_non_root_through_the_entrypoint():
    path = ROOT / "docker" / "Dockerfile"
    if not path.exists():
        pytest.skip("docker/Dockerfile is not in this checkout")
    text = path.read_text(encoding="utf-8")
    assert re.search(r"^FROM python:\$\{PYTHON_VERSION\}-slim$", text, flags=re.M)
    assert re.search(r"^USER 10001:10001$", text, flags=re.M)
    assert 'ENTRYPOINT ["python", "/usr/local/lib/tez/entrypoint.py"]' in text
    assert "HEALTHCHECK" in text and "/healthz" in text
    rules = [line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    assert rules[0] == "*"                                               # a whitelist: GGUFs in tools/ never sent
    assert {"!tez/", "!pyproject.toml", "!README.md", "!LICENSE", "!docker/entrypoint.py"} <= set(rules)


def test_entrypoint_reads_file_secrets(tmp_path):
    ep = load_script("docker/entrypoint.py")
    key = tmp_path / "key"
    key.write_text("s3cret\n", encoding="utf-8")
    env = ep.resolve_env({"TEZ_API_KEY_FILE": str(key), "TEZ_PORT": "9000", "OTHER": "x"})
    assert env == {"TEZ_API_KEY": "s3cret", "TEZ_PORT": "9000", "OTHER": "x"}
    with pytest.raises(ep.ConfigError, match="not both"):
        ep.resolve_env({"TEZ_API_KEY": "a", "TEZ_API_KEY_FILE": str(key)})
    with pytest.raises(ep.ConfigError, match="cannot read TEZ_SCHEMAS_FILE"):
        ep.resolve_env({"TEZ_SCHEMAS_FILE": str(tmp_path / "missing")})
    (tmp_path / "empty").write_text("\n", encoding="utf-8")
    with pytest.raises(ep.ConfigError, match="is empty"):
        ep.resolve_env({"TEZ_BACKEND_FILE": str(tmp_path / "empty")})


def test_entrypoint_command_line():
    ep = load_script("docker/entrypoint.py")
    assert ep.serve_argv({}) == ["tez", "serve", "--host", "0.0.0.0", "--port", "8787"]
    argv = ep.serve_argv({"TEZ_PORT": "9000", "TEZ_HOST": "127.0.0.1", "TEZ_SCHEMAS": "/schemas", "TEZ_DATA_DIR": "/data"},
                         ["--log-level", "warning"])
    assert argv == ["tez", "serve", "--host", "127.0.0.1", "--port", "9000", "--schemas", "/schemas", "--data-dir", "/data",
                    "--log-level", "warning"]
    for bad in ("http", "0", "70000", "-1"):
        with pytest.raises(ep.ConfigError, match="TEZ_PORT"):
            ep.serve_argv({"TEZ_PORT": bad})
    from tez.cli import build_parser               # every flag the entrypoint writes exists on `tez serve`
    args = build_parser().parse_args(argv[1:])
    assert (args.port, args.host, args.schemas, args.data_dir, args.log_level) == (9000, "127.0.0.1", "/schemas", "/data", "warning")


# ---------------------------------------------------------------------------------------------- scripts
def test_banned_calls_check_passes_on_the_runtime():
    check = load_script("scripts/check_banned_calls.py")
    assert check.check_paths([ROOT / "tez"]) == []


@pytest.mark.parametrize("code, reason", [
    ("import pickle", "import of pickle"),
    ("from joblib import load", "import from joblib"),
    ("import dill as d", "import of dill"),
    ("import torch\ntorch.load('m.pt')", "torch.load"),
    ("import yaml\nyaml.load(s)", "without a safe Loader"),
    ("import yaml\nyaml.load(s, Loader=yaml.Loader)", "without a safe Loader"),
    ("import yaml\nyaml.load_all(s, yaml.FullLoader)", "without a safe Loader"),
    ("import yaml\nyaml.unsafe_load(s)", "unsafe YAML"),
    ("from yaml import full_load\nfull_load(s)", "unsafe YAML"),
    ("eval('1 + 1')", "call to eval()"),
    ("exec(\n  'x = 1'\n)", "call to exec()"),
    ("import numpy as np\nnp.load(p)", "allow_pickle=False"),
    ("import numpy\nnumpy.load(p, allow_pickle=True)", "allow_pickle=False"),
])
def test_banned_calls_are_found(code, reason):
    check = load_script("scripts/check_banned_calls.py")
    found = check.check_source(code)
    assert found and reason in found[0][1]


@pytest.mark.parametrize("code", [
    "import yaml\nyaml.safe_load(s)",
    "import yaml\nyaml.load(s, Loader=yaml.SafeLoader)",
    "from yaml import CSafeLoader, load\nload(s, Loader=CSafeLoader)",
    "import yaml\nclass L(yaml.SafeLoader):\n    pass\nclass M(L):\n    pass\nyaml.load(s, Loader=M)",
    "import numpy as np\nnp.load(\n    p,\n    allow_pickle=False,\n)",
    "model.eval()\nevaluate(x)\n'pickle and eval( in a string'  # import pickle",
])
def test_safe_code_is_allowed(code):
    assert load_script("scripts/check_banned_calls.py").check_source(code) == []


def test_version_check():
    check = load_script("scripts/check_version.py")
    from tez import __version__
    assert check.check(f"v{__version__}") == []
    assert len(check.check("v99.0.0")) == 2
    assert "not v<version>" in check.check(__version__)[0]
    assert check.main([f"v{__version__}"]) == 0 and check.main(["v99.0.0"]) == 1


def test_presets_are_package_data():
    tomllib = pytest.importorskip("tomllib")
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]
    assert config["package-data"]["tez.presets"] == ["*.yaml"]
    assert "tez.*" in config["packages"]["find"]["include"]
    shipped = sorted(p.name for p in (ROOT / "tez" / "presets").glob("*.yaml"))
    assert len(shipped) == 10
    required = load_script("scripts/check_dist.py").REQUIRED
    assert "tez/presets/__init__.py" in required and any(r.startswith("tez/presets/") and r.endswith(".yaml") for r in required)


def test_extras():
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    assert extras["langchain"] == ["langchain-core>=0.3"]
    assert extras["otel"] == ["opentelemetry-api>=1.24"]
    assert extras["mcp"] == ["mcp>=1.2"]
    others = [dep for name, deps in extras.items() if name not in ("all", "test") for dep in deps]
    assert sorted(extras["all"]) == sorted(others)
    assert "langchain-core" not in " ".join(project["dependencies"])       # optional stays optional
