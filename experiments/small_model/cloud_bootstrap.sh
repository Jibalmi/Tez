#!/usr/bin/env bash
# Run only inside the dedicated Linux Pod after unpacking the checksum-verified project bundle.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
mkdir -p runs/cloud-runtime/models runs/cloud-runtime/llama
python -m pip install 'numpy==2.5.3' 'PyYAML==6.0.3' 'psutil==7.2.2' 'requests>=2.33.0'
python - <<'PY'
import hashlib, json, pathlib, tarfile, urllib.request
root = pathlib.Path.cwd() / 'runs/cloud-runtime'
sha = '047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795'
model = root / 'models' / (sha + '.gguf')
def download(url, path):
    temporary = path.with_suffix(path.suffix + '.part')
    with urllib.request.urlopen(url, timeout=120) as source, temporary.open('wb') as target:
        while block := source.read(8 * 1024 * 1024):
            target.write(block)
    temporary.replace(path)
def checksum(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()
if not model.exists():
    download('https://registry.ollama.ai/v2/library/gemma4/blobs/sha256:' + sha, model)
if checksum(model) != sha:
    raise SystemExit('Model checksum mismatch; refusing to run')
request = urllib.request.Request('https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b11100',
                                 headers={'User-Agent': 'tez-pilot-cloud'})
release = json.load(urllib.request.urlopen(request, timeout=30))
for name in ('llama-b11100-bin-ubuntu-cuda-12.8-x64.tar.gz', 'cudart-llama-b11100-bin-ubuntu-cuda-12.8-x64.tar.gz'):
    asset = next(a for a in release['assets'] if a['name'] == name)
    target = root / name
    if not target.exists():
        download(asset['browser_download_url'], target)
    expected = asset.get('digest')
    if not expected or not expected.startswith('sha256:') or checksum(target) != expected.split(':', 1)[1]:
        raise SystemExit('Release asset checksum missing or mismatched')
    with tarfile.open(target, 'r:gz') as archive:
        archive.extractall(root / 'llama', filter='data')
(root / 'sources.json').write_text(json.dumps({'model_sha256': sha, 'release': release['tag_name'],
    'assets': [{'name': a['name'], 'digest': a.get('digest')} for a in release['assets'] if a['name'] in (
        'llama-b11100-bin-ubuntu-cuda-12.8-x64.tar.gz','cudart-llama-b11100-bin-ubuntu-cuda-12.8-x64.tar.gz')]}, indent=2))
PY
export TEZ_PILOT_GGUF="$ROOT/runs/cloud-runtime/models/047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795.gguf"
export TEZ_PILOT_LLAMA_LIB
TEZ_PILOT_LLAMA_LIB=$(find "$ROOT/runs/cloud-runtime/llama" -name libllama.so -print -quit)
test -n "$TEZ_PILOT_LLAMA_LIB"
LIB_DIRS=$(find "$ROOT/runs/cloud-runtime/llama" -type d -printf '%p:' )
export LD_LIBRARY_PATH="${LIB_DIRS}${LD_LIBRARY_PATH:-}"
export RUN_HEAVY_THREADS=6 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python experiments/small_model/cloud_teacher.py run "$@"
