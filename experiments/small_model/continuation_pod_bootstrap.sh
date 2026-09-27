#!/usr/bin/env bash
# Run the registered Qwen continuation's training segments on a RunPod GPU, matching the laptop's software exactly.
# Usage on the Pod, after extracting bundle.tar into /workspace/tez-pilot:
#   bash experiments/small_model/continuation_pod_bootstrap.sh setup
#   bash experiments/small_model/continuation_pod_bootstrap.sh train <attempt> <checkpoint> <checkpoint-sha256>
set -euo pipefail
cd /workspace/tez-pilot
VENV=/workspace/venv
PY=$VENV/bin/python
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1

setup() {
  # 1. every bundled file against the laptop's manifest
  python3 - <<'EOF'
import hashlib, json, pathlib, sys
manifest = json.loads(pathlib.Path("/workspace/manifest.json").read_text())
bad = [n for n, h in manifest.items() if hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest() != h]
print(f"manifest: {len(manifest) - len(bad)}/{len(manifest)} files match")
sys.exit(1 if bad else 0)
EOF
  # 2. the laptop's interpreter line and pinned packages; no FLA or causal-conv1d (the laptop runs PyTorch's reference paths)
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  uv venv --python 3.14 "$VENV"
  uv pip install --python "$PY" torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
  uv pip install --python "$PY" transformers==5.17.0 peft==0.21.0 tokenizers==0.23.2 huggingface-hub==1.33.0 \
    accelerate==1.15.0 safetensors==0.8.0 numpy==2.5.3 psutil==7.2.2 PyYAML==6.0.3 regex==2026.9.10 scipy==1.18.1 \
    tqdm==4.70.1 requests==2.34.2 hf-xet==1.6.0 packaging==26.3 filelock==3.32.3 fsspec==2026.6.0 Jinja2==3.1.6
  if "$PY" -c "import fla" 2>/dev/null || "$PY" -c "import causal_conv1d" 2>/dev/null; then
    echo "optional kernels present; the registered runs use the reference paths" >&2; exit 1
  fi
  # 3. the pod's library files must equal the registered laptop copies byte for byte
  SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  for f in transformers/models/qwen3_5/modeling_qwen3_5.py peft/peft_model.py peft/tuners/lora/layer.py; do
    cmp "$SITE/$f" ".venv-pilot/Lib/site-packages/$f"
  done
  echo "library sources: identical to the registered laptop files"
  # 4. the pinned base model, then its registered hash
  "$PY" - <<'EOF'
import hashlib, json, pathlib
from huggingface_hub import snapshot_download
spec = json.loads(pathlib.Path("experiments/small_model/qwen_continuation_registration.json").read_text())
base = pathlib.Path(snapshot_download("Qwen/Qwen3.5-0.8B", revision=spec["base_revision"]))
digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in base.glob("*.safetensors")]
assert digests == [spec["base_sha256"]], "base weights differ from the registration"
import torch
print("base model verified;", torch.__version__, torch.cuda.get_device_name(), "CUDA", torch.version.cuda)
EOF
}

train() {
  local attempt=$1 checkpoint=$2 digest=$3
  RUN_HEAVY_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCH_NUM_THREADS=4 \
    "$PY" experiments/small_model/qwen_continuation.py --operation train --target-step 150 --attempt "$attempt" \
      --checkpoint "$checkpoint" --checkpoint-sha256 "$digest"
}

case "${1:-}" in
  setup) setup ;;
  train) shift; train "$@" ;;
  *) echo "usage: $0 setup | train <attempt> <checkpoint> <sha256>" >&2; exit 2 ;;
esac
