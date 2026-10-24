#!/usr/bin/env bash
# One-time setup on the college GPU server (Linux). Safe to re-run: finished steps are skipped.
#   ./setup_server.sh            create .venv, install PyTorch + requirements, download models + data, check the GPU
# Everything stays inside this folder (venv, pip cache, Hugging Face cache), which lives in your team home folder.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"
export HF_HOME="$HERE/.hf_cache"
export PIP_CACHE_DIR="$HERE/.pip_cache"
export PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false
T0=$(date +%s)
say() { echo -e "\n==== $* ($(( ($(date +%s) - T0) / 60 )) min elapsed) ===="; }

say "GPU and driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then echo "[FAIL] nvidia-smi not found: is this the GPU node?"; exit 1; fi
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
DMAJ=${DRV%%.*}
df -h "$HERE" | tail -1

say "Internet access"
for u in https://pypi.org https://download.pytorch.org https://huggingface.co; do
  if python3 -c "import urllib.request,sys; urllib.request.urlopen(sys.argv[1], timeout=10)" "$u" 2>/dev/null; then
    echo "[ok] $u"; else echo "[warn] cannot reach $u"; fi
done

say "Python"
PY=""
for c in python3.12 python3.11 python3.13 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    PY="$c"; break
  fi
done
SYS_TORCH=0
if [ -n "$PY" ] && "$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  SYS_TORCH=1
  echo "[ok] the server already has PyTorch $("$PY" -c 'import torch; print(torch.__version__)') with CUDA: reusing it (no 3 GB download)"
  if [ -x .venv/bin/python ] && ! .venv/bin/python -c 'import torch' 2>/dev/null; then rm -rf .venv; fi
fi
if [ ! -x .venv/bin/python ]; then
  if [ -n "$PY" ]; then
    echo "using $PY ($($PY --version))"
    rm -rf .venv
    # --without-pip works even when the system lacks ensurepip/python3-venv (no sudo needed); pip is added below
    if [ "$SYS_TORCH" = 1 ]; then "$PY" -m venv --without-pip --system-site-packages .venv
    else "$PY" -m venv --without-pip .venv; fi
  elif command -v conda >/dev/null 2>&1; then
    echo "[info] no Python >= 3.10 found; creating a conda env in ./.venv"
    conda create -y -q -p "$HERE/.venv" python=3.12
  else
    echo "[FAIL] Python 3.10+ not found and conda is not available. Send the output of 'python3 --version' and 'ls /usr/bin/python*'."
    exit 1
  fi
fi
VPY="$HERE/.venv/bin/python"
"$VPY" --version
if ! "$VPY" -m pip --version >/dev/null 2>&1; then
  echo "[info] adding pip to the virtual environment (system Python has no ensurepip)"
  "$VPY" - <<'PYBOOT'
import json, os, subprocess, sys, tempfile, urllib.request
def get(url, timeout=60):
    return urllib.request.urlopen(url, timeout=timeout).read()
d = tempfile.mkdtemp()
try:   # 1) official bootstrap script
    p = os.path.join(d, "get-pip.py")
    open(p, "wb").write(get("https://bootstrap.pypa.io/get-pip.py"))
    sys.exit(subprocess.call([sys.executable, p, "-q"]))
except Exception as e:
    print(f"[info] get-pip.py unavailable ({type(e).__name__}); installing the pip wheel from PyPI")
info = json.loads(get("https://pypi.org/pypi/pip/json"))
url = next(f["url"] for f in info["urls"] if f["filename"].endswith(".whl"))
whl = os.path.join(d, url.rsplit("/", 1)[1])
open(whl, "wb").write(get(url))
sys.exit(subprocess.call([sys.executable, os.path.join(whl, "pip"), "install", "-q", "--no-index", whl]))
PYBOOT
fi
"$VPY" -m pip --version
"$VPY" -m pip install -q --upgrade pip

say "PyTorch (driver $DRV)"
if "$VPY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  echo "[ok] PyTorch already installed: $("$VPY" -c 'import torch; print(torch.__version__)')"
else
  if   [ "$DMAJ" -ge 570 ]; then IDX=cu128
  elif [ "$DMAJ" -ge 560 ]; then IDX=cu126
  else echo "[FAIL] NVIDIA driver $DRV is too old for current PyTorch (need >= 560; Blackwell GPUs need >= 570)"; exit 1
  fi
  echo "installing torch (a ~3 GB download, done once; it stays installed for later slots)"
  if [ "$DMAJ" -ge 580 ] && "$VPY" -m pip install --progress-bar on "torch==2.11.0"; then
    echo "[ok] torch from PyPI (CUDA 13 build; driver $DRV supports it)"
  elif ! "$VPY" -m pip install "torch==2.11.0" --index-url "https://download.pytorch.org/whl/$IDX"; then
    echo "[warn] torch 2.11.0/$IDX failed; trying the newest torch on the same index"
    if ! "$VPY" -m pip install torch --index-url "https://download.pytorch.org/whl/$IDX"; then
      if [ "$DMAJ" -ge 580 ]; then
        echo "[warn] download.pytorch.org unreachable; using the PyPI build (CUDA 13, driver $DRV is new enough)"
        "$VPY" -m pip install "torch==2.11.0"
      else
        echo "[FAIL] could not install PyTorch (no access to download.pytorch.org?)"; exit 1
      fi
    fi
  fi
fi

say "Python packages"
"$VPY" -m pip install -q -r requirements.txt

say "Models and data (Llama-3.2-3B ~6.4 GB, wikitext-2)"
if [ "${WITH_1B:-0}" = 1 ]; then
  "$VPY" scripts/prefetch.py --config configs/server_3b.yaml --config configs/server_1b.yaml
else
  "$VPY" scripts/prefetch.py --config configs/server_3b.yaml
fi

say "GPU + attention + 3B model check (offline, exactly how the runs will load the models)"
if HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "$VPY" scripts/s0_check_env.py --config configs/server_3b.yaml --model-check \
   && { [ "${WITH_1B:-0}" != 1 ] || HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "$VPY" scripts/s0_check_env.py --config configs/server_1b.yaml; }; then
  echo "[ok] offline loading works"
else
  echo "[warn] check failed in offline mode; retrying online (runs will then use the network)"
  rm -f .prefetch_ok
  "$VPY" scripts/s0_check_env.py --config configs/server_3b.yaml --model-check
fi

say "Setup finished"
echo "Next: ./run_server.sh --quick            (3-5 min smoke test)"
echo "      ./run_server.sh --until HH:MM       (full baseline; HH:MM = when your slot ends, e.g. 00:00)"
