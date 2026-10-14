#!/usr/bin/env bash
# Build the NDT3 environment inside the Ubuntu-24.04 WSL distro (stored on H:\wsl\ndt3). Run as root:
#   wsl -d Ubuntu-24.04 -u root -- bash "/mnt/g/codes/invasive bci/scripts/ndt3/setup_wsl.sh"
# Mirrors the authors' setup (https://github.com/joel99/ndt3 README): Python 3.11, torch 2.1.0+cu118, FlashAttention 2,
# with the official prebuilt FlashAttention wheel for that combination (no compilation). NDT3 code pinned to a commit.
set -euo pipefail
trap 'echo "EXIT=$?"' EXIT
NDT3_COMMIT=877f9f5
FA_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

apt-get update -qq
apt-get install -y -qq git curl build-essential > /dev/null
if ! command -v uv > /dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

cd /opt
[ -d ndt3 ] || git clone -q https://github.com/joel99/ndt3.git
git -C ndt3 checkout -q "$NDT3_COMMIT"

uv python install 3.11
[ -d /opt/ndt3-venv ] || uv venv -q --python 3.11 /opt/ndt3-venv
. /opt/ndt3-venv/bin/activate
uv pip install -q "setuptools==69.5.1" wheel packaging ninja
uv pip install -q torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
uv pip install -q "numpy<2"
uv pip install -q "$FA_WHEEL"
uv pip install -q -e /opt/ndt3 --extra-index-url https://download.pytorch.org/whl/cu118
# The unpinned requirements resolve to releases that need torch >= 2.5 (transformers), and the PyTorch index only
# hosts torchmetrics <= 1.0.3 (lacks text.EditDistance). Pin mid-2024 releases from PyPI that support torch 2.1;
# tensordict 0.2.1 is the release built for torch 2.1 (NDT3 uses only TensorDict in analysis helpers);
# --no-deps so torch 2.1.0+cu118 is never replaced; `uv pip check` below verifies the set is consistent.
uv pip install -q --no-deps torchmetrics==1.4.0.post0 lightning==2.2.5 pytorch-lightning==2.2.5 transformers==4.41.2   peft==0.11.1 tensordict==0.2.1 timm==1.0.3 tokenizers==0.19.1 lightning-utilities==0.11.2 fsspec==2024.6.1 packaging==24.1
uv pip install -q "huggingface_hub<0.24" h5py
# Allowed: NDT3's setup.py asks tensordict>=0.3.0, but 0.3.x needs torch 2.2 (see above). Anything else fails.
chk=$(uv pip check 2>&1 || true); echo "$chk"
if echo "$chk" | grep "requires" | grep -v "requires \`tensordict>=0.3.0\`" | grep -q .; then echo "dependency conflict"; exit 1; fi

python - <<'EOF'
import torch, flash_attn
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("flash_attn", flash_attn.__version__)
from flash_attn import flash_attn_func
q = torch.randn(1, 64, 4, 32, device="cuda", dtype=torch.bfloat16)
print("flash_attn_func ok", tuple(flash_attn_func(q, q, q).shape))
import context_general_bci.model  # noqa: F401
print("NDT3 imports ok")
EOF
echo "NDT3 environment ready: /opt/ndt3-venv (code /opt/ndt3 @ $NDT3_COMMIT)"
