#!/usr/bin/env bash
# Run ONCE on a LOGIN node (needs internet). Installs: lpips (+VGG weights) into the main env; a separate DA3 env with
# its own torch/xformers (never touches the MeshSplatting env); downloads DA3-LARGE-1.1 weights into $HF_HOME.
source "$(dirname "$0")/env.sh"
echo "== [1/4] main env: lpips + scikit-image check"
main_env
python -c "import skimage, scipy, PIL; print('skimage', skimage.__version__, 'scipy', scipy.__version__)" || pip install scikit-image
python -c "import lpips" 2>/dev/null || pip install --no-deps lpips
python -c "import lpips, torch; lpips.LPIPS(net='vgg', verbose=False); print('lpips + VGG weights cached')" || fail "lpips warm-up"
echo "== [2/4] DA3 env at $DA3_ENV (separate; python 3.11)"
if [ ! -x "$DA3_ENV/bin/python" ]; then conda create -y -p $DA3_ENV python=3.11 || fail "conda create"; fi
da3_env
python -c "import torch, xformers" 2>/dev/null || pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu126 || fail "torch/xformers for DA3"
echo "== [3/4] DA3 code"
if [ ! -d $MM/Depth-Anything-3 ]; then git clone https://github.com/ByteDance-Seed/Depth-Anything-3 $MM/Depth-Anything-3 || fail "git clone DA3"; fi
(cd $MM/Depth-Anything-3 && pip install -e .) || fail "pip install DA3"
python -c "import numpy, PIL, scipy" 2>/dev/null || pip install scipy
echo "== [4/4] DA3 weights -> $HF_HOME"
python -c "from huggingface_hub import snapshot_download; p=snapshot_download('depth-anything/DA3-LARGE-1.1'); print('weights at', p)" || fail "weights download"
python -c "from depth_anything_3.api import DepthAnything3; print('DA3 import OK')" || fail "DA3 import"
echo "== install_login.sh finished (check above for FAILED lines)"
