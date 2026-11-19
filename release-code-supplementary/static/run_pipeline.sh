#!/usr/bin/env bash
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAB=$(realpath "$1")
DATA=$(realpath "$2")
IMAGES=$3
TARGET_MB=$4
XYZ=$5
OUT=$(realpath -m "$6")
INDOOR=${7:-0}

export PYTHONPATH="$ROOT/codec/renderer:$ROOT/codec:$ROOT/codec/codec:$LAB/upstream/submodules/effrdel/src:${PYTHONPATH:-}"
FLAG=""
[ "$INDOOR" = 1 ] && FLAG="--indoor"
MODEL=$OUT/original
CKPT=$MODEL/point_cloud/iteration_30000/point_cloud_state_dict.pt
MESH=$OUT/mesh
mkdir -p "$OUT"

LAM=$(python -c "print('%.4e' % min(1e-1, max(1e-4, 6.2e-3 * (9.0 / $TARGET_MB) ** 1.67)))")
python "$ROOT/training/train_bytemesh.py" --lab "$LAB" -s "$DATA" -i "$IMAGES" -r 1 --eval --data_device cpu -m "$MODEL" --iterations 30000 \
  --lam "$LAM" --target_mb "$TARGET_MB" --bytes_per_vertex 10.0 --bytes_per_face 2.0 --faces_per_vertex 2.5 --dd_mode error --dd_norm mpx \
  --phase_a_overshoot 1.35 --probe_floor 0.9999 --probe_scaling 4 --sigma_prior 0.5 \
  --prune_iters $(seq -s " " 12500 500 29500) --prune_mu 1.0 --prune_max_frac 0.15 --probe_extra_iters --probe_a_every 1000 \
  --probe_cmd "python $ROOT/training/size_probe.py --stage2 $ROOT/codec" --probe_k 4096 --probe_r 256 --probe_xyz "$XYZ" --probe_dc q7 \
  --trim_exact 5 --trim_hold --allow_any_prune_iter $FLAG

(cd "$ROOT/postprocess" && python vertex_footprint.py --lab "$LAB" --model "$MODEL" --dataset "$DATA" --images "$IMAGES" --out "$OUT/footprint.npz")
VERTICES=$(python -c "import torch; print(len(torch.load('$CKPT', map_location='cpu', weights_only=False)['triangles_points']))")
(cd "$ROOT/postprocess" && python coarse_vis.py --source "$CKPT" --footprint "$OUT/footprint.npz" --base uniform --target-vertices "$VERTICES" --output "$MESH" \
  --geo-weight sens_pos --app-weight none --weight-power 0.5 --weight-clip 0.1,10 --weight-divide-pixel-scale)
python "$ROOT/postprocess/finetune.py" --lab "$LAB" --model "$MODEL" --dataset "$DATA" --images "$IMAGES" --coarse "$MESH/coarse-initial.pt" --steps 3000 --out "$MESH/finetune"

(cd "$ROOT/codec" && python "$ROOT/encode.py" --stage2 "$ROOT/codec" --coarse "$MESH/finetune/best.pt" --output "$MESH/deliver" --historical-config none \
  --source "$CKPT" --dataset "$DATA" --lab "$LAB" --images "$IMAGES" --k 4096 --r 256 --xyz "$XYZ" --dc q7 --dictionary-rounds 4)

ASSET=$(find "$MESH/deliver" -name "*.mslb" | head -1)
python "$ROOT/decode.py" --asset "$ASSET" --out "$OUT/decoded.pt"
ls -l "$ASSET"
