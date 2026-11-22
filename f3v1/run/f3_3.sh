#!/usr/bin/env bash
# F3-3: Mip-360 room, real photos (images_2, COLMAP poses, MeshSplatting --eval split), DA3 depth. GPU node.
source "$(dirname "$0")/env.sh"; cd $F3
ROOM=/scratch/zr2245/datasets/m360/room
[ -d $ROOM/images_2 ] || { fail "room dataset not found at $ROOM"; return 1 2>/dev/null || exit 1; }
if [ -z "$MS_CKPT" ]; then
  echo "== looking for a MeshSplatting room checkpoint (override with MS_CKPT=...)"
  find /scratch/zr2245 -maxdepth 7 -path "*room*" -path "*iteration_30000*" -name point_cloud_state_dict.pt 2>/dev/null | head -n 10
  MS_CKPT=$(find /scratch/zr2245 -maxdepth 7 -path "*room*" -path "*iteration_30000*" -name point_cloud_state_dict.pt 2>/dev/null | grep -i "indoor" | head -n 1)
fi
echo "   MeshSplatting checkpoint for the off-trajectory comparison: ${MS_CKPT:-<none, comparison skipped>}"
echo "== F3-3 DA3 depth for room train views"
da3_env; export HF_HUB_OFFLINE=1
python scripts/depth_da3.py --kind colmap --data $ROOM --images images_2 --out $OUT/da3-room.npz > $OUT/da3-room.log 2>&1
grep -q "DONE" $OUT/da3-room.log || { fail "DA3 room" $OUT/da3-room.log; return 1 2>/dev/null || exit 1; }
grep "scale vs sparse\|DONE" $OUT/da3-room.log
main_env
echo "== F3-3 smoke"
python scripts/run_f3.py --kind colmap --data $ROOM --images images_2 --depth $OUT/da3-room.npz --out $OUT/f3-3-smoke --smoke > $OUT/f3-3-smoke.log 2>&1
grep -q "== DONE" $OUT/f3-3-smoke.log || { fail "F3-3 smoke" $OUT/f3-3-smoke.log; return 1 2>/dev/null || exit 1; }
echo "== F3-3 full (log: $OUT/f3-3.log)"
python scripts/run_f3.py --kind colmap --data $ROOM --images images_2 --depth $OUT/da3-room.npz --out $OUT/f3-3 ${MS_CKPT:+--ms_ckpt $MS_CKPT} > $OUT/f3-3.log 2>&1
grep -q "== DONE" $OUT/f3-3.log || { fail "F3-3 full" $OUT/f3-3.log; return 1 2>/dev/null || exit 1; }
grep "^\[5\]\|^\[2\] surface\|^\[3\]" $OUT/f3-3.log
