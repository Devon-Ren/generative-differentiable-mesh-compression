#!/usr/bin/env python3
"""make_batch38_files_v1.py (2026-11-14): the generic codecs on the NeRF-Synthetic objects.  Writes generic-rd-v3.sbatch from generic-rd-v2.sbatch:
adds the eight objects (black background, split 'train', positions q14, cap runs under <scene>-budgetb-<cap>-best) and lets a job also take
the model trained at the default cap ('full').  The captured scenes behave as in v2.
  python3 make_batch38_files_v1.py --sbatch <generic-rd-v2.sbatch> --out <dir>"""
import argparse, hashlib, pathlib, sys
def md5(s): return hashlib.md5(s.encode()).hexdigest()
def rep(s, old, new, what):
    n = s.count(old)
    if n != 1: sys.exit(f'FAILED: edit "{what}": anchor found {n} times (expected 1)')
    return s.replace(old, new)
ap = argparse.ArgumentParser(); ap.add_argument('--sbatch', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); s = pathlib.Path(a.sbatch).read_text()
if md5(s) != 'e7e3bd69c56855efda4dbd86d7c4de7d': sys.exit(f'FAILED: {a.sbatch} md5 {md5(s)}')
OBJ = "  lego|chair|drums|ficus|hotdog|materials|mic|ship) ORIG=/scratch/zr2245/$SCENE-blender4-best; DATA=$D/nerf_synthetic/$SCENE; IMAGES=train; QP=14; BUD=\"\"; CDIR=budgetb; CSUF=-best;;\n"
s = rep(s, "  *) echo \"FAILED: no settings for scene $SCENE\"; exit 1;;\n", OBJ + "  *) echo \"FAILED: no settings for scene $SCENE\"; exit 1;;\n", 'object cases')
s = rep(s, "SCENE=${SCENE:?set SCENE via --export}; BITS=${BITS:-\"8 6 4\"}\n", "SCENE=${SCENE:?set SCENE via --export}; BITS=${BITS:-\"8 6 4\"}; CDIR=budget; CSUF=\"\"\n", 'defaults')
s = rep(s, "  MK=/scratch/zr2245/$SCENE-budget-$M/axisB/v1.0/finetune/best.pt; [ -f $MK ] || MK=/scratch/zr2245/$SCENE-budget-$M/original/point_cloud/iteration_30000/point_cloud_state_dict.pt\n",
        "  CG=/scratch/zr2245/$SCENE-$CDIR-$M$CSUF; MK=$CG/axisB/v1.0/finetune/best.pt; [ -f $MK ] || MK=$CG/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; TAG=cap$M\n"
        "  if [ $M = full ]; then MK=$CK; TAG=full; fi\n", 'mesh path')
s = rep(s, "  O=$OUT/cap$M; mkdir -p $O\n", "  O=$OUT/$TAG; mkdir -p $O\n", 'out dir')
s = rep(s, "  python - \"$SCENE\" \"cap$M\" \"$O\" <<'PY' | tee -a $TAB\n", "  python - \"$SCENE\" \"$TAG\" \"$O\" <<'PY' | tee -a $TAB\n", 'row tag')
s = rep(s, "VERSION genrd-v2:", "VERSION genrd-v3 (adds the objects and 'full'); v2:", 'version')
(out / 'generic-rd-v3.sbatch').write_text(s); print(f'generic-rd-v3.sbatch: {len(s.splitlines())} lines md5 {md5(s)}')
