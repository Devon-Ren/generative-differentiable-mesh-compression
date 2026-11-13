#!/usr/bin/env python3
"""make_batch36_files_v1.py (2026-11-13): the generic codecs (zstd, Draco, MPEG TFAN) on meshes TRAINED with a vertex cap, instead of the FP32
model simplified after training (batch33).  Writes generic-rd-v2.sbatch from generic-rd-v1.sbatch: per scene three cap-trained meshes
(600,000 / the scene's cap near 1.08 million / 3,000,000), the fine-tuned mesh that our codec also encodes when it exists, else the trained one.
Rows go to the same table with 'mesh cap<N>'.   python3 make_batch36_files_v1.py --sbatch <generic-rd-v1.sbatch> --out <dir>"""
import argparse, hashlib, pathlib, sys
def md5(s): return hashlib.md5(s.encode()).hexdigest()
def rep(s, old, new, what):
    n = s.count(old)
    if n != 1: sys.exit(f'FAILED: edit "{what}": anchor found {n} times (expected 1)')
    return s.replace(old, new)
ap = argparse.ArgumentParser(); ap.add_argument('--sbatch', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); s = pathlib.Path(a.sbatch).read_text()
if md5(s) != 'a9ad66a9e7bd9a740b82f5b7bff42681': sys.exit(f'FAILED: {a.sbatch} md5 {md5(s)}')
MID = {'room': 1078000, 'kitchen': 1086000, 'counter': 1083000, 'bonsai': 1073000, 'garden': 1073000, 'bicycle': 1080000, 'stump': 1069000, 'flowers': 1071000, 'treehill': 1079000, 'truck': 1078000, 'train': 1085000}
CASE = 'case $SCENE in ' + ' '.join('%s) MIDCAP=%d;;' % kv for kv in MID.items()) + ' esac; CAPS=${CAPS:-"600000 $MIDCAP 3000000"}\n'
s = rep(s, "RUN=$ORIG/original; CK=", CASE + "RUN=$ORIG/original; CK=", 'caps')
s = rep(s, "simplified at $BUD |", "cap-trained meshes $CAPS |", 'banner')
s = rep(s, "for M in full $BUD; do\n", "for M in $CAPS; do\n", 'loop')
s = rep(s, "  if [ $M = full ]; then MK=$CK; else MK=$ORIG/axisB/g0ctrl-$M/finetune/best.pt; fi\n",
        "  MK=/scratch/zr2245/$SCENE-budget-$M/axisB/v1.0/finetune/best.pt; [ -f $MK ] || MK=/scratch/zr2245/$SCENE-budget-$M/original/point_cloud/iteration_30000/point_cloud_state_dict.pt\n"
        "  echo \"mesh cap $M: $MK\"\n", 'mesh path')
s = rep(s, "  O=$OUT/$M; mkdir -p $O\n", "  O=$OUT/cap$M; mkdir -p $O\n", 'out dir')
s = rep(s, "  python - \"$SCENE\" \"$M\" \"$O\" <<'PY' | tee -a $TAB\n", "  python - \"$SCENE\" \"cap$M\" \"$O\" <<'PY' | tee -a $TAB\n", 'row tag')
s = rep(s, "VERSION genrd-v1)", "VERSION genrd-v2: meshes trained with a vertex cap; text below is v1's)", 'version')
(out / 'generic-rd-v2.sbatch').write_text(s); print(f'generic-rd-v2.sbatch: {len(s.splitlines())} lines md5 {md5(s)}')
