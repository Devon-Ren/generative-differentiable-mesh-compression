#!/usr/bin/env python3
"""make_batch43_files_v1.py (2026-11-15): generic-rd-v3.sbatch + the two Deep Blending scenes -> generic-rd-v4.sbatch.  Nothing else changes.
  python3 make_batch43_files_v1.py --sbatch <generic-rd-v3.sbatch> --out <dir>"""
import argparse, hashlib, pathlib, sys
IN = '88b8342f718d39217f76889717b6a6e9'; md5 = lambda s: hashlib.md5(s.encode()).hexdigest()
ap = argparse.ArgumentParser(); ap.add_argument('--sbatch', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
s = pathlib.Path(a.sbatch).read_text()
if md5(s) != IN: sys.exit(f'FAILED: {a.sbatch} md5 {md5(s)} != {IN}')
A = '  *) echo "FAILED: no settings for scene $SCENE"; exit 1;;\n'
if s.count(A) != 1: sys.exit('FAILED: anchor not found exactly once')
s = s.replace(A, '  drjohnson|playroom) ORIG=/scratch/zr2245/$SCENE-indoor; DATA=$D/db/$SCENE; IMAGES=images; QP=15; BUD="";;   # batch43: Deep Blending\n' + A)
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / 'generic-rd-v4.sbatch').write_text(s)
print('written generic-rd-v4.sbatch  lines %d  md5 %s' % (s.count('\n'), md5(s)))
