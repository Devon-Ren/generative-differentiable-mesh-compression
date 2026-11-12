#!/usr/bin/env python3
"""make_batch27_files_v1.py (2026-11-12): adds ONE variant to the batch24 round-2 sbatch files:
  adaptcap : the byte probe and growth plan decide HOW MANY triangles are split at each step (as in BudgetMesh), but WHICH triangles
             is MeshSplatting's own importance sampling (--dd_mode importance); no collapse rounds (they need the per-triangle error).
             = "measure the file as you go and update the vertex cap".  Run dir $SCENE-abl24-adaptcap-t${T}mb, rows tagged abl24-adaptcap.
  python3 make_batch27_files_v1.py --sbatch <ablation-b24-r2.sbatch> --sbatch_blender <ablation-b24-r2-blender.sbatch> --out <dir>"""
import argparse, hashlib, pathlib, sys
IN = {'sbatch': '2adf57a024da31f1eb41781b174cb49b', 'blender': 'b4315505e93cc9137141d81eb8a37f60'}
def md5(s): return hashlib.md5(s.encode()).hexdigest()
def rep(s, old, new, what):
    n = s.count(old)
    if n != 1: sys.exit(f'FAILED: edit "{what}": anchor found {n} times (expected 1)')
    return s.replace(old, new)
ap = argparse.ArgumentParser(); ap.add_argument('--sbatch', required=True); ap.add_argument('--sbatch_blender', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
for key, src, name in [('sbatch', a.sbatch, 'ablation-b24-r3.sbatch'), ('blender', a.sbatch_blender, 'ablation-b24-r3-blender.sbatch')]:
    s = pathlib.Path(src).read_text()
    if md5(s) != IN[key]: sys.exit(f'FAILED: {src} md5 {md5(s)} != {IN[key]} (not the batch24 round-2 file)')
    s = rep(s, "constlam|repeat via --export}", "constlam|repeat|adaptcap via --export}", 'ABL prompt')
    s = rep(s, "  constants) PAE=0 ;;\n", "  constants) PAE=0 ;;\n  adaptcap) PRUNE=\"\"; EXTRA=\"\"; DDMODE=importance ;;\n", 'adaptcap case')
    s = s.replace("VERSION abl24-r2", "VERSION abl24-r3").replace("VERSION abl24b-r2", "VERSION abl24b-r3")
    (out / name).write_text(s); print(f'{name}: {len(s.splitlines())} lines md5 {md5(s)}')
