"""Scene-scale numbers behind the XYZ precision choice: bounding-box extent, median edge length and the quantisation
step at q13..q16 relative to the median edge, plus flipped-face counts from precision/sweep.json when present.
    python geom_precision.py <deliver-dir> [<deliver-dir> ...]      (CPU, reads shared-final.pt)
"""
import sys, json, torch, numpy as np
from pathlib import Path
for d in map(Path, sys.argv[1:]):
    w = torch.load(d / 'shared-final.pt', map_location='cpu', weights_only=False)
    x = w['triangles_points'].numpy().astype(np.float64); f = np.asarray(w['_triangle_indices']).astype(np.int64)
    span = x.max(0) - x.min(0); extent = span.max()
    rng = np.random.default_rng(0); sel = rng.choice(len(f), min(len(f), 1_000_000), replace=False)
    e = np.concatenate([f[sel][:, [0, 1]], f[sel][:, [1, 2]], f[sel][:, [2, 0]]])
    L = np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1); med = float(np.median(L)); p10 = float(np.percentile(L, 10))
    flips = {}
    sw = d / 'precision' / 'sweep.json'
    if sw.exists():
        for r in json.loads(sw.read_text())['rows']: flips[r['xyz'] + '/' + r['dc']] = r['quantization_geometry']['flipped_faces']
    print(f'{d}\n  vertices {len(x):,} faces {len(f):,} | bbox span {np.round(span, 3)} extent {extent:.3f} | median edge {med:.5f} p10 edge {p10:.5f}')
    for b in (13, 14, 15, 16):
        step = extent / (2 ** b - 1)
        print(f'  q{b}: step {step:.6f} = {step / med:.3f} x median edge = {step / p10:.3f} x p10 edge | flips {[(k, v) for k, v in flips.items() if k.startswith(f"q{b}/")]}')
