"""qbits_rule_v1.py (2026-11-18): the position bit depth that the stated rule gives, against the one used.  Rule: the smallest b such that the
grid step (longest bounding-box side / (2^b - 1), the quantiser of the codec) is at most a tenth of the median edge length of the mesh.
    python qbits_rule_v1.py --ckpt <finetune/best.pt> --used 15 --label garden"""
import argparse, math
import numpy as np
p = argparse.ArgumentParser(); p.add_argument('--ckpt', required=True); p.add_argument('--used', type=int, required=True); p.add_argument('--label', default=''); p.add_argument('--frac', type=float, default=0.1); a = p.parse_args()
import torch
s = torch.load(a.ckpt, map_location='cpu', weights_only=False); g = lambda n: (s[n].detach().cpu().numpy() if hasattr(s[n], 'detach') else np.asarray(s[n]))
xyz = g('triangles_points').reshape(-1, 3).astype(np.float64); f = g('_triangle_indices').reshape(-1, 3).astype(np.int64)
e = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1), axis=0); L = np.linalg.norm(xyz[e[:, 0]] - xyz[e[:, 1]], axis=1)
med = float(np.median(L)); ext = float((xyz.max(0) - xyz.min(0)).max()); b = math.ceil(math.log2(ext / (a.frac * med) + 1))
step = lambda bb: ext / (2 ** bb - 1)
print('QBITS %-9s | V %d edges %d | longest side %.4g | median edge %.4g | rule (step <= %.2f x median edge) -> %d bits | used %d bits | step used / median edge %.3f' % (
    a.label, len(xyz), len(e), ext, med, a.frac, b, a.used, step(a.used) / med), flush=True)
