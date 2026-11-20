"""make_fig_wire_v2.py (2026-11-20): fig_triangles_v2 -- where the triangles are, at about the same file size.  One row per scene:
the test view | screen area of the triangle under each pixel for MeshSplatting FP32, MeshSplatting-cap (middle cap) and ByteMesh (10 MB).
(v1 also had wireframe crops; they were stored at image resolution and are not readable.)  Data: paper/figures/wire-src.  Run from the repository root."""
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
SC = [('garden', 'DSC08076'), ('room', 'DSCF4715'), ('bicycle', '_DSC8727'), ('truck', '000049')]
COLS = [('FP32', 'MeshSplatting FP32'), ('CAP', 'MeshSplatting-cap'), ('OURS', 'ByteMesh (ours)')]
ST = {}
for s, v in SC:
    for l in open('paper/figures/wire-src/%s/stats.txt' % s):
        p = l.split('|'); ST[(s, p[0].split()[2])] = (int(p[1].split()[1]), int(p[4].split()[-1]), float(l.split('median ')[1].split()[0]))
W = 7.1; cw = W / 4; gap = 0.03; head = 0.2; foot = 0.42
load = lambda s, v, k: np.asarray(Image.open('paper/figures/wire-src/%s/%s-%s' % (s, v, k)))
hs = [cw * load(s, v, 'gt.jpg').shape[0] / load(s, v, 'gt.jpg').shape[1] for s, v in SC]
H = head + foot + sum(h + gap for h in hs); fig = plt.figure(figsize=(W, H)); y = H - head; im0 = None
for (s, v), hh in zip(SC, hs):
    for c in range(4):
        ax = fig.add_axes([c * cw / W + 0.003, (y - hh) / H, cw / W - 0.006, hh / H]); ax.axis('off')
        if c == 0:
            ax.imshow(load(s, v, 'gt.jpg')); ax.text(0.02, 0.05, s, transform=ax.transAxes, ha='left', va='bottom', fontsize=7.5, color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.55, pad=1.5, lw=0))
            if s == SC[0][0]: ax.text(0.5, 1.03, 'Test view', transform=ax.transAxes, ha='center', va='bottom', fontsize=7)
            continue
        k, lab = COLS[c - 1]; V, vis, med = ST[(s, k)]
        im0 = ax.imshow(load(s, v, k + '-area.png').astype(np.float32) / 255, cmap='viridis_r', vmin=0.25, vmax=0.875, interpolation='bilinear')
        ax.text(0.985, 0.97, '%.2f M vertices\n%.0f k triangles visible\nmedian area %.1f px' % (V / 1e6, vis / 1e3, med), transform=ax.transAxes, ha='right', va='top', fontsize=5.6, color='white', linespacing=1.08, bbox=dict(facecolor='black', alpha=0.62, pad=1.5, lw=0))
        if s == SC[0][0]: ax.text(0.5, 1.03, lab, transform=ax.transAxes, ha='center', va='bottom', fontsize=7, fontweight='bold' if k == 'OURS' else 'normal')
    y -= hh + gap
cax = fig.add_axes([0.42, 0.2 / H, 0.46, 0.07 / H]); cb = fig.colorbar(im0, cax=cax, orientation='horizontal'); cb.set_ticks([0.25, 0.5, 0.75]); cb.set_ticklabels(['1', '10', '100']); cb.ax.tick_params(labelsize=6.5, length=2)
fig.text(0.41, 0.235 / H, 'screen area of the triangle under each pixel (pixels)', ha='right', va='center', fontsize=6.5, color='#1f2430')
fig.savefig('paper/figures/fig_triangles_v2.pdf', dpi=250); fig.savefig('paper/figures/fig_triangles_v2.png', dpi=115); print('ok', W, round(H, 2))
