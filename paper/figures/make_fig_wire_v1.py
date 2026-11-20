"""make_fig_wire_v1.py (2026-11-20): fig_triangles_v1 -- where the triangles are, at about the same file size.  Per scene two rows:
  row 1: the test view (crop marked) | screen area of the triangle under each pixel, for MeshSplatting FP32, MeshSplatting-cap (middle cap) and ByteMesh (10 MB)
  row 2: the crop of the view        | the triangle edges inside the crop, same three meshes
Data: paper/figures/wire-src (render_ids_v1.py, batch51): edge coverage and log area per image pixel, from the triangle-id map of the renderer.
Run from the repository root."""
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mp
SC = [('garden', 'DSC08076', (520, 330, 230, 150)), ('bicycle', '_DSC8727', (180, 430, 230, 150)), ('truck', '000049', (440, 255, 215, 140))]
COLS = [('FP32', 'MeshSplatting FP32'), ('CAP', 'MeshSplatting-cap'), ('OURS', 'ByteMesh (ours)')]
ST = {}
for s, v, _ in SC:
    for l in open('paper/figures/wire-src/%s/stats.txt' % s):
        p = l.split('|'); n = p[0].split()[2]; ST[(s, n)] = (int(p[1].split()[1]), int(p[4].split()[-1]), float(l.split('median ')[1].split()[0]))
W = 7.1; cw = W / 4; gap = 0.025; head = 0.2; foot = 0.42
load = lambda s, v, k: np.asarray(Image.open('paper/figures/wire-src/%s/%s-%s' % (s, v, k)))
hs = [(cw * load(s, v, 'gt.jpg').shape[0] / load(s, v, 'gt.jpg').shape[1], cw * c[3] / c[2]) for s, v, c in SC]
H = head + foot + sum(a + b + 3 * gap for a, b in hs); fig = plt.figure(figsize=(W, H)); y = H - head; im0 = None
for (s, v, (x0, y0, w, h)), (ha, hb) in zip(SC, hs):
    gt = load(s, v, 'gt.jpg')
    for row, hh in ((0, ha), (1, hb)):
        for c in range(4):
            ax = fig.add_axes([c * cw / W + 0.003, (y - hh) / H, cw / W - 0.006, hh / H]); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if c == 0:
                if row == 0:
                    ax.imshow(gt); ax.add_patch(mp.Rectangle((x0, y0), w, h, fill=False, ec='#ffd21f', lw=1.0))
                    ax.text(0.02, 0.05, s, transform=ax.transAxes, ha='left', va='bottom', fontsize=7.5, color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.55, pad=1.5, lw=0))
                else: ax.imshow(gt[y0:y0 + h, x0:x0 + w], interpolation='nearest')
                continue
            k, lab = COLS[c - 1]; V, vis, med = ST[(s, k)]
            if row == 0:
                a = load(s, v, k + '-area.png').astype(np.float32) / 255; im0 = ax.imshow(a, cmap='viridis_r', vmin=0.25, vmax=0.875)
                ax.add_patch(mp.Rectangle((x0, y0), w, h, fill=False, ec='white', lw=0.8))
                ax.text(0.985, 0.97, '%.2f M vertices\nmedian %.1f px' % (V / 1e6, med), transform=ax.transAxes, ha='right', va='top', fontsize=5.8, color='white', linespacing=1.05, bbox=dict(facecolor='black', alpha=0.6, pad=1.5, lw=0))
                if s == SC[0][0]: ax.text(0.5, 1.03, lab, transform=ax.transAxes, ha='center', va='bottom', fontsize=7, fontweight='bold' if k == 'OURS' else 'normal')
            else:
                e = load(s, v, k + '-edges.png').astype(np.float32)[y0:y0 + h, x0:x0 + w] / 255; ax.imshow(1 - np.clip(e * 1.3, 0, 1), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                for sp in ax.spines.values(): sp.set_visible(True); sp.set_color('#9aa0a8'); sp.set_linewidth(0.5)
        y -= hh + gap
    y -= gap
cax = fig.add_axes([0.38, 0.2 / H, 0.5, 0.07 / H]); cb = fig.colorbar(im0, cax=cax, orientation='horizontal'); cb.set_ticks([0.25, 0.5, 0.75]); cb.set_ticklabels(['1', '10', '100']); cb.ax.tick_params(labelsize=6.5, length=2)
fig.text(0.37, 0.235 / H, 'screen area of the triangle under each pixel (pixels)', ha='right', va='center', fontsize=6.5, color='#1f2430')
fig.savefig('paper/figures/fig_triangles_v1.pdf', dpi=230); fig.savefig('paper/figures/fig_triangles_v1.png', dpi=115); print('ok', W, round(H, 2), ST)
