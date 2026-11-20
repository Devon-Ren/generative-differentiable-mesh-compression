"""make_fig_wire_v4.py (2026-11-20): fig_triangles_v4 = v3 with calm colours, and an export of every panel at the resolution of the test image.
  density maps : light grey -> dark slate (one hue)
  ratio map    : the two colours of the paper: blue where ByteMesh has more triangles, orange where MeshSplatting-cap has more, near white at 1
Export (paper/figures/triangles-export/): <scene>-view.png, <scene>-density-cap.png, <scene>-density-bytemesh.png, <scene>-ratio.png (coloured, no text),
and <scene>-data.npz with the raw arrays (density_cap, density_bytemesh in visible triangles per pixel; log2_ratio).  Run from the repository root."""
import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
SC = [('garden', 'DSC08076', 11.64, 10.00), ('room', 'DSCF4715', 10.67, 9.99), ('bicycle', '_DSC8727', 10.04, 9.78), ('truck', '000049', 9.95, 9.90)]
DENS = LinearSegmentedColormap.from_list('dens', ['#f4f5f7', '#c9d1dc', '#8a9bb5', '#4f6283', '#27314a'])
RATIO = LinearSegmentedColormap.from_list('ratio', ['#eb6834', '#f3a583', '#f7f5f2', '#8ab4e8', '#2a78d6'])
ST = {}
for s, v, _, _ in SC:
    for l in open('paper/figures/wire-src/%s/stats.txt' % s):
        p = l.split('|'); ST[(s, p[0].split()[2])] = (int(p[1].split()[1]), int(p[4].split()[-1]))
load = lambda s, v, k: np.asarray(Image.open('paper/figures/wire-src/%s/%s-%s' % (s, v, k)))
EXP = 'paper/figures/triangles-export'; os.makedirs(EXP, exist_ok=True); W = 7.1; cw = W / 4; gap = 0.03; head = 0.2; foot = 0.52; DV = (0.0, 2.0); RV = (-2.0, 2.0)
hs = [cw * load(s, v, 'gt.jpg').shape[0] / load(s, v, 'gt.jpg').shape[1] for s, v, _, _ in SC]
H = head + foot + sum(h + gap for h in hs); fig = plt.figure(figsize=(W, H)); y = H - head; imd = imr = None
rgb = lambda a, cm, lo, hi: (cm(np.clip((a - lo) / (hi - lo), 0, 1))[..., :3] * 255).round().astype(np.uint8)
for (s, v, mbc, mbo), hh in zip(SC, hs):
    gt = load(s, v, 'gt.jpg')
    D = {k: gaussian_filter(1 / 10 ** (load(s, v, k + '-area.png').astype(np.float64) / 255 * 4 - 1), 8) for k in ('CAP', 'OURS')}; r = np.log2(D['OURS'] / D['CAP'])
    Image.fromarray(gt).save('%s/%s-view.png' % (EXP, s)); Image.fromarray(rgb(np.log10(D['CAP'] * 100), DENS, *DV)).save('%s/%s-density-cap.png' % (EXP, s))
    Image.fromarray(rgb(np.log10(D['OURS'] * 100), DENS, *DV)).save('%s/%s-density-bytemesh.png' % (EXP, s)); Image.fromarray(rgb(r, RATIO, *RV)).save('%s/%s-ratio.png' % (EXP, s))
    np.savez_compressed('%s/%s-data.npz' % (EXP, s), density_cap=D['CAP'].astype(np.float16), density_bytemesh=D['OURS'].astype(np.float16), log2_ratio=r.astype(np.float16))
    for c in range(4):
        ax = fig.add_axes([c * cw / W + 0.003, (y - hh) / H, cw / W - 0.006, hh / H]); ax.axis('off')
        if c == 0:
            ax.imshow(gt); ax.text(0.02, 0.05, s, transform=ax.transAxes, ha='left', va='bottom', fontsize=7.5, color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.55, pad=1.5, lw=0)); lab = 'Test view'
        elif c < 3:
            k = ('CAP', 'OURS')[c - 1]; imd = ax.imshow(np.log10(D[k] * 100), cmap=DENS, vmin=DV[0], vmax=DV[1]); V, vis = ST[(s, k)]
            ax.text(0.985, 0.97, '%.1f MB, %.2f M vertices\n%.0f k triangles visible' % ((mbc, mbo)[c - 1], V / 1e6, vis / 1e3), transform=ax.transAxes, ha='right', va='top', fontsize=5.6, color='#1f2430', linespacing=1.08, bbox=dict(facecolor='white', alpha=0.8, pad=1.5, lw=0))
            lab = ('MeshSplatting-cap', 'ByteMesh (ours)')[c - 1]
        else:
            imr = ax.imshow(r, cmap=RATIO, vmin=RV[0], vmax=RV[1]); lab = 'ByteMesh / MeshSplatting-cap'
        if s == SC[0][0]: ax.text(0.5, 1.03, lab, transform=ax.transAxes, ha='center', va='bottom', fontsize=7, fontweight='bold' if c == 2 else 'normal')
    y -= hh + gap
c1 = fig.add_axes([0.30, 0.30 / H, 0.36, 0.06 / H]); cb = fig.colorbar(imd, cax=c1, orientation='horizontal'); cb.set_ticks([0, 1, 2]); cb.set_ticklabels(['1', '10', '100']); cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.4)
fig.text(0.48, 0.02 / H, 'visible triangles per 100 pixels', ha='center', va='bottom', fontsize=6.5, color='#1f2430')
c2 = fig.add_axes([0.765, 0.30 / H, 0.22, 0.06 / H]); cb = fig.colorbar(imr, cax=c2, orientation='horizontal'); cb.set_ticks([-2, -1, 0, 1, 2]); cb.set_ticklabels(['1/4', '1/2', '1', '2', '4']); cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.4)
fig.text(0.875, 0.02 / H, 'ratio of the two densities', ha='center', va='bottom', fontsize=6.5, color='#1f2430')
fig.savefig('paper/figures/fig_triangles_v4.pdf', dpi=300); fig.savefig('paper/figures/fig_triangles_v4.png', dpi=115); print('ok', W, round(H, 2))
