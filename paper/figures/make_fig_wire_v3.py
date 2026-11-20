"""make_fig_wire_v3.py (2026-11-20): fig_triangles_v3 -- triangle density of ByteMesh against MeshSplatting-cap at about the same file size.
One row per scene: test view | local triangle density of MeshSplatting-cap | of ByteMesh | their ratio (ByteMesh / MeshSplatting-cap).
Local density = Gaussian average (sigma 8 image pixels) of 1 / (screen area of the triangle under the pixel) = visible triangles per pixel.
Data: paper/figures/wire-src (batch51).  Prints, per view, the L1 error of both meshes where ByteMesh is denser (ratio > 1.4) and coarser (< 0.7).
Run from the repository root."""
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
SC = [('garden', 'DSC08076', 11.64, 10.00), ('room', 'DSCF4715', 10.67, 9.99), ('bicycle', '_DSC8727', 10.04, 9.78), ('truck', '000049', 9.95, 9.90)]
ST = {}
for s, v, _, _ in SC:
    for l in open('paper/figures/wire-src/%s/stats.txt' % s):
        p = l.split('|'); ST[(s, p[0].split()[2])] = (int(p[1].split()[1]), int(p[4].split()[-1]))
load = lambda s, v, k: np.asarray(Image.open('paper/figures/wire-src/%s/%s-%s' % (s, v, k)))
W = 7.1; cw = W / 4; gap = 0.03; head = 0.2; foot = 0.52
hs = [cw * load(s, v, 'gt.jpg').shape[0] / load(s, v, 'gt.jpg').shape[1] for s, v, _, _ in SC]
H = head + foot + sum(h + gap for h in hs); fig = plt.figure(figsize=(W, H)); y = H - head; imd = imr = None
for (s, v, mbc, mbo), hh in zip(SC, hs):
    gt = load(s, v, 'gt.jpg').astype(np.float64) / 255
    D = {k: gaussian_filter(1 / 10 ** (load(s, v, k + '-area.png').astype(np.float64) / 255 * 4 - 1), 8) for k in ('CAP', 'OURS')}; r = np.log2(D['OURS'] / D['CAP'])
    E = {k: np.abs(load(s, v, k + '-render.jpg').astype(np.float64) / 255 - gt).mean(2) for k in D}; up, dn = r > 0.5, r < -0.5
    print('%-8s denser on %.0f %% of the pixels, coarser on %.0f %% | L1 error where denser: cap %.4f ours %.4f | where coarser: cap %.4f ours %.4f' % (s, 100 * up.mean(), 100 * dn.mean(), E['CAP'][up].mean(), E['OURS'][up].mean(), E['CAP'][dn].mean(), E['OURS'][dn].mean()))
    for c in range(4):
        ax = fig.add_axes([c * cw / W + 0.003, (y - hh) / H, cw / W - 0.006, hh / H]); ax.axis('off')
        if c == 0:
            ax.imshow(gt); ax.text(0.02, 0.05, s, transform=ax.transAxes, ha='left', va='bottom', fontsize=7.5, color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.55, pad=1.5, lw=0)); lab = 'Test view'
        elif c < 3:
            k = ('CAP', 'OURS')[c - 1]; imd = ax.imshow(np.log10(D[k] * 100), cmap='magma', vmin=0, vmax=1.7); V, vis = ST[(s, k)]
            ax.text(0.985, 0.97, '%.1f MB, %.2f M vertices\n%.0f k triangles visible' % ((mbc, mbo)[c - 1], V / 1e6, vis / 1e3), transform=ax.transAxes, ha='right', va='top', fontsize=5.6, color='white', linespacing=1.08, bbox=dict(facecolor='black', alpha=0.62, pad=1.5, lw=0))
            lab = ('MeshSplatting-cap', 'ByteMesh (ours)')[c - 1]
        else:
            imr = ax.imshow(r, cmap='RdBu_r', vmin=-2, vmax=2); lab = 'ByteMesh / MeshSplatting-cap'
        if s == SC[0][0]: ax.text(0.5, 1.03, lab, transform=ax.transAxes, ha='center', va='bottom', fontsize=7, fontweight='bold' if c == 2 else 'normal')
    y -= hh + gap
c1 = fig.add_axes([0.30, 0.30 / H, 0.36, 0.07 / H]); cb = fig.colorbar(imd, cax=c1, orientation='horizontal'); cb.set_ticks([0, 1, np.log10(50)]); cb.set_ticklabels(['1', '10', '50']); cb.ax.tick_params(labelsize=6.5, length=2)
fig.text(0.48, 0.02 / H, 'visible triangles per 100 pixels', ha='center', va='bottom', fontsize=6.5, color='#1f2430')
c2 = fig.add_axes([0.765, 0.30 / H, 0.22, 0.07 / H]); cb = fig.colorbar(imr, cax=c2, orientation='horizontal'); cb.set_ticks([-2, -1, 0, 1, 2]); cb.set_ticklabels(['1/4', '1/2', '1', '2', '4']); cb.ax.tick_params(labelsize=6.5, length=2)
fig.text(0.875, 0.02 / H, 'ratio of the two densities', ha='center', va='bottom', fontsize=6.5, color='#1f2430')
fig.savefig('paper/figures/fig_triangles_v3.pdf', dpi=250); fig.savefig('paper/figures/fig_triangles_v3.png', dpi=115); print('ok', W, round(H, 2))
