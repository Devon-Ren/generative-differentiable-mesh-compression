"""make_fig_wire_v6.py (2026-11-20): fig_triangles_v6 -- triangle density of MeshSplatting-cap and of ByteMesh at about the same file size and their
ratio, on one test view of four scenes that the paper does not show: bicycle (Mip-NeRF 360), truck (Tanks and Temples), drjohnson (Deep Blending,
about 15 MB) and chair (NeRF-Synthetic, about 2.5 MB).  Colours as fig_triangles_v5 (viridis density, orange-blue ratio).
Density = visible triangles per pixel = 1 / (screen area of the triangle that covers the pixel), Gaussian-smoothed (sigma 8 px; chair 5 px).
chair: only the pixels of the object count (pixels covered by a triangle of less than 1,000 px; the black background is left blank).
Data: paper/figures/wire-src (batch51 for bicycle and truck, batch52 for drjohnson and chair).  Also writes every panel at image resolution to
paper/figures/triangles-export.  Run from the repository root."""
import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
#      scene, view, MB cap, MB ours, sigma, object only, crop (y0, y1, x0, x1) or None, black columns added left and right
SC = [('bicycle', '_DSC8727', 10.04, 9.78, 8, False, None, 0), ('truck', '000049', 9.95, 9.90, 8, False, None, 0),
      ('drjohnson', 'IMG_6552', 14.53, 14.87, 8, False, None, 0), ('chair', 'r_143', 2.63, 2.51, 5, True, (60, 780, 49, 769), 108)]
INK, MUTED, SURF = '#0b0b0b', '#52514e', '#ffffff'
plt.rcParams.update({'font.size': 7, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'text.color': INK})
DENS = plt.get_cmap('viridis').copy(); DENS.set_bad('#eeeeee'); RATIO = LinearSegmentedColormap.from_list('ratio', ['#eb6834', '#f3a583', '#f7f5f2', '#8ab4e8', '#2a78d6']); RATIO.set_bad('#eeeeee')
load = lambda s, v, k: np.asarray(Image.open('paper/figures/wire-src/%s/%s-%s' % (s, v, k)))
ST = {}
for s, v, *_ in SC:
    for l in open('paper/figures/wire-src/%s/stats.txt' % s):
        p = l.split('|'); n = p[0].split()
        if n[1] == v: ST[(s, n[2])] = (int(p[1].split()[1]), int(p[4].split()[-1]))
def shape(a, crop, pad, fill):
    if crop: a = a[crop[0]:crop[1], crop[2]:crop[3]]
    if pad: a = np.concatenate([np.full((a.shape[0], pad) + a.shape[2:], fill, a.dtype), a, np.full((a.shape[0], pad) + a.shape[2:], fill, a.dtype)], 1)
    return a
def density(s, v, k, sig, obj):
    code = load(s, v, k + '-area.png').astype(np.float64); area = 10 ** (code / 255 * 4 - 1); m = np.ones(code.shape)
    if obj: m = ((load(s, v, k + '-valid.png') > 0) & (code < 255)).astype(np.float64)
    w = gaussian_filter(m, sig); d = gaussian_filter(m / area, sig) / np.maximum(w, 1e-9); d[w < 0.5] = np.nan; return d
EXP = 'paper/figures/triangles-export'; os.makedirs(EXP, exist_ok=True); W = 7.1; cw = W / 4; gap = 0.03; head = 0.2; foot = 0.52; DV = (0.0, 2.0); RV = (-2.0, 2.0); P = {}
for s, v, mbc, mbo, sig, obj, crop, pad in SC:
    gt = load(s, v, 'gt.jpg'); D = {k: density(s, v, k, sig, obj) for k in ('CAP', 'OURS')}; r = np.log2(D['OURS'] / D['CAP']); ok = np.isfinite(r); up, dn = ok & (r > 0.5), ok & (r < -0.5)
    msg = '%-9s vertices ours/cap %.3f | visible triangles ours/cap %.3f | ByteMesh >= 1.41x denser on %.0f %% of the pixels, <= 0.71x on %.0f %%' % (s, ST[(s, 'OURS')][0] / ST[(s, 'CAP')][0], ST[(s, 'OURS')][1] / ST[(s, 'CAP')][1], 100 * up.sum() / ok.sum(), 100 * dn.sum() / ok.sum())
    if os.path.exists('paper/figures/wire-src/%s/%s-CAP-render.jpg' % (s, v)):
        E = {k: np.abs(load(s, v, k + '-render.jpg').astype(np.float64) / 255 - gt.astype(np.float64) / 255).mean(2) for k in D}
        msg += ' | L1 error where denser: cap %.4f ours %.4f | where coarser: cap %.4f ours %.4f' % (E['CAP'][up].mean(), E['OURS'][up].mean(), E['CAP'][dn].mean(), E['OURS'][dn].mean())
    print(msg)
    P[s] = (shape(gt, crop, pad, 0), {k: shape(D[k], crop, pad, np.nan) for k in D}, shape(r, crop, pad, np.nan))
    rgb = lambda a, cm, lo, hi: (cm(np.ma.masked_invalid(np.clip((a - lo) / (hi - lo), 0, 1)))[..., :3] * 255).round().astype(np.uint8)
    Image.fromarray(P[s][0]).save('%s/%s-view.png' % (EXP, s)); Image.fromarray(rgb(np.log10(P[s][1]['CAP'] * 100), DENS, *DV)).save('%s/%s-density-cap.png' % (EXP, s))
    Image.fromarray(rgb(np.log10(P[s][1]['OURS'] * 100), DENS, *DV)).save('%s/%s-density-bytemesh.png' % (EXP, s)); Image.fromarray(rgb(P[s][2], RATIO, *RV)).save('%s/%s-ratio.png' % (EXP, s))
    np.savez_compressed('%s/%s-data.npz' % (EXP, s), density_cap=P[s][1]['CAP'].astype(np.float16), density_bytemesh=P[s][1]['OURS'].astype(np.float16), log2_ratio=P[s][2].astype(np.float16))
hs = [cw * P[s][0].shape[0] / P[s][0].shape[1] for s, *_ in SC]
H = head + foot + sum(h + gap for h in hs); fig = plt.figure(figsize=(W, H)); y = H - head; imd = imr = None
for (s, v, mbc, mbo, *_), hh in zip(SC, hs):
    gt, D, r = P[s]
    for c in range(4):
        ax = fig.add_axes([c * cw / W + 0.003, (y - hh) / H, cw / W - 0.006, hh / H]); ax.axis('off')
        if c == 0:
            ax.imshow(gt); ax.text(0.02, 0.05, s, transform=ax.transAxes, ha='left', va='bottom', fontsize=7.5, color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.55, pad=1.5, lw=0)); lab = 'Test view'
        elif c < 3:
            k = ('CAP', 'OURS')[c - 1]; imd = ax.imshow(np.ma.masked_invalid(np.log10(D[k] * 100)), cmap=DENS, vmin=DV[0], vmax=DV[1]); V, vis = ST[(s, k)]
            ax.text(0.985, 0.97, '%.1f MB, %.2f M vertices\n%.0f k triangles visible' % ((mbc, mbo)[c - 1], V / 1e6, vis / 1e3), transform=ax.transAxes, ha='right', va='top', fontsize=5.6, color='#1f2430', linespacing=1.08, bbox=dict(facecolor='white', alpha=0.8, pad=1.5, lw=0))
            lab = ('MeshSplatting-cap', 'ByteMesh (ours)')[c - 1]
        else:
            imr = ax.imshow(np.ma.masked_invalid(r), cmap=RATIO, vmin=RV[0], vmax=RV[1]); lab = 'ByteMesh / MeshSplatting-cap'
        if s == SC[0][0]: ax.text(0.5, 1.03, lab, transform=ax.transAxes, ha='center', va='bottom', fontsize=7, fontweight='bold' if c == 2 else 'normal')
    y -= hh + gap
c1 = fig.add_axes([0.30, 0.30 / H, 0.36, 0.06 / H]); cb = fig.colorbar(imd, cax=c1, orientation='horizontal'); cb.set_ticks([0, 1, 2]); cb.set_ticklabels(['1', '10', '100']); cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.4)
fig.text(0.48, 0.02 / H, 'visible triangles per 100 pixels', ha='center', va='bottom', fontsize=6.5, color='#1f2430')
c2 = fig.add_axes([0.765, 0.30 / H, 0.22, 0.06 / H]); cb = fig.colorbar(imr, cax=c2, orientation='horizontal'); cb.set_ticks([-2, -1, 0, 1, 2]); cb.set_ticklabels(['1/4', '1/2', '1', '2', '4']); cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.4)
fig.text(0.875, 0.02 / H, 'ratio of the two densities', ha='center', va='bottom', fontsize=6.5, color='#1f2430')
fig.savefig('paper/figures/fig_triangles_v6.pdf', dpi=300); fig.savefig('paper/figures/fig_triangles_v6.png', dpi=115); print('ok', W, round(H, 2))
