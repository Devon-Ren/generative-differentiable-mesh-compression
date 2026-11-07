"""Table-1 figure: generic codecs on the SAME MeshSplatting asset vs our two routes.  Room + garden, test PSNR vs delivered MB (log).
usage: python plot_generic_codecs_v1.py <out-stem>   (reads generic-codecs-2026-10-24.txt, seven-scene-points-v2.csv, room-lambda-vs-cap-v2.csv, lambda-7scenes-v1.csv)"""
import csv, re, sys, math
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker as mt
from scipy.interpolate import PchipInterpolator
import numpy as np
out = sys.argv[1]
OLD, NEW, PRUNE, GEN, REF, SURF = '#eb6834', '#1baf7a', '#0f6e4c', '#8a5cd6', '#2b2b2b', '#fcfcfb'
ORIG = {'room': (742, 28.263), 'garden': (740, 24.732)}
def upper_hull(pts):
    pts = sorted(set(pts)); h = []
    for p in pts:
        while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) >= 0: h.pop()
        h.append(p)
    top = max(range(len(h)), key=lambda i: h[i][1]); return h[:top + 1]
def smooth(pts):
    h = upper_hull(pts)
    if len(h) < 2: return h
    x = np.log10([p[0] for p in h]); f = PchipInterpolator(x, [p[1] for p in h]); xs = np.linspace(x[0], x[-1], 100); return [(10 ** a, float(f(a))) for a in xs]
gen = {}
for line in open('generic-codecs-2026-10-24.txt'):
    parts = [s.strip() for s in line.split('|')]; sc, name = parts[0].split()
    mb = re.match(r'([\d.]+) MB', parts[1]); ps = re.search(r'test ([\d.]+) /', line)
    if mb and ps: gen.setdefault(sc, {})[name] = (float(mb.group(1)), float(ps.group(1)))
seven = list(csv.DictReader(open('seven-scene-points-v2.csv'))); room = list(csv.DictReader(open('room-lambda-vs-cap-v2.csv'))); lam7 = list(csv.DictReader(open('lambda-7scenes-v1.csv')))
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#999', 'axes.linewidth': 0.6, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
labels = {'quant_ply_zstd19': 'quantised PLY + zstd', 'draco_full_sh8': 'Draco, SH as 8-bit attributes', 'draco_full_sh16': 'Draco, SH 16-bit', 'draco_rgb8': 'Draco, RGB only (no SH)', 'texture_8192': 'texture route: xatlas + Draco + 8k JPEG (no SH)', 'texture_4096': 'texture route, 4k JPEG'}
for ax, sc in zip(axes, ('room', 'garden')):
    mb0, p0 = ORIG[sc]
    old = [(float(r['MB']), float(r['test_psnr'])) for r in seven if r['scene'] == sc and r['series'] == 'collapse']
    if sc == 'room': new = [(float(r['MB']), float(r['test_psnr'])) for r in room if r['series'] in ('lambda', 'prune')]
    else: new = [(float(r['MB']), float(r['test_psnr'])) for r in lam7 if r['scene'] == sc]
    ax.plot(*zip(*old), 'o', ms=3, color=OLD, alpha=0.35, mec='none', zorder=2, label='ours, old route: codec + post-hoc collapse (configs)' if sc == 'room' else None)
    oh = smooth(old); ax.plot(*zip(*oh), '-', lw=2, color=OLD, zorder=3, label='old route hull' if sc == 'room' else None)
    ax.plot(*zip(*new), 'o', ms=5, color=NEW, mec=SURF, mew=1, zorder=5, label='ours, new route: rate-controlled growth (λ)' if sc == 'room' else None)
    nh = smooth(new) if len(new) > 2 else new; ax.plot(*zip(*nh), '-', lw=2, color=NEW, zorder=4, label='new route hull' if sc == 'room' else None)
    ax.plot([mb0], [p0], '*', ms=11, color=REF, zorder=6, label='FP32 original' if sc == 'room' else None)
    ax.annotate(f'FP32 {mb0} MB', (mb0, p0), textcoords='offset points', xytext=(0, 9), fontsize=7, ha='center', color=REF)
    for name, mk in (('quant_ply_zstd19', 's'), ('draco_full_sh8', 'D'), ('draco_full_sh16', 'd'), ('draco_rgb8', 'v'), ('texture_4096', '^'), ('texture_8192', 'P')):
        if name not in gen[sc]: continue
        mb, ps = gen[sc][name]
        ax.plot([mb], [ps], mk, ms=7, color=GEN, mec=SURF, mew=0.8, zorder=6, label=labels[name] if sc == 'room' else None)
    g = gen[sc]
    ax.annotate(f'{g["quant_ply_zstd19"][0]:.0f} MB', g['quant_ply_zstd19'], textcoords='offset points', xytext=(0, -12), fontsize=7, ha='center', color=GEN)
    ax.annotate(f'{g["draco_full_sh8"][0]:.0f} MB', g['draco_full_sh8'], textcoords='offset points', xytext=(0, -12), fontsize=7, ha='center', color=GEN)
    ax.annotate(f'{g["draco_full_sh16"][0]:.0f} MB', g['draco_full_sh16'], textcoords='offset points', xytext=(0, -12), fontsize=7, ha='center', color=GEN)
    ax.annotate(f'RGB only: {g["draco_rgb8"][0]:.0f} MB, {g["draco_rgb8"][1] - p0:+.1f} dB', g['draco_rgb8'], textcoords='offset points', xytext=(-8, -4), fontsize=7, ha='right', color=GEN)
    ax.annotate(f'texture 8k: {g["texture_8192"][0]:.0f} MB', g['texture_8192'], textcoords='offset points', xytext=(8, -4), fontsize=7, ha='left', color=GEN)
    # ours callouts
    if sc == 'room':
        ax.annotate('13.7 MB / 27.99', (13.699, 27.985), textcoords='offset points', xytext=(-8, 2), fontsize=7, ha='right', color=NEW)
        ax.annotate('5.7 MB / 27.37', (5.695, 27.37), textcoords='offset points', xytext=(-6, -12), fontsize=7, ha='left', color=NEW)
        ax.annotate('30.9 MB / 28.06', (30.932, 28.061), textcoords='offset points', xytext=(4, -12), fontsize=7, ha='left', color=OLD)
    else:
        ax.annotate('24.8 MB / 24.45', (24.781, 24.445), textcoords='offset points', xytext=(-8, -3), fontsize=7, ha='right', color=NEW)
        ax.annotate('10.7 MB / 23.64', (10.685, 23.637), textcoords='offset points', xytext=(-6, -12), fontsize=7, ha='left', color=NEW)
        ax.annotate('32.5 MB / 24.46', (32.539, 24.458), textcoords='offset points', xytext=(4, -12), fontsize=7, ha='left', color=OLD)
    ax.axhline(p0, color=REF, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.set_xscale('log'); ax.set_xticks([3, 10, 30, 100, 300, 1000]); ax.xaxis.set_major_formatter(mt.ScalarFormatter()); ax.xaxis.set_minor_formatter(mt.NullFormatter())
    ax.set_xlabel('delivered size (MB, log)'); ax.set_ylabel('test PSNR (dB)'); ax.set_title(sc, fontsize=10, loc='left'); ax.margins(y=0.12)
    ax.grid(color='#e6e6e3', lw=0.5, zorder=0); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
fig.suptitle('Generic codecs on the same MeshSplatting asset vs ours: at equal quality ours is 5× smaller than quantised-PLY+zstd and 15–17× smaller than Draco; the RGB-only / texture tier loses 3–3.7 dB', fontsize=8.5, y=1.01)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, loc='lower center', ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(); fig.savefig(out + '.png', dpi=200, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight'); print('wrote', out)
