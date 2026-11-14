"""make_figures_v8.py (2026-11-14): appendix, the same RD figure on the NeRF-Synthetic objects (objects with complete rows; see the json); from make_figures_v5.py: the main RD figure with every baseline trained with a vertex cap (no simplification of ours).
  fig_rd_objects_v1 : Mip-NeRF 360 + T&T (11 scenes), PSNR against file size: BudgetMesh (hold controller), MeshSplatting with a vertex cap + our
                  codec, and MPEG TFAN / zstd / Draco on the same cap-trained meshes (caps 0.6M, about 1.08M, 3M and the default 4M; SH at 8 / 6 / 4 bits).
Data: release-batch38-generic-objects/results/batch38-means.json (analyze_batch36_v1.py)."""
import json, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker as mt
from scipy.interpolate import PchipInterpolator
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#dddcd6', '#fcfcfb'
C5 = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']   # palette slots 1-5 in fixed order
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
D = json.load(open('release-batch38-generic-objects/results/batch38-means.json'))
def smooth(pts, n=80):
    p = sorted(pts); lx = np.log([q[0] for q in p]); y = np.array([q[1] for q in p])
    if len(p) < 3: return np.exp(lx), y
    g = np.linspace(lx[0], lx[-1], n); return np.exp(g), PchipInterpolator(lx, y)(g)
def at_psnr(curve, p):
    c = sorted(curve, key=lambda x: x[1])
    for (a, pa), (b, pb) in zip(c, c[1:]):
        if pa <= p <= pb and pb > pa: return math.exp(math.log(a) + (math.log(b) - math.log(a)) * (p - pa) / (pb - pa))
fig, ax = plt.subplots(figsize=(7.0, 3.9))
def curve(pts, hull, color, ls, mk, label, lw=1.8, ms=5.0, z=4):
    hull = [tuple(q) for q in hull]; rest = [tuple(q) for q in pts if tuple(q) not in hull]
    ax.plot(*smooth(hull), ls=ls, color=color, lw=lw, zorder=z)
    if rest: ax.plot([q[0] for q in rest], [q[1] for q in rest], ls='none', marker=mk, color=color, ms=ms - 1.6, mec='none', alpha=0.45, zorder=z - 1)
    ax.plot([q[0] for q in hull], [q[1] for q in hull], ls='none', marker=mk, color=color, ms=ms, mec=SURF, mew=0.9, zorder=z + 1)
    ax.plot([], [], ls=ls, marker=mk, color=color, lw=lw, ms=ms, mec=SURF, mew=0.9, label=label)
bm, co = D['budgetmesh'], D['cap_ours']
curve(bm, bm, C5[0], '-', 'o', 'BudgetMesh: our controller + our codec', lw=2.2, ms=6, z=8)
curve(co, co, C5[1], '-', 's', 'MeshSplatting with a vertex cap + our codec', lw=2.0, ms=5.5, z=7)
for (k, name, ls, mk), col in zip((('tfan', 'MPEG TFAN', '-', 'D'), ('zstd', 'quantized PLY + zstd', '-', '^'), ('draco', 'Draco', '-', 'v')), C5[2:]):
    curve(D['points'][k], D['hull'][k], col, ls, mk, 'MeshSplatting with a vertex cap + ' + name)
ax.annotate('BudgetMesh', bm[1], textcoords='offset points', xytext=(-10, 9), ha='center', fontsize=8, color=INK, fontweight='bold')
ax.set_xscale('log'); ax.set_xlim(0.6, 60); ax.set_ylim(27.4, 32.7)
ax.xaxis.set_major_locator(mt.FixedLocator([1, 2, 5, 10, 20, 50])); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: '%g' % v)); ax.xaxis.set_minor_formatter(mt.NullFormatter())
ax.set_xlabel('file size (MB, log scale)'); ax.set_ylabel('PSNR (dB)'); ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
ax.tick_params(length=2); ax.set_title('NeRF-Synthetic, mean over %d objects' % len(D['objects']), loc='left', fontsize=9, color=INK, pad=6)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=7.2, handlelength=2.6, columnspacing=1.8)
fig.tight_layout(); fig.savefig('paper/figures/fig_rd_objects_v1.pdf'); fig.savefig('paper/figures/fig_rd_objects_v1.png', dpi=170); print('ok', D['objects'])
