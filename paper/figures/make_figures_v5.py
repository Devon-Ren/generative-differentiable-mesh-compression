"""make_figures_v5.py (2026-11-14): the main RD figure with every baseline trained with a vertex cap (no simplification of ours).
  fig_rd_cap_v1 : Mip-NeRF 360 + T&T (11 scenes), PSNR against file size: BudgetMesh (hold controller), MeshSplatting with a vertex cap + our
                  codec, and MPEG TFAN / zstd / Draco on the same cap-trained meshes (caps 0.6M, about 1.08M, 3M and the default 4M; SH at 8 / 6 / 4 bits).
Data: release-batch36-generic-on-cap/results/batch36-means.json (analyze_batch36_v1.py)."""
import json, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker as mt
from scipy.interpolate import PchipInterpolator
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#dddcd6', '#fcfcfb'
C5 = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']   # palette slots 1-5 in fixed order
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
D = json.load(open('release-batch36-generic-on-cap/results/batch36-means.json'))
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
for (k, name, ls, mk), col in zip((('tfan', 'MPEG TFAN', '-', 'D'), ('zstd', 'quantized PLY + zstd', (0, (5, 2)), '^'), ('draco', 'Draco', (0, (1.5, 1.8)), 'v')), C5[2:]):
    curve(D['points'][k], D['hull'][k], col, ls, mk, 'MeshSplatting with a vertex cap + ' + name)
ax.annotate('BudgetMesh', bm[1], textcoords='offset points', xytext=(-8, 15), ha='center', fontsize=8, color=INK, fontweight='bold')
ax.annotate('cap + our codec', co[0], textcoords='offset points', xytext=(2, -8), ha='center', va='top', fontsize=7.5, color=INK)
ax.text(11.3, 21.9, 'cap + TFAN', ha='right', va='center', fontsize=7.5, color=INK)
ax.text(18.0, 21.9, 'cap + zstd', ha='left', va='center', fontsize=7.5, color=INK)
ax.text(64.0, 21.9, 'cap + Draco', ha='left', va='center', fontsize=7.5, color=INK)
# the two horizontal gaps the text quotes, measured on these curves
p1 = co[1][1]; x1 = at_psnr(D['hull']['tfan'], p1); ax.annotate('', (x1, p1), (co[1][0], p1), arrowprops=dict(arrowstyle='<->', color=MUTED, lw=0.8, shrinkA=5, shrinkB=5))
ax.text(math.sqrt(x1 * co[1][0]), p1 - 0.13, 'codec alone: x%.1f' % (x1 / co[1][0]), ha='center', va='top', fontsize=7, color=MUTED)
p2 = bm[2][1]; x2 = at_psnr(D['hull']['tfan'], p2); ax.annotate('', (x2, p2), (bm[2][0], p2), arrowprops=dict(arrowstyle='<->', color=MUTED, lw=0.8, shrinkA=6, shrinkB=5))
ax.text(math.sqrt(x2 * bm[2][0]), p2 + 0.07, 'whole system: x%.1f' % (x2 / bm[2][0]), ha='center', va='bottom', fontsize=7, color=MUTED)
ax.set_xscale('log'); ax.set_xlim(3.6, 520); ax.set_ylim(20.2, 24.35)
ax.xaxis.set_major_locator(mt.FixedLocator([5, 10, 20, 50, 100, 200, 400])); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: '%g' % v)); ax.xaxis.set_minor_formatter(mt.NullFormatter())
ax.set_xlabel('file size (MB, log scale)'); ax.set_ylabel('PSNR (dB)'); ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
ax.tick_params(length=2); ax.set_title('Mip-NeRF 360 + Tanks and Temples, mean over 11 scenes', loc='left', fontsize=9, color=INK, pad=6)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=7.2, handlelength=2.6, columnspacing=1.8)
fig.tight_layout(); fig.savefig('paper/figures/fig_rd_cap_v1.pdf'); fig.savefig('paper/figures/fig_rd_cap_v1.png', dpi=170); print('ok x%.2f x%.2f' % (x1 / co[1][0], x2 / bm[2][0]))
