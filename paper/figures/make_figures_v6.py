"""make_figures_v6.py (2026-11-14): the two RD panels side by side (two-column figure), Mip-NeRF 360 + T&T, mean over 11 scenes.
  (a) against generic codecs, every curve from meshes trained with a vertex cap (batch36)
  (b) when the size is decided, our codec everywhere: simplified after training / trained with a vertex cap / BudgetMesh
Output paper/figures/fig_rd_pair_v1.{pdf,png}."""
import json, math, re, collections, statistics as st
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker as mt
from scipy.interpolate import PchipInterpolator
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#dddcd6', '#fcfcfb'
C5 = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']; VIOLET = '#6250d6'
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
MIP = ['room','kitchen','counter','bonsai','garden','bicycle','stump','flowers','treehill','truck','train']
D = json.load(open('release-batch36-generic-on-cap/results/batch36-means.json')); bm, co = D['budgetmesh'], D['cap_ours']
# simplified after training + our codec: the POST-HOC-COLLAPSE controls (three per scene, ranked by size) and the full mesh
simp = collections.defaultdict(list)
for f in ['release-g0-refresh-v1/results/g0-refresh-2026-11-09.txt', 'release-g0-refresh-v1/results/g0-refresh-indoor-2026-11-09.txt', 'release-batch21-v1/results/batch21-collect-2026-11-09-2110-raw.txt', 'release-batch24-ablation-v1/results/batch24-collect-2026-11-11-0641.txt']:
    for l in open(f):
        m = re.match(r'(\w+) budget (\d+) POST-HOC-COLLAPSE \| .*?\| delivered ([\d.]+) MB test ([\d.]+)', l)
        if m and m[1] in MIP: simp[m[1]].append((float(m[3]), float(m[4])))
K = min(len(set(v)) for v in simp.values()); ph = []
for k in range(K): p = [sorted(set(simp[s]), reverse=True)[k] for s in MIP]; ph.append((st.mean(q[0] for q in p), st.mean(q[1] for q in p)))
print('post-hoc levels per scene (min)', K, [(round(a, 1), round(b, 2)) for a, b in ph])
def smooth(pts, n=80):
    p = sorted(pts); lx = np.log([q[0] for q in p]); y = np.array([q[1] for q in p])
    if len(p) < 3: return np.exp(lx), y
    g = np.linspace(lx[0], lx[-1], n); return np.exp(g), PchipInterpolator(lx, y)(g)
def hull(p):
    P = sorted((math.log(x), y) for x, y in p); H = []
    for q in P:
        while len(H) >= 2 and (H[-1][0] - H[-2][0]) * (q[1] - H[-2][1]) - (H[-1][1] - H[-2][1]) * (q[0] - H[-2][0]) >= 0: H.pop()
        H.append(q)
    k = max(range(len(H)), key=lambda t: H[t][1]); return [(math.exp(a), b) for a, b in H[:k + 1]]
def at_psnr(curve, p):
    c = sorted(curve, key=lambda x: x[1])
    for (a, pa), (b, pb) in zip(c, c[1:]):
        if pa <= p <= pb and pb > pa: return math.exp(math.log(a) + (math.log(b) - math.log(a)) * (p - pa) / (pb - pa))
def at_size(curve, mb):
    c = sorted(curve)
    for (a, pa), (b, pb) in zip(c, c[1:]):
        if a <= mb <= b: return pa + (pb - pa) * math.log(mb / a) / math.log(b / a)
def curve(ax, pts, hl, color, ls, mk, label, lw=1.8, ms=5.0, z=4):
    hl = [tuple(q) for q in hl]; rest = [tuple(q) for q in pts if tuple(q) not in hl]
    ax.plot(*smooth(hl), ls=ls, color=color, lw=lw, zorder=z)
    if rest: ax.plot([q[0] for q in rest], [q[1] for q in rest], ls='none', marker=mk, color=color, ms=ms - 1.6, mec='none', alpha=0.45, zorder=z - 1)
    ax.plot([q[0] for q in hl], [q[1] for q in hl], ls='none', marker=mk, color=color, ms=ms, mec=SURF, mew=0.9, zorder=z + 1)
    return ax.plot([], [], ls=ls, marker=mk, color=color, lw=lw, ms=ms, mec=SURF, mew=0.9, label=label)[0]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 3.3), gridspec_kw=dict(width_ratios=[1.45, 1]))
h = [curve(a, bm, bm, C5[0], '-', 'o', 'BudgetMesh (ours)', lw=2.2, ms=6, z=8), curve(a, co, co, C5[1], '-', 's', 'vertex cap + our codec', lw=2.0, ms=5.5, z=7)]
for (k, name, ls, mk), col in zip((('tfan', 'vertex cap + MPEG TFAN', '-', 'D'), ('zstd', 'vertex cap + zstd', (0, (5, 2)), '^'), ('draco', 'vertex cap + Draco', (0, (1.5, 1.8)), 'v')), C5[2:]):
    h.append(curve(a, D['points'][k], D['hull'][k], col, ls, mk, name))
a.annotate('BudgetMesh', bm[1], textcoords='offset points', xytext=(-4, 8), ha='center', fontsize=7.5, color=INK, fontweight='bold')
a.text(11.3, 21.9, 'TFAN', ha='right', va='center', fontsize=7.5, color=INK); a.text(18.5, 21.9, 'zstd', ha='left', va='center', fontsize=7.5, color=INK); a.text(66.0, 21.9, 'Draco', ha='left', va='center', fontsize=7.5, color=INK)
p1 = co[1][1]; x1 = at_psnr(D['hull']['tfan'], p1); a.annotate('', (x1, p1), (co[1][0], p1), arrowprops=dict(arrowstyle='<->', color=MUTED, lw=0.8, shrinkA=5, shrinkB=5))
a.text(math.sqrt(x1 * co[1][0]), p1 - 0.3, 'x%.1f' % (x1 / co[1][0]), ha='center', va='top', fontsize=7, color=MUTED)
p2 = bm[2][1]; x2 = at_psnr(D['hull']['tfan'], p2); a.annotate('', (x2, p2), (bm[2][0], p2), arrowprops=dict(arrowstyle='<->', color=MUTED, lw=0.8, shrinkA=6, shrinkB=5))
a.text(math.sqrt(x2 * bm[2][0]), p2 + 0.1, 'x%.1f' % (x2 / bm[2][0]), ha='center', va='bottom', fontsize=7, color=MUTED)
a.set_xscale('log'); a.set_xlim(3.6, 520); a.xaxis.set_major_locator(mt.FixedLocator([5, 10, 20, 50, 100, 200, 400]))
a.set_title('(a) against generic codecs, same cap-trained meshes', loc='left', fontsize=8.5, color=INK, pad=5); a.set_ylabel('PSNR (dB)')
hb = [curve(b, bm, bm, C5[0], '-', 'o', 'BudgetMesh (ours)', lw=2.2, ms=6, z=8), curve(b, co, co, C5[1], '-', 's', 'vertex cap + our codec', lw=2.0, ms=5.5, z=7),
      curve(b, ph, hull(ph), VIOLET, (0, (4, 1.5)), 'X', 'simplified after training + our codec', lw=1.8, ms=5.5, z=6)]
b.annotate('BudgetMesh', bm[1], textcoords='offset points', xytext=(-6, 8), ha='center', fontsize=7.5, color=INK, fontweight='bold')
b.annotate('vertex cap', co[0], textcoords='offset points', xytext=(2, -8), ha='center', va='top', fontsize=7.5, color=INK)
b.text(8.6, 20.0, 'simplified after\ntraining', ha='left', va='center', fontsize=7.5, color=INK)
for mb, p in bm:
    q = at_size(hull(ph), mb)
    if q: print('BudgetMesh %.1f MB: %+.2f dB over simplified-after' % (mb, p - q))
b.set_xscale('log'); b.set_xlim(3.6, 34); b.xaxis.set_major_locator(mt.FixedLocator([5, 10, 20])); b.tick_params(labelleft=False)
b.set_title('(b) when the size is set, our codec', loc='left', fontsize=8.5, color=INK, pad=5)
for ax in (a, b):
    ax.set_ylim(17.6, 24.4); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: '%g' % v)); ax.xaxis.set_minor_formatter(mt.NullFormatter()); ax.set_xlabel('file size (MB, log scale)')
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=2)
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
fig.legend(handles=h + [hb[2]], loc='lower center', ncol=3, frameon=False, fontsize=7, handlelength=2.6, columnspacing=1.6, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.0); fig.savefig('paper/figures/fig_rd_pair_v1.pdf'); fig.savefig('paper/figures/fig_rd_pair_v1.png', dpi=170); print('ok')
