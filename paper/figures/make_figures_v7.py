"""make_figures_v7.py (2026-11-14): the two RD panels side by side, Mip-NeRF 360 + T&T, mean over 11 scenes.
  (a) meshes trained with a vertex cap: BudgetMesh, cap + our codec, cap + TFAN / zstd / Draco (batch36).  All lines solid.
  (b) meshes simplified after training: BudgetMesh (solid), and the FP32 model simplified after training + our codec / TFAN / zstd / Draco
      (batch33), all dashed; the uncompressed FP32 model as a star.
Colour = codec in both panels; line style = how the mesh got its size (solid: during training, dashed: after training).
Output paper/figures/fig_rd_pair_v2.{pdf,png}."""
import json, math, re, collections, statistics as st
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker as mt
from scipy.interpolate import PchipInterpolator
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#dddcd6', '#fcfcfb'
C5 = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']; DASH = (0, (4.5, 2.2))
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
MIP = ['room','kitchen','counter','bonsai','garden','bicycle','stump','flowers','treehill','truck','train']
D = json.load(open('release-batch36-generic-on-cap/results/batch36-means.json')); bm, co = D['budgetmesh'], D['cap_ours']
G33 = json.load(open('release-batch33-generic-rd/results/batch33-collect-2026-11-13-0109-means.json'))['points']
simp = collections.defaultdict(list)
for f in ['release-g0-refresh-v1/results/g0-refresh-2026-11-09.txt', 'release-g0-refresh-v1/results/g0-refresh-indoor-2026-11-09.txt', 'release-batch21-v1/results/batch21-collect-2026-11-09-2110-raw.txt', 'release-batch24-ablation-v1/results/batch24-collect-2026-11-11-0641.txt']:
    for l in open(f):
        m = re.match(r'(\w+) budget (\d+) POST-HOC-COLLAPSE \| .*?\| delivered ([\d.]+) MB test ([\d.]+)', l)
        if m and m[1] in MIP: simp[m[1]].append((float(m[3]), float(m[4])))
K = min(len(set(v)) for v in simp.values()); ph = []
for k in range(K): p = [sorted(set(simp[s]), reverse=True)[k] for s in MIP]; ph.append((st.mean(q[0] for q in p), st.mean(q[1] for q in p)))
FP = {'room': (597, 28.48), 'counter': (520, 26.48), 'kitchen': (507, 27.46), 'bonsai': (678, 28.28), 'garden': (740, 24.73), 'bicycle': (680, 23.03), 'stump': (698, 24.93), 'flowers': (668, 19.45), 'treehill': (692, 20.39), 'truck': (432, 22.39), 'train': (485, 18.90)}
ring = (D['default_ours_mb'], 23.66)   # our codec on the FP32 model with no vertex removed (make_figures_v4: mean of the per-scene table)
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
def curve(ax, pts, color, ls, mk, lw=1.8, ms=5.0, z=4, hl=None):
    pts = [tuple(q) for q in pts]; hl = [tuple(q) for q in (hl if hl is not None else hull(pts))]; rest = [q for q in pts if q not in hl]
    ax.plot(*smooth(hl), ls=ls, color=color, lw=lw, zorder=z)
    if rest: ax.plot([q[0] for q in rest], [q[1] for q in rest], ls='none', marker=mk, color=color, ms=ms - 1.6, mec='none', alpha=0.45, zorder=z - 1)
    ax.plot([q[0] for q in hl], [q[1] for q in hl], ls='none', marker=mk, color=color, ms=ms, mec=SURF, mew=0.9, zorder=z + 1)
fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 3.45))
SPEC = (('tfan', 'MPEG TFAN', 'D', C5[2]), ('zstd', 'quantized PLY + zstd', '^', C5[3]), ('draco', 'Draco', 'v', C5[4]))
# ---- (a) all solid
curve(a, bm, C5[0], '-', 'o', lw=2.2, ms=6, z=8, hl=bm); curve(a, co, C5[1], '-', 's', lw=2.0, ms=5.5, z=7, hl=co)
for k, name, mk, col in SPEC: curve(a, D['points'][k], col, '-', mk, hl=D['hull'][k])
a.annotate('BudgetMesh', bm[1], textcoords='offset points', xytext=(-4, 8), ha='center', fontsize=7.5, color=INK, fontweight='bold')
a.annotate('our codec', co[0], textcoords='offset points', xytext=(4, -8), ha='center', va='top', fontsize=7.5, color=INK)
a.text(11.0, 21.5, 'TFAN', ha='right', va='center', fontsize=7.5, color=INK); a.text(19.0, 21.5, 'zstd', ha='left', va='center', fontsize=7.5, color=INK); a.text(64.0, 21.5, 'Draco', ha='left', va='center', fontsize=7.5, color=INK)
# ---- (b) BudgetMesh solid, everything else dashed
curve(b, bm, C5[0], '-', 'o', lw=2.2, ms=6, z=8, hl=bm)
pho = ph + [ring]; curve(b, pho, C5[1], DASH, 's', lw=2.0, ms=5.5, z=7)
for k, name, mk, col in SPEC: curve(b, [(v[k], v['psnr']) for v in G33.values()], col, DASH, mk)
fx, fy = st.mean(FP[s][0] for s in MIP), st.mean(FP[s][1] for s in MIP)
b.plot([fx], [fy], '*', color=INK, ms=10, zorder=6); b.annotate('uncompressed\n%.0f MB' % fx, (fx, fy), textcoords='offset points', xytext=(0, 7), ha='center', va='bottom', fontsize=7, color=INK)
b.annotate('BudgetMesh', bm[1], textcoords='offset points', xytext=(-4, 8), ha='center', fontsize=7.5, color=INK, fontweight='bold')
for ax, cap in ((a, '(a) Baselines trained with a vertex cap'), (b, '(b) Baselines simplified after training')):
    ax.set_xscale('log'); ax.set_xlim(3.3, 900); ax.set_ylim(17.4, 25.1)
    ax.xaxis.set_major_locator(mt.FixedLocator([5, 10, 20, 50, 100, 200, 500])); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: '%g' % v)); ax.xaxis.set_minor_formatter(mt.NullFormatter())
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=2)
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.set_xlabel('file size (MB, log scale)\n\n' + cap, color=MUTED); ax.xaxis.label.set_linespacing(1.25)
a.set_ylabel('PSNR (dB)'); b.tick_params(labelleft=False)
from matplotlib.lines import Line2D
L = lambda c, mk, lab: Line2D([], [], color=c, marker=mk, ms=5, mec=SURF, mew=0.8, lw=1.8, label=lab)
hs = [L(C5[0], 'o', 'BudgetMesh (ours)'), L(C5[1], 's', 'our codec'), L(C5[2], 'D', 'MPEG TFAN'), L(C5[3], '^', 'zstd'), L(C5[4], 'v', 'Draco'),
      Line2D([], [], color=MUTED, lw=1.6, ls='-', label='size set during training'), Line2D([], [], color=MUTED, lw=1.6, ls=DASH, label='size set after training')]
fig.legend(handles=hs, loc='upper center', ncol=5, frameon=False, fontsize=7, handlelength=2.4, columnspacing=1.6, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.89), w_pad=1.2); fig.savefig('paper/figures/fig_rd_pair_v2.pdf'); fig.savefig('paper/figures/fig_rd_pair_v2.png', dpi=170); print('ok')
