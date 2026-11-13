"""make_figures_v4.py (2026-11-13): two figures for the system framing.
  fig_rd_system_v1      : Mip-NeRF 360 + T&T (11 scenes), PSNR against file size: BudgetMesh at its targets, the FP32 model simplified after training
                          (same fine-tuning and codec), generic codecs on the full FP32 mesh, the FP32 model.  No vertex-cap curve.
  fig_size_control_v1   : size error per scene, 21 scenes, one target per dataset: vertex cap (prior from the other scenes, first training) and
                          BudgetMesh (final design, one training)."""
import re, math, statistics as st, collections
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#dddcd6', '#fcfcfb'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
MIP = ['room','kitchen','counter','bonsai','garden','bicycle','stump','flowers','treehill','truck','train']; DB = ['drjohnson','playroom']; OBJ = ['lego','chair','drums','ficus','hotdog','materials','mic','ship']
FP = {'room': (597, 28.48), 'counter': (520, 26.48), 'kitchen': (507, 27.46), 'bonsai': (678, 28.28), 'garden': (740, 24.73), 'bicycle': (680, 23.03), 'stump': (698, 24.93), 'flowers': (668, 19.45), 'treehill': (692, 20.39), 'truck': (432, 22.39), 'train': (485, 18.90)}
ctrl = {}
for f in ['release-adapt-v9b/results/adapt-v9b-2026-11-08.txt', 'release-batch21-v1/results/batch21-collect-2026-11-09-2110-raw.txt']:
    for l in open(f):
        m = re.match(r'(\w+) \| target ([\d.]+) MB \|.*?delivered ([\d.]+) MB \(hit ([\d.]+) x target.*?\| test ([\d.]+)', l)
        if m: ctrl[(m[1], float(m[2]))] = (float(m[3]), float(m[5]))
simp = collections.defaultdict(list)
for f in ['release-g0-refresh-v1/results/g0-refresh-2026-11-09.txt', 'release-g0-refresh-v1/results/g0-refresh-indoor-2026-11-09.txt', 'release-batch21-v1/results/batch21-collect-2026-11-09-2110-raw.txt', 'release-batch24-ablation-v1/results/batch24-collect-2026-11-11-0641.txt']:
    for l in open(f):
        m = re.match(r'(\w+) budget (\d+) POST-HOC-COLLAPSE \| .*?\| delivered ([\d.]+) MB test ([\d.]+)', l)
        if m and m[1] in MIP: simp[m[1]].append((float(m[3]), float(m[4])))
gen = collections.defaultdict(dict)   # route -> scene -> (MB, PSNR)
for l in open('release-batch22-v1/results/generic-codecs-all-scenes-2026-11-10.md'):
    c = [x.strip() for x in l.split('|')]
    if len(c) > 8 and c[1] in MIP:
        s = c[1]; fp = FP[s][1]
        def cell(x, dflt=None):
            m = re.match(r'([\d.]+)(?: \(([-+\d.]+)\))?', x)
            if not m: return None
            return (float(m[1]), fp + (float(m[2]) if m[2] else dflt))
        z = cell(c[4]); gen['quantized PLY + zstd'][s] = z; gen['our codec, all vertices'][s] = cell(c[3]); gen['Draco, 8-bit SH'][s] = cell(c[5])
        mp = cell(c[6], z[1] - fp)
        if mp: gen['MPEG TFAN, 8-bit SH'][s] = mp
        gen['Draco, RGB only'][s] = cell(c[7]); gen['baked texture + Draco'][s] = cell(c[8])
def mean(d): return (st.mean(v[0] for v in d.values()), st.mean(v[1] for v in d.values()), len(d))
# ---------------------------------------------------------------- figure 1
import numpy as np
from scipy.interpolate import PchipInterpolator
def smooth(xs_, ys_, n=80):   # monotone (PCHIP) curve through the points, in log size
    o = np.argsort(xs_); lx = np.log(np.asarray(xs_, float)[o]); yy = np.asarray(ys_, float)[o]
    if len(lx) < 3: return np.exp(lx), yy
    g = np.linspace(lx[0], lx[-1], n); return np.exp(g), PchipInterpolator(lx, yy)(g)
fig, ax = plt.subplots(figsize=(7.0, 3.9))
C5 = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']   # palette slots 1-5 in order, assigned left to right (validated: adjacent pairs pass; low-contrast hues carry direct labels and their own marker and line style)
def upper_hull(pts):   # operational RD curve: upper convex hull in (log size, PSNR)
    P = sorted((math.log(x), y) for x, y in pts); H = []
    for q in P:
        while len(H) >= 2 and (H[-1][0] - H[-2][0]) * (q[1] - H[-2][1]) - (H[-1][1] - H[-2][1]) * (q[0] - H[-2][0]) >= 0: H.pop()
        H.append(q)
    k = max(range(len(H)), key=lambda t: H[t][1]); return [(math.exp(a), b) for a, b in H[:k + 1]]
def curve(pts, color, ls, mk, label, lw=1.8, ms=5.0, z=4, ring=None):
    h = upper_hull(pts); rest = [q for q in pts if q not in h]
    ax.plot(*smooth([q[0] for q in h], [q[1] for q in h]), ls=ls, color=color, lw=lw, zorder=z)
    if rest: ax.plot([q[0] for q in rest], [q[1] for q in rest], ls='none', marker=mk, color=color, ms=ms - 1.6, mec='none', alpha=0.45, zorder=z - 1)
    ax.plot([q[0] for q in h if q != ring], [q[1] for q in h if q != ring], ls='none', marker=mk, color=color, ms=ms, mec=SURF, mew=0.9, zorder=z + 1)
    ax.plot([], [], ls=ls, marker=mk, color=color, lw=lw, ms=ms, mec=SURF, mew=0.9, label=label); return h
xs, ys = zip(*[(st.mean(ctrl[(s, t)][0] for s in MIP), st.mean(ctrl[(s, t)][1] for s in MIP)) for t in (5, 7.5, 10, 15, 20)])
HULL = {'BudgetMesh': curve(list(zip(xs, ys)), C5[0], '-', 'o', 'BudgetMesh (target 5 to 20 MB)', lw=2.2, ms=6, z=7)}
ax.annotate('BudgetMesh', (xs[1], ys[1]), textcoords='offset points', xytext=(0, 9), ha='center', fontsize=8, color=INK, fontweight='bold')
K = min(len(v) for v in simp.values()); gx, gy = [], []
for k in range(K): pts = [sorted(simp[s], reverse=True)[k] for s in MIP]; gx.append(st.mean(p[0] for p in pts)); gy.append(st.mean(p[1] for p in pts))
x0, y0, _ = mean(gen['our codec, all vertices'])
HULL['FP32 + our codec'] = curve([(x0, y0)] + list(zip(gx, gy)), C5[1], '-', 's', 'FP32 model + our codec: simplified after training (ring: no vertices removed)', lw=2.0, ms=5.5, z=5, ring=(x0, y0))
ax.plot([x0], [y0], 'o', color=SURF, ms=6.5, mec=C5[1], mew=1.8, zorder=7); ax.annotate('no vertices removed, %.0f MB' % x0, (x0, y0), textcoords='offset points', xytext=(6, 8), ha='center', va='bottom', fontsize=7, color=INK)
ax.annotate('simplified after\ntraining + our codec', (gx[4], gy[4]), textcoords='offset points', xytext=(-8, 2), ha='right', va='center', fontsize=7.5, color=INK)
fx, fy = st.mean(FP[s][0] for s in MIP), st.mean(FP[s][1] for s in MIP)
ax.plot([fx], [fy], '*', color=INK, ms=11, zorder=6); ax.annotate('FP32 model\n%.0f MB' % fx, (fx, fy), textcoords='offset points', xytext=(0, 8), ha='center', va='bottom', fontsize=7.5, color=INK)
import json
G33 = json.load(open('release-batch33-generic-rd/results/batch33-collect-2026-11-13-0109-means.json'))['points']
for (codec, name, ls, mk), col in zip((('tfan', 'MPEG TFAN', '-', 'D'), ('zstd', 'quantized PLY + zstd', (0, (5, 2)), '^'), ('draco', 'Draco', (0, (1.5, 1.8)), 'v')), C5[2:]):
    HULL[codec] = curve([(v[codec], v['psnr']) for v in G33.values()], col, ls, mk, name + ' (generic codec)')
ax.set_xscale('log'); ax.set_xlim(3.2, 900)
import matplotlib.ticker as mt
ax.xaxis.set_major_locator(mt.FixedLocator([5, 10, 20, 50, 100, 200, 500])); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: '%g' % v)); ax.xaxis.set_minor_formatter(mt.NullFormatter())
ax.set_xlabel('file size (MB, log scale)'); ax.set_ylabel('PSNR (dB)'); ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
ax.tick_params(length=2); ax.set_title('Mip-NeRF 360 + Tanks and Temples, mean over 11 scenes', loc='left', fontsize=9, color=INK, pad=6)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=7.2, handlelength=2.6, columnspacing=1.8)
ax.set_ylim(17.4, 25.2)
fig.tight_layout(); fig.savefig('paper/figures/fig_rd_system_v1.pdf'); fig.savefig('paper/figures/fig_rd_system_v1.png', dpi=170); plt.close(fig)
# ---------------------------------------------------------------- figure 2
txt = open('release-batch30-overnight/results/batch30-collect-2026-11-12-2007.txt').read(); sec = {m.group(1): m.group(2) for m in re.finditer(r'===== ([^\n]+)\n(.*?)(?=\n===== |\Z)', txt, re.S)}
T0 = lambda s: 2.5 if s in OBJ else 10.0
ours = {}
for l in sec['controller rows'].splitlines():
    m = re.match(r'(\w+) abl24-final \| target ([\d.]+) MB \|.*?delivered ([\d.]+) MB', l)
    if m and float(m[2]) == T0(m[1]): ours[m[1]] = 100 * (float(m[3]) / T0(m[1]) - 1)
capE = dict(room=6.7, kitchen=16.3, counter=-6.9, bonsai=-16.9, garden=3.0, bicycle=0.4, stump=2.1, flowers=-5.8, treehill=6.0, truck=-0.5, train=3.6, drjohnson=15.7, playroom=9.6, lego=11.6, chair=5.4, drums=9.5, ficus=-39.6, hotdog=9.9, materials=1.4, mic=-18.4, ship=2.9)
order = MIP + DB + OBJ; yt = []
fig, ax = plt.subplots(figsize=(7.0, 4.3))
ax.axvspan(-10, 10, color='#eeede8', zorder=0); ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
for i, s in enumerate(order):
    y = len(order) - 1 - i + (0.0 if s in MIP else (-0.6 if s in DB else -1.2))
    ax.plot([capE[s], ours[s]], [y, y], '-', color=GRID, lw=1.4, zorder=2)
    ax.plot([capE[s]], [y], 'D', color=ORANGE, ms=5.5, mec=SURF, mew=1.0, zorder=3, label='vertex cap, prior from the other scenes of the same kind (first training)' if i == 0 else None)
    ax.plot([ours[s]], [y], 'o', color=BLUE, ms=6.2, mec=SURF, mew=1.2, zorder=4, label='BudgetMesh (one training)' if i == 0 else None)
    yt.append(y)
    if abs(capE[s]) > 10: ax.annotate('%+.0f%%' % capE[s], (capE[s], y), textcoords='offset points', xytext=(7 if capE[s] > 0 else -7, 0), ha='left' if capE[s] > 0 else 'right', va='center', fontsize=7, color=INK)
yb = lambda grp: [len(order) - 1 - order.index(s) + (0.0 if s in MIP else (-0.6 if s in DB else -1.2)) for s in grp]
for name, grp in (('Mip-NeRF 360 + T&T, target 10 MB', MIP), ('Deep Blending, 10 MB', DB), ('NeRF-Synthetic, 2.5 MB', OBJ)): ax.text(27, max(yb(grp)), name, va='center', ha='left', fontsize=7.2, color=MUTED)
yf = yb(['ficus'])[0]; ax.annotate('ficus cannot fill 2.5 MB', (ours['ficus'], yf), textcoords='offset points', xytext=(0, -11), ha='center', fontsize=6.8, color=MUTED)
ax.text(0, len(order) - 0.15, 'within 10% of the target', ha='center', va='bottom', fontsize=7, color=MUTED)
ax.set_xlim(-46, 62); ax.set_ylim(-2.0, len(order) + 0.4); ax.set_yticks(yt); ax.set_yticklabels(order, fontsize=7.5, color=INK); ax.tick_params(axis='y', length=0); ax.set_xticks([-40, -30, -20, -10, 0, 10, 20]); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: '%+d%%' % v if v else '0'))
ax.set_xlabel('size error of the delivered file (file size / target - 1)')
for s_ in ('top', 'right', 'left'): ax.spines[s_].set_visible(False)
ax.tick_params(length=2); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), frameon=False, fontsize=7.5, ncol=2, handletextpad=0.4, columnspacing=1.6)
fig.tight_layout(); fig.savefig('paper/figures/fig_size_control_v1.pdf'); fig.savefig('paper/figures/fig_size_control_v1.png', dpi=170); plt.close(fig)
for name, d in gen.items(): print('%-26s %6.1f MB  %.2f dB  (%d scenes)' % ((name,) + mean(d)))
print('hulls', {c: [(round(a, 1), round(b, 2)) for a, b in h] for c, h in HULL.items()}); print('BudgetMesh', [(round(a, 1), round(b, 2)) for a, b in zip(xs, ys)]); print('simplified', [(round(a, 1), round(b, 2)) for a, b in zip(gx, gy)]); print('FP32 %.0f MB %.2f' % (fx, fy))
