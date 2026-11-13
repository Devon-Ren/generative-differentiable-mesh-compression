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
fig, ax = plt.subplots(figsize=(7.0, 3.1))
xs, ys = zip(*[(st.mean(ctrl[(s, t)][0] for s in MIP), st.mean(ctrl[(s, t)][1] for s in MIP)) for t in (5, 7.5, 10, 15, 20)])
ax.plot(xs, ys, '-', color=BLUE, lw=2, zorder=5); ax.plot(xs, ys, 'o', color=BLUE, ms=6, mec=SURF, mew=1.2, zorder=6, label='BudgetMesh (target 5 to 20 MB)')
ax.annotate('BudgetMesh', (xs[1], ys[1]), textcoords='offset points', xytext=(0, 9), ha='center', fontsize=8, color=INK, fontweight='bold')
K = min(len(v) for v in simp.values()); gx, gy = [], []
for k in range(K): pts = [sorted(simp[s], reverse=True)[k] for s in MIP]; gx.append(st.mean(p[0] for p in pts)); gy.append(st.mean(p[1] for p in pts))
x0, y0, _ = mean(gen['our codec, all vertices'])  # the same FP32 model with no vertex removed: the end point of this family
ax.plot([x0] + gx, [y0] + gy, '-', color=AQUA, lw=2, zorder=4); ax.plot(gx, gy, 's', color=AQUA, ms=5.5, mec=SURF, mew=1.2, zorder=4, label='FP32 model + our codec: no vertices removed (ring), simplified after training (squares)')
ax.annotate('simplified after training', (gx[3], gy[3]), textcoords='offset points', xytext=(9, -4), ha='left', fontsize=8, color=INK)
fx, fy = st.mean(FP[s][0] for s in MIP), st.mean(FP[s][1] for s in MIP)
ax.plot([fx], [fy], '*', color=INK, ms=11, zorder=6); ax.annotate('FP32 model\n%.0f MB' % fx, (fx, fy), textcoords='offset points', xytext=(0, -9), ha='center', va='top', fontsize=7.5, color=INK)
off = {'our codec, all vertices': (2, 9, 'center'), 'quantized PLY + zstd': (0, 9, 'center'), 'Draco, 8-bit SH': (0, 9, 'center'), 'MPEG TFAN, 8-bit SH': (0, -9, 'center'), 'Draco, RGB only': (-9, 0, 'right'), 'baked texture + Draco': (9, 0, 'left')}
first = True
for name, d in gen.items():
    x, y, n = mean(d)
    if name == 'our codec, all vertices': ax.plot([x], [y], 'o', color=SURF, ms=6.5, mec=AQUA, mew=1.8, zorder=6)
    else: ax.plot([x], [y], 'D', color=ORANGE, ms=5.5, mec=SURF, mew=1.0, zorder=5, label='generic codecs on the full FP32 mesh' if first else None); first = False
    dx, dy, ha = off[name]; ax.annotate({'our codec, all vertices': 'FP32 model + our codec,\nno vertices removed'}.get(name, name) + (' (10 scenes)' if n == 10 else '') + ('\n%.0f MB' % x if name != 'our codec, all vertices' else ', %.0f MB' % x), (x, y), textcoords='offset points', xytext=(dx, dy), ha=ha, va='bottom' if dy > 0 else ('top' if dy < 0 else 'center'), fontsize=7, color=INK)
ax.set_xscale('log'); ax.set_xlim(3.2, 900); ax.set_ylim(18.2, 25.9)
import matplotlib.ticker as mt
ax.xaxis.set_major_locator(mt.FixedLocator([5, 10, 20, 50, 100, 200, 500])); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: '%g' % v)); ax.xaxis.set_minor_formatter(mt.NullFormatter())
ax.set_xlabel('file size (MB, log scale)'); ax.set_ylabel('PSNR (dB)'); ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
ax.tick_params(length=2); ax.set_title('Mip-NeRF 360 + Tanks and Temples, mean over 11 scenes', loc='left', fontsize=9, color=INK, pad=6)
ax.legend(loc='lower right', frameon=False, fontsize=7.0, handlelength=1.6)
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
print('BudgetMesh', [(round(a, 1), round(b, 2)) for a, b in zip(xs, ys)]); print('simplified', [(round(a, 1), round(b, 2)) for a, b in zip(gx, gy)]); print('FP32 %.0f MB %.2f' % (fx, fy))
