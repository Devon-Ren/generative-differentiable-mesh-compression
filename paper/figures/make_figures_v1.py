"""make_figures_v1.py (2026-11-10): the four experiment figures from repo tables. Run from the repo root."""
import re, json, statistics, math
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = 'paper/figures/'
FP = {'room': (597, 28.48), 'counter': (520, 26.48), 'kitchen': (507, 27.46), 'bonsai': (678, 28.28), 'garden': (740, 24.73), 'bicycle': (680, 23.03), 'stump': (698, 24.93), 'flowers': (668, 19.45), 'treehill': (692, 20.39), 'truck': (432, 22.39), 'train': (485, 18.90), 'drjohnson': (778.4, 26.737), 'playroom': (765.0, 27.718)}
BLV = {'ficus': (74947, 245535, 32.131), 'chair': (219507, 820226, 32.033), 'mic': (133739, 465114, 33.40), 'drums': (240891, 1094114, 25.472), 'materials': (224941, 775298, 28.231), 'hotdog': (287754, 1205503, 35.682), 'lego': (497529, 1857294, 33.515), 'ship': (387201, 1550245, 29.463)}
for s, (v, f, p) in BLV.items(): FP[s] = ((v * 208 + f * 12) / 1e6, p)
MIP = ['room', 'kitchen', 'counter', 'bonsai', 'garden', 'bicycle', 'stump', 'flowers', 'treehill', 'truck', 'train']; DB = ['drjohnson', 'playroom']; BL = list(BLV)
DS = {'Mip-NeRF 360 + T&T': MIP, 'Deep Blending': DB, 'NeRF-Synthetic': BL}
# --- controller rows
ctrl = {}
for f in ['release-adapt-v9b/results/adapt-v9b-2026-11-08.txt', 'release-adapt-v9b/results/adapt-v9b-blender-2026-11-08.txt', 'release-batch21-v1/results/batch21-collect-2026-11-09-2110-raw.txt']:
    for l in open(f):
        m = re.match(r'(\w+) \| target ([\d.]+) MB \|.*?lam_final \(shadow\) ([\d.e+-]+) \|.*?delivered ([\d.]+) MB \(hit ([\d.]+) x target.*?\| test ([\d.]+)', l)
        if m and m[1] in FP:
            cap = float(m[5]) < 0.9 and float(m[3]) <= 1.1e-4
            ctrl[(m[1], float(m[2]))] = dict(mb=float(m[4]), hit=float(m[5]), psnr=float(m[6]), capped=cap)
# --- vertex-cap rows (train-at-budget deliveries)
capr = {}
for f in ['release-g0-refresh-v1/results/g0-refresh-2026-11-09.txt', 'release-g0-refresh-v1/results/g0-refresh-indoor-2026-11-09.txt', 'release-batch21-v1/results/batch21-collect-2026-11-09-2110-raw.txt']:
    for l in open(f):
        m = re.match(r'(\w+) budget (\d+) TRAIN-AT-BUDGET(?:-BLENDER)? \| .*?\| delivered ([\d.]+) MB test ([\d.]+)', l)
        if m and m[1] in FP: capr.setdefault(m[1], []).append((float(m[3]), float(m[4])))
# ---------- Figure 1: RD per dataset (means over scenes)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
PLOT = {'NeRF-Synthetic': (['chair', 'drums', 'hotdog', 'lego', 'ship'], [1.5, 2.5, 3, 4])}   # same scene set at every target (objects at the capacity limit and the missing materials-3 MB run excluded)
for ax, (name, scenes) in zip(axes, DS.items()):
    scenes, targets = PLOT.get(name, (scenes, sorted({t for (s, t) in ctrl if s in scenes})))
    xs, ys = [], []
    for t in targets:
        pts = [ctrl[(s, t)] for s in scenes if (s, t) in ctrl and not ctrl[(s, t)]['capped']]
        if len(pts) >= max(2, len(scenes) // 2): xs.append(statistics.mean(p['mb'] for p in pts)); ys.append(statistics.mean(p['psnr'] for p in pts))
    ax.plot(xs, ys, 'o-', color='#d9822b', lw=2, label='BudgetMesh (target sizes)')
    # cap curve: tiers = sorted budgets per scene, averaged by rank
    K = max(len(v) for s, v in capr.items() if s in scenes); cx, cy = [], []
    for k in range(K):
        pts = [sorted(capr[s], reverse=True)[k] for s in scenes if s in capr and len(capr[s]) > k]
        if len(pts) >= max(2, len(scenes) // 2): cx.append(statistics.mean(p[0] for p in pts)); cy.append(statistics.mean(p[1] for p in pts))
    ax.plot(cx, cy, 's--', color='#5b7a99', lw=1.6, label='MeshSplatting, vertex cap + our codec')
    fx = statistics.mean(FP[s][0] for s in scenes); fy = statistics.mean(FP[s][1] for s in scenes)
    ax.plot([fx], [fy], '*', color='k', ms=11, label='FP32 model'); ax.annotate(f'{fx:.0f} MB', (fx, fy), textcoords='offset points', xytext=(-6, 6), ha='right', fontsize=8)
    ax.set_xscale('log'); ax.set_title(name + (' (5 objects)' if name in PLOT else '')); ax.set_xlabel('file size (MB, log)'); ax.grid(alpha=.3)
    import matplotlib.ticker as mt; ax.xaxis.set_major_locator(mt.LogLocator(base=10, subs=(1, 2, 5))); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f'{v:g}')); ax.xaxis.set_minor_formatter(mt.NullFormatter())
    for x, y in zip(xs, ys): ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=7, color='#d9822b')
axes[0].set_ylabel('PSNR (dB), mean over scenes'); axes[0].legend(fontsize=8, loc='lower right')
fig.tight_layout(); fig.savefig(OUT + 'fig_rd_per_dataset.pdf'); fig.savefig(OUT + 'fig_rd_per_dataset.png', dpi=160); plt.close(fig)
# ---------- Figure 2: size-error histogram per dataset
fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=False)
for ax, (name, scenes) in zip(axes, DS.items()):
    e = [100 * (ctrl[k]['hit'] - 1) for k in ctrl if k[0] in scenes and not ctrl[k]['capped']]
    ax.hist(e, bins=range(-15, 16, 2), color='#d9822b', alpha=.85, edgecolor='white'); ax.axvline(0, color='k', lw=.8); ax.axvline(-10, color='gray', ls='--', lw=.8); ax.axvline(10, color='gray', ls='--', lw=.8)
    ax.set_title(f'{name}: {len(e)} runs, {sum(abs(x) <= 10 for x in e)} within ±10 %'); ax.set_xlabel('size error (%) = (file − target)/target'); ax.grid(alpha=.3)
axes[0].set_ylabel('runs'); fig.tight_layout(); fig.savefig(OUT + 'fig_size_error_hist.pdf'); fig.savefig(OUT + 'fig_size_error_hist.png', dpi=160); plt.close(fig)
# ---------- Figure 3: bytes per stream, room 20 and 5 MB
streams = ['connectivity', 'positions', 'appearance indices', 'DC colour', 'dictionaries + header']
b20 = [8904794, 4204635, 2367714 + 1816079, 2675255, 609573 + 39570 + 15811 + 96 + (20640878 - 20633431)]; b5 = [2084546, 1082631, 518093 + 390373, 575060, 603964 + 40283 + 3630 + 96 + (5305014 - 5298676)]
fig, ax = plt.subplots(figsize=(6.5, 3.2)); cols = ['#5b7a99', '#8fa8c2', '#d9822b', '#f0b27a', '#9a9a94']
for i, (lab, tot) in enumerate([('room, 20 MB target', b20), ('room, 5 MB target', b5)]):
    left = 0; T = sum(tot)
    for sname, b, c in zip(streams, tot, cols):
        ax.barh(i, 100 * b / T, left=left, color=c, label=sname if i == 0 else None); 
        if b / T > 0.06: ax.text(left + 50 * b / T, i, f'{100*b/T:.0f}%', ha='center', va='center', fontsize=8, color='white')
        left += 100 * b / T
ax.set_yticks([0, 1]); ax.set_yticklabels(['room, 20.6 MB', 'room, 5.3 MB']); ax.set_xlabel('share of the file (%)'); ax.legend(fontsize=7, ncol=3, loc='upper center', bbox_to_anchor=(.5, -.35)); ax.set_xlim(0, 100)
fig.tight_layout(); fig.savefig(OUT + 'fig_bytes_per_stream.pdf'); fig.savefig(OUT + 'fig_bytes_per_stream.png', dpi=160); plt.close(fig)
# ---------- Figure 4: controller traces
tr = [json.loads(l) for l in open('release-batch23-render-v1/results/traces-b21-2026-11-10.jsonl')]
groups = {'Mip-NeRF 360 indoor + Deep Blending': lambda r: r.split('-')[0] in ['room', 'kitchen', 'counter', 'bonsai', 'drjohnson', 'playroom'], 'Mip-NeRF 360 outdoor + T&T': lambda r: r.split('-')[0] in ['garden', 'bicycle', 'stump', 'flowers', 'treehill', 'truck', 'train'], 'NeRF-Synthetic': lambda r: 'adapt9xb' in r}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, (name, sel) in zip(axes, groups.items()):
    n = 0
    for t in tr:
        if not sel(t['run']) or not t['rounds'] or not t['target']: continue
        it = [r[0] / 1000 for r in t['rounds']]; est = [r[2] / t['target'] for r in t['rounds']]; n += 1
        ax.plot(it, est, '-', color='#d9822b', alpha=.35, lw=1)
        ax.plot([it[-1], 30], [est[-1], t['final'] / t['target']], ':', color='#5b7a99', alpha=.5, lw=1)
    ax.axhline(1.0, color='k', lw=.9); ax.axhspan(.9, 1.1, color='gray', alpha=.12); ax.set_ylim(.6, 1.6); ax.set_title(f'{name} ({n} runs)'); ax.set_xlabel('training iteration (k)'); ax.grid(alpha=.3)
axes[0].set_ylabel('estimated landing size / target'); fig.tight_layout(); fig.savefig(OUT + 'fig_controller_traces.pdf'); fig.savefig(OUT + 'fig_controller_traces.png', dpi=160); plt.close(fig)
print('figures written')
