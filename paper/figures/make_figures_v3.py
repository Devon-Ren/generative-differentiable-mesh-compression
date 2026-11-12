"""make_figures_v3.py (2026-11-12): = v2 at print size (CVPR text width 7.0 in, 8 pt fonts, no scaling in LaTeX) + the simplification-after-training curve on every panel. = v1 with the RD figure redrawn: smooth (PCHIP in log size) curves, the per-scene landing spread drawn as whiskers at each target, 3DGS-class compressed codecs on the Mip-360 panel, mesh-class points limited to 1 GB; the vertex-cap curves use the batch24 cap runs at the controller counts. Run from the repo root."""
import re, json, statistics, math
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'pdf.fonttype': 42})
OUT = 'paper/cvpr/figures/'
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
for f in ['release-g0-refresh-v1/results/g0-refresh-2026-11-09.txt', 'release-g0-refresh-v1/results/g0-refresh-indoor-2026-11-09.txt', 'release-batch21-v1/results/batch21-collect-2026-11-09-2110-raw.txt', 'release-batch24-ablation-v1/results/batch24-collect-2026-11-11-0641.txt']:
    for l in open(f):
        m = re.match(r'(\w+) budget (\d+) TRAIN-AT-BUDGET(?:-BLENDER)? \| .*?\| delivered ([\d.]+) MB test ([\d.]+)', l)
        if m and m[1] in FP: capr.setdefault(m[1], []).append((float(m[3]), float(m[4])))
simp = {}
for f in ['release-g0-refresh-v1/results/g0-refresh-2026-11-09.txt', 'release-g0-refresh-v1/results/g0-refresh-indoor-2026-11-09.txt', 'release-batch21-v1/results/batch21-collect-2026-11-09-2110-raw.txt', 'release-batch24-ablation-v1/results/batch24-collect-2026-11-11-0641.txt']:
    for l in open(f):
        m = re.match(r'(\w+) budget (\d+) POST-HOC-COLLAPSE(?:-BLENDER)? \| .*?\| delivered ([\d.]+) MB test ([\d.]+)', l)
        if m and m[1] in FP: simp.setdefault(m[1], []).append((float(m[3]), float(m[4])))
# ---------- Figure 1: RD per dataset, smooth curves, landing whiskers, published points
from scipy.interpolate import PchipInterpolator
import numpy as np
DS4 = {'Mip-NeRF 360': MIP[:9], 'T&T': ['truck', 'train'], 'Deep Blending': DB, 'NeRF-Synthetic': BL}
MESHPTS = {'Mip-NeRF 360': {'MeshSplatting (paper)': (100, 24.78), 'MiLo': (253, 24.09), 'BOG (7 sc.)': (913, 25.62)}, 'T&T': {'MiLo': (253, 21.46)}}
GSPTS = {'HAC++': [(8.34, 27.60), (18.48, 27.82)], 'HAC': [(15.26, 27.53), (21.87, 27.77)], 'CompGS': [(8.83, 26.37), (16.5, 27.26)], 'Compact3DGS': [(48.8, 27.08)]}
PLOT = {'NeRF-Synthetic': (['chair', 'drums', 'hotdog', 'lego', 'ship'], [1.5, 2.5, 3, 4])}
def smooth(xs, ys):
    d = {}
    for x, y in zip(xs, ys): d.setdefault(round(float(x), 4), []).append(float(y))
    xs = np.array(sorted(d)); ys = np.array([np.mean(d[x]) for x in xs])
    if len(xs) < 3: return xs, ys
    lx = np.log(xs); f = PchipInterpolator(lx, ys); g = np.linspace(lx[0], lx[-1], 60); return np.exp(g), f(g)
fig, axes2 = plt.subplots(2, 2, figsize=(7.0, 4.6)); axes = axes2.ravel()
for ax, (name, scenes) in zip(axes, DS4.items()):
    scenes, targets = PLOT.get(name, (scenes, sorted({t for (s, t) in ctrl if s in scenes})))
    xs, ys, lo, hi = [], [], [], []
    for t in targets:
        pts = [ctrl[(s, t)] for s in scenes if (s, t) in ctrl and not ctrl[(s, t)]['capped']]
        if len(pts) >= max(1, len(scenes) // 2):
            xs.append(statistics.mean(p['mb'] for p in pts)); ys.append(statistics.mean(p['psnr'] for p in pts)); lo.append(min(p['mb'] for p in pts)); hi.append(max(p['mb'] for p in pts))
    sx, sy = smooth(xs, ys); ax.plot(sx, sy, '-', color='#d9822b', lw=2, zorder=5)
    ax.errorbar(xs, ys, xerr=[np.array(xs) - np.array(lo), np.array(hi) - np.array(xs)], fmt='o', color='#d9822b', ecolor='#d9822b', elinewidth=1.0, capsize=2, ms=4, zorder=6, label='BudgetMesh at the target sizes (whiskers: per-scene file-size range)')
    for t in targets: ax.axvline(t, color='#d9822b', lw=0.6, ls=':', alpha=0.7)
    K = min(len(v) for s, v in capr.items() if s in scenes); cx, cy = [], []
    for k in range(K):
        pts = [sorted(capr[s], reverse=True)[k] for s in scenes if s in capr and len(capr[s]) > k]
        if len(pts) >= max(1, len(scenes) // 2): cx.append(statistics.mean(p[0] for p in pts)); cy.append(statistics.mean(p[1] for p in pts))
    cxs, cys = smooth(cx, cy); ax.plot(cxs, cys, '--', color='#5b7a99', lw=1.3); ax.plot(cx, cy, 's', color='#5b7a99', ms=3.5, label='MeshSplatting, vertex cap + our codec')
    Ks = min(len(v) for s_, v in simp.items() if s_ in scenes) if any(s_ in simp for s_ in scenes) else 0; gx, gy = [], []
    for k in range(Ks):
        pts = [sorted(simp[s_], reverse=True)[k] for s_ in scenes if s_ in simp and len(simp[s_]) > k]
        if len(pts) >= max(1, len(scenes) // 2): gx.append(statistics.mean(p[0] for p in pts)); gy.append(statistics.mean(p[1] for p in pts))
    if gx: gxs, gys = smooth(gx, gy); ax.plot(gxs, gys, ':', color='#9a9a9a', lw=1.3); ax.plot(gx, gy, 'x', color='#9a9a9a', ms=4, label='MeshSplatting FP32, simplified after training + our codec')
    fx = statistics.mean(FP[s][0] for s in scenes); fy = statistics.mean(FP[s][1] for s in scenes)
    ax.plot([fx], [fy], '*', color='k', ms=8, label='MeshSplatting FP32 (our reproduction)'); ax.annotate(f'{fx:.0f} MB', (fx, fy), textcoords='offset points', xytext=(-4, 4), ha='right', fontsize=6.5)
    for i, (k, (x, y)) in enumerate(MESHPTS.get(name, {}).items()):
        ax.scatter([x], [y], marker='D', s=18, color='#7a5c99', zorder=4, label='other mesh methods (published)' if i == 0 else None)
        ax.annotate(k, (x, y), textcoords='offset points', xytext={'MeshSplatting (paper)': (0, -9), 'MiLo': (0, -9), 'BOG (7 sc.)': (0, 5)}.get(k, (0, 5)), ha='center', fontsize=6.5, color='#5a4273')
    if name == 'Mip-NeRF 360':
        for i, (k, pts) in enumerate(GSPTS.items()):
            px = [p[0] for p in pts]; py = [p[1] for p in pts]
            ax.plot(px, py, '^-', color='#8a8a8a', ms=3.5, lw=0.8, label='3DGS-class codecs (published)' if i == 0 else None, zorder=3)
            off = {'HAC++': (4, 1), 'HAC': (4, -4), 'CompGS': (-4, -8), 'Compact3DGS': (4, -3)}[k]
            ax.annotate(k, (px[-1], py[-1]), textcoords='offset points', xytext=off, ha='left' if off[0] >= 0 else 'right', fontsize=6, color='#6a6a6a')
    ax.margins(y=0.12); ax.set_xscale('log'); ax.set_title(name + (' (5 objects)' if name in PLOT else '')); ax.set_xlabel('file size (MB, log)'); ax.grid(alpha=.3); ax.tick_params(length=2, pad=1.5)
    import matplotlib.ticker as mt; ax.xaxis.set_major_locator(mt.LogLocator(base=10, subs=(1, 2, 5))); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f'{v:g}' if v in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) else '')); ax.xaxis.set_minor_formatter(mt.NullFormatter())
    if name == 'Mip-NeRF 360': ax.set_xlim(3.5, 1100)
axes[0].set_ylabel('PSNR (dB), mean over scenes'); axes[2].set_ylabel('PSNR (dB), mean over scenes')
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=1.8, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.0, h_pad=1.2); fig.savefig(OUT + 'fig_rd_per_dataset.pdf', bbox_inches='tight'); fig.savefig(OUT + 'fig_rd_per_dataset.png', dpi=160, bbox_inches='tight'); plt.close(fig)
# ---------- Figure 2: size-error histogram per dataset
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=False)
for ax, (name, scenes) in zip(axes, DS.items()):
    e = [100 * (ctrl[k]['hit'] - 1) for k in ctrl if k[0] in scenes and not ctrl[k]['capped']]
    ax.hist(e, bins=range(-15, 16, 2), color='#d9822b', alpha=.85, edgecolor='white'); ax.axvline(0, color='k', lw=.8); ax.axvline(-10, color='gray', ls='--', lw=.8); ax.axvline(10, color='gray', ls='--', lw=.8)
    ax.set_title(f'{name}: {len(e)} runs, {sum(abs(x) <= 10 for x in e)} within ±10 %'); ax.set_xlabel('size error (%) = (file − target)/target'); ax.grid(alpha=.3)
axes[0].set_ylabel('runs'); fig.tight_layout(); fig.savefig(OUT + 'fig_size_error_hist.pdf'); fig.savefig(OUT + 'fig_size_error_hist.png', dpi=160); plt.close(fig)
# ---------- Figure 3: bytes per stream, room 20 and 5 MB
streams = ['connectivity', 'positions', 'appearance indices', 'DC colour', 'dictionaries + header']
b20 = [8904794, 4204635, 2367714 + 1816079, 2675255, 609573 + 39570 + 15811 + 96 + (20640878 - 20633431)]; b5 = [2084546, 1082631, 518093 + 390373, 575060, 603964 + 40283 + 3630 + 96 + (5305014 - 5298676)]
fig, ax = plt.subplots(figsize=(3.3, 2.0)); cols = ['#5b7a99', '#8fa8c2', '#d9822b', '#f0b27a', '#9a9a94']
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
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
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

# ---------- Figure 2b: size error as a strip plot (replaces the histogram)
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True, gridspec_kw=dict(width_ratios=[5, 3, 5]))
import random; random.seed(0)
for ax, (name, scenes) in zip(axes, DS.items()):
    targets = sorted({t for (s, t) in ctrl if s in scenes})
    for i, t in enumerate(targets):
        e = [100 * (ctrl[(s, t)]['hit'] - 1) for s in scenes if (s, t) in ctrl and not ctrl[(s, t)]['capped']]
        ec = [100 * (ctrl[(s, t)]['hit'] - 1) for s in scenes if (s, t) in ctrl and ctrl[(s, t)]['capped']]
        ax.scatter([i + random.uniform(-.12, .12) for _ in e], e, s=22, color='#d9822b', alpha=.8, edgecolor='white', lw=.5, zorder=3)
        if ec: ax.scatter([i + random.uniform(-.12, .12) for _ in ec], [max(v, -32) for v in ec], s=22, marker='v', color='#9a9a94', alpha=.8, zorder=3)
        if e: ax.plot([i - .22, i + .22], [statistics.median(e)] * 2, color='k', lw=1.6, zorder=4)
    ax.axhspan(-10, 10, color='gray', alpha=.12); ax.axhline(0, color='k', lw=.7)
    ax.set_xticks(range(len(targets))); ax.set_xticklabels([f'{t:g} MB' for t in targets]); ax.set_title(name); ax.grid(axis='y', alpha=.3); ax.set_ylim(-32, 20)
axes[0].set_ylabel('size error (%)\n(file − target) / target')
axes[0].scatter([], [], s=22, color='#d9822b', label='one scene'); axes[0].plot([], [], color='k', lw=1.6, label='median'); axes[0].scatter([], [], marker='v', color='#9a9a94', label='capacity limit (price at floor)'); axes[0].legend(fontsize=7.5, loc='lower left')
fig.tight_layout(); fig.savefig(OUT + 'fig_size_error_strip.pdf'); fig.savefig(OUT + 'fig_size_error_strip.png', dpi=160); plt.close(fig)

# ---------- Figure 5: cross-representation points, Mip-NeRF 360 means (published numbers; baselines-v2.md)
mip9 = MIP[:9]
xs, ys = [], []
for t in [5, 7.5, 10, 15, 20]:
    pts = [ctrl[(s, t)] for s in mip9 if (s, t) in ctrl]; xs.append(statistics.mean(p['mb'] for p in pts)); ys.append(statistics.mean(p['psnr'] for p in pts))
fx = statistics.mean(FP[s][0] for s in mip9); fy = statistics.mean(FP[s][1] for s in mip9)
mesh_pts = {'MeshSplatting (paper, RGB-only PLY)': (100, 24.78), 'MeshSplatting FP32 (our reproduction)': (fx, fy), 'MiLo': (253, 24.09), 'RaDe-GS mesh': (1100, 23.56), 'GOF mesh': (1500, 20.78), 'BakedSDF (7-scene set)': (434, 25.09), 'Binary Opacity Grids (7-scene set)': (870, 26.09)}
gs_pts = {'3DGS': (750.9, 27.46), 'Scaffold-GS': (253.9, 27.50), 'Compact3DGS': (48.8, 27.08), 'LightGaussian': (44.54, 27.00), 'RDO-Gaussian': (23.46, 27.05), 'HAC': (15.26, 27.53), 'HAC (high)': (21.87, 27.77), 'HAC++': (8.34, 27.60), 'HAC++ (high)': (18.48, 27.82), 'CompGS': (8.83, 26.37), 'CompGS (high)': (16.5, 27.26)}
fig, ax = plt.subplots(figsize=(3.3, 2.6))
ax.plot(xs, ys, 'o-', color='#d9822b', lw=2.2, ms=6, label='BudgetMesh (ours), 5 … 20 MB targets', zorder=5)
for i, (k, (x, y)) in enumerate(mesh_pts.items()):
    off = {'MeshSplatting (paper, RGB-only PLY)': (-6, 5, 'right'), 'BakedSDF (7-scene set)': (6, 7, 'left'), 'MeshSplatting FP32 (our reproduction)': (6, -11, 'left'), 'GOF mesh': (-6, 4, 'right'), 'MiLo': (6, -11, 'left'), 'RaDe-GS mesh': (6, -4, 'left'), 'Binary Opacity Grids (7-scene set)': (-6, 4, 'right')}.get(k, (6, 4, 'left'))
    ax.scatter([x], [y], marker='s', s=55, color='#5b7a99', zorder=4, label='mesh class (published, single points)' if i == 0 else None); ax.annotate(k, (x, y), textcoords='offset points', xytext=off[:2], ha=off[2], fontsize=7.5, color='#3f5566')
for i, (k, (x, y)) in enumerate(gs_pts.items()):
    ax.scatter([x], [y], marker='^', s=50, color='#2f8a5b', zorder=4, label='3DGS class (published, not a mesh)' if i == 0 else None)
    if '(high)' not in k: ax.annotate(k, (x, y), textcoords='offset points', xytext=(6, 4 if i % 2 else -9), fontsize=7.5, color='#2f8a5b')
ax.set_xscale('log'); ax.set_xlabel('file size (MB, log; Mip-NeRF 360 mean)'); ax.set_ylabel('PSNR (dB, Mip-NeRF 360 mean)'); ax.grid(alpha=.3); ax.legend(fontsize=8, loc='lower left')
import matplotlib.ticker as mt; ax.xaxis.set_major_locator(mt.LogLocator(base=10, subs=(1, 2, 5))); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f'{v:g}')); ax.xaxis.set_minor_formatter(mt.NullFormatter())
ax.set_ylim(20, 28.6); ax.set_xlim(3.5, 3000)
fig.tight_layout(); fig.savefig(OUT + 'fig_cross_methods_mip360.pdf'); fig.savefig(OUT + 'fig_cross_methods_mip360.png', dpi=160); plt.close(fig)
print('strip + cross-method figures written')
