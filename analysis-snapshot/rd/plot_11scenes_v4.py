"""11 scenes + dataset-average panels (2026-11-01 evening, v4 of the figure: indoor rerun rows): per scene the v3 Pareto hull (non-dominated points joined,
dominated points hollow), level-1.0 / collapse curve, FP32 line, and published 3DGS-class points per scene when a baseline
csv row exists; two extra panels with the Mip-360 (9-scene) and T&T (2-scene) averages against published RD curves.
usage: python plot_11scenes_v4.py rd-11scenes-v3.csv baselines-rd-v1.csv out-prefix
baselines csv columns: dataset(m360|tnt|<scene>),method,label,MB,psnr   (dataset = scene name for per-scene points)"""
import csv, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.ticker
import matplotlib.pyplot as plt
rows = list(csv.DictReader(open(sys.argv[1]))); base = list(csv.DictReader(open(sys.argv[2]))); out = sys.argv[3]
SCENES = [('room', 28.480, 597), ('counter', 26.479, 520), ('kitchen', 27.465, 507), ('bonsai', 28.280, 678),
          ('garden', 24.732, 740), ('bicycle', 23.033, 680), ('stump', 24.928, 698), ('flowers', 19.451, 668),
          ('treehill', 20.387, 692), ('truck', 22.387, 432), ('train', 18.900, 485)]
M360 = [s[0] for s in SCENES if s[0] not in ('truck', 'train')]; TNT = ['truck', 'train']
C = {'collapse': '#eb6834', 'v3': '#1baf7a'}
BC = {'HAC++': '#6f4bd8', 'HAC': '#b06fd0', 'RDO-Gaussian': '#2a78d6', '3DGS': '#2b2b2b', 'LightGaussian': '#9a6b00', 'Compact3DGS': '#c2185b',
      'Scaffold-GS': '#555', 'ContextGS': '#0b7285', 'CompGS': '#7a7a00', 'MeshSplatting (paper)': '#1baf7a'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#999', 'axes.linewidth': 0.6, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})

def pareto(pts):
    """pts sorted by MB; keep a point iff no smaller point has >= psnr"""
    keep, best = [], -1e9
    for x, y, lab in pts:
        if y > best: keep.append((x, y, lab)); best = y
    return keep

def draw_ours(ax, pts, omb, annotate=True):
    hull = pareto(pts); dom = [p for p in pts if p not in hull]
    ax.plot([p[0] for p in hull], [p[1] for p in hull], '-o', color=C['v3'], lw=1.7, ms=5, mec='#fcfcfb', mew=1, zorder=4, label='ours: rate-controlled growth + λ-prune (Pareto hull)')
    if dom:
        ax.plot([p[0] for p in dom], [p[1] for p in dom], 'o', mfc='none', mec=C['v3'], ms=5, mew=1.2, zorder=4, label='ours: dominated λ point (more vertices, no test gain)')
    if annotate:
        for k, (x, y, lab) in enumerate(pts):
            ax.annotate(f'{omb / x:.0f}×', (x, y), textcoords='offset points', xytext=(0, -13 if k % 2 == 0 else 7), ha='center', fontsize=7.2, color='#0f6e4c')

def draw_base(ax, key, ms=5, curves_only=False):
    for m in dict.fromkeys(r['method'] for r in base if r['dataset'] == key):
        pts = sorted([(float(r['MB']), float(r['psnr'])) for r in base if r['dataset'] == key and r['method'] == m])
        if curves_only and len(pts) < 2: continue  # per-scene panels: only published RD curves (HAC); single points stay in the average panels
        col = BC.get(m, '#777'); mk = 's' if len(pts) == 1 else 'D'
        ax.plot([p[0] for p in pts], [p[1] for p in pts], '-' + mk if len(pts) > 1 else mk, color=col, lw=1.1, ms=ms if len(pts) == 1 else ms - 1, mec='#fcfcfb', mew=0.8, zorder=3, label=m)

fig, axes = plt.subplots(4, 4, figsize=(14.5, 12.6)); axes = axes.ravel()
for ax, (sc, orig, omb) in zip(axes, SCENES):
    cp = sorted([(float(r['MB']), float(r['test'])) for r in rows if r['scene'] == sc and r['series'] == 'collapse'])
    if cp:
        ax.plot([p[0] for p in cp], [p[1] for p in cp], '-o' if len(cp) > 1 else 'o', color=C['collapse'], lw=1.4, ms=4, mec='#fcfcfb', mew=1, zorder=3,
                label='dense training + codec (level 1.0) / + post-hoc collapse')
    pts = sorted([(float(r['MB']), float(r['test']), r['label']) for r in rows if r['scene'] == sc and r['series'] == 'v3'])
    draw_ours(ax, pts, omb)
    draw_base(ax, sc, curves_only=True)
    ax.set_xscale('log')
    xs = [float(r['MB']) for r in rows if r['scene'] == sc] + [float(r['MB']) for r in base if r['dataset'] == sc and r['method'] == 'HAC']
    ticks = [t for t in (2, 3, 5, 10, 20, 30, 50) if min(xs) * 0.9 <= t <= max(xs) * 1.1]
    ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axhline(orig, color='#2b2b2b', lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.text(0.5, orig, f'FP32 original {orig:.2f} ({omb} MB)', transform=ax.get_yaxis_transform(), ha='center', va='bottom', fontsize=7, color='#2b2b2b')
    ax.set_title(sc + ('  (T&T)' if sc in TNT else ''), fontsize=10, loc='left')
    ys = [float(r['test']) for r in rows if r['scene'] == sc] + [orig] + [float(r['psnr']) for r in base if r['dataset'] == sc and r['method'] == 'HAC']
    lo, hi = min(ys), max(ys); span = max(hi - lo, 1.6); mid = (lo + hi) / 2
    ax.set_ylim(mid - span / 2 - 0.12 * span, mid + span / 2 + 0.1 * span)
    ax.grid(color='#e6e6e3', lw=0.5, zorder=0); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)

# dataset-average panels
def avg_series(scenes, series, label):
    v = [float(r['test']) for r in rows if r['scene'] in scenes and r['series'] == series and r['label'] == label]
    m = [float(r['MB']) for r in rows if r['scene'] in scenes and r['series'] == series and r['label'] == label]
    return (sum(m) / len(m), sum(v) / len(v)) if len(v) == len(scenes) else None
for ax, (key, scenes, name, omb, orig) in zip(axes[11:13], [('m360', M360, 'Mip-NeRF 360 average (9 scenes)', 642, None), ('tnt', TNT, 'Tanks & Temples average (truck, train)', 459, None)]):
    o = sum(dict((s, p) for s, p, _ in SCENES)[s] for s in scenes) / len(scenes)
    lams = ['λ 0.00062', 'λ 0.00185', 'λ 0.0062', 'λ 0.0185']
    pts = sorted([avg_series(scenes, 'v3', l) + (l,) for l in lams if avg_series(scenes, 'v3', l)])
    draw_ours(ax, pts, omb, annotate=False)
    l1 = [(float(r['MB']), float(r['test'])) for r in rows if r['scene'] in scenes and r['series'] == 'collapse' and r['label'].startswith('v1.0')]
    if len(l1) == len(scenes):
        ax.plot(sum(p[0] for p in l1) / len(l1), sum(p[1] for p in l1) / len(l1), 'o', color=C['collapse'], ms=4.5, mec='#fcfcfb', mew=1, zorder=3, label='dense training + codec (level 1.0)')
    ax.plot(omb, o, '*', color='#2b2b2b', ms=9, zorder=3, label=f'FP32 original ({omb} MB)')
    draw_base(ax, key, ms=6)
    ax.set_xscale('log'); ax.set_title(name, fontsize=10, loc='left')
    ax.set_xticks([1, 3, 10, 30, 100, 300, 1000]); ax.set_xticklabels(['1', '3', '10', '30', '100', '300', '1000']); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('size (MB, log)'); ax.grid(color='#e6e6e3', lw=0.5, zorder=0, which='both'); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
for i in (0, 4, 8, 12): axes[i].set_ylabel('test PSNR (dB)')
for i in (8, 9, 10): axes[i].set_xlabel('delivered size (MB, log)')
axes[13].axis('off'); axes[14].axis('off'); axes[15].axis('off')
h, l = [], []
for ax in (axes[0], axes[11]):
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in l: h.append(hh); l.append(ll)
axes[13].legend(h, l, frameon=False, loc='upper left', fontsize=8, bbox_to_anchor=(-0.05, 1.02), ncol=1)
axes[15].text(0.0, 1.0, 'ours, 11-scene means (MB, ratio vs FP32, Δ test PSNR); indoor scenes = --indoor rerun\nλ 6.2e-4:  27.0 MB,  23×, −0.29 dB  (Mip-360: −0.24)\nλ 1.85e-3: 20.4 MB,  32×, −0.34 dB\nλ 6.2e-3:    9.4 MB,  70×, −0.77 dB\nλ 1.85e-2:   4.5 MB, 143×, −1.47 dB\nlevel 1.0 (dense + codec): 26.3 MB, 23×, −0.38 dB\n\n3DGS-class numbers are published values: HAC per-scene and\naverage curves (HAC appendix, λe 0.5e-3…4e-3), HAC++ low/high\nand single points from HAC++ Table II. Per-scene panels show\nonly the HAC curve; the PSNR gap to the 3DGS class is the\nopaque-mesh representation gap, not the codec.',
              transform=axes[15].transAxes, fontsize=8.4, va='top', color='#2b2b2b')
fig.suptitle('Eleven scenes, one codec, four λ values; dataset averages against published 3DGS-class compressors', fontsize=10.5, y=1.0)
fig.tight_layout(); fig.savefig(out + '.png', dpi=200, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight'); print('wrote', out)
