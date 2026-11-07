"""figstyle_v1 (2026-11-01): one visual language for every paper/report figure + shared data loaders.
Palette: ours = green, post-hoc / dense+codec = orange, fixed cap = grey, mesh class = black hollow circles with text labels,
Gaussian class = cool colours, generic codecs = purple.  Circles everywhere (no exotic markers); labels are text next to points.
x axes: absolute delivered MB, log, plain tick labels; a 'better' arrow in every RD panel; curves through our points are PCHIP in log-MB."""
import csv, re, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker as mt
from scipy.interpolate import PchipInterpolator
D = '/mnt/project-files/analysis/rd/'
OURS, POST, CAP, INK, GEN, STAR = '#1a9e6c', '#e8702a', '#8c8c8c', '#262626', '#7a52c7', '#262626'
GAUSS = {'HAC++': '#5b3fd1', 'HAC': '#9b6fd6', 'ContextGS': '#0f7c8c', 'RDO-Gaussian': '#2a6fd4', 'Compact3DGS': '#b8336a', 'LightGaussian': '#8a6d00', '3DGS': '#555555', 'Scaffold-GS': '#777777'}
M360 = ['room', 'counter', 'kitchen', 'bonsai', 'garden', 'bicycle', 'stump', 'flowers', 'treehill']; TNT = ['truck', 'train']
M360_7 = ['room', 'counter', 'kitchen', 'bonsai', 'garden', 'bicycle', 'stump']
LAMS = [0.00062, 0.00185, 0.0062, 0.0185]
# FP32 originals: pipeline-evaluated test PSNR (same evaluator as every delivered asset) and FP32 size (V*208 + F*12 bytes)
ORIG = {'room': (28.263, 741.7), 'counter': (25.944, 575.0), 'kitchen': (26.454, 456.0), 'bonsai': (27.291, 662.2), 'garden': (24.732, 739.8),
        'bicycle': (23.033, 680.0), 'stump': (24.928, 698.0), 'flowers': (19.451, 667.7), 'treehill': (20.387, 691.8), 'truck': (22.387, 432.4), 'train': (18.900, 484.7)}
def rc():
    plt.rcParams.update({'font.size': 8.5, 'font.family': 'DejaVu Sans', 'axes.edgecolor': '#9a9a9a', 'axes.linewidth': 0.6, 'axes.titlesize': 9.5, 'axes.titleweight': 'medium',
                         'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.6, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'savefig.dpi': 220})
def logx(ax, lo=2.5, hi=2000, ticks=(3, 10, 30, 100, 300, 1000)):
    ax.set_xscale('log'); t = [v for v in ticks if lo <= v <= hi]; ax.set_xticks(t); ax.xaxis.set_major_formatter(mt.FixedFormatter([str(v) for v in t]))
    ax.xaxis.set_minor_formatter(mt.NullFormatter()); ax.set_xlim(lo, hi); ax.grid(color='#ededeb', lw=0.5, which='major'); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
def better(ax, up=True, text='better'):
    y0, y1 = (0.885, 0.985) if up else (0.115, 0.015)
    ax.annotate('', xy=(0.02, y1), xytext=(0.14, y0), xycoords='axes fraction', arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.2))
    ax.text(0.155, (y0 + y1) / 2, text, transform=ax.transAxes, fontsize=7.6, va='center', color=INK)
def smooth(ax, pts, color=OURS, lw=2.0, ms=5.5, label=None, zorder=5, ls='-'):
    """pts = [(MB, y)]: PCHIP through the points in log-MB, markers on the data points."""
    p = sorted(pts); x = np.log10([q[0] for q in p]); y = np.array([q[1] for q in p])
    if len(p) >= 3 and np.all(np.diff(x) > 0):
        xs = np.linspace(x[0], x[-1], 120); ax.plot(10 ** xs, PchipInterpolator(x, y)(xs), ls, color=color, lw=lw, zorder=zorder, label=label, solid_capstyle='round')
        ax.plot([q[0] for q in p], y, 'o', color=color, ms=ms, mec='white', mew=1, zorder=zorder + 1)
    else: ax.plot([q[0] for q in p], y, ls + 'o', color=color, lw=lw, ms=ms, mec='white', mew=1, zorder=zorder, label=label)
def point(ax, x, y, name, color=INK, hollow=True, dx=6, dy=0, ha='left', ms=6.5, fs=7.4, zorder=4, marker='o', **kw):
    ax.plot(x, y, marker, color=color, mfc='white' if hollow else color, ms=ms, mew=1.3 if hollow else 0.8, mec=color if hollow else 'white', zorder=zorder)
    if name: ax.annotate(name, (x, y), textcoords='offset points', xytext=(dx, dy), ha=ha, va='center', fontsize=fs, color=color, zorder=zorder + 1, **kw)
RATIO_LABELS = False  # user 11/01: compression ratios go in the tables, not on the figures
def label_ratio(ax, x, y, ratio, color=OURS, dy=-11, fs=7.2):
    if not RATIO_LABELS: return
    ax.annotate(f'{ratio:.0f}×', (x, y), textcoords='offset points', xytext=(0, dy), ha='center', fontsize=fs, color=color)
# ---------- data ----------
def load_rows():
    rows = list(csv.DictReader(open(D + 'rd-11scenes-v2.csv')))
    ssim = {}
    for f in ('rd-prune3-7scenes-2026-10-24.csv', 'rd-prune3-new4-2026-10-24.csv', 'rd-prune3-hull2-2026-11-01.csv'):
        for r in csv.DictReader(open(D + f)):
            if r.get('scene') and r.get('ssim'): ssim[(r['scene'], round(float(r['lam']), 6))] = (float(r['ssim']), float(r['lpips']))
    return rows, ssim
def ours_lambda(rows, scene, lam):
    lab = 'λ ' + ('0.00062' if lam == 0.00062 else '0.00185' if lam == 0.00185 else '0.0062' if lam == 0.0062 else '0.0185')
    r = [x for x in rows if x['scene'] == scene and x['series'] == 'v3' and x['label'].startswith(lab)]
    assert len(r) == 1, (scene, lam, len(r)); return float(r[0]['MB']), float(r[0]['test'])
def level1(rows, scene):
    r = [x for x in rows if x['scene'] == scene and x['series'] == 'collapse' and x['label'].startswith('v1.0')][0]; return float(r['MB']), float(r['test'])
def mean_lambda(rows, scenes):
    out = []
    for l in LAMS:
        p = [ours_lambda(rows, s, l) for s in scenes]; out.append((sum(q[0] for q in p) / len(p), sum(q[1] for q in p) / len(p)))
    return out
def mean_level1(rows, scenes):
    p = [level1(rows, s) for s in scenes]; return sum(q[0] for q in p) / len(p), sum(q[1] for q in p) / len(p)
def mean_orig(scenes): return sum(ORIG[s][1] for s in scenes) / len(scenes), sum(ORIG[s][0] for s in scenes) / len(scenes)
def generic_room():
    g = {}
    for line in open(D + 'generic-codecs-2026-10-24.txt'):
        parts = [s.strip() for s in line.split('|')]
        if len(parts) < 3: continue
        sc_name = parts[0].split()
        if len(sc_name) != 2: continue
        sc, name = sc_name; mb = re.match(r'([\d.]+) MB', parts[1]); ps = re.search(r'test ([\d.]+) /', line)
        if sc == 'room' and mb and ps: g[name] = (float(mb.group(1)), float(ps.group(1)))
    return g
def baselines(dataset):
    out = {}
    for r in csv.DictReader(open(D + 'baselines-rd-v1.csv')):
        if r['dataset'] == dataset: out.setdefault(r['method'], []).append((float(r['MB']), float(r['psnr'])))
    return {k: sorted(v) for k, v in out.items()}
def save(fig, name):
    fig.savefig(name + '.png', bbox_inches='tight'); fig.savefig(name + '.pdf', bbox_inches='tight'); print('wrote', name)
