"""Main-paper RD figure, v1 (2026-11-01): Mip-NeRF 360 averages.
(a) relative: ΔPSNR vs each method's OWN uncompressed representation against compression ratio (log) — the codec question.
(b) absolute: size (log) vs test PSNR among engine-ready meshes (ours, MeshSplatting, MiLo, BakedSDF, BOG, RaDe-GS, GOF);
    the Gaussian-splat class is drawn grey and small for honesty (not a mesh, +3 dB class gap).
usage: python plot_rd_main_v1.py rd-11scenes-v2.csv baselines-rd-v1.csv out-prefix"""
import csv, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.ticker as mt
import matplotlib.pyplot as plt
rows = list(csv.DictReader(open(sys.argv[1]))); base = list(csv.DictReader(open(sys.argv[2]))); out = sys.argv[3]
ORIG = {'room': (28.263, 741.7), 'counter': (25.944, 575.0), 'kitchen': (26.454, 456.0), 'bonsai': (27.291, 662.2), 'garden': (24.732, 739.8),
        'bicycle': (23.033, 680.0), 'stump': (24.928, 698.0), 'flowers': (19.451, 667.7), 'treehill': (20.387, 691.8)}
M360 = list(ORIG); o_psnr = sum(v[0] for v in ORIG.values()) / 9; o_mb = sum(v[1] for v in ORIG.values()) / 9
def ours(series, label):
    r = [x for x in rows if x['scene'] in M360 and x['series'] == series and x['label'].startswith(label)]
    assert len(r) == 9, (series, label, len(r)); return sum(float(x['MB']) for x in r) / 9, sum(float(x['test']) for x in r) / 9
lam = [ours('v3', l) for l in ('λ 0.00062', 'λ 0.00185', 'λ 0.0062', 'λ 0.0185')]; l1 = ours('collapse', 'v1.0')
def bpts(method): return sorted([(float(r['MB']), float(r['psnr'])) for r in base if r['dataset'] == 'm360' and r['method'] == method])
# each 3DGS-class method relative to the 3DGS reference its own paper reports
# anchor-based coders (HAC, HAC++, ContextGS) compress Scaffold-GS, so their own uncompressed representation is Scaffold-GS (253.9 MB / 27.50, HAC++ Table II);
# RDO-Gaussian, LightGaussian, Compact3DGS compress vanilla 3DGS (their papers' 3DGS reference)
REF = {'HAC': (253.9, 27.50), 'HAC++': (253.9, 27.50), 'ContextGS': (253.9, 27.50), 'RDO-Gaussian': (745.8, 27.49), 'LightGaussian': (750.9, 27.46), 'Compact3DGS': (750.9, 27.46)}
BASE = {'HAC': 'Scaffold-GS', 'HAC++': 'Scaffold-GS', 'ContextGS': 'Scaffold-GS', 'RDO-Gaussian': '3DGS', 'LightGaussian': '3DGS', 'Compact3DGS': '3DGS'}
MESH = [('MeshSplatting, authors\' RGB-only PLY (SH dropped)', 100, 24.78, 'o'), ('MiLo (SIGGRAPH Asia 25)', 253, 24.09, 's'), ('BakedSDF (SIGGRAPH 23)', 434, 24.51, '^'),
        ('Binary Opacity Grids (SIGGRAPH 24)', 913, 25.62, 'v'), ('RaDe-GS mesh', 1100, 23.56, 'P'), ('GOF mesh', 1500, 20.78, 'X')]
GS = [('3DGS', [(750.9, 27.46)]), ('HAC++', bpts('HAC++')), ('HAC', bpts('HAC')), ('RDO-Gaussian', bpts('RDO-Gaussian'))]
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#999', 'axes.linewidth': 0.6, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
fig, (a, b) = plt.subplots(1, 2, figsize=(12.4, 4.6))
G = '#1baf7a'; OR = '#eb6834'
# (a) relative
a.axhline(0, color='#2b2b2b', lw=0.8, ls=(0, (4, 3)))
a.plot([o_mb / p[0] for p in lam], [p[1] - o_psnr for p in lam], '-o', color=G, lw=2, ms=6, mec='white', mew=1, zorder=5, label='ours: rate-controlled growth + λ-prune, vs FP32 MeshSplatting')
a.plot(o_mb / l1[0], l1[1] - o_psnr, 'o', color=OR, ms=6, mec='white', mew=1, zorder=5, label='ours: dense training + same codec (level 1.0)')
for x, y, t in zip([o_mb / p[0] for p in lam], [p[1] - o_psnr for p in lam], ['λ 6.2e-4', '1.85e-3', '6.2e-3', '1.85e-2']):
    a.annotate(t, (x, y), textcoords='offset points', xytext=(6, -11), fontsize=7.5, color='#0f6e4c')
cols = {'HAC++': '#6f4bd8', 'HAC': '#b06fd0', 'RDO-Gaussian': '#2a78d6', 'ContextGS': '#0b7285', 'LightGaussian': '#9a6b00', 'Compact3DGS': '#c2185b'}
for m in ('HAC++', 'HAC', 'ContextGS', 'RDO-Gaussian', 'LightGaussian', 'Compact3DGS'):
    pts = bpts(m); rm, rp = REF[m]
    a.plot([rm / p[0] for p in pts], [p[1] - rp for p in pts], ('-D' if len(pts) > 1 else 's'), color=cols[m], lw=1.1, ms=4.5, mec='white', mew=0.7, zorder=3, label=m + ' vs ' + BASE[m] + (' (its base model)' if BASE[m] == 'Scaffold-GS' else ''))
a.set_xscale('log'); a.set_xticks([5, 10, 20, 50, 100, 200, 500]); a.xaxis.set_major_formatter(mt.FixedFormatter(['5', '10', '20', '50', '100', '200', '500'])); a.xaxis.set_minor_formatter(mt.NullFormatter())
a.set_xlabel('compression ratio vs the method\'s own uncompressed representation (log)'); a.set_ylabel('Δ test PSNR vs own uncompressed (dB)')
a.set_title('(a) quality kept per compression ratio, Mip-NeRF 360 (9 scenes)', loc='left', fontsize=9.5)
a.legend(frameon=False, fontsize=7.4, loc='lower left'); a.grid(color='#ececea', lw=0.5, which='both'); a.set_axisbelow(True)
for sp in ('top', 'right'): a.spines[sp].set_visible(False)
# (b) absolute among meshes
b.plot([p[0] for p in lam], [p[1] for p in lam], '-o', color=G, lw=2, ms=6, mec='white', mew=1, zorder=5, label='ours (4 λ)')
b.plot(l1[0], l1[1], 'o', color=OR, ms=6, mec='white', mew=1, zorder=5, label='ours, dense + codec (level 1.0)')
b.plot(o_mb, o_psnr, '*', color='#2b2b2b', ms=11, zorder=5, label=f'MeshSplatting FP32, our reproduction ({o_mb:.0f} MB)')
for name, mb, p, mk in MESH:
    b.plot(mb, p, mk, color='#2b2b2b', ms=7, mfc='#fcfcfb', mew=1.3, zorder=4, label=name)
for name, pts in GS:
    b.plot([p[0] for p in pts], [p[1] for p in pts], '-' if len(pts) > 1 else '', marker='.', color='#b5b5b5', lw=1, ms=5, zorder=2)
b.annotate('Gaussian-splat class (3DGS, HAC, HAC++, RDO-Gaussian):\nnot a mesh; +3 dB representation gap, shown for scale', (12, 27.0), fontsize=7.3, color='#8a8a8a', ha='left', va='top')
b.set_xscale('log'); b.set_xticks([3, 10, 30, 100, 300, 1000]); b.xaxis.set_major_formatter(mt.FixedFormatter(['3', '10', '30', '100', '300', '1000'])); b.xaxis.set_minor_formatter(mt.NullFormatter())
b.set_xlabel('asset size (MB, log)'); b.set_ylabel('test PSNR (dB)'); b.set_ylim(20.2, 28.3)
b.set_title('(b) engine-ready meshes: size vs quality, Mip-NeRF 360', loc='left', fontsize=9.5)
b.legend(frameon=False, fontsize=7.2, loc='lower right'); b.grid(color='#ececea', lw=0.5, which='both'); b.set_axisbelow(True)
for sp in ('top', 'right'): b.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(out + '.png', dpi=200, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight')
print('ours M360 means:', [(round(p[0], 1), round(o_mb / p[0]), round(p[1] - o_psnr, 2)) for p in lam], 'level1.0', (round(l1[0], 1), round(o_mb / l1[0]), round(l1[1] - o_psnr, 2)), 'FP32', round(o_mb), round(o_psnr, 2))
