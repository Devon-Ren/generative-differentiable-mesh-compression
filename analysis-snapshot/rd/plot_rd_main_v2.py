"""Main-paper RD figure, v2 (2026-11-01, VERSION v2b): three panels, x = absolute delivered size (MB, log) everywhere, a "better" arrow in every panel.
(a) same asset (Room FP32 MeshSplatting): generic codecs vs ours.   (b) engine-ready meshes, Mip-NeRF 360 average: size vs PSNR, mesh methods as points.
(c) Gaussian-class compressors vs ours: ΔPSNR vs the full-precision model each method trains and compresses (stated per method), against size.
usage: python plot_rd_main_v2.py rd-11scenes-v2.csv baselines-rd-v1.csv out-prefix"""
import csv, sys, re
import matplotlib; matplotlib.use('Agg'); import matplotlib.ticker as mt
import matplotlib.pyplot as plt
rows = list(csv.DictReader(open(sys.argv[1]))); base = list(csv.DictReader(open(sys.argv[2]))); out = sys.argv[3]
ORIG = {'room': (28.263, 741.7), 'counter': (25.944, 575.0), 'kitchen': (26.454, 456.0), 'bonsai': (27.291, 662.2), 'garden': (24.732, 739.8),
        'bicycle': (23.033, 680.0), 'stump': (24.928, 698.0), 'flowers': (19.451, 667.7), 'treehill': (20.387, 691.8)}
M360 = list(ORIG); o_psnr = sum(v[0] for v in ORIG.values()) / 9; o_mb = sum(v[1] for v in ORIG.values()) / 9
LAMS = ('λ 0.00062', 'λ 0.00185', 'λ 0.0062', 'λ 0.0185'); LT = ['6.2e-4', '1.85e-3', '6.2e-3', '1.85e-2']
def mean9(series, label):
    r = [x for x in rows if x['scene'] in M360 and x['series'] == series and x['label'].startswith(label)]
    assert len(r) == 9, (series, label, len(r)); return sum(float(x['MB']) for x in r) / 9, sum(float(x['test']) for x in r) / 9
def scene(sc, series, label):
    r = [x for x in rows if x['scene'] == sc and x['series'] == series and x['label'].startswith(label)][0]; return float(r['MB']), float(r['test'])
lam = [mean9('v3', l) for l in LAMS]; l1 = mean9('collapse', 'v1.0')
def bpts(method): return sorted([(float(r['MB']), float(r['psnr'])) for r in base if r['dataset'] == 'm360' and r['method'] == method])
G, OR, REFC, GEN = '#1baf7a', '#eb6834', '#2b2b2b', '#8a5cd6'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#999', 'axes.linewidth': 0.6, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})
fig, (a, b, c) = plt.subplots(1, 3, figsize=(17.4, 4.9))
def logx(ax, ticks):
    ax.set_xscale('log'); ax.set_xticks(ticks); ax.xaxis.set_major_formatter(mt.FixedFormatter([str(t) for t in ticks])); ax.xaxis.set_minor_formatter(mt.NullFormatter())
    ax.grid(color='#ececea', lw=0.5, which='both'); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
def better(ax):
    ax.annotate('', xy=(0.02, 0.985), xytext=(0.15, 0.885), xycoords='axes fraction', arrowprops=dict(arrowstyle='-|>', color='#2b2b2b', lw=1.4))
    ax.text(0.165, 0.95, 'better (smaller, higher PSNR)', transform=ax.transAxes, fontsize=7.6, va='center', ha='left', color='#2b2b2b')

# ---------- (a) same asset, Room ----------
gen = {}
for line in open('generic-codecs-2026-10-24.txt'):
    parts = [s.strip() for s in line.split('|')]; sc, name = parts[0].split()
    mb = re.match(r'([\d.]+) MB', parts[1]); ps = re.search(r'test ([\d.]+) /', line)
    if sc == 'room' and mb and ps: gen[name] = (float(mb.group(1)), float(ps.group(1)))
GL = [('quant_ply_zstd19', 'quantised PLY + zstd-19', 's'), ('draco_full_sh8', 'Draco (SH as 8-bit attributes)', 'D'), ('draco_rgb8', 'Draco, RGB only (SH dropped)', 'v'), ('texture_8192', 'texture route: xatlas + Draco + 8k JPEG', '^')]
for key, name, mk in GL:
    a.plot(gen[key][0], gen[key][1], mk, color=GEN, ms=7, mec='white', mew=0.8, zorder=4, label=name)
rl = [scene('room', 'v3', l) for l in LAMS]; rl1 = scene('room', 'collapse', 'v1.0')
a.plot([p[0] for p in rl], [p[1] for p in rl], '-o', color=G, lw=2, ms=6, mec='white', mew=1, zorder=5, label='ours: rate-controlled growth + λ-prune (4 λ)')
for (x, y), t in zip(rl, LT): a.annotate(f'{741.7 / x:.0f}×', (x, y), textcoords='offset points', xytext=(0, -12), ha='center', fontsize=7.3, color='#0f6e4c')
a.plot(rl1[0], rl1[1], 'o', color=OR, ms=6, mec='white', mew=1, zorder=5, label='ours: same codec on the dense asset (level 1.0)')
a.annotate(f'{741.7 / rl1[0]:.0f}×', rl1, textcoords='offset points', xytext=(8, -2), ha='left', va='center', fontsize=7.3, color=OR)
a.plot(741.7, 28.263, '*', color=REFC, ms=12, zorder=6, label='FP32 MeshSplatting asset (742 MB)')
logx(a, [3, 10, 30, 100, 300, 1000]); a.set_xlabel('delivered size (MB, log)'); a.set_ylabel('test PSNR (dB)'); a.set_ylim(24.8, 29.3)
a.set_title('(a) one asset, every codec: Room (Mip-NeRF 360)', loc='left', fontsize=9.5); a.legend(frameon=False, fontsize=7.2, loc='center right', bbox_to_anchor=(1.0, 0.5)); better(a)

# ---------- (b) engine-ready meshes, Mip-360 average ----------
MESH = [("MeshSplatting, authors' RGB-only PLY (SH dropped)", 100, 24.78, 'o'), ('MiLo (SIGGRAPH Asia 2025)', 253, 24.09, 's'), ('BakedSDF (SIGGRAPH 2023)', 434, 24.51, '^'),
        ('Binary Opacity Grids (SIGGRAPH 2024)', 913, 25.62, 'v'), ('RaDe-GS mesh', 1100, 23.56, 'P'), ('GOF mesh', 1500, 20.78, 'X')]
b.plot([p[0] for p in lam], [p[1] for p in lam], '-o', color=G, lw=2, ms=6, mec='white', mew=1, zorder=5, label='ours (4 λ, 9-scene mean)')
for (x, y), t in zip(lam, LT): b.annotate(f'{o_mb / x:.0f}×', (x, y), textcoords='offset points', xytext=(0, -12), ha='center', fontsize=7.3, color='#0f6e4c')
b.plot(l1[0], l1[1], 'o', color=OR, ms=6, mec='white', mew=1, zorder=5, label='ours, dense asset + codec (level 1.0)')
b.plot(o_mb, o_psnr, '*', color=REFC, ms=12, zorder=5, label=f'MeshSplatting FP32, our reproduction ({o_mb:.0f} MB)')
for name, mb, p, mk in MESH: b.plot(mb, p, mk, color=REFC, ms=7, mfc='#fcfcfb', mew=1.3, zorder=4, label=name)
logx(b, [3, 10, 30, 100, 300, 1000]); b.set_xlabel('asset size (MB, log)'); b.set_ylabel('test PSNR (dB)'); b.set_ylim(20.3, 26.8)
b.set_title('(b) engine-ready meshes, Mip-NeRF 360 (9-scene mean)', loc='left', fontsize=9.5); b.legend(frameon=False, fontsize=7.2, loc='lower left'); better(b)

# ---------- (c) Gaussian-class compressors, Δ vs the model each method compresses ----------
REF = {'HAC': (27.50, 'Scaffold-GS'), 'HAC++': (27.50, 'Scaffold-GS'), 'ContextGS': (27.50, 'Scaffold-GS'), 'RDO-Gaussian': (27.49, '3DGS'), 'LightGaussian': (27.46, '3DGS'), 'Compact3DGS': (27.46, '3DGS')}
cols = {'HAC++': '#6f4bd8', 'HAC': '#b06fd0', 'ContextGS': '#0b7285', 'RDO-Gaussian': '#2a78d6', 'LightGaussian': '#9a6b00', 'Compact3DGS': '#c2185b'}
c.axhline(0, color=REFC, lw=0.8, ls=(0, (4, 3)))
c.plot([p[0] for p in lam], [p[1] - o_psnr for p in lam], '-o', color=G, lw=2, ms=6, mec='white', mew=1, zorder=5, label='ours vs FP32 MeshSplatting (657 MB)')
for (x, y), t in zip(lam, LT): c.annotate(f'{o_mb / x:.0f}×', (x, y - o_psnr), textcoords='offset points', xytext=(0, -12), ha='center', fontsize=7.3, color='#0f6e4c')
c.plot(l1[0], l1[1] - o_psnr, 'o', color=OR, ms=6, mec='white', mew=1, zorder=5, label='ours, dense asset + codec (level 1.0)')
for m in ('HAC++', 'HAC', 'ContextGS', 'RDO-Gaussian', 'LightGaussian', 'Compact3DGS'):
    pts = bpts(m); rp, rn = REF[m]
    c.plot([p[0] for p in pts], [p[1] - rp for p in pts], ('-D' if len(pts) > 1 else 's'), color=cols[m], lw=1.1, ms=4.5, mec='white', mew=0.7, zorder=3,
           label=f'{m} vs {rn} ({"254" if rn == "Scaffold-GS" else "751"} MB)')
logx(c, [1, 3, 10, 30, 100]); c.set_xlabel('delivered size (MB, log)'); c.set_ylabel('Δ test PSNR vs the full-precision model the method compresses (dB)')
c.set_ylim(-3.4, 0.9); c.set_title('(c) quality kept by Gaussian-class compressors vs ours, Mip-NeRF 360', loc='left', fontsize=9.5); c.legend(frameon=False, fontsize=7.0, loc='lower right'); better(c)
fig.tight_layout(); fig.savefig(out + '.png', dpi=200, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight'); print('wrote', out)
