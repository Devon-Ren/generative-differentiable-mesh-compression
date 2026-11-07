"""PROVISIONAL (2026-11-01 evening, VERSION v2: indoor rerun rows replace room/kitchen/bonsai/counter): PSNR / SSIM / LPIPS vs delivered size for our four lambda points (dataset means) against
published points.  Our SSIM/LPIPS come from the delivery logs (training_report evaluator = upstream train.py, lpips VGG on [0,1]);
to be replaced by the one-pass unified evaluation.  Level 1.0 / FP32 rows are drawn only for T&T (Mip-360 originals lack SSIM/LPIPS for 5 scenes).
usage: python plot_rd_metrics_v1.py out-prefix"""
import csv, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.ticker as mt
import matplotlib.pyplot as plt
out = sys.argv[1]
M360 = ['room', 'counter', 'kitchen', 'bonsai', 'garden', 'bicycle', 'stump', 'flowers', 'treehill']; TNT = ['truck', 'train']
LAMS = [0.00062, 0.00185, 0.0062, 0.0185]
rows = []
INDOOR = ('room', 'kitchen', 'bonsai', 'counter')
for f in ('rd-prune3-7scenes-2026-10-24.csv', 'rd-prune3-new4-2026-10-24.csv', 'rd-prune3-hull2-2026-11-01.csv', 'rd-prune3-indoor-2026-11-01.csv'):
    for r in csv.DictReader(open(f)):
        if r.get('scene') and r.get('ssim') and not (r['scene'] in INDOOR and 'indoor' not in f): rows.append(dict(scene=r['scene'], lam=float(r['lam']), MB=float(r['MB']), psnr=float(r['test']), ssim=float(r['ssim']), lpips=float(r['lpips'])))
def mean(scenes, lam):
    rr = [r for r in rows if r['scene'] in scenes and abs(r['lam'] - lam) < 1e-6]
    assert len(rr) == len(scenes), (lam, [r['scene'] for r in rr]); n = len(rr)
    return {k: sum(r[k] for r in rr) / n for k in ('MB', 'psnr', 'ssim', 'lpips')}
ours = {'m360': [mean(M360, l) for l in LAMS], 'tnt': [mean(TNT, l) for l in LAMS]}
# T&T FP32 originals and level 1.0 (originals-new4-2026-10-24.txt): MB, psnr, ssim, lpips
TNT_FP32 = dict(MB=(432.4 + 484.7) / 2, psnr=(22.387 + 18.900) / 2, ssim=(0.8069 + 0.7119) / 2, lpips=(0.2256 + 0.3189) / 2)
TNT_L1 = dict(MB=(20.036 + 22.316) / 2, psnr=(22.308 + 18.738) / 2, ssim=(0.7909 + 0.6868) / 2, lpips=(0.2688 + 0.3671) / 2)
# published points (paper/baselines-v2.md: HAC++ Table II for the 3DGS class; MeshSplatting Tables 1-2 for the mesh class); (MB, PSNR, SSIM, LPIPS)
G = {'m360': {'3DGS (ref.)': (750.9, 27.46, .812, .222), 'HAC++ low/high': [(8.34, 27.60, .803, .253), (18.48, 27.82, .811, .231)],
              'HAC low/high': [(15.26, 27.53, .807, .238), (21.87, 27.77, .811, .230)], 'ContextGS low/high': [(12.68, 27.62, .808, .237), (18.41, 27.75, .811, .231)],
              'RDO-Gaussian': (23.46, 27.05, .802, .239), 'Compact3DGS': (48.80, 27.08, .798, .247), 'LightGaussian': (44.54, 27.00, .799, .249)},
     'tnt': {'3DGS (ref.)': (431.0, 23.69, .844, .178), 'HAC++ low/high': [(5.18, 24.22, .849, .190), (8.63, 24.32, .854, .178)],
             'HAC low/high': [(8.10, 24.04, .846, .187), (11.24, 24.40, .853, .177)], 'ContextGS low/high': [(7.05, 24.20, .852, .184), (11.80, 24.29, .855, .176)],
             'RDO-Gaussian': (12.03, 23.34, .835, .195), 'Compact3DGS': (39.43, 23.32, .831, .201), 'LightGaussian': (22.43, 22.83, .822, .242)}}
MESH = {'m360': {"MeshSplatting (paper, RGB-only PLY)": (100, 24.78, .728, .310), 'MiLo (2025)': (253, 24.09, .688, .323), 'RaDe-GS mesh (2024)': (1100, 23.56, .668, .361), 'GOF mesh (2024)': (1500, 20.78, .573, .465)},
        'tnt': {"MeshSplatting (paper, RGB-only PLY)": (100, 20.52, .745, .287), 'MiLo (2025)': (253, 21.46, .706, .348), 'RaDe-GS mesh (2024)': (1100, 20.51, .659, .344), 'GOF mesh (2024)': (1500, 21.69, .690, .326)}}
GC = {'3DGS (ref.)': '#555555', 'HAC++ low/high': '#6f4bd8', 'HAC low/high': '#b06fd0', 'ContextGS low/high': '#0b7285', 'RDO-Gaussian': '#2a78d6', 'Compact3DGS': '#c2185b', 'LightGaussian': '#9a6b00'}
MM = {"MeshSplatting (paper, RGB-only PLY)": 'o', 'MiLo (2025)': 's', 'RaDe-GS mesh (2024)': 'P', 'GOF mesh (2024)': 'X'}
GREEN, OR = '#1baf7a', '#eb6834'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': '#999', 'axes.linewidth': 0.6, 'figure.facecolor': 'white'})
fig, axs = plt.subplots(3, 2, figsize=(10.5, 10.2), sharex='col')
def logx(ax):
    ax.set_xscale('log'); t = [3, 10, 30, 100, 300, 1000]; ax.set_xticks(t); ax.xaxis.set_major_formatter(mt.FixedFormatter([str(v) for v in t])); ax.xaxis.set_minor_formatter(mt.NullFormatter())
    ax.set_xlim(2.5, 2000); ax.grid(color='#ececea', lw=0.5, which='both'); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
def better(ax, up=True):
    y0, y1 = (0.885, 0.985) if up else (0.115, 0.015)
    ax.annotate('', xy=(0.02, y1), xytext=(0.15, y0), xycoords='axes fraction', arrowprops=dict(arrowstyle='-|>', color='#2b2b2b', lw=1.3))
    ax.text(0.165, (y0 + y1) / 2, 'better', transform=ax.transAxes, fontsize=7.5, va='center', color='#2b2b2b')
for j, (ds, title) in enumerate((('m360', 'Mip-NeRF 360 (9-scene mean)'), ('tnt', 'Tanks & Temples (truck, train)'))):
    for i, (k, lab) in enumerate((('psnr', 'test PSNR (dB)'), ('ssim', 'SSIM'), ('lpips', 'LPIPS (VGG)'))):
        ax = axs[i, j]; o = ours[ds]
        ax.plot([p['MB'] for p in o], [p[k] for p in o], '-o', color=GREEN, lw=2, ms=6, mec='white', mew=1, zorder=5, label='ours, 4 λ (provisional metrics)')
        if ds == 'tnt':
            ax.plot(TNT_L1['MB'], TNT_L1[k], 'o', color=OR, ms=6, mec='white', mew=1, zorder=5, label='ours, dense + codec (level 1.0)')
            ax.plot(TNT_FP32['MB'], TNT_FP32[k], '*', color='#2b2b2b', ms=12, zorder=5, label='MeshSplatting FP32, our reproduction')
        for name, v in MESH[ds].items():
            ax.plot(v[0], v[1 + i], MM[name], color='#2b2b2b', mfc='#fcfcfb', ms=7, mew=1.2, zorder=4, label=name)
        for name, v in G[ds].items():
            if isinstance(v, list): ax.plot([p[0] for p in v], [p[1 + i] for p in v], '-D', color=GC[name], lw=1.1, ms=4.5, mec='white', mew=0.7, zorder=3, label=name)
            else: ax.plot(v[0], v[1 + i], 'D', color=GC[name], ms=4.5, mec='white', mew=0.7, zorder=3, label=name)
        logx(ax); ax.set_ylabel(lab); better(ax, up=(k != 'lpips'))
        if i == 0: ax.set_title(title, loc='left', fontsize=9.5)
        if i == 2: ax.set_xlabel('delivered size (MB, log)')
h, l = axs[0, 1].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=7.6, bbox_to_anchor=(0.5, -0.005))
fig.suptitle('PROVISIONAL: our SSIM/LPIPS from the delivery logs (upstream train.py evaluator); final numbers come from the one-pass unified evaluation', fontsize=8.5, color='#8a3b12', y=0.995)
fig.tight_layout(rect=(0, 0.06, 1, 0.98)); fig.savefig(out + '.png', dpi=180, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight')
for ds in ('m360', 'tnt'):
    for l, p in zip(LAMS, ours[ds]): print(ds, l, ' '.join(f"{k} {p[k]:.3f}" for k in ('MB', 'psnr', 'ssim', 'lpips')))
print('wrote', out)
