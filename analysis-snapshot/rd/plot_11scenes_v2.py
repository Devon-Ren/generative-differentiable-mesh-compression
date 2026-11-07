"""11 scenes (Mip-360 9 + T&T 2), same codec (K4096 R256, scene XYZ, DC q7, stage2-v2): post-hoc collapse curve or
level-1.0 point, v2 lambda-only, v3 growth + prune at FOUR lambdas (6.2e-4, 1.85e-3, 6.2e-3, 1.85e-2; 2026-11-01). usage: python plot_11scenes_v2.py rd-11scenes-v2.csv out-prefix"""
import csv, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows = list(csv.DictReader(open(sys.argv[1]))); out = sys.argv[2]
SCENES = [('room', 28.263, 742), ('counter', 25.944, 575), ('kitchen', 26.454, 456), ('bonsai', 27.291, 662),
          ('garden', 24.732, 740), ('bicycle', 23.033, 680), ('stump', 24.928, 698), ('flowers', 19.451, 668),
          ('treehill', 20.387, 692), ('truck', 22.387, 432), ('train', 18.900, 485)]
C = {'collapse': '#eb6834', 'lambda-v2': '#2a78d6', 'v3': '#1baf7a'}
NAME = {'collapse': 'dense training + post-hoc collapse (level 1.0 only for the 4 new scenes)',
        'lambda-v2': 'rate-controlled growth, λ only (v2)', 'v3': 'rate-controlled growth + λ-prune (v3, λ 6.2e-4 / 1.85e-3 / 6.2e-3 / 1.85e-2)'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#999', 'axes.linewidth': 0.6, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
fig, axes = plt.subplots(3, 4, figsize=(14.5, 9.6)); axes = axes.ravel()
for ax, (sc, orig, omb) in zip(axes, SCENES):
    for s in ('collapse', 'lambda-v2', 'v3'):
        pts = sorted([(float(r['MB']), float(r['test'])) for r in rows if r['scene'] == sc and r['series'] == s])
        if not pts: continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], '-o', color=C[s], lw=1.6, ms=4.8 if s == 'v3' else 4, mec='#fcfcfb', mew=1,
                label=NAME[s], zorder=4 if s == 'v3' else 3)
        if s == 'v3':
            for k, (x, y) in enumerate(pts):
                ax.annotate(f'{omb / x:.0f}×', (x, y), textcoords='offset points', xytext=(0, -13 if k % 2 == 0 else 7), ha='center', fontsize=7.5, color='#0f6e4c')
    ax.axhline(orig, color='#2b2b2b', lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.text(0.5, orig, f'original {orig:.2f} ({omb} MB)', transform=ax.get_yaxis_transform(), ha='center', va='bottom', fontsize=7, color='#2b2b2b')
    ax.set_title(sc + ('  (T&T)' if sc in ('truck', 'train') else ''), fontsize=10, loc='left')
    ys = [float(r['test']) for r in rows if r['scene'] == sc] + [orig]
    lo, hi = min(ys), max(ys); span = max(hi - lo, 1.6); mid = (lo + hi) / 2
    ax.set_ylim(mid - span / 2 - 0.12 * span, mid + span / 2 + 0.1 * span)
    ax.grid(color='#e6e6e3', lw=0.5, zorder=0); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
for i in (0, 4, 8): axes[i].set_ylabel('test PSNR (dB)')
for i in (8, 9, 10): axes[i].set_xlabel('delivered size (MB)')
h, l = axes[0].get_legend_handles_labels()
axes[11].axis('off')
axes[11].legend(h, l, frameon=False, loc='upper left', fontsize=8.2, bbox_to_anchor=(0, 1.0))
axes[11].text(0.0, 0.46, 'v3 means over 11 scenes (MB, ratio, Δ test PSNR vs FP32)\nλ 6.2e-4:  28.2 MB,  22×, −0.11 dB  (Mip-360: −0.03)\nλ 1.85e-3: 20.7 MB,  32×, −0.22 dB\nλ 6.2e-3:    9.4 MB,  73×, −0.59 dB\nλ 1.85e-2:   4.5 MB, 149×, −1.25 dB\nlevel 1.0 (dense + codec): 26.6 MB, 23×, −0.25 dB',
              transform=axes[11].transAxes, fontsize=8.6, va='top', color='#2b2b2b')
fig.suptitle('Eleven scenes, one codec, four λ values: original FP32 432–742 MB → 3–36 MB', fontsize=10.5, y=1.0)
fig.tight_layout(); fig.savefig(out + '.png', dpi=200, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight'); print('wrote', out)
