"""Seven Mip-360 scenes: rate-controlled growth (lambda) vs vertex cap vs post-hoc collapse, same delivery format.
(K4096 R256, scene XYZ precision / q7, stage2-v2 const/ctx). Labels = vertex count.
usage: python plot_budget_vs_collapse_7scenes.py budget-vs-collapse-7scenes-v1.csv out-prefix"""
import csv, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows = list(csv.DictReader(open(sys.argv[1]))); out = sys.argv[2]
# scene, original test PSNR, original FP32 MB
SCENES = [('room', 28.263, 742), ('garden', 24.732, 740), ('bonsai', 27.291, 680), ('counter', 25.944, 575),
          ('kitchen', 26.454, 456), ('bicycle', 23.033, 680), ('stump', 24.928, 698)]
C = {'budget': '#2a78d6', 'collapse': '#eb6834', 'lambda': '#1baf7a'}
NAME = {'budget': 'vertex cap (--max_points)', 'collapse': 'dense training, collapsed afterwards', 'lambda': 'rate-controlled growth (λ per Mpx)'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#999', 'axes.linewidth': 0.6, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.6))
axes = axes.ravel()
for ax, (sc, orig, omb) in zip(axes, SCENES):
    for s in ('collapse', 'budget', 'lambda'):
        pts = sorted([(float(r['MB']), float(r['test_psnr']), r['label']) for r in rows if r['scene'] == sc and r['series'] == s])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], '-o', color=C[s], lw=1.6, ms=4.5, mec='#fcfcfb', mew=1, label=NAME[s], zorder=3)
        for x, y, l in pts:
            if l.startswith('λ'):
                ax.annotate(l, (x, y), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=7, color=C[s])
    ax.axhline(orig, color='#2b2b2b', lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.text(0.98, orig, f'original {orig} ({omb} MB)', transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=7, color='#2b2b2b')
    ax.set_title(sc, fontsize=10, loc='left'); ax.set_xlabel('delivered size (MB)')
    ax.margins(y=0.14); ax.grid(color='#e6e6e3', lw=0.5, zorder=0); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
for i in (0, 4): axes[i].set_ylabel('test PSNR (dB)')
h, l = axes[0].get_legend_handles_labels()
axes[7].axis('off'); axes[7].legend(h, l, frameon=False, loc='center', fontsize=9)
fig.suptitle('Seven scenes, same codec: rate-controlled growth (one λ for all scenes, no per-scene cap) sits on or above the cap curve at equal bytes on every comparable point (+0.0 to +0.57 dB)', fontsize=9.5, y=1.0)
fig.tight_layout(); fig.savefig(out + '.png', dpi=200, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight'); print('wrote', out)
