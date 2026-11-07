"""Three scenes: train-at-budget vs post-hoc collapse at the same vertex count, same delivery format (K4096 R256, scene XYZ / q7).
usage: python plot_budget_vs_collapse_scenes.py budget-vs-collapse-3scenes-v1.csv out-prefix"""
import csv, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows = list(csv.DictReader(open(sys.argv[1]))); out = sys.argv[2]
SCENES = [('room', 28.263, 742), ('garden', 24.732, 740), ('bonsai', 27.291, 680)]
C = {'budget': '#2a78d6', 'collapse': '#eb6834'}
NAME = {'budget': 'grown under a vertex budget', 'collapse': 'dense training, collapsed afterwards'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#999', 'axes.linewidth': 0.6, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, (sc, orig, omb) in zip(axes, SCENES):
    for s in ('collapse', 'budget'):
        pts = sorted([(float(r['MB']), float(r['test_psnr']), r['label']) for r in rows if r['scene'] == sc and r['series'] == s])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], '-o', color=C[s], lw=1.6, ms=4.5, mec='#fcfcfb', mew=1, label=NAME[s], zorder=3)
        for x, y, l in pts:
            if l[0].isdigit():
                ax.annotate(l, (x, y), textcoords='offset points', xytext=(0, 6 if s == 'budget' else -11), ha='center', fontsize=7, color=C[s])
    ax.axhline(orig, color='#2b2b2b', lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.text(0.98, orig, f'original {orig} ({omb} MB)', transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=7, color='#2b2b2b')
    ax.set_title(sc, fontsize=10, loc='left'); ax.set_xlabel('delivered size (MB)')
    ax.grid(color='#e6e6e3', lw=0.5, zorder=0); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
axes[0].set_ylabel('test PSNR (dB)'); axes[0].legend(frameon=False, loc='lower right', fontsize=8)
fig.suptitle('Same vertex count, same codec: growing the mesh under a budget beats collapsing a dense mesh by 0.8-4.3 dB (labels = vertices)', fontsize=9, y=1.02)
fig.tight_layout(); fig.savefig(out + '.png', dpi=200, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight'); print('wrote', out)
