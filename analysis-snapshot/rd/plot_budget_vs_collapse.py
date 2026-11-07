"""Room: train-at-budget vs post-hoc collapse, same delivery format (K4096 R256 q14/q7, stage2-v2).
usage: python plot_budget_vs_collapse.py budget-vs-collapse-room-v1.csv out-prefix"""
import csv, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows = list(csv.DictReader(open(sys.argv[1]))); out = sys.argv[2]
ORIG_PSNR, ORIG_MB = 28.263, 742
C = {'budget': '#2a78d6', 'collapse': '#eb6834'}
NAME = {'budget': 'train at vertex budget (--max_points)', 'collapse': 'dense training, post-hoc collapse'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#999', 'axes.linewidth': 0.6, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
for ax, xkey, xlabel in ((axes[0], 'MB', 'delivered size (MB)'), (axes[1], 'vertices', 'vertices (M)')):
    for s in ('collapse', 'budget'):
        pts = sorted([(float(r[xkey]) / (1e6 if xkey == 'vertices' else 1), float(r['test_psnr']), r['label']) for r in rows if r['series'] == s])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], '-o', color=C[s], lw=1.6, ms=4.5, mec='#fcfcfb', mew=1, label=NAME[s], zorder=3)
        for x, y, l in pts:
            if l in ('B1.6M', 'B1.0M', 'B700k', 'B500k', 'ctrl-1.35M', 'ctrl-0.97M', 'v0.5', 'v0.8'):
                ax.annotate(l.replace('ctrl-', '').replace('B', ''), (x, y), textcoords='offset points', xytext=(0, 6 if s == 'budget' else -11), ha='center', fontsize=7, color=C[s])
    ax.axhline(ORIG_PSNR, color='#2b2b2b', lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.text(ax.get_xlim()[1], ORIG_PSNR + 0.03, f'dense original {ORIG_PSNR} (742 MB)', ha='right', va='bottom', fontsize=7, color='#2b2b2b')
    ax.set_xlabel(xlabel); ax.grid(color='#e6e6e3', lw=0.5, zorder=0); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
axes[0].set_ylabel('test PSNR (dB)'); axes[0].legend(frameon=False, loc='lower right', fontsize=8)
if True:
    ax = axes[0]; sec = ax.secondary_xaxis('top', functions=(lambda mb: ORIG_MB / mb, lambda r: ORIG_MB / r))
    sec.set_xlabel('compression ratio vs 742 MB', fontsize=8); sec.tick_params(labelsize=7); sec.set_xticks([25, 50, 100, 150])
fig.suptitle('Room: the same vertex count is worth 1.3-2 dB more when it is grown under a budget than when it is collapsed afterwards', fontsize=9, y=1.02)
fig.tight_layout()
fig.savefig(out + '.png', dpi=200, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight'); print('wrote', out)
