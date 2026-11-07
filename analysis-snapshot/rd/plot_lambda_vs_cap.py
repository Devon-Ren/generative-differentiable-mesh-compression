"""Room: rate-controlled growth (lambda sweep) vs vertex cap vs post-hoc collapse, same codec (K4096 R256 q14/q7).
usage: python plot_lambda_vs_cap.py room-lambda-vs-cap-v1.csv out-stem"""
import csv, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows = list(csv.DictReader(open(sys.argv[1]))); out = sys.argv[2]
ORIG, OMB = 28.263, 742
C = {'lambda': '#1baf7a', 'cap': '#2a78d6', 'collapse': '#eb6834'}
NAME = {'lambda': 'rate-controlled growth (lambda sweep)', 'cap': 'vertex cap (--max_points)', 'collapse': 'dense, collapsed afterwards'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#999', 'axes.linewidth': 0.6, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, xkey, xlabel in ((axes[0], 'MB', 'delivered size (MB)'), (axes[1], 'vertices', 'vertices (millions)')):
    for s in ('collapse', 'cap', 'lambda'):
        pts = sorted([(float(r[xkey]) / (1e6 if xkey == 'vertices' else 1), float(r['test_psnr']), r['label']) for r in rows if r['series'] == s])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], '-o', color=C[s], lw=1.6, ms=4.5, mec='#fcfcfb', mew=1, label=NAME[s], zorder=3 if s != 'lambda' else 4)
        if s == 'lambda':
            for x, y, l in pts:
                ax.annotate(f'λ={l}', (x, y), textcoords='offset points', xytext=(6, -10 if float(l) < 2e-3 else 5), fontsize=7, color=C[s])
    ax.axhline(ORIG, color='#2b2b2b', lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.text(0.98, ORIG, f'original {ORIG} ({OMB} MB)', transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=7, color='#2b2b2b')
    ax.set_xlabel(xlabel); ax.margins(y=0.12); ax.grid(color='#e6e6e3', lw=0.5, zorder=0); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
axes[0].set_ylabel('test PSNR (dB)'); axes[0].legend(frameon=False, loc='lower right', fontsize=8)
axes[0].set_xscale('log'); axes[0].set_xticks([3, 5, 10, 20, 30]); axes[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()); axes[0].get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.suptitle('Room: the vertex count as an output of λ. Equal to the cap at 1.3M, above it below 0.5M (λ 1e-2: +0.25 dB; λ 3e-2: 3.0 MB / 246x at -1.75 dB)', fontsize=9, y=1.0)
fig.tight_layout(); fig.savefig(out + '.png', dpi=200, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight'); print('wrote', out)
