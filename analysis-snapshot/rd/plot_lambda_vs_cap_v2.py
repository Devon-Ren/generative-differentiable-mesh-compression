"""Room v2 (10/19): lambda sweep (v2 units, per Mpx) vs cap vs post-hoc collapse + matched-count importance control + phase-B lambda-prune.
usage: python plot_lambda_vs_cap_v2.py room-lambda-vs-cap-v2.csv out-stem"""
import csv, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows = list(csv.DictReader(open(sys.argv[1]))); out = sys.argv[2]
ORIG, OMB = 28.263, 742
C = {'lambda': '#1baf7a', 'cap': '#2a78d6', 'collapse': '#eb6834', 'prune': '#0f6e4c', 'control': '#8a5cd6'}
NAME = {'lambda': 'rate-controlled growth (λ sweep)', 'cap': 'vertex cap (--max_points)', 'collapse': 'dense, collapsed afterwards',
        'prune': 'λ growth + phase-B λ-prune', 'control': 'importance sampling on the λ=6.2e-3 schedule (control)'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#999', 'axes.linewidth': 0.6, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
for ax, xkey, xlabel in ((axes[0], 'MB', 'delivered size (MB)'), (axes[1], 'vertices', 'vertices (millions)')):
    for s in ('collapse', 'cap', 'lambda', 'prune', 'control'):
        pts = sorted([(float(r[xkey]) / (1e6 if xkey == 'vertices' else 1), float(r['test_psnr']), r['label']) for r in rows if r['series'] == s])
        style = dict(lw=1.6, ms=4.5, mec='#fcfcfb', mew=1, color=C[s], label=NAME[s])
        if s == 'prune': ax.plot([p[0] for p in pts], [p[1] for p in pts], 's--', zorder=5, **style)
        elif s == 'control': ax.plot([p[0] for p in pts], [p[1] for p in pts], 'D', zorder=6, **{k: v for k, v in style.items() if k != 'lw'})
        else: ax.plot([p[0] for p in pts], [p[1] for p in pts], '-o', zorder=4 if s == 'lambda' else 3, **style)
        if s == 'lambda':
            for x, y, l in pts:
                if l in ('0', '1.9e-4'):
                    continue
                if l == '6.2e-4':
                    ax.annotate('λ ≤ 6.2e-4: limiter-bound, 2.9M', (x, y), textcoords='offset points', xytext=(-4, 9), fontsize=7, color=C[s], ha='right')
                else:
                    ax.annotate(f'λ={l}', (x, y), textcoords='offset points', xytext=(7, -9 if l == '6.2e-3' else 5), fontsize=7, color=C[s])
        if s == 'control' and xkey == 'MB':
            ax.annotate('same 18 split counts,\nimportance-sampled: −0.12 dB vs λ,\n+0.3 vs cap at equal MB', pts[0][:2], textcoords='offset points', xytext=(8, -30), fontsize=7, color=C[s])
        if s == 'prune' and xkey == 'MB':
            ax.annotate('+ phase-B λ-prune: −8 % MB, same PSNR', pts[-1][:2], textcoords='offset points', xytext=(-6, 9), fontsize=7, color=C[s], ha='right')
    ax.axhline(ORIG, color='#2b2b2b', lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.text(0.02, ORIG, f'original {ORIG} ({OMB} MB)', transform=ax.get_yaxis_transform(), ha='left', va='bottom', fontsize=7, color='#2b2b2b')
    ax.set_xlabel(xlabel); ax.margins(y=0.12); ax.grid(color='#e6e6e3', lw=0.5, zorder=0); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
axes[0].set_ylabel('test PSNR (dB)'); axes[0].legend(frameon=False, loc='lower right', fontsize=7.5)
axes[0].set_xscale('log'); axes[0].set_xticks([3, 5, 10, 20, 30]); axes[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()); axes[0].get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.suptitle('Room: vertex count as an output of λ (per-Mpx units). Below 0.5M vertices λ beats the cap; the schedule gives ~0.3 dB, error ranking another 0.12; phase-B λ-prune returns 8 % of the bytes for free', fontsize=8.5, y=1.0)
fig.tight_layout(); fig.savefig(out + '.png', dpi=200, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight'); print('wrote', out)
