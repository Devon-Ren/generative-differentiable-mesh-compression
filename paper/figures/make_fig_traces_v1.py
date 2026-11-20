"""make_fig_traces_v1.py (2026-11-20): fig_controller_traces_v2 -- measured file size / target at every round of Phase 2, for the 65 ByteMesh
runs on the 13 captured scenes (traces-b50-2026-11-20.txt), with the size at 30k as the last point.  Single column.  Run from the repository root."""
import re
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker as mt
from matplotlib.lines import Line2D
CAP = ['bicycle','flowers','garden','stump','treehill','room','counter','kitchen','bonsai','truck','train','drjohnson','playroom']
INK, MUTED, GRID, BLUE, ORANGE = '#1f2430', '#5b6270', '#e3e6ea', '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 7.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(3.4, 2.35)); n = [0, 0]
ax.axhspan(0.9, 1.1, color='#eef1f4', zorder=0); ax.axhline(1.0, color=INK, lw=0.7, zorder=1)
for l in open('release-batch50-traces/results/traces-b50-2026-11-20.txt'):
    m = re.match(r'TRACE (\w+) ([\d.]+) \| (.*?) \| final (\S+) \|', l)
    if m[1] not in CAP: continue
    T = float(m[2]); pr = [t.split(':') for t in m[3].split()]; x = [int(p[0]) / 1000 for p in pr] + [30.0]; y = [float(p[1]) / T for p in pr] + [float(m[4]) / T]
    above = y[0] >= 1.0; n[0 if above else 1] += 1
    ax.plot(x, y, color=BLUE if above else ORANGE, lw=0.55, alpha=0.55, zorder=3 if above else 2)
ax.set_xlim(12.3, 30.2); ax.set_ylim(0.78, 1.25); ax.set_xlabel('iteration (thousands)'); ax.set_ylabel('measured file size / target')
ax.xaxis.set_major_locator(mt.FixedLocator([12.5, 15, 20, 25, 30])); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: '%g' % v)); ax.yaxis.set_major_locator(mt.MultipleLocator(0.1))
ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=2)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.legend(handles=[Line2D([], [], color=BLUE, lw=1.4, label='above the target at 12.5k (%d runs)' % n[0]), Line2D([], [], color=ORANGE, lw=1.4, label='below the target at 12.5k (%d runs)' % n[1])],
          loc='upper right', frameon=False, fontsize=6.5, handlelength=1.6)
fig.tight_layout(pad=0.4); fig.savefig('paper/figures/fig_controller_traces_v2.pdf'); fig.savefig('paper/figures/fig_controller_traces_v2.png', dpi=220); print('ok', n)
