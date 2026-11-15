"""make_cap_fig_single_v1.py: single-column version of the cap figure (file size against vertex cap, all 21 scenes), drawn from the
coordinates in paper/figures/cap-onefig-block-v3.tex (output of paper/yao-note/make_cap_onefig_v3.py). Output fig_cap_instability.{pdf,png}."""
import re
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#dddcd6', '#fcfcfb'
COL = {'cA': '#2a78d6', 'cB': '#eb6834', 'cC': '#1baf7a'}
plt.rcParams.update({'font.size': 7.5, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
fig, ax = plt.subplots(figsize=(3.35, 2.45))
for x in (0.6, 1.08, 3.0, 4.0): ax.axvline(x, color=GRID, lw=3.5, zorder=1)
for l in open('paper/figures/cap-onefig-block-v3.tex'):
    m = re.match(r'\\addplot\[ds, (c[ABC]),(.*?)\] coordinates \{(.*?)\}', l)
    if not m: continue
    pts = [tuple(map(float, p.split(','))) for p in re.findall(r'\(([\d.]+,[\d.]+)\)', m[3])]
    dashed = 'dash pattern' in m[2]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COL[m[1]], lw=0.9, ls=(0, (3, 2)) if dashed else '-', marker='o', ms=1.8, mec='none', zorder=3, alpha=0.9)
for x, t, ha in ((0.66, '5.8–7.5', 'right'), (1.0, '8.3–11.6', 'left'), (3.0, '19.9–31.6', 'center'), (4.0, '20.0–39.8', 'center')):
    ax.text(x, 43.2, t, ha=ha, va='bottom', fontsize=6.0, color=INK)
ax.text(-0.12, 43.2, 'MB:', ha='right', va='bottom', fontsize=6.3, color=MUTED)
ax.set_xlim(0, 4.35); ax.set_ylim(0, 43); ax.set_xticks([0, 0.6, 1.08, 2, 3, 4]); ax.set_xticklabels(['0', '0.6', '1.08', '2', '3', '4']); ax.set_yticks([0, 10, 20, 30, 40])
ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=2)
for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
ax.set_xlabel('vertex cap (millions)'); ax.set_ylabel('file size (MB)')
hs = [Line2D([], [], color=COL['cA'], lw=1.4, label='Mip-NeRF 360, T&T (11)'), Line2D([], [], color=COL['cB'], lw=1.4, label='Deep Blending (2)'), Line2D([], [], color=COL['cC'], lw=1.4, label='NeRF-Synthetic (8)')]
ax.legend(handles=hs, loc='upper left', bbox_to_anchor=(0.0, 0.93), frameon=False, fontsize=6.3, handlelength=1.6, labelspacing=0.25)
fig.tight_layout(pad=0.4); fig.savefig('paper/figures/fig_cap_instability.pdf'); fig.savefig('paper/figures/fig_cap_instability.png', dpi=200); print('ok')
