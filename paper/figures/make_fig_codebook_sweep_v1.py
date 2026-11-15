"""make_fig_codebook_sweep_v1.py (2026-11-15): appendix figure, codebook-size sweep.
Source analysis-snapshot/rd/lambda-k-fan-v2.csv (Nov 5-6): 11 Mip-NeRF 360 / T&T scenes x 4 mesh sizes (the four rate settings of the earlier
fixed-price training) x 6 codec settings (K + K_r, DC bits).  Each plotted point is the mean over the 11 scenes of one (mesh size, codec
setting) pair: 24 points.  Drawn: all points, the upper convex hull over them (in log file size), and the line of the default K 4096 + K_r 256.
Prints the gap between the default line and the hull.  Output paper/figures/fig_codebook_sweep_v1.{pdf,png}."""
import csv, math, collections, statistics as st
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker as mt
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#dddcd6', '#fcfcfb'
plt.rcParams.update({'font.size': 7.5, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
rows = list(csv.DictReader(open('analysis-snapshot/rd/lambda-k-fan-v2.csv'))); scenes = sorted({r['scene'] for r in rows}); assert len(scenes) == 11
G = collections.defaultdict(list)
for r in rows: G[(r['lam'], r['lever'])].append((float(r['MB']), float(r['psnr'])))
assert all(len(v) == 11 for v in G.values()), {k: len(v) for k, v in G.items()}
P = {k: (st.mean(x[0] for x in v), st.mean(x[1] for x in v)) for k, v in G.items()}
LEV = [('K4096 R256 q7', r'4096 + 256 (default)', '#2a78d6', 'o'), ('K4096 R1 q7', r'4096 + 0', '#eb6834', 's'), ('K1024 R256 q7', r'1024 + 256', '#1baf7a', 'D'),
       ('K4096 R1024 q7', r'4096 + 1024', '#eda100', '^'), ('K16384 R1024 q7', r'16384 + 1024', '#e87ba4', 'v'), ('K4096 R256 q8', r'4096 + 256, 8-bit DC', '#7a5cc9', 'P')]
lams = sorted({k[0] for k in P}, key=float, reverse=True)
def hull(pts):
    Q = sorted((math.log(x), y) for x, y in pts); H = []
    for q in Q:
        while len(H) >= 2 and (H[-1][0] - H[-2][0]) * (q[1] - H[-2][1]) - (H[-1][1] - H[-2][1]) * (q[0] - H[-2][0]) >= 0: H.pop()
        H.append(q)
    return H
H = hull(P.values()); hx = np.array([h[0] for h in H]); hy = np.array([h[1] for h in H])
print('hull members:', [min(P, key=lambda k: abs(math.log(P[k][0]) - a_) + abs(P[k][1] - b_)) for a_, b_ in H])
print('hull:', [(round(math.exp(a), 2), round(b, 3)) for a, b in H])
gaps = []
for lam in lams:
    x, y = P[(lam, 'K4096 R256 q7')]; g = float(np.interp(math.log(x), hx, hy)) - y; gaps.append(g); print('default at %.2f MB: %.3f dB, hull %.3f, gap %.3f dB' % (x, y, y + g, g))
print('default line below the hull: max %.3f dB, mean %.3f dB' % (max(gaps), st.mean(gaps)))
fig, ax = plt.subplots(figsize=(3.35, 2.6))
ax.plot(np.exp(hx), hy, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=2, label='convex hull of all points')
d = [P[(lam, 'K4096 R256 q7')] for lam in lams]; ax.plot([q[0] for q in d], [q[1] for q in d], color='#2a78d6', lw=1.6, zorder=3)
for key, lab, col, mk in LEV:
    pts = [P[(lam, key)] for lam in lams]; ax.plot([q[0] for q in pts], [q[1] for q in pts], ls='none', marker=mk, color=col, ms=5 if key.startswith('K4096 R256 q7') else 4.2, mec=SURF, mew=0.6, zorder=6 if key in ('K4096 R1024 q7', 'K4096 R256 q8') else (5 if key == 'K4096 R256 q7' else 4), label=lab)
ax.set_xscale('log'); ax.xaxis.set_major_locator(mt.FixedLocator([4, 6, 10, 15, 20, 30])); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: '%g' % v)); ax.xaxis.set_minor_formatter(mt.NullFormatter())
ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=2)
for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
ax.set_xlabel('file size (MB, log scale)'); ax.set_ylabel('PSNR (dB)')
ax.legend(loc='lower right', frameon=False, fontsize=6.2, handlelength=1.6, labelspacing=0.25, title='$K + K_r$', title_fontsize=6.5)
fig.tight_layout(pad=0.4); fig.savefig('paper/figures/fig_codebook_sweep_v1.pdf'); fig.savefig('paper/figures/fig_codebook_sweep_v1.png', dpi=220)
