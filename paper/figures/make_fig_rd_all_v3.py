"""make_fig_rd_all_v3.py (2026-11-20): fig_rd_all_v3 -- rate-distortion in PSNR, SSIM and LPIPS on the four datasets (4 rows x 3 columns),
drawn like the main RD figure (make_figures_v22.py): each baseline curve is the convex hull of its points over log size, the line is a
PCHIP interpolation through the hull points over log size, and the points off the hull are drawn faintly.  ByteMesh and
MeshSplatting-cap + BM-Codec pass through all of their points.  An arrow in each panel marks the better direction.
Data loading is the first part of make_fig_rd_all_v1.py.  Run from the repository root."""
import math
import numpy as np
from scipy.interpolate import PchipInterpolator
src = open('paper/figures/make_fig_rd_all_v1.py').read(); exec(compile(src[:src.index('fig, axs = plt.subplots')], 'v1data', 'exec'))
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#dddcd6', '#fcfcfb'
plt.rcParams.update({'font.size': 7.5, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
def smooth(pts, n=80):
    p = sorted(pts); lx = np.log([q[0] for q in p]); y = np.array([q[1] for q in p])
    if len(p) < 3: return np.exp(lx), y
    g = np.linspace(lx[0], lx[-1], n); return np.exp(g), PchipInterpolator(lx, y)(g)
def hull(p, sign):      # convex hull over log size on the better side, up to the best point (sign = +1: higher is better, -1: lower is better)
    P = sorted((math.log(x), sign * y) for x, y in p); H = []
    for q in P:
        while len(H) >= 2 and (H[-1][0] - H[-2][0]) * (q[1] - H[-2][1]) - (H[-1][1] - H[-2][1]) * (q[0] - H[-2][0]) >= 0: H.pop()
        H.append(q)
    k = max(range(len(H)), key=lambda t: H[t][1]); return [(math.exp(a), sign * b) for a, b in H[:k + 1]]
def curve(ax, pts, color, mk, sign, lw=1.6, ms=4.2, z=4, allpts=False):
    pts = [tuple(q) for q in pts]; hl = pts if allpts else hull(pts, sign); near = lambda q: any(abs(q[0] - h[0]) < 1e-9 * h[0] and abs(q[1] - h[1]) < 1e-9 for h in hl); rest = [q for q in pts if not near(q)]
    ax.plot(*smooth(hl), color=color, lw=lw, zorder=z)
    if rest: ax.plot([q[0] for q in rest], [q[1] for q in rest], ls='none', marker=mk, color=color, ms=ms - 1.5, mec='none', alpha=0.4, zorder=z - 1)
    ax.plot([q[0] for q in hl], [q[1] for q in hl], ls='none', marker=mk, color=color, ms=ms, mec=SURF, mew=0.8, zorder=z + 1)
fig, axs = plt.subplots(4, 3, figsize=(7.1, 8.6))
for r, (name, S, xl, tk) in enumerate((('Mip-NeRF 360 (9 scenes)', MIP, (3.3, 520), [5, 10, 20, 50, 100, 200, 500]), ('Tanks and Temples (2 scenes)', TT, (3.3, 520), [5, 10, 20, 50, 100, 200, 500]),
                                       ('Deep Blending (2 scenes)', DB, (3.3, 520), [5, 10, 20, 50, 100, 200, 500]), ('NeRF-Synthetic (8 objects)', OBJ, (0.9, 60), [1, 2, 5, 10, 20, 50]))):
    L = [levels(s) for s in S]; n = min(len(x) for x in L); assert n >= 2, (name, [len(x) for x in L])
    T = sorted(set.intersection(*[set(H[s]) for s in S]))
    for c, (lab, sign) in enumerate((('PSNR (dB)', 1), ('SSIM', 1), ('LPIPS', -1))):
        ax = axs[r][c]
        for k in ('draco', 'zstd', 'tfan'):
            pts = [(mean([G[(s, L[i][j])][b][k] for i, s in enumerate(S)]), mean([G[(s, L[i][j])][b]['q'][c] for i, s in enumerate(S)])) for j in range(n) for b in (8, 6, 4) if all(b in G[(s, L[i][j])] for i, s in enumerate(S))]
            curve(ax, pts, COL[k], MK[k], sign)
        cp = [(mean([O[s][L[i][j]][0] for i, s in enumerate(S)]), mean([O[s][L[i][j]][1][c] for i, s in enumerate(S)])) for j in range(n)]
        bm = [(mean([H[s][t][0] for s in S]), mean([H[s][t][1][c] for s in S])) for t in T]
        curve(ax, cp, COL['cap'], MK['cap'], sign, lw=2.0, ms=4.8, z=7, allpts=True); curve(ax, bm, COL['bm'], MK['bm'], sign, lw=2.2, ms=5.0, z=8, allpts=True)
        ax.set_xscale('log'); ax.set_xlim(*xl); ax.xaxis.set_major_locator(mt.FixedLocator(tk)); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: '%g' % v)); ax.xaxis.set_minor_locator(mt.NullLocator())
        ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=2, pad=2); ax.set_ylabel(lab)
        for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
        # better direction: a small arrow into the upper left corner (PSNR, SSIM) or the lower left corner (LPIPS); the y range is extended so that the corner is free
        lo, hi = ax.get_ylim(); m = 0.15 * (hi - lo); ax.set_ylim((lo, hi + m) if sign > 0 else (lo - m, hi))
        y0, y1, yt = (0.905, 0.975, 0.94) if sign > 0 else (0.095, 0.025, 0.06)
        ax.annotate('', xy=(0.03, y1), xytext=(0.095, y0), xycoords='axes fraction', textcoords='axes fraction', arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=0.8, mutation_scale=6, shrinkA=0, shrinkB=0), zorder=9)
        ax.text(0.11, yt, 'better', transform=ax.transAxes, ha='left', va='center', fontsize=5.6, color=MUTED, style='italic')
        if c == 1: ax.set_title(name, fontsize=8, color=INK, pad=4)
        if r == 3: ax.set_xlabel('file size (MB)', color=MUTED)
LG = lambda k, lab: Line2D([], [], color=COL[k], marker=MK[k], ms=4.5, mec=SURF, mew=0.7, lw=1.7, label=lab)
fig.legend(handles=[LG('bm', 'ByteMesh (ours)'), LG('cap', 'MeshSplatting-cap + BM-Codec'), LG('tfan', '+ MPEG TFAN'), LG('zstd', '+ zstd'), LG('draco', '+ Draco')], loc='upper center', ncol=5, frameon=False, fontsize=7.2, handlelength=2.0, columnspacing=1.4)
fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.2, w_pad=1.0); fig.savefig('paper/figures/fig_rd_all_v3.pdf'); fig.savefig('paper/figures/fig_rd_all_v3.png', dpi=130); print('ok')
