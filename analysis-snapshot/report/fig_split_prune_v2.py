# VERSION fig-split-prune-v2 (2026-11-03, notation B_a / B_f, additive only): worked example of the priced split (stage 1) and the lambda edge-collapse (stage 2).
# Numbers are illustrative; rules and byte prices follow analysis/rdgrowth/rd_growth_v2.py and rd_prune.py.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch

GREEN = '#1f7750'; GREENF = '#e5f3ec'; GREY = '#5a5a5a'; RED = '#c0392b'; REDF = '#f6dcd8'; BLUEF = '#dde8f5'
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

# ---------------- (a) split ----------------
ax = axes[0]
A, B, C = (0.0, 0.0), (2.0, 0.0), (1.0, 1.7)
ax.add_patch(Polygon([A, B, C], closed=True, fc=REDF, ec=GREY, lw=1.2))
for p, n in zip([A, B, C], 'ABC'):
    ax.plot(*p, 'o', color=GREY, ms=5); ax.text(p[0] + (-0.12 if n == 'A' else 0.08), p[1] - 0.14 if n != 'C' else p[1] + 0.08, n, color=GREY)
ax.text(1.0, 0.55, 'triangle $t$', ha='center', fontsize=9)
off = 3.0
A2, B2, C2 = (A[0] + off, A[1]), (B[0] + off, B[1]), (C[0] + off, C[1])
mAB = ((A2[0] + B2[0]) / 2, (A2[1] + B2[1]) / 2); mBC = ((B2[0] + C2[0]) / 2, (B2[1] + C2[1]) / 2); mCA = ((C2[0] + A2[0]) / 2, (C2[1] + A2[1]) / 2)
for tri in ([A2, mAB, mCA], [mAB, B2, mBC], [mCA, mBC, C2], [mAB, mBC, mCA]):
    ax.add_patch(Polygon(tri, closed=True, fc='white', ec=GREY, lw=1.0))
for p in (A2, B2, C2):
    ax.plot(*p, 'o', color=GREY, ms=5)
for p in (mAB, mBC, mCA):
    ax.plot(*p, 'o', color=GREEN, ms=7)
ax.text(off + 1.0, -0.32, '3 new vertices (edge midpoints), 1 parent -> 4 children', ha='center', color=GREEN, fontsize=8.5)
ax.add_patch(FancyArrowPatch((2.15, 0.85), (2.85, 0.85), arrowstyle='-|>', mutation_scale=14, color=GREEN, lw=1.4))
ax.text(2.5, 1.0, 'split', ha='center', color=GREEN, fontweight='bold')
txt = ('Stage 1, every 500 it until 10k: split $t$ iff $\\Delta D_t / \\Delta R_t > \\lambda$\n\n'
       '$\\Delta D_t$: in each training view, sum the L1 error of the pixels\n'
       '   whose front triangle is $t$, divide by image Mpx, average over views\n'
       '   e.g. 40 px x 0.05 = 2.0 ; / 1.6 Mpx = 1.25 ; visible in 30 % of views -> 0.375\n'
       '$\\Delta R_t = 3B_a + 6B_f$ = 3 x 6 B + 6 x 2 B = 30 B\n'
       '   (3 new vertices; after the Delaunay rebuild each vertex brings ~2 faces)\n'
       'ratio = 0.375 / 30 = 0.0125\n'
       '   $\\lambda$ = 6.2e-3 : 0.0125 > $\\lambda$  -> split\n'
       '   $\\lambda$ = 1.85e-2: 0.0125 < $\\lambda$  -> keep (growth stops here earlier)')
ax.text(-0.1, -0.75, txt, va='top', fontsize=8.6, family='DejaVu Sans')
ax.set_xlim(-0.3, 5.3); ax.set_ylim(-3.55, 2.0); ax.set_aspect('equal'); ax.axis('off')
ax.set_title('(a) priced split: does this triangle deserve 30 more bytes?', fontsize=10, loc='left')

# ---------------- (b) collapse ----------------
ax = axes[1]
u, v = (-0.5, 0.0), (0.5, 0.0)
t, b = (0.0, 0.95), (0.0, -0.95)
ul, l, dl = (-1.1, 0.85), (-1.6, 0.0), (-1.1, -0.85)
ur, r, dr = (1.1, 0.85), (1.6, 0.0), (1.1, -0.85)
ring = [(u, t, ul), (u, ul, l), (u, l, dl), (u, dl, b), (v, ur, t), (v, r, ur), (v, dr, r), (v, b, dr)]
edgef = [(u, b, v), (u, v, t)]
for f in ring:
    ax.add_patch(Polygon(f, closed=True, fc=BLUEF, ec=GREY, lw=1.0))
for f in edgef:
    ax.add_patch(Polygon(f, closed=True, fc=REDF, ec=GREY, lw=1.0))
ax.plot([u[0], v[0]], [u[1], v[1]], color=RED, lw=2.4)
for p in (t, b, ul, l, dl, ur, r, dr):
    ax.plot(*p, 'o', color=GREY, ms=4.5)
ax.plot(*u, 'o', color=RED, ms=6.5); ax.plot(*v, 'o', color=RED, ms=6.5)
ax.text(u[0] - 0.05, u[1] + 0.12, 'u', color=RED, ha='right'); ax.text(v[0] + 0.05, v[1] + 0.12, 'v', color=RED)
off = 4.3
def S(p): return (p[0] + off, p[1])
m = S((0.0, 0.0))
after = [(m, S(t), S(ul)), (m, S(ul), S(l)), (m, S(l), S(dl)), (m, S(dl), S(b)), (m, S(ur), S(t)), (m, S(r), S(ur)), (m, S(dr), S(r)), (m, S(b), S(dr))]
for f in after:
    ax.add_patch(Polygon(f, closed=True, fc=BLUEF, ec=GREY, lw=1.0))
for p in (t, b, ul, l, dl, ur, r, dr):
    ax.plot(*S(p), 'o', color=GREY, ms=4.5)
ax.plot(*m, 'o', color=GREEN, ms=7.5)
ax.text(m[0], -1.35, 'merged vertex (green) at the midpoint', ha='center', color=GREEN, fontsize=8.3)
ax.add_patch(FancyArrowPatch((1.8, 0.0), (2.55, 0.0), arrowstyle='-|>', mutation_scale=14, color=GREEN, lw=1.4))
ax.text(2.17, 0.15, 'collapse', ha='center', color=GREEN, fontweight='bold')
ax.text(0.0, -1.35, '10 faces in the two rings\n(red = the 2 faces on edge uv)', ha='center', va='top', fontsize=8.3, color=GREY)
ax.text(off, -1.62, '8 faces: -1 vertex, -2 faces', ha='center', va='top', fontsize=8.3, color=GREY)
txt = ('Stage 2, at 12k / 14k / 16k / 18k: collapse $(u,v)$ iff $\\Delta D_{uv} / \\Delta R_{uv} < \\lambda$\n\n'
       '$\\Delta D_{uv}$ = sum of $\\Delta D_f$ over all faces $f$ touching $u$ or $v$\n'
       '   (same per-view, per-Mpx units as the split), e.g. 10 faces x 0.004 = 0.04\n'
       '$\\Delta R_{uv} = B_a + n_{uv} B_f$ = bytes the collapse removes\n'
       '   1 vertex + the $n_{uv}$ faces on the edge: 6 B + 2 x 2 B = 10 B (interior)\n'
       '   boundary edge ($n_{uv}$ = 1): 8 B\n'
       'score = 0.04 / 10 = 0.004\n'
       '   $\\lambda$ = 6.2e-3 : 0.004 < $\\lambda$  -> collapse\n'
       '   $\\lambda$ = 1.85e-3: 0.004 > $\\lambda$  -> keep\n'
       'per round: no vertex in two collapses, cheapest first, <= 15 % of V,\n'
       'reject if a surviving face would flip')
ax.text(-1.75, -2.25, txt, va='top', fontsize=8.6)
ax.set_xlim(-1.9, 6.1); ax.set_ylim(-5.1, 1.3); ax.set_aspect('equal'); ax.axis('off')
ax.set_title('(b) lambda-prune: is this vertex worth its 10 bytes?', fontsize=10, loc='left')

fig.text(0.5, 0.01, 'Two independent byte prices: $B_a$ = 6 B attributes of one vertex (position residual, DC colour, SH indices); $B_f$ = 2 B connectivity of one face.  Numbers are illustrative.', ha='center', fontsize=8.2, color=GREY)
plt.tight_layout(rect=(0, 0.03, 1, 1))
out = '/mnt/project-files/analysis/report/fig-split-prune-v2'
fig.savefig(out + '.png', dpi=170); fig.savefig(out + '.pdf')
print('wrote', out)
