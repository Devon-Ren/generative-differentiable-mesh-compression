"""make_figures_v14.py (2026-11-16): main RD figure as three panels, one variable per panel (11 Mip-NeRF 360 / T&T scenes, means).
  (a) codec       : the same meshes (MeshSplatting with vertex caps) written by BM-Codec, TFAN, zstd, Draco
  (b) training    : the same codec (BM-Codec); ByteMesh training against MeshSplatting with vertex caps
  (c) simplifying : the same codec (BM-Codec); ByteMesh training against the FP32 model simplified after training
Data loading is the first part of make_figures_v7.py.  Output paper/figures/fig_rd_three_v1.{pdf,png}."""
src = open('paper/figures/make_figures_v7.py').read(); cut = src.index('fig, (a, b) = plt.subplots'); exec(compile(src[:cut], 'v7data', 'exec'))
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 7.5})
fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.1, 2.55))
SPEC = (('tfan', 'TFAN', 'D', C5[2]), ('zstd', 'zstd', '^', C5[3]), ('draco', 'Draco', 'v', C5[4]))
# (a) codec
curve(a, co, C5[1], '-', 's', lw=2.0, ms=5, z=7, hl=co)
for k, name, mk, col in SPEC: curve(a, D['points'][k], col, '-', mk, lw=1.6, ms=4.5, hl=D['hull'][k])
a.set_xlim(4.5, 520); a.set_ylim(20.3, 24.1); a.xaxis.set_major_locator(mt.FixedLocator([5, 10, 20, 50, 100, 200, 500]))
# (b) training
curve(b, bm, C5[0], '-', 'o', lw=2.2, ms=5.5, z=8, hl=bm); curve(b, co, C5[1], '-', 's', lw=2.0, ms=5, z=7, hl=co)
b.set_xlim(4, 27); b.set_ylim(22.6, 23.95); b.xaxis.set_major_locator(mt.FixedLocator([5, 7.5, 10, 15, 20, 25]))
# (c) simplification
pho = ph + [ring]; curve(c, bm, C5[0], '-', 'o', lw=2.2, ms=5.5, z=8, hl=bm); curve(c, pho, C5[5] if len(C5) > 5 else '#7a5cc9', DASH, 'P', lw=2.0, ms=5.5, z=7)
c.set_xlim(3.3, 32); c.set_ylim(17.4, 24.4); c.xaxis.set_major_locator(mt.FixedLocator([5, 10, 20, 30]))
for ax, cap in ((a, '(a) Codec: same meshes'), (b, '(b) Training: same codec'), (c, '(c) Simplifying: same codec')):
    ax.set_xscale('log'); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: '%g' % v)); ax.xaxis.set_minor_formatter(mt.NullFormatter())
    ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=2, pad=2)
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.set_xlabel('file size (MB)\n' + cap, color=MUTED); ax.xaxis.label.set_linespacing(1.5)
for ax, lim, tk in ((a, (4.5, 520), [5, 10, 20, 50, 100, 200, 500]), (b, (4, 27), [5, 7.5, 10, 15, 20, 25]), (c, (3.3, 32), [5, 10, 20, 30])):
    ax.set_xlim(*lim); ax.xaxis.set_major_locator(mt.FixedLocator(tk)); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: '%g' % v)); ax.xaxis.set_minor_locator(mt.NullLocator())
a.set_ylabel('PSNR (dB)')
L = lambda col, mk, lab, ls='-': Line2D([], [], color=col, marker=mk, ms=4.5, mec=SURF, mew=0.7, lw=1.7, ls=ls, label=lab)
a.legend(handles=[L(C5[1], 's', 'BM-Codec'), L(C5[2], 'D', 'TFAN'), L(C5[3], '^', 'zstd'), L(C5[4], 'v', 'Draco')], loc='lower right', frameon=True, facecolor=SURF, edgecolor='none', framealpha=0.9, fontsize=6.5, handlelength=2, labelspacing=0.25, title='MeshSplatting with cap +', title_fontsize=6.5)
b.legend(handles=[L(C5[0], 'o', 'ByteMesh'), L(C5[1], 's', 'MeshSplatting with cap\n+ BM-Codec')], loc='lower right', frameon=False, fontsize=6.5, handlelength=2, labelspacing=0.4)
c.legend(handles=[L(C5[0], 'o', 'ByteMesh'), L('#7a5cc9', 'P', 'FP32 simplified\n+ BM-Codec', ls=DASH)], loc='lower right', frameon=False, fontsize=6.5, handlelength=2.4, labelspacing=0.4)
fig.tight_layout(pad=0.4, w_pad=1.0); fig.savefig('paper/figures/fig_rd_three_v1.pdf'); fig.savefig('paper/figures/fig_rd_three_v1.png', dpi=200); print('ok')
