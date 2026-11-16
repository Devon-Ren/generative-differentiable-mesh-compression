"""make_figures_v21.py (2026-11-16): main RD figure, three panels in the order the advisor asked for (11 Mip-NeRF 360 / T&T scenes, means).
  (a) codecs on the meshes from MeshSplatting-cap           (b) codecs on the meshes from collapse after training
  (c) training: ByteMesh against MeshSplatting-cap, both written by BM-Codec
In (a) and (b) the grey line is the same meshes without a codec (32-bit floats).
Data loading is the first part of make_figures_v7.py.  Output paper/figures/fig_rd_three_v6.{pdf,png}."""
src = open('paper/figures/make_figures_v7.py').read(); cut = src.index('fig, (a, b) = plt.subplots'); exec(compile(src[:cut], 'v7data', 'exec'))
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 7.5})
fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.1, 2.95))
SPEC = (('tfan', 'D', C5[2]), ('zstd', '^', C5[3]), ('draco', 'v', C5[4])); co4 = co + [ring]; pho = ph + [ring]
RAWA = [(129.6, 22.91), (216.1, 23.36), (519.4, 23.92), (608.7, 23.96)]; RAWB = [(92, 18.62), (199, 21.30), (392, 23.23), (608.7, 23.96)]
def raw(ax, pts): ax.plot([q[0] for q in pts], [q[1] for q in pts], color=MUTED, lw=1.2, marker='s', ms=4.2, mfc=SURF, mec=MUTED, mew=1.0, zorder=3)
curve(a, co4, C5[1], '-', 's', lw=2.0, ms=5, z=7, hl=co4)
for k, mk, col in SPEC: curve(a, D['points'][k], col, '-', mk, lw=1.6, ms=4.5, hl=D['hull'][k])
raw(a, RAWA)
curve(b, pho, C5[1], '-', 's', lw=2.0, ms=5, z=7)
for k, mk, col in SPEC: curve(b, [(v[k], v['psnr']) for v in G33.values()], col, '-', mk, lw=1.6, ms=4.5)
raw(b, RAWB)
curve(c, bm, C5[0], '-', 'o', lw=2.2, ms=5.5, z=8, hl=bm); curve(c, co4, C5[1], '-', 's', lw=2.0, ms=5, z=7, hl=co4); raw(c, RAWA)
CAPS = ((a, '(a) Codecs on meshes from\nMeshSplatting-cap'), (b, '(b) Codecs on meshes from\ncollapse after training'), (c, '(c) Training, same codec\n(BM-Codec)'))
for (ax, cap), lim, yl, tk in zip(CAPS, ((4.5, 900), (3.3, 900), (3.8, 900)), ((20.3, 24.2), (17.4, 24.4), (22.6, 24.1)), ([5, 10, 20, 50, 100, 200, 500], [5, 10, 20, 50, 100, 200, 500], [5, 10, 20, 50, 100, 200, 500])):
    ax.set_xscale('log'); ax.set_xlim(*lim); ax.set_ylim(*yl); ax.xaxis.set_major_locator(mt.FixedLocator(tk)); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: '%g' % v)); ax.xaxis.set_minor_locator(mt.NullLocator())
    ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=2, pad=2)
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.set_xlabel('file size (MB)\n' + cap, color=MUTED); ax.xaxis.label.set_linespacing(1.45)
a.set_ylabel('PSNR (dB)')
L = lambda col, mk, lab: Line2D([], [], color=col, marker=mk, ms=4.5, mec=SURF, mew=0.7, lw=1.7, label=lab)
hs = [L(C5[1], 's', 'BM-Codec (ours)'), L(C5[2], 'D', 'MPEG TFAN'), L(C5[3], '^', 'zstd'), L(C5[4], 'v', 'Draco'), Line2D([], [], color=MUTED, lw=1.2, marker='s', ms=4.2, mfc=SURF, mec=MUTED, mew=1.0, label='no codec (32-bit floats)')]
fig.legend(handles=hs, loc='upper left', bbox_to_anchor=(0.045, 1.0), ncol=5, frameon=False, fontsize=6.8, handlelength=2.0, columnspacing=1.0)
c.legend(handles=[L(C5[0], 'o', 'ByteMesh + BM-Codec'), L(C5[1], 's', 'MeshSplatting-cap + BM-Codec')], loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False, fontsize=6.8, handlelength=2.0, labelspacing=0.3, borderaxespad=0.2)
fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.4, w_pad=1.0); fig.savefig('paper/figures/fig_rd_three_v6.pdf'); fig.savefig('paper/figures/fig_rd_three_v6.png', dpi=200); print('ok')
