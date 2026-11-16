"""make_figures_v18.py (2026-11-16): one figure for all three dataset groups, 2 rows x 3 columns.
  row 1, codec   : the same meshes (MeshSplatting-cap) written by BM-Codec, TFAN, zstd, Draco
  row 2, training: the same codec (BM-Codec); ByteMesh, MeshSplatting-cap, and the FP32 model with collapse after training
  columns        : Mip-NeRF 360 + T&T (11 scenes), Deep Blending (2), NeRF-Synthetic (8).  Dataset means.
Data loading: make_figures_v13.py (which runs make_figures_v7.py).  Output paper/figures/fig_rd_grid_v1.{pdf,png}."""
import io, contextlib
src = open('paper/figures/make_figures_v13.py').read(); cut = src.index('fig, (p, q) = plt.subplots')
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src[:cut], 'v13data', 'exec'))
OBJ = J['objects']
PH = re.compile(r'^(\w+) budget (\d+) POST-HOC-COLLAPSE(?:-BLENDER)? \| .*?\| delivered ([\d.]+) MB test ([\d.]+)'); sim = collections.defaultdict(set)
for f in sorted(glob.glob('release-*/results/*.txt')):
    for l in open(f, errors='ignore'):
        m = PH.match(l)
        if m: sim[m[1]].add((float(m[3]), float(m[4])))
def by_rank(scenes):      # mean of the k-th largest simplified file over the scenes
    K = min(len(sim[x]) for x in scenes); return [(mean([sorted(sim[x], reverse=True)[k][0] for x in scenes]), mean([sorted(sim[x], reverse=True)[k][1] for x in scenes])) for k in range(K)]
col_db, col_obj = by_rank(DB), by_rank(OBJ); co4 = co + [ring]; pho = ph + [ring]
print('collapse after training, means: DB', [(round(x, 1), round(y, 2)) for x, y in col_db], '| objects', [(round(x, 2), round(y, 2)) for x, y in col_obj])
plt.rcParams.update({'font.size': 7.2}); PUR = '#7a5cc9'
fig, ax = plt.subplots(2, 3, figsize=(7.1, 3.75))
COD = [(co4, D['points'], D['hull']), (db_co, db_pts, None), (J['cap_ours'], J['points'], J['hull'])]
TRN = [(bm, co4, pho), (db_bm, db_co, col_db), (J['budgetmesh'], J['cap_ours'], col_obj)]
for j, (c_, pts, hl) in enumerate(COD):
    curve(ax[0, j], c_, C5[1], '-', 's', lw=1.8, ms=4.5, z=7, hl=hull(c_))
    for k, name, mk, col in SPEC: curve(ax[0, j], pts[k], col, '-', mk, lw=1.4, ms=4, hl=(hl[k] if hl else None))
for j, (b_, c_, s_) in enumerate(TRN):
    curve(ax[1, j], b_, C5[0], '-', 'o', lw=2.0, ms=4.8, z=8, hl=b_); curve(ax[1, j], c_, C5[1], '-', 's', lw=1.8, ms=4.5, z=7, hl=hull(c_)); curve(ax[1, j], s_, PUR, DASH, 'P', lw=1.8, ms=4.8, z=6)
XL0 = [(4.5, 520), (5, 800), (1.0, 60)]; YL0 = [(20.3, 24.1), (21.8, 27.4), (28.4, 31.7)]; TK0 = [[5, 10, 20, 50, 100, 200, 500], [5, 10, 20, 50, 100, 200, 500], [1, 2, 5, 10, 20, 50]]
XL1 = [(4, 30), (4, 45), (0.9, 7)]; YL1 = [(21.0, 24.0), (24.2, 27.3), (26.0, 31.6)]; TK1 = [[5, 10, 20], [5, 10, 20, 40], [1, 2, 4, 6]]
NAMES = ['Mip-NeRF 360 and T&T (11 scenes)', 'Deep Blending (2 scenes)', 'NeRF-Synthetic (8 objects)']
for i, (XL, YL, TK) in enumerate(((XL0, YL0, TK0), (XL1, YL1, TK1))):
    for j in range(3):
        a_ = ax[i, j]; a_.set_xscale('log'); a_.set_xlim(*XL[j]); a_.set_ylim(*YL[j]); a_.xaxis.set_major_locator(mt.FixedLocator(TK[j])); a_.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: '%g' % v)); a_.xaxis.set_minor_locator(mt.NullLocator())
        a_.grid(axis='y', color=GRID, lw=0.5); a_.set_axisbelow(True); a_.tick_params(length=2, pad=2)
        for s_ in ('top', 'right'): a_.spines[s_].set_visible(False)
        if i == 0: a_.set_title(NAMES[j], fontsize=7.5, color=INK, pad=4)
        if i == 1: a_.set_xlabel('file size (MB)', color=MUTED)
ax[0, 0].set_ylabel('(a) codecs,\nsame meshes\nPSNR (dB)'); ax[1, 0].set_ylabel('(b) training,\nsame codec\nPSNR (dB)')
LG = lambda col, mk, lab, ls='-': Line2D([], [], color=col, marker=mk, ms=4.5, mec=SURF, mew=0.7, lw=1.7, ls=ls, label=lab)
fig.legend(handles=[LG(C5[1], 's', 'BM-Codec (ours)'), LG(C5[2], 'D', 'MPEG TFAN'), LG(C5[3], '^', 'zstd'), LG(C5[4], 'v', 'Draco')], loc='upper center', bbox_to_anchor=(0.54, 1.0), ncol=4, frameon=False, fontsize=6.8, handlelength=2.2, columnspacing=1.6, title='(a) meshes from MeshSplatting-cap, written by', title_fontsize=6.8)
fig.legend(handles=[LG(C5[0], 'o', 'ByteMesh (rate-controlled training)'), LG(C5[1], 's', 'MeshSplatting-cap'), LG(PUR, 'P', 'MeshSplatting FP32, collapse after training', ls=DASH)], loc='upper center', bbox_to_anchor=(0.54, 0.515), ncol=3, frameon=False, fontsize=6.8, handlelength=2.2, columnspacing=1.6, title='(b) written by BM-Codec, meshes from', title_fontsize=6.8)
fig.tight_layout(rect=(0, 0, 1, 0.9), pad=0.4, w_pad=0.8, h_pad=3.6); fig.savefig('paper/figures/fig_rd_grid_v1.pdf'); fig.savefig('paper/figures/fig_rd_grid_v1.png', dpi=200); print('ok')
