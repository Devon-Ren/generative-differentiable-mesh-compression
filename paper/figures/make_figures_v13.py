"""make_figures_v13.py (2026-11-16): figures after the advisor's comments on the experiments.
  fig_rd_pair_v5   : the two RD panels (11 Mip-NeRF 360 / T&T scenes).  'our codec' -> 'BM-Codec'; the uncompressed (FP32) size and PSNR of the
                     meshes behind each panel are drawn as hollow grey squares (cap 0.6M / 1.08M / 3M in (a); three simplified levels in (b)).
  fig_rd_db_obj_v1 : the same comparison as panel (a) on Deep Blending (2 scenes) and NeRF-Synthetic (8 objects).
Data: as make_figures_v7.py; release-batch43-overnight/results (Deep Blending); release-batch38-generic-objects/results/batch38-means.json."""
import re, glob, json, math, collections, statistics as st
import numpy as np
import matplotlib; matplotlib.use('Agg')
s = open('paper/figures/make_figures_v7.py').read()
def rep(a, b):
    global s
    assert s.count(a) >= 1, a[:60]; s = s.replace(a, b)
rep('BudgetMesh', 'ByteMesh'); rep('fig_rd_pair_v2', 'fig_rd_pair_v5')
rep("ax.set_xscale('log'); ax.set_xlim(3.3, 900); ax.set_ylim(17.4, 25.1)", "ax.set_xscale('log'); ax.set_xlim(3.3, 900); ax.set_ylim(*((20.3, 24.3) if ax is a else (17.4, 25.1)))")
rep("a.set_ylabel('PSNR (dB)'); b.tick_params(labelleft=False)", "a.set_ylabel('PSNR (dB)'); b.set_ylabel('PSNR (dB)')")
rep("a.annotate('our codec', co[0]", "a.annotate('BM-Codec', co[0]")
rep("a.text(11.0, 21.5, 'TFAN', ha='right', va='center', fontsize=7.5, color=INK); a.text(19.0, 21.5, 'zstd', ha='left', va='center', fontsize=7.5, color=INK); a.text(64.0, 21.5, 'Draco', ha='left', va='center', fontsize=7.5, color=INK)",
    "a.text(11.2, 21.6, 'TFAN', ha='right', va='center', fontsize=7.5, color=INK); a.text(17.5, 21.6, 'zstd', ha='left', va='center', fontsize=7.5, color=INK); a.text(52.0, 21.6, 'Draco', ha='left', va='center', fontsize=7.5, color=INK)")
rep("L(C5[1], 's', 'our codec')", "L(C5[1], 's', 'BM-Codec')")
RAW = ("RAWA = [(130, 22.91), (216, 23.36), (519, 23.92)]; RAWB = [(92, 18.62), (199, 21.30), (392, 23.23)]\n"
       "for ax_, R_ in ((a, RAWA), (b, RAWB)): ax_.plot([q[0] for q in R_], [q[1] for q in R_], ls='none', marker='s', ms=5.2, mfc='none', mec=MUTED, mew=1.1, zorder=5)\n"
       "a.plot([fx], [fy], '*', color=INK, ms=10, zorder=6)\n")
rep("for ax, cap in ((a, '(a) Baselines trained with a vertex cap'), (b, '(b) Baselines simplified after training')):", RAW + "for ax, cap in ((a, '(a) Meshes from MeshSplatting with different vertex caps'), (b, '(b) Meshes from simplifying the MeshSplatting FP32 model')):")
rep("Line2D([], [], color=MUTED, lw=1.6, ls='-', label='size set during training'), Line2D([], [], color=MUTED, lw=1.6, ls=DASH, label='size set after training')]",
    "Line2D([], [], color=MUTED, lw=1.6, ls='-', label='size set during training'), Line2D([], [], color=MUTED, lw=1.6, ls=DASH, label='size set after training'), Line2D([], [], ls='none', marker='s', ms=5, mfc='none', mec=MUTED, mew=1.1, label='uncompressed mesh'), Line2D([], [], ls='none', marker='*', ms=8, color=INK, label='uncompressed FP32 model')]")
rep("fig.legend(handles=hs, loc='upper center', ncol=5,", "fig.legend(handles=hs, loc='upper center', ncol=5,")
exec(compile(s, 'v7mod', 'exec'))
# ------------------------------------------------------------------ Deep Blending + objects
DB = ['drjohnson', 'playroom']; mean = lambda v: sum(v) / len(v)
GR = re.compile(r'(\w+) GENERIC-RD \| mesh (\w+) \| V (\d+) F (\d+) \| sh(\d) \| zstd ([\d.]+) MB \| draco ([\d.]+) MB \| tfan ([\d.]+) MB \| test ([\d.]+)')
Gd = collections.defaultdict(dict)
for l in open('release-batch43-overnight/results/batch43-db-generic-rows.txt'):
    m = GR.match(l)
    if m: Gd[(m[1], m[2])][int(m[5])] = dict(zstd=float(m[6]), draco=float(m[7]), tfan=float(m[8]), psnr=float(m[9]))
A = re.compile(r'^(\w+) budget (\d+) TRAIN-AT-BUDGET \| vertices (\d+) .*?\| delivered ([\d.]+) MB test ([\d.]+)'); Od = collections.defaultdict(dict); Fd = {}; Hd = collections.defaultdict(dict)
for f in sorted(glob.glob('release-*/results/*.txt')):
    for l in open(f, errors='ignore'):
        m = A.match(l)
        if m and m[1] in DB: Od[m[1]][int(m[2])] = (float(m[4]), float(m[5]))
        m = re.match(r'^(\w+) ORIGINAL-INDOOR \|.*?delivered ([\d.]+) MB test ([\d.]+)', l)
        if m and m[1] in DB: Fd[m[1]] = (float(m[2]), float(m[3]))
        m = re.match(r'(\w+) abl24-hold \| target ([\d.]+) MB \|.*?delivered ([\d.]+) MB.*?\| test ([\d.]+)', l)
        if m and m[1] in DB: Hd[m[1]][float(m[2])] = (float(m[3]), float(m[4]))
MESH = ['cap600000', 'cap1081000', 'cap3000000', 'full']
db_bm = [(mean([Hd[x][T][0] for x in DB]), mean([Hd[x][T][1] for x in DB])) for T in sorted(Hd['playroom'])]
db_co = [(mean([(Fd[x] if m_ == 'full' else Od[x][int(m_[3:])])[0] for x in DB]), mean([(Fd[x] if m_ == 'full' else Od[x][int(m_[3:])])[1] for x in DB])) for m_ in MESH]
db_pts = {k: [(mean([Gd[(x, m_)][b_][k] for x in DB]), mean([Gd[(x, m_)][b_]['psnr'] for x in DB])) for m_ in MESH for b_ in (8, 6, 4)] for k in ('tfan', 'zstd', 'draco')}
J = json.load(open('release-batch38-generic-objects/results/batch38-means.json'))
print('Deep Blending means: ByteMesh', [(round(x, 2), round(y, 2)) for x, y in db_bm], '| cap + BM-Codec', [(round(x, 2), round(y, 2)) for x, y in db_co])
fig, (p, q) = plt.subplots(1, 2, figsize=(7.1, 3.3))
curve(p, db_bm, C5[0], '-', 'o', lw=2.2, ms=6, z=8, hl=db_bm); curve(p, db_co, C5[1], '-', 's', lw=2.0, ms=5.5, z=7, hl=hull(db_co))
for k, name, mk, col in SPEC: curve(p, db_pts[k], col, '-', mk)
curve(q, J['budgetmesh'], C5[0], '-', 'o', lw=2.2, ms=6, z=8, hl=J['budgetmesh']); curve(q, J['cap_ours'], C5[1], '-', 's', lw=2.0, ms=5.5, z=7, hl=J['cap_ours'])
for k, name, mk, col in SPEC: curve(q, J['points'][k], col, '-', mk, hl=J['hull'][k])
for ax, cap, xl, yl, tk in ((p, '(a) Deep Blending, mean over 2 scenes', (3.3, 900), (21.8, 27.6), [5, 10, 20, 50, 100, 200, 500]), (q, '(b) NeRF-Synthetic, mean over 8 objects', (0.9, 60), (28.4, 31.8), [1, 2, 5, 10, 20, 50])):
    ax.set_xscale('log'); ax.set_xlim(*xl); ax.set_ylim(*yl); ax.xaxis.set_major_locator(mt.FixedLocator(tk)); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: '%g' % v)); ax.xaxis.set_minor_formatter(mt.NullFormatter())
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=2); ax.set_ylabel('PSNR (dB)')
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.set_xlabel('file size (MB, log scale)\n\n' + cap, color=MUTED); ax.xaxis.label.set_linespacing(1.25)
LG = lambda c, mk, lab: Line2D([], [], color=c, marker=mk, ms=5, mec=SURF, mew=0.8, lw=1.8, label=lab)
fig.legend(handles=[LG(C5[0], 'o', 'ByteMesh (ours)'), LG(C5[1], 's', 'MeshSplatting with cap + BM-Codec'), LG(C5[2], 'D', '+ MPEG TFAN'), LG(C5[3], '^', '+ zstd'), LG(C5[4], 'v', '+ Draco')], loc='upper center', ncol=5, frameon=False, fontsize=7, handlelength=2.4, columnspacing=1.4, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.4); fig.savefig('paper/figures/fig_rd_db_obj_v1.pdf'); fig.savefig('paper/figures/fig_rd_db_obj_v1.png', dpi=170); print('ok 2')
