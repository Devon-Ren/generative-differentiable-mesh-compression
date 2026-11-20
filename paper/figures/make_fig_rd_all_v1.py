"""make_fig_rd_all_v1.py (2026-11-20): fig_rd_all_v1 -- rate-distortion in PSNR, SSIM and LPIPS on the four datasets (4 rows x 3 columns).
Curves: ByteMesh (mean over the scenes per target); MeshSplatting-cap + BM-Codec (mean per cap, three caps); MeshSplatting-cap meshes written by
MPEG TFAN / zstd / Draco (mean per cap and SH precision; the line is the envelope of these points, the other points are drawn faintly).
All numbers are read from the result rows in the repository.  Run from the repository root."""
import re, glob, collections
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker as mt
from matplotlib.lines import Line2D
MIP = ['bicycle','flowers','garden','stump','treehill','room','counter','kitchen','bonsai']; TT = ['truck','train']; DB = ['drjohnson','playroom']; OBJ = ['lego','chair','drums','ficus','hotdog','materials','mic','ship']
INK, MUTED, GRID, SURF = '#1f2430', '#5b6270', '#e3e6ea', '#ffffff'; COL = dict(bm='#2a78d6', cap='#eb6834', tfan='#1baf7a', zstd='#eda100', draco='#e87ba4'); MK = dict(bm='o', cap='s', tfan='D', zstd='^', draco='v')
plt.rcParams.update({'font.size': 7.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'pdf.fonttype': 42})
mean = lambda v: sum(v) / len(v)
G = collections.defaultdict(dict)       # (scene, budget) -> bits -> (tfan MB, zstd MB, draco MB, (psnr, ssim, lpips))
R = re.compile(r'(\w+) GENERIC-RD \| mesh cap(\d+) \| V \d+ F \d+ \| sh(\d) \| zstd ([\d.]+) MB \| draco ([\d.]+) MB \| tfan ([\d.]+) MB \| test ([\d.]+) ([\d.]+) ([\d.]+)')
C = re.compile(r'^(\w+) budget (\d+) TRAIN-AT-BUDGET(?:-BLENDER)? \| vertices \d+ .*?\| delivered ([\d.]+) MB test ([\d.]+) ([\d.]+) ([\d.]+)')
O = collections.defaultdict(dict)       # scene -> budget -> (MB, (psnr, ssim, lpips))
for f in sorted(glob.glob('release-*/results/*.txt')):
    for l in open(f, errors='ignore'):
        m = R.match(l)
        if m: G[(m[1], int(m[2]))][int(m[3])] = dict(zstd=float(m[4]), draco=float(m[5]), tfan=float(m[6]), q=(float(m[7]), float(m[8]), float(m[9])))
        m = C.match(l)
        if m: O[m[1]][int(m[2])] = (float(m[3]), (float(m[4]), float(m[5]), float(m[6])))
H = collections.defaultdict(dict)
for fn in ('release-batch31-hold/results/batch31-collect-2026-11-13-0126.txt', 'release-batch37-more-points/results/batch37-rows-2026-11-14.txt', 'release-batch31-hold/results/batch31-ship-rows-2026-11-14.txt'):
    for l in open(fn):
        m = re.match(r'(\w+) abl24-hold \| target ([\d.]+) MB \|.*?delivered ([\d.]+) MB.*?\| test ([\d.]+) ([\d.]+) ([\d.]+)', l)
        if m: H[m[1]][float(m[2])] = (float(m[3]), (float(m[4]), float(m[5]), float(m[6])))
def levels(s):      # budgets of this scene that have baseline rows and a BM-Codec row, ascending
    return sorted(b for (sc, b) in G if sc == s and b in O[s])
def envelope(p, sign):
    out = []
    for x, y in sorted(p):
        if not out or sign * y > sign * out[-1][1]: out.append((x, y))
    return out
fig, axs = plt.subplots(4, 3, figsize=(7.1, 8.6))
for r, (name, S, xl, tk) in enumerate((('Mip-NeRF 360 (9 scenes)', MIP, (3.3, 520), [5, 10, 20, 50, 100, 200, 500]), ('Tanks and Temples (2 scenes)', TT, (3.3, 520), [5, 10, 20, 50, 100, 200, 500]),
                                       ('Deep Blending (2 scenes)', DB, (3.3, 520), [5, 10, 20, 50, 100, 200, 500]), ('NeRF-Synthetic (8 objects)', OBJ, (0.9, 60), [1, 2, 5, 10, 20, 50]))):
    L = [levels(s) for s in S]; n = min(len(x) for x in L); assert n >= 2, (name, [len(x) for x in L])
    T = sorted(set.intersection(*[set(H[s]) for s in S]))
    print(name, '| cap levels per scene', [len(x) for x in L], '-> used', n, '| targets', T)
    for c, (lab, sign) in enumerate((('PSNR (dB)', 1), ('SSIM', 1), ('LPIPS (lower is better)', -1))):
        ax = axs[r][c]
        for k in ('draco', 'zstd', 'tfan'):
            pts = [(mean([G[(s, L[i][j])][b][k] for i, s in enumerate(S)]), mean([G[(s, L[i][j])][b]['q'][c] for i, s in enumerate(S)])) for j in range(n) for b in (8, 6, 4) if all(b in G[(s, L[i][j])] for i, s in enumerate(S))]
            e = envelope(pts, sign); ax.plot([p[0] for p in pts], [p[1] for p in pts], ls='none', marker=MK[k], ms=2.6, color=COL[k], alpha=0.3, mec='none')
            ax.plot([p[0] for p in e], [p[1] for p in e], color=COL[k], lw=1.3, marker=MK[k], ms=3.4, mec=SURF, mew=0.5)
        cp = [(mean([O[s][L[i][j]][0] for i, s in enumerate(S)]), mean([O[s][L[i][j]][1][c] for i, s in enumerate(S)])) for j in range(n)]
        bm = [(mean([H[s][t][0] for s in S]), mean([H[s][t][1][c] for s in S])) for t in T]
        ax.plot([p[0] for p in cp], [p[1] for p in cp], color=COL['cap'], lw=1.7, marker=MK['cap'], ms=4.2, mec=SURF, mew=0.6, zorder=6)
        ax.plot([p[0] for p in bm], [p[1] for p in bm], color=COL['bm'], lw=1.9, marker=MK['bm'], ms=4.4, mec=SURF, mew=0.6, zorder=7)
        ax.set_xscale('log'); ax.set_xlim(*xl); ax.xaxis.set_major_locator(mt.FixedLocator(tk)); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: '%g' % v)); ax.xaxis.set_minor_formatter(mt.NullFormatter())
        ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=2); ax.set_ylabel(lab)
        for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
        if c == 1: ax.set_title(name, fontsize=8, color=INK, pad=4)
        if r == 3: ax.set_xlabel('file size (MB, log scale)', color=MUTED)
LG = lambda k, lab: Line2D([], [], color=COL[k], marker=MK[k], ms=4.5, mec=SURF, mew=0.6, lw=1.7, label=lab)
fig.legend(handles=[LG('bm', 'ByteMesh (ours)'), LG('cap', 'MeshSplatting-cap + BM-Codec'), LG('tfan', '+ MPEG TFAN'), LG('zstd', '+ zstd'), LG('draco', '+ Draco')], loc='upper center', ncol=5, frameon=False, fontsize=7.5, handlelength=2.0, columnspacing=1.6)
fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.2, w_pad=1.0); fig.savefig('paper/figures/fig_rd_all_v1.pdf'); fig.savefig('paper/figures/fig_rd_all_v1.png', dpi=130); print('ok')
