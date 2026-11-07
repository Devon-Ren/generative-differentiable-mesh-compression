"""Report figure 2 (2026-11-01, v1): among engine-ready meshes.  (a) size vs PSNR, Mip-NeRF 360 9-scene mean, published methods as labelled points;
(b) 'size for the same quality': each published mesh vs the size of OUR asset at the same PSNR (PCHIP through our 4 lambda points + FP32, in log MB)."""
import sys; sys.path.insert(0, '/mnt/project-files/analysis/report'); from figstyle_v1 import *
rc(); rows, _ = load_rows()
lam = mean_lambda(rows, M360); l1 = mean_level1(rows, M360); o_mb, o_psnr = mean_orig(M360)
MESH = [("MeshSplatting, authors' PLY (SH dropped)", 100, 24.78, 2025), ('MiLo', 253, 24.09, 2025), ('Binary Opacity Grids', 913, 25.62, 2024), ('RaDe-GS mesh', 1100, 23.56, 2024), ('GOF mesh', 1500, 20.78, 2024)]
fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
smooth(a, lam, label='ours (4 λ, mean of 9 scenes)')
for i, (x, y) in enumerate(lam): label_ratio(a, x, y, o_mb / x, dy=(9 if i == 0 else -11))
a.plot(l1[0], l1[1], 'o', color=POST, ms=5.5, mec='white', mew=1, zorder=6, label='ours, codec only (all vertices)')
a.plot(o_mb, o_psnr, '*', color=STAR, ms=12, zorder=6); a.annotate('MeshSplatting FP32\n(our reproduction)', (o_mb, o_psnr), textcoords='offset points', xytext=(9, 0), ha='left', va='center', fontsize=7.2, color=INK)
for name, mb, p, yr in MESH:
    dx, dy, ha = (7, 0, 'left') if name not in ('GOF mesh', 'RaDe-GS mesh', 'Binary Opacity Grids', 'MiLo') else (-7, 0, 'right')
    point(a, mb, p, name, dx=dx, dy=dy, ha=ha)
logx(a, 2.5, 2600); a.set_ylim(20.3, 26.6); a.set_xlabel('asset size (MB, log)'); a.set_ylabel('test PSNR (dB), Mip-NeRF 360 mean'); a.set_title('(a) engine-ready meshes: size vs quality', loc='left'); better(a, text='better')
a.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.0, 0.08))
# (b) size for the same quality: our curve = PCHIP through the 4 lambda points (log MB -> PSNR)
lam_s = sorted(lam); xs = np.log10([q[0] for q in lam_s]); ys = np.array([q[1] for q in lam_s]); f = PchipInterpolator(xs, ys); grid = np.linspace(xs[0], xs[-1], 4000); v = f(grid)
rowsb = [('MeshSplatting FP32 (our reproduction)', o_mb, o_psnr), ("MeshSplatting, authors' PLY (SH dropped)", 100, 24.78), ('Binary Opacity Grids', 913, 25.62), ('MiLo', 253, 24.09), ('RaDe-GS mesh', 1100, 23.56), ('GOF mesh', 1500, 20.78)]
yy = np.arange(len(rowsb))[::-1]
for k, (name, mb, p) in enumerate(rowsb):
    y = yy[k]
    if ys[0] <= p <= ys[-1]: s_ = 10 ** grid[int(np.argmin(np.abs(v - p)))]; txt = f'{mb / s_:.0f}× smaller at equal PSNR'
    elif p > ys[-1]: s_ = lam_s[-1][0]; txt = f'{mb / s_:.0f}× smaller at −{p - ys[-1]:.2f} dB (our best point)'
    else: s_ = lam_s[0][0]; txt = f'{mb / s_:.0f}× smaller and +{ys[0] - p:.1f} dB (our smallest point)'
    b.plot([s_, mb], [y, y], '-', color='#d9d9d9', lw=2.2, zorder=2); b.plot(mb, y, 'o', color=INK, mfc='white', ms=7, mew=1.3, zorder=4); b.plot(s_, y, 'o', color=OURS, ms=7, mec='white', mew=1, zorder=5)
    b.annotate(txt, (s_, y), textcoords='offset points', xytext=(-8, 0), ha='right', va='center', fontsize=7.4, color=OURS)
    b.annotate(f'{name}  {mb:.0f} MB @ {p:.2f} dB', (mb, y), textcoords='offset points', xytext=(0, 9), ha='center', va='bottom', fontsize=7.2, color=INK)
logx(b, 2.5, 2600); b.set_yticks([]); b.set_ylim(-0.7, len(rowsb) - 0.3); b.set_xlabel('asset size (MB, log)')
for sp in ('left',): b.spines[sp].set_visible(False)
b.set_title('(b) size needed for the same PSNR: published mesh (black) vs ours (green)', loc='left')
fig.tight_layout(); save(fig, 'fig-mesh-class-v1')
