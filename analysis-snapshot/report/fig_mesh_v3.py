"""v3 (11/05, Yao's comment on Fig 4): panel (a) only -- (b) 'size needed for the same PSNR' dropped. Every published point is that method's own reported single asset (no rate control); ours is the only curve."""
import sys; sys.path.insert(0, '/mnt/project-files/analysis/report'); from figstyle_v2 import *
rc(); rows, _ = load_rows()
lam = mean_lambda(rows, M360); l1 = mean_level1(rows, M360); o_mb, o_psnr = mean_orig(M360)
MESH = [("MeshSplatting, authors' PLY (SH dropped)", 100, 24.78, 2025), ('MiLo', 253, 24.09, 2025), ('Binary Opacity Grids', 913, 25.62, 2024), ('RaDe-GS mesh', 1100, 23.56, 2024), ('GOF mesh', 1500, 20.78, 2024)]
fig, a = plt.subplots(1, 1, figsize=(6.0, 3.9))
smooth(a, lam, label='ours (4 λ, mean of 9 scenes)')
for i, (x, y) in enumerate(lam): label_ratio(a, x, y, o_mb / x, dy=(9 if i == 0 else -11))
a.plot(l1[0], l1[1], 'o', color=POST, ms=5.5, mec='white', mew=1, zorder=6, label='ours, codec only (all vertices)')
a.plot(o_mb, o_psnr, '*', color=STAR, ms=12, zorder=6); a.annotate('MeshSplatting FP32\n(our reproduction)', (o_mb, o_psnr), textcoords='offset points', xytext=(-9, 9), ha='right', va='bottom', fontsize=7.2, color=INK)
for name, mb, p, yr in MESH:
    dx, dy, ha = (7, -11, 'left') if name.startswith('MeshSplatting') else (-7, 0, 'right')
    point(a, mb, p, name, dx=dx, dy=dy, ha=ha)
logx(a, 2.5, 2600); a.set_ylim(20.3, 26.6); a.set_xlabel('asset size (MB, log)'); a.set_ylabel('test PSNR (dB), Mip-NeRF 360 mean'); a.set_title('engine-ready meshes: size vs quality (published single assets vs our λ curve)', loc='left', fontsize=8.5); better(a, text='better')
a.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.0, 0.08))
fig.tight_layout(); save(fig, 'fig-mesh-class-v3')
