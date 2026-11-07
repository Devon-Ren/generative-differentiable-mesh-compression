"""Report figure 1 (2026-11-01, v2, +MPEG row): the same FP32 MeshSplatting asset (Room, 741.7 MB) through every codec.  MPEG SC3DMC/TFAN (Open3DGC) row added 11/01, GPU eval pending (PSNR by construction = quant PLY row)."""
import sys; sys.path.insert(0, '/mnt/project-files/analysis/report'); from figstyle_v1 import *
rc(); rows, _ = load_rows(); g = generic_room()
fig, ax = plt.subplots(figsize=(5.2, 3.6))
GL = [('quant_ply_zstd19', 'quantised PLY + zstd', 0, 10, 'center'), ('draco_full_sh8', 'Draco,\nSH as attributes', 0, -16, 'center'), ('draco_rgb8', 'Draco, RGB only (SH dropped)', -8, 0, 'right'), ('texture_8192', 'UV atlas + 8k JPEG (SH dropped)', 8, 0, 'left')]
for key, name, dx, dy, ha in GL: point(ax, g[key][0], g[key][1], name, color=GEN, hollow=False, dx=dx, dy=dy, ha=ha)
# MPEG-4 Part 16 SC3DMC/TFAN via Open3DGC (11/01): same 8-bit SH values and 14-bit positions as the quantised-PLY row, lossless int attributes -> same quality by construction (GPU check pending)
MPEG_MB = 107.981; point(ax, MPEG_MB, g['quant_ply_zstd19'][1], 'MPEG SC3DMC/TFAN\n(Open3DGC)', color=GEN, hollow=False, dx=0, dy=-16, ha='center')
rl = [ours_lambda(rows, 'room', l) for l in LAMS]; l1 = level1(rows, 'room')
smooth(ax, rl, label='ours: rate-controlled growth + λ-prune')
for x, y in rl: label_ratio(ax, x, y, 741.7 / x)
point(ax, l1[0], l1[1], 'ours, codec only', color=POST, hollow=False, dx=7, dy=7, ms=5.5)
ax.plot(741.7, 28.263, '*', color=STAR, ms=12, zorder=6); ax.annotate('FP32 asset, 742 MB', (741.7, 28.263), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=7.4, color=INK)
logx(ax, 2.2, 1300); ax.set_ylim(24.8, 29.1); ax.set_xlabel('delivered size (MB, log)'); ax.set_ylabel('test PSNR (dB)')
ax.set_title('One asset, every codec (Room, Mip-NeRF 360)', loc='left'); better(ax, text='better (smaller, higher PSNR)')
ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.0, 0.14), handlelength=2.2)
fig.tight_layout(); save(fig, 'fig-codecs-room-v2')
