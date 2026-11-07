"""fig_mean_rd_v1 (11/05, Yao: show the average RD curves next to Table 1): two panels, Mip-NeRF 360 9-scene mean and T&T 2-scene mean; x = delivered MB (log), y = test PSNR; our 4 lambda points (curve), codec-only point, FP32 star."""
import sys; sys.path.insert(0, '/mnt/project-files/analysis/report'); from figstyle_v2 import *
rc(); rows, _ = load_rows()
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), gridspec_kw=dict(width_ratios=[1, 1]))
for ax, scenes, title in zip(axes, (M360, TNT), ('(a) Mip-NeRF 360, mean of 9 scenes', '(b) Tanks & Temples, mean of truck and train')):
    lam = mean_lambda(rows, scenes); o_mb, o_psnr = mean_orig(scenes)
    # Pareto hull through the non-dominated lambda points; dominated points (a larger asset with no higher PSNR) hollow
    pts = sorted(lam); hull = []; best = -1e9
    for x, y in pts:
        if y > best: hull.append((x, y)); best = y
    dom = [p for p in pts if p not in hull]
    smooth(ax, hull, label='ours, 4 λ (same codec, K4096 R256, DC 7 bit)')
    for x, y in dom: ax.plot(x, y, 'o', color=OURS, mfc='white', ms=5.5, mew=1.3, zorder=6); ax.annotate('dominated', (x, y), textcoords='offset points', xytext=(6, 0), ha='left', va='center', fontsize=7, color=OURS)
    try:
        l1 = mean_level1(rows, scenes); ax.plot(l1[0], l1[1], 'o', color=POST, ms=5.5, mec='white', mew=1, zorder=6, label='ours, codec only (all vertices)')
    except Exception as e: print('no level-1.0 mean for', scenes, e)
    ax.plot(o_mb, o_psnr, '*', color=STAR, ms=12, zorder=6); ax.annotate('FP32 (our reproduction)', (o_mb, o_psnr), textcoords='offset points', xytext=(0, -11), ha='center', va='top', fontsize=7.2, color=INK)
    ax.axhline(o_psnr, color='#d0d0d0', lw=0.7, ls=':', zorder=1)
    logx(ax, 2.5, 1100); ax.set_xlabel('delivered size (MB, log)'); ax.set_ylabel('test PSNR (dB)'); ax.set_title(title, loc='left'); better(ax, text='better')
    ax.legend(frameon=False, loc='lower right', fontsize=7.4)
fig.tight_layout(); save(fig, 'fig-mean-rd-v1')
