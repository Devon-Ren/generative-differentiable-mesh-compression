"""fig_adapt_v1 (11/06): rate control (train_rd_adapt_v1) operating points on the fixed-lambda curves. Four panels (room, kitchen, garden, bicycle):
the lambda-only curve (K4096 R256, DC 7 bit; green, 4 lambda), the three rate-controlled runs (orange squares, labelled with their target), and the
targets as vertical dotted lines. x = delivered MB (log)."""
import sys, csv; sys.path.insert(0, '/mnt/project-files/analysis/report'); from figstyle_v2 import *
rc()
fan = [r for r in csv.DictReader(open(D + 'lambda-k-fan-v2.csv')) if r['base'] == 'True']
ad = list(csv.DictReader(open(D + 'adapt-v1.csv')))
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.3))
for ax, S, title in zip(axes, ('room', 'kitchen', 'garden', 'bicycle'), ('(a) room', '(b) kitchen', '(c) garden', '(d) bicycle')):
    base = sorted([(float(r['MB']), float(r['psnr']), float(r['lam'])) for r in fan if r['scene'] == S])
    smooth(ax, [(m, p) for m, p, _ in base], label='fixed λ (K4096 R256, DC 7 bit)')
    for m, p, l in base: ax.annotate(f"λ {l:g}", (m, p), textcoords='offset points', xytext=(6, -8), ha='left', va='top', fontsize=6.8, color=OURS)
    pts = sorted([(float(r['delivered']), float(r['psnr']), float(r['target']), float(r['lam_final'])) for r in ad if r['scene'] == S])
    ax.plot([q[0] for q in pts], [q[1] for q in pts], 's', color=POST, ms=6, mec='white', mew=0.8, zorder=7, label='rate-controlled λ (target 5 / 10 / 20 MB)')
    for m, p, t, l in pts:
        ax.axvline(t, color=POST, ls=':', lw=0.8, zorder=1)
        ax.annotate(f"target {t:g} MB\n{m:.1f} MB", (m, p), textcoords='offset points', xytext=(-6, 7), ha='right', va='bottom', fontsize=6.6, color=POST)
    lo = min(min(q[0] for q in pts), base[0][0]) * 0.7; hi = max(base[-1][0], 20) * 1.3
    logx(ax, lo, hi, ticks=(2, 3, 5, 10, 20, 30, 50)); ax.set_xlabel('delivered size (MB, log)'); ax.set_title(title, loc='left')
    if S == 'room': ax.set_ylabel('test PSNR (dB)')
    better(ax)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc='lower center', ncol=3, fontsize=7.6, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, 'fig-adapt-v1')
