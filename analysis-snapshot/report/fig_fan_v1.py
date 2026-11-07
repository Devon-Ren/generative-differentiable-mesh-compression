"""fig_fan_v1 (11/05): lambda x dictionary fan on room / kitchen / garden. Per panel: the lambda-only curve (K4096 R256, DC 7 bit; green),
the five re-export levers at every lambda (small markers, one colour per lever), and the joint hull over all 24 points (dashed). x = delivered MB (log)."""
import sys, csv; sys.path.insert(0, '/mnt/project-files/analysis/report'); from figstyle_v2 import *
def hull(pts):
    pts = sorted(pts, key=lambda r: (r['MB'], -r['psnr'])); par = []; best = -1e9
    for r in pts:
        if r['psnr'] > best: par.append(r); best = r['psnr']
    h = []
    for r in par:
        while len(h) >= 2 and (h[-1]['psnr'] - h[-2]['psnr']) / (h[-1]['MB'] - h[-2]['MB']) <= (r['psnr'] - h[-1]['psnr']) / (r['MB'] - h[-1]['MB']): h.pop()
        h.append(r)
    return h
rc()
rows = list(csv.DictReader(open(D + 'lambda-k-fan-v1.csv')))
for r in rows: r['MB'] = float(r['MB']); r['psnr'] = float(r['psnr']); r['lam'] = float(r['lam']); r['base'] = r['base'] == 'True'
LEV = [('K4096 R1 q7', '#b8336a', 'K4096 R1 (no residual)'), ('K1024 R256 q7', '#e8702a', 'K1024 R256'), ('K4096 R256 q8', '#2a6fd4', 'K4096 R256, DC 8 bit'),
       ('K4096 R1024 q7', '#7a52c7', 'K4096 R1024'), ('K16384 R1024 q7', '#0f7c8c', 'K16384 R1024')]
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
for ax, S, title in zip(axes, ('room', 'kitchen', 'garden'), ('(a) room', '(b) kitchen', '(c) garden')):
    pts = [r for r in rows if r['scene'] == S]; base = sorted([r for r in pts if r['base']], key=lambda r: r['MB'])
    H = hull(pts)
    ax.plot([r['MB'] for r in H], [r['psnr'] for r in H], '--', color=INK, lw=1.0, zorder=3, label='joint hull (all 24 points)')
    smooth(ax, [(r['MB'], r['psnr']) for r in base], label='λ only: K4096 R256, DC 7 bit')
    for key, col, lab in LEV:
        q = [r for r in pts if r['lever'] == key]
        ax.plot([r['MB'] for r in q], [r['psnr'] for r in q], 'o', color=col, ms=4, mec='white', mew=0.6, zorder=6, label=lab)
    for r in base: ax.annotate(f"λ {r['lam']:g}", (r['MB'], r['psnr']), textcoords='offset points', xytext=(7, -7), ha="left", va="top", fontsize=6.8, color=OURS)
    lo = min(r['MB'] for r in pts) * 0.8; hi = max(r['MB'] for r in pts) * 1.25
    logx(ax, lo, hi, ticks=(2, 3, 5, 10, 20, 30, 50)); ax.set_xlabel('delivered size (MB, log)'); ax.set_title(title, loc='left')
    if S == 'room': ax.set_ylabel('test PSNR (dB)')
    better(ax)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc='lower center', ncol=7, fontsize=7.2, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, 'fig-fan-v1')
