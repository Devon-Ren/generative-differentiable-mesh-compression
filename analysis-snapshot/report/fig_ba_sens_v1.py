"""fig_ba_sens_v1 (11/06): B_a sensitivity. Two panels (kitchen, garden): the bpv-10 lambda curve (green, 4 lambda) and the bpv 8 / 13 runs at
lambda 6.2e-3 and 6.2e-4 (hollow / filled purple circles), labelled with B_v. x = delivered MB (log)."""
import sys, csv; sys.path.insert(0, '/mnt/project-files/analysis/report'); from figstyle_v2 import *
rc()
fan = [r for r in csv.DictReader(open(D + 'lambda-k-fan-v2.csv')) if r['base'] == 'True']
bs = list(csv.DictReader(open(D + 'ba-sens-v1.csv')))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
for ax, S, title in zip(axes, ('kitchen', 'garden'), ('(a) kitchen', '(b) garden')):
    base = sorted([(float(r['MB']), float(r['psnr']), float(r['lam'])) for r in fan if r['scene'] == S])
    smooth(ax, [(m, p) for m, p, _ in base], label='B$_v$ = 10 B (default): λ curve')
    for m, p, l in base: ax.annotate(f"λ {l:g}", (m, p), textcoords='offset points', xytext=(6, -8), ha='left', va='top', fontsize=6.8, color=OURS)
    for bpv, mfc, lab in ((8.0, 'white', 'B$_v$ = 8 B'), (13.0, GEN, 'B$_v$ = 13 B')):
        q = [(float(r['MB']), float(r['psnr']), float(r['lam'])) for r in bs if r['scene'] == S and float(r['bpv']) == bpv]
        ax.plot([x[0] for x in q], [x[1] for x in q], 'o', color=GEN, mfc=mfc, ms=6, mew=1.3, zorder=7, label=lab + ' at λ 6.2e-3 / 6.2e-4')
    lo = base[0][0] * 0.75; hi = max(base[-1][0], max(float(r['MB']) for r in bs if r['scene'] == S)) * 1.3
    logx(ax, lo, hi, ticks=(3, 5, 10, 20, 30, 50)); ax.set_xlabel('delivered size (MB, log)'); ax.set_title(title, loc='left')
    if S == 'kitchen': ax.set_ylabel('test PSNR (dB)')
    better(ax)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc='lower center', ncol=3, fontsize=7.4, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, 'fig-ba-sens-v1')
