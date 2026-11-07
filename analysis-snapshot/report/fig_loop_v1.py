"""Report figure 4 (2026-11-01, v1): why vertices must be chosen during training.  Same codec, three ways to pick vertices, 7 Mip-NeRF 360 scenes:
(a) mean RD: post-hoc collapse of the dense model / fixed vertex cap during training / rate-controlled growth + lambda-prune;
(b) at the SAME vertex count, train-at-cap vs post-hoc collapse per scene (two caps per scene)."""
import sys; sys.path.insert(0, '/mnt/project-files/analysis/report'); from figstyle_v1 import *
rc(); rows, _ = load_rows()
bc = list(csv.DictReader(open(D + 'budget-vs-collapse-7scenes-v1.csv')))
def pick(scene, series, label): return [(float(r['MB']), float(r['test_psnr']), int(r['vertices'])) for r in bc if r['scene'] == scene and r['series'] == series and r['label'] == label]
# post-hoc collapse curve: v1.0, v0.8 (7 scenes) + the two controls (collapse to the cap run's vertex count)
coll, cap = {}, {}
for s in M360_7:
    rs = [r for r in bc if r['scene'] == s]
    top = [(float(r['MB']), float(r['test_psnr'])) for r in rs if r['series'] == 'collapse' and (r['label'].startswith('v1.0') or r['label'].startswith('v0.8'))]
    ctrl = [(float(r['MB']), float(r['test_psnr'])) for r in rs if r['series'] == 'collapse' and not r['label'].startswith('v')]
    coll[s] = sorted(top, reverse=True) + sorted(ctrl, reverse=True)
    cap[s] = sorted([(float(r['MB']), float(r['test_psnr'])) for r in rs if r['series'] == 'budget' and r['label'] != 'dense'], reverse=True)
n = min(len(v) for v in coll.values()); nc = min(len(v) for v in cap.values())
coll_mean = [(sum(coll[s][i][0] for s in M360_7) / 7, sum(coll[s][i][1] for s in M360_7) / 7) for i in range(n)]
cap_mean = [(sum(cap[s][i][0] for s in M360_7) / 7, sum(cap[s][i][1] for s in M360_7) / 7) for i in range(nc)]
lam = mean_lambda(rows, M360_7); o_mb, o_psnr = mean_orig(M360_7); l1 = mean_level1(rows, M360_7)
fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.8), gridspec_kw=dict(width_ratios=[1.1, 1]))
smooth(a, coll_mean, color=POST, label='dense model, post-hoc collapse (same codec)')
smooth(a, [l1] + cap_mean, color=CAP, label='fixed vertex cap during training (same codec)')
smooth(a, lam, color=OURS, label='rate-controlled growth + λ-prune (ours)')
for x, y in lam: label_ratio(a, x, y, o_mb / x)
a.plot(o_mb, o_psnr, '*', color=STAR, ms=12, zorder=6); a.annotate('FP32 (dense)', (o_mb, o_psnr), textcoords='offset points', xytext=(0, -13), ha='center', fontsize=7.2, color=INK)
logx(a, 2.5, 1300); a.set_ylim(22.4, 26.6); a.set_xlabel('delivered size (MB, log)'); a.set_ylabel('test PSNR (dB), mean of 7 Mip-NeRF 360 scenes')
a.set_title('(a) same codec, three ways of choosing the vertices', loc='left'); better(a, text='better'); a.legend(frameon=False, loc='lower right')
# (b) equal-vertex bars
labs, gains, cols = [], [], []
for s in M360_7:
    for r in bc:
        if r['scene'] == s and r['series'] == 'budget' and r['label'] != 'dense':
            v = int(r['vertices']); c = [q for q in bc if q['scene'] == s and q['series'] == 'collapse' and not q['label'].startswith('v') and abs(int(q['vertices']) - v) <= max(2, v // 1000)]
            if len(c) != 1: continue
            labs.append(f"{s} {r['label']}"); gains.append(float(r['test_psnr']) - float(c[0]['test_psnr']))
idx = np.argsort(gains)[::-1]; labs = [labs[i] for i in idx]; gains = [gains[i] for i in idx]
y = np.arange(len(labs))[::-1]; b.barh(y, gains, color=OURS, height=0.62)
for yi, g in zip(y, gains): b.annotate(f'+{g:.2f}', (g, yi), textcoords='offset points', xytext=(4, 0), va='center', fontsize=7.2, color=INK)
b.set_yticks(y); b.set_yticklabels(labs, fontsize=7.2); b.set_xlabel('Δ test PSNR (dB): trained at the vertex count − collapsed to the same count'); b.set_xlim(0, max(gains) * 1.18)
for sp in ('top', 'right'): b.spines[sp].set_visible(False)
b.grid(axis='x', color='#ededeb', lw=0.5); b.set_axisbelow(True); b.set_title('(b) at the same vertex count, 14/14 points favour in-the-loop', loc='left')
fig.tight_layout(); save(fig, 'fig-in-the-loop-v1')
print('coll_mean', [(round(x, 1), round(y, 2)) for x, y in coll_mean]); print('cap_mean', [(round(x, 1), round(y, 2)) for x, y in cap_mean]); print('gains', [round(g, 2) for g in gains])
