"""Build the 11-scene 6-point RD table (2026-11-01): original FP32, level-1.0 codec point, v3 growth+prune at lambda
6.2e-4 / 1.85e-3 / 6.2e-3 / 1.85e-2.  Reads rd-11scenes-v1.csv (collapse + lambda-v2 + v3 two lambdas) and
rd-prune3-hull2-2026-11-01.csv (two new lambdas).  Writes rd-11scenes-v2.csv (plot input) and rd-hull6-v1-table.md."""
import csv
D = '/mnt/project-files/analysis/rd/'
ORIG = {'room': (28.263, 741.7), 'counter': (25.944, 575.0), 'kitchen': (26.454, 456.0), 'bonsai': (27.291, 662.2),
        'garden': (24.732, 739.8), 'bicycle': (23.033, 680.0), 'stump': (24.928, 698.0), 'flowers': (19.451, 667.7),
        'treehill': (20.387, 691.8), 'truck': (22.387, 432.4), 'train': (18.900, 484.7)}
LAMS = ['0.00062', '0.00185', '0.0062', '0.0185']
rows = list(csv.DictReader(open(D + 'rd-11scenes-v1.csv')))
for r in csv.DictReader(open(D + 'rd-prune3-hull2-2026-11-01.csv')):
    rows.append(dict(scene=r['scene'], series='v3', label='λ ' + r['lam'], MB=r['MB'], test=r['test'], lpips=r['lpips']))
with open(D + 'rd-11scenes-v2.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, ['scene', 'series', 'label', 'MB', 'test', 'lpips']); w.writeheader(); w.writerows(rows)

def pt(sc, series, label):
    for r in rows:
        if r['scene'] == sc and r['series'] == series and r['label'] == label:
            return float(r['MB']), float(r['test'])
    return None

L = ['# 11 scenes × 6 points (FP32 original, level-1.0 codec, v3 λ 6.2e-4 / 1.85e-3 / 6.2e-3 / 1.85e-2), 2026-11-01',
     '', 'Codec K4096 R256, scene XYZ (room q14; garden/bonsai/counter/flowers q15; kitchen/bicycle/stump/treehill q16; truck/train q18), DC q7. '
     'ratio = FP32 MB / delivered MB; Δ = test PSNR − FP32 original test PSNR. Garden level-1.0 is the q14 point of 10/16.',
     '', '| scene | FP32 | level 1.0 | λ 6.2e-4 | λ 1.85e-3 | λ 6.2e-3 | λ 1.85e-2 |', '|---|---|---|---|---|---|---|']
agg = {k: [] for k in LAMS}; l1 = []
for sc, (o, fp) in ORIG.items():
    cells = [f'{fp:.0f} MB / {o:.2f}']
    v10 = pt(sc, 'collapse', 'v1.0') or pt(sc, 'collapse', 'v1.0 q14')
    cells.append(f'{v10[0]:.1f} MB {fp / v10[0]:.0f}× {v10[1]:.2f} ({v10[1] - o:+.2f})'); l1.append((v10[0], fp / v10[0], v10[1] - o))
    for lam in LAMS:
        p = pt(sc, 'v3', 'λ ' + lam)
        cells.append(f'{p[0]:.1f} MB {fp / p[0]:.0f}× {p[1]:.2f} ({p[1] - o:+.2f})'); agg[lam].append((p[0], fp / p[0], p[1] - o, sc))
    L.append(f'| {sc} | ' + ' | '.join(cells) + ' |')
def mean(v, i): return sum(a[i] for a in v) / len(v)
L += ['', '**Means over 11 scenes**', '', '| point | MB | ratio | Δ dB | ratio range | Δ range |', '|---|---|---|---|---|---|',
      f'| level 1.0 | {mean(l1, 0):.1f} | {mean(l1, 1):.0f}× | {mean(l1, 2):+.2f} | {min(a[1] for a in l1):.0f}–{max(a[1] for a in l1):.0f}× | {min(a[2] for a in l1):+.2f}…{max(a[2] for a in l1):+.2f} |']
for lam in LAMS:
    v = agg[lam]
    L.append(f'| λ {lam} | {mean(v, 0):.1f} | {mean(v, 1):.0f}× | {mean(v, 2):+.2f} | {min(a[1] for a in v):.0f}–{max(a[1] for a in v):.0f}× | {min(a[2] for a in v):+.2f}…{max(a[2] for a in v):+.2f} |')
# Pareto / monotonicity check of the v3 series per scene
L += ['', '**Dominated points** (a smaller v3 asset with ≥ test PSNR exists in the same scene):', '']
for sc in ORIG:
    pts = sorted([(float(r['MB']), float(r['test']), r['label']) for r in rows if r['scene'] == sc and r['series'] == 'v3'])
    dom = [p[2] for i, p in enumerate(pts) if any(q[1] >= p[1] for q in pts[:i])]
    if dom: L.append(f'- {sc}: ' + ', '.join(dom))
M360 = [s for s in ORIG if s not in ('truck', 'train')]
L += ['', '**Means over the 9 Mip-360 scenes**', '']
for lam in LAMS:
    v = [a for a in agg[lam] if a[3] in M360]
    L.append(f'- λ {lam}: {mean(v, 0):.1f} MB, {mean(v, 1):.0f}×, {mean(v, 2):+.2f} dB')
open(D + 'rd-hull6-v1-table.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
