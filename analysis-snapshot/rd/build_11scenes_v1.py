"""Build the 11-scene RD table (2026-10-24): original, post-hoc collapse curve (7 scenes) or level-1.0 point (4 new scenes),
v2 lambda-only (7 scenes), v3 growth+prune (11 scenes, lambda 1.85e-3 / 6.2e-3). Writes rd-11scenes-v1.csv and prints the summary."""
import csv
D = '/mnt/project-files/analysis/rd/'
# scene: (original test PSNR, SSIM, LPIPS or None, FP32 MB = V*208 + F*12)
ORIG = {'room': (28.263, 0.8833, 0.3389, 741.7), 'garden': (24.732, 0.7624, 0.2485, 739.8), 'bonsai': (27.291, None, None, 662.2),
        'counter': (25.944, None, None, 575.0), 'kitchen': (26.454, None, None, 456.0), 'bicycle': (23.033, None, None, 680.0),
        'stump': (24.928, None, None, 698.0), 'flowers': (19.451, 0.4923, 0.4086, 667.7), 'treehill': (20.387, 0.5393, 0.4270, 691.8),
        'truck': (22.387, 0.8069, 0.2256, 432.4), 'train': (18.900, 0.7119, 0.3189, 484.7)}
NEW4L1 = {'flowers': (26.914, 18.949), 'treehill': (28.495, 20.27), 'truck': (20.036, 22.308), 'train': (22.316, 18.738)}
out = []
for r in csv.DictReader(open(D + 'lambda-vs-cap-7scenes-v1.csv')):
    if r['series'] in ('collapse', 'lambda'):
        out.append(dict(scene=r['scene'], series='collapse' if r['series'] == 'collapse' else 'lambda-v2', label=r['label'], MB=float(r['MB']), test=float(r['test_psnr'])))
for s, (mb, p) in NEW4L1.items():
    out.append(dict(scene=s, series='collapse', label='v1.0', MB=mb, test=p))
for f in ('rd-prune3-7scenes-2026-10-24.csv', 'rd-prune3-new4-2026-10-24.csv'):
    for r in csv.DictReader(open(D + f)):
        if r.get('scene'):
            out.append(dict(scene=r['scene'], series='v3', label='λ ' + r['lam'], MB=float(r['MB']), test=float(r['test']), lpips=float(r['lpips'])))
with open(D + 'rd-11scenes-v1.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, ['scene', 'series', 'label', 'MB', 'test', 'lpips']); w.writeheader(); w.writerows(out)
print(f"{'scene':9s} {'lam':8s} {'MB':>6s} {'ratio':>6s} {'dB':>6s}")
agg = {}
for sc in ORIG:
    o, _, _, fp = ORIG[sc]
    for r in sorted([r for r in out if r['scene'] == sc and r['series'] == 'v3'], key=lambda r: r['MB']):
        x = fp / r['MB']; d = r['test'] - o; agg.setdefault(r['label'], []).append((r['MB'], x, d, sc))
        print(f"{sc:9s} {r['label']:8s} {r['MB']:6.2f} {x:5.0f}x {d:+6.2f}")
for k, v in agg.items():
    n = len(v); print(k, 'n', n, 'mean MB %.2f ratio %.1f dB %.3f' % (sum(a[0] for a in v) / n, sum(a[1] for a in v) / n, sum(a[2] for a in v) / n),
                      '| ratio range %.0f-%.0f' % (min(a[1] for a in v), max(a[1] for a in v)))
