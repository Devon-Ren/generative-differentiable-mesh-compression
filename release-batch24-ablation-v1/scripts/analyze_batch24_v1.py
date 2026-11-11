#!/usr/bin/env python3
"""analyze_batch24_v1.py (2026-11-11): the training-rule ablation table from the batch24 rows.  Run from the repo root.
  python3 release-batch24-ablation-v1/scripts/analyze_batch24_v1.py <collect.txt> [<collect2.txt> ...]
Rows: 'abl24-<variant>' rows (ablation-b24.txt / -blender.txt), the paper's controller rows (adapt-v9b*.txt, batch21 collect) as 'full'.
Quality at the same file size: each scene's constant-price curve = log-linear interpolation of the scene's two abl24 constlam points
(extrapolated beyond them, marked *).  dPSNR(variant) = PSNR - curve(file size); 'vs full' = dPSNR(variant) - dPSNR(full at its own size)."""
import re, sys, math, statistics as st, collections
MIP = ['room', 'kitchen', 'counter', 'bonsai', 'garden', 'bicycle', 'stump', 'flowers', 'treehill', 'truck', 'train']; BL = ['lego', 'chair', 'drums', 'ficus', 'hotdog', 'materials', 'mic', 'ship']
R = re.compile(r'^(\w+)(?: abl24-(\S+))? \| target ([\d.]+) MB \|.*?delivered ([\d.]+) MB \(hit ([\d.]+).*?\| test ([\d.]+) ([\d.]+) ([\d.]+)')
rows = collections.defaultdict(dict)   # (scene, target) -> variant -> dict
for f in sys.argv[1:] + ['release-adapt-v9b/results/adapt-v9b-2026-11-08.txt', 'release-adapt-v9b/results/adapt-v9b-blender-2026-11-08.txt', 'release-batch21-v1/results/batch21-collect-2026-11-09-2110-raw.txt']:
    for l in open(f):
        m = R.match(l)
        if not m: continue
        s, var, t, mb, hit, p, ss, lp = m[1], m[2] or 'full', float(m[3]), float(m[4]), float(m[5]), float(m[6]), float(m[7]), float(m[8])
        if s not in MIP + BL: continue
        rows[(s, t)].setdefault(var, dict(mb=mb, hit=hit, psnr=p, ssim=ss, lpips=lp))   # first occurrence wins (the paper's row)
curve = {}  # scene -> sorted [(mb, psnr)] from abl24 constlam rows
for (s, t), d in rows.items():
    for var, r in d.items():
        if var.startswith('constlam-lam'): curve.setdefault(s, []).append((r['mb'], r['psnr']))
for s in curve: curve[s].sort()
def on_curve(s, mb):
    pts = curve.get(s)
    if not pts or len(pts) < 2: return None, True
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    if len(pts) > 2:
        for a, b in zip(pts, pts[1:]):
            if a[0] <= mb <= b[0]: (x0, y0), (x1, y1) = a, b
    f = (math.log(mb) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return y0 + f * (y1 - y0), not (x0 <= mb <= x1)
def fmt(v, n=2, sign=True): return ('%+.*f' if sign else '%.*f') % (n, v)
out = []
for name, scenes, targets in [('Mip-NeRF 360 / T&T', MIP, [10.0, 20.0]), ('NeRF-Synthetic', BL, [2.5, 1.5])]:
    for t in targets:
        have = [s for s in scenes if (s, t) in rows and 'full' in rows[(s, t)]]
        variants = ['full', 'nocollapse', 'constants', 'matched']
        out.append(f'\n## {name}, target {t:g} MB ({len(have)} scenes with a controller row)\n')
        out.append('| variant | scenes | size error mean | range | within 10 % | dPSNR vs constant-price curve: mean (range) | vs full: mean (range), better |')
        out.append('|---|---|---|---|---|---|---|')
        dfull = {}
        for s in have:
            r = rows[(s, t)]['full']; c, ex = on_curve(s, r['mb']); dfull[s] = (r['psnr'] - c) if c is not None else None
        for var in variants:
            sc = [s for s in have if var in rows[(s, t)]]
            if not sc: out.append(f'| {var} | 0 | not run yet | | | | |'); continue
            errs = [rows[(s, t)][var]['hit'] - 1 for s in sc]
            d = []; rel = []; ext = 0
            for s in sc:
                r = rows[(s, t)][var]; c, ex = on_curve(s, r['mb'])
                if c is None: continue
                ext += ex; d.append(r['psnr'] - c)
                if dfull.get(s) is not None: rel.append(r['psnr'] - c - dfull[s])
            dcol = f'{fmt(st.mean(d))} ({fmt(min(d))} .. {fmt(max(d))}){"*" if ext else ""}' if d else '-'
            rcol = ('0 (reference)' if var == 'full' else (f'{fmt(st.mean(rel))} ({fmt(min(rel))} .. {fmt(max(rel))}), {sum(1 for x in rel if x > 0)}/{len(rel)}' if rel else '-'))
            out.append(f'| {var} | {len(sc)} | {fmt(100 * st.mean(errs), 1)} % | {fmt(100 * min(errs), 1)} .. {fmt(100 * max(errs), 1)} % | {sum(1 for e in errs if abs(e) <= 0.10)}/{len(errs)} | {dcol} | {rcol} |')
        # matched vs nocollapse (same split counts, both without collapse): direct PSNR difference and size
        pairs = [(s, rows[(s, t)]['matched'], rows[(s, t)]['nocollapse']) for s in have if 'matched' in rows[(s, t)] and 'nocollapse' in rows[(s, t)]]
        if pairs:
            dd = [n['psnr'] - m['psnr'] for s, m, n in pairs]; dm = [n['mb'] / m['mb'] - 1 for s, m, n in pairs]
            out.append(f'\nrate-aware ranking vs MeshSplatting importance sampling at the same split counts (both without collapse): nocollapse - matched = {fmt(st.mean(dd))} dB mean ({fmt(min(dd))} .. {fmt(max(dd))}), rate-aware better on {sum(1 for x in dd if x > 0)}/{len(dd)}; file size of the rate-aware mesh / importance mesh - 1 = {fmt(100 * st.mean(dm), 1)} % ({fmt(100 * min(dm), 1)} .. {fmt(100 * max(dm), 1)} %)')
    # constant-price rows: spread of file size across scenes at one price
    out.append(f'\n### constant price, {name}: file size spread across scenes at one price (abl24 constlam rows)\n')
    lam_rows = collections.defaultdict(list)
    for (s, t), d in rows.items():
        if s in scenes:
            for var, r in d.items():
                if var.startswith('constlam-lam'): lam_rows[var].append((s, r['mb'], r['psnr']))
    for var, lst in sorted(lam_rows.items()):
        mbs = [x[1] for x in lst]; out.append(f'- {var}: {len(lst)} scenes, file size {min(mbs):.1f} .. {max(mbs):.1f} MB (max/min {max(mbs)/min(mbs):.1f}x)')
out.append('\n### per-scene rows (file size MB / PSNR / size error; dPSNR vs the scene\'s constant-price curve)\n')
for (s, t) in sorted(rows, key=lambda k: (k[0] not in MIP, MIP.index(k[0]) if k[0] in MIP else BL.index(k[0]), k[1])):
    d = rows[(s, t)]
    if 'full' not in d or not any(v != 'full' and not v.startswith('constlam') for v in d): continue
    cells = []
    for var in ['full', 'nocollapse', 'constants', 'matched']:
        if var in d:
            r = d[var]; c, ex = on_curve(s, r['mb']); cells.append(f"{var}: {r['mb']:.2f} / {r['psnr']:.2f} / {fmt(100*(r['hit']-1),1)} % / {fmt(r['psnr']-c) if c is not None else '-'}{'*' if ex else ''}")
    cl = [f"{v[12:]}: {r['mb']:.2f} / {r['psnr']:.2f}" for v, r in sorted(d.items()) if v.startswith('constlam')]
    out.append(f"- {s} {t:g} MB: " + ' | '.join(cells) + (' | constlam ' + ', '.join(cl) if cl else ''))
print('\n'.join(out))
