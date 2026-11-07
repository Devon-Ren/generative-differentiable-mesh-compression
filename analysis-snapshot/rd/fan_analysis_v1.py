"""lambda x dictionary fan analysis v1 (2026-11-05): per-scene joint hull over 4 lambda x {K4096R256 q7 (base), q8, K4096R1, K1024R256, K4096R1024, K16384R1024}."""
import re, csv, sys, collections
rows = []
for line in open('lambda-k-fan-v1-2026-11-05.txt'):
    p = [x.strip() for x in line.split('|')]
    S = p[0]; lam = float(p[1].split()[1]); lever = p[2].replace(' (base)', ''); mb = float(p[3].split()[0])
    t = p[4].split(); psnr, ssim, lp = float(t[1]), float(t[2]), float(t[3])
    rows.append(dict(scene=S, lam=lam, lever=lever, MB=mb, psnr=psnr, ssim=ssim, lpips=lp, base='(base)' in p[2]))
with open('lambda-k-fan-v1.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def hull(pts):  # upper-left Pareto convex hull in (MB, psnr): increasing MB, concave
    pts = sorted(pts, key=lambda r: (r['MB'], -r['psnr']))
    # pareto first
    par = []; best = -1e9
    for r in pts:
        if r['psnr'] > best: par.append(r); best = r['psnr']
    # convex (concave upper) hull
    h = []
    for r in par:
        while len(h) >= 2:
            a, b = h[-2], h[-1]
            if (b['psnr'] - a['psnr']) / (b['MB'] - a['MB']) <= (r['psnr'] - b['psnr']) / (r['MB'] - b['MB']): h.pop()
            else: break
        h.append(r)
    return h

out = []
for S in ('room', 'kitchen', 'garden'):
    pts = [r for r in rows if r['scene'] == S]
    base = sorted([r for r in pts if r['base']], key=lambda r: r['MB'])
    H = hull(pts); hb = hull(base)
    out.append(f"\n=== {S}: joint hull ({len(H)} of {len(pts)} points) ===")
    for i, r in enumerate(H):
        sl = '' if i == 0 else f"{(r['psnr'] - H[i-1]['psnr']) / (r['MB'] - H[i-1]['MB']):.3f} dB/MB"
        out.append(f"  {r['MB']:7.3f} MB  {r['psnr']:.3f}  lam {r['lam']:.5f}  {r['lever']:16s} {sl}")
    # gain of the joint hull over the lambda-only (base) hull at each base point's bytes (interpolate on joint hull)
    def interp(h, x):
        if x <= h[0]['MB'] or x >= h[-1]['MB']: return None
        for a, b in zip(h, h[1:]):
            if a['MB'] <= x <= b['MB']: return a['psnr'] + (b['psnr'] - a['psnr']) * (x - a['MB']) / (b['MB'] - a['MB'])
    out.append(f"  base (lambda-only, K4096R256 q7) points vs joint hull at the same bytes:")
    for r in base:
        y = interp(H, r['MB']); g = '' if y is None else f"joint hull {y:.3f} (+{y - r['psnr']:.3f})"
        out.append(f"    lam {r['lam']:.5f}  {r['MB']:7.3f} MB  {r['psnr']:.3f}  {g}")
    # lever prices per lambda: dB/MB of each lever vs the lambda chord slope at that point
    out.append(f"  lever prices (dB/MB) per lambda; lambda-chord slope = towards neighbouring lambda:")
    for i, b in enumerate(base):
        up = f"{(base[i+1]['psnr'] - b['psnr']) / (base[i+1]['MB'] - b['MB']):.3f}" if i + 1 < len(base) else '  -  '
        dn = f"{(b['psnr'] - base[i-1]['psnr']) / (b['MB'] - base[i-1]['MB']):.3f}" if i > 0 else '  -  '
        lv = [r for r in pts if r['lam'] == b['lam'] and not r['base']]
        s = '  '.join(f"{r['lever']}:{(r['psnr'] - b['psnr']) / (r['MB'] - b['MB']):.3f}" for r in sorted(lv, key=lambda r: r['MB']))
        out.append(f"    lam {b['lam']:.5f} ({b['MB']:.1f} MB): chord up {up} / down {dn} | {s}")
txt = '\n'.join(out); print(txt); open('fan-analysis-v1.txt', 'w').write(txt + '\n')
