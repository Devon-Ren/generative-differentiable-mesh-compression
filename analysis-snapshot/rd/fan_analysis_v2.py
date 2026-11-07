"""lambda x dictionary fan analysis v2 (2026-11-06): all 11 scenes (lambda-k-fan-v2-2026-11-06.txt = 72 rows of 11/05 + 192 rows of the 11/05 overnight
batch).  Per scene: joint upper-concave Pareto hull over 4 lambda x {K4096R256 q7 (base), q8, K4096R1, K1024R256, K4096R1024, K16384R1024},
gain of the joint hull over the lambda-only hull at the base points' bytes, lever prices (dB/MB) vs the lambda chord.
Outputs: lambda-k-fan-v2.csv, fan-analysis-v2.txt, fan-table-v2.tex (44 rows, same columns as fan-table-v1.tex), fan-summary-v2.tex (11 rows)."""
import csv, collections
SCENES = ['room', 'counter', 'kitchen', 'bonsai', 'garden', 'bicycle', 'stump', 'flowers', 'treehill', 'truck', 'train']
rows = []
for line in open('lambda-k-fan-v2-2026-11-06.txt'):
    p = [x.strip() for x in line.split('|')]
    S = p[0]; lam = float(p[1].split()[1]); lever = p[2].replace(' (base)', ''); mb = float(p[3].split()[0])
    t = p[4].split(); psnr, ssim, lp = float(t[1]), float(t[2]), float(t[3])
    rows.append(dict(scene=S, lam=lam, lever=lever, MB=mb, psnr=psnr, ssim=ssim, lpips=lp, base='(base)' in p[2]))
assert len(rows) == 264, len(rows)
with open('lambda-k-fan-v2.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def hull(pts):  # upper-left Pareto concave hull in (MB, psnr)
    pts = sorted(pts, key=lambda r: (r['MB'], -r['psnr'])); par = []; best = -1e9
    for r in pts:
        if r['psnr'] > best: par.append(r); best = r['psnr']
    h = []
    for r in par:
        while len(h) >= 2 and (h[-1]['psnr'] - h[-2]['psnr']) / (h[-1]['MB'] - h[-2]['MB']) <= (r['psnr'] - h[-1]['psnr']) / (r['MB'] - h[-1]['MB']): h.pop()
        h.append(r)
    return h
def interp(h, x):
    if x <= h[0]['MB'] or x >= h[-1]['MB']: return None
    for a, b in zip(h, h[1:]):
        if a['MB'] <= x <= b['MB']: return a['psnr'] + (b['psnr'] - a['psnr']) * (x - a['MB']) / (b['MB'] - a['MB'])
LEVERS = ['K4096 R256 q8', 'K4096 R1 q7', 'K1024 R256 q7', 'K4096 R1024 q7', 'K16384 R1024 q7']
out, tex, summ = [], [], []
price = collections.defaultdict(list)  # lever -> list of dB/MB over all scenes x lambda
for S in SCENES:
    pts = [r for r in rows if r['scene'] == S]; base = sorted([r for r in pts if r['base']], key=lambda r: r['MB'])
    H = hull(pts); hb = hull(base)
    def interp_in(h, x):  # inclusive interpolation on a hull
        if x < h[0]['MB'] or x > h[-1]['MB']: return None
        for a, b in zip(h, h[1:]):
            if a['MB'] <= x <= b['MB']: return a['psnr'] + (b['psnr'] - a['psnr']) * (x - a['MB']) / (b['MB'] - a['MB'])
        return h[0]['psnr'] if len(h) == 1 else None
    # gain of the joint hull over the lambda-only hull, evaluated at every joint-hull vertex inside the lambda-only range
    gh = [(r['MB'], interp_in(H, r['MB']) - interp_in(hb, r['MB'])) for r in H if interp_in(hb, r['MB']) is not None]
    out.append(f"\n=== {S}: joint hull ({len(H)} of {len(pts)} points); lambda-only hull {len(hb)} of 4 ===")
    for i, r in enumerate(H):
        sl = '' if i == 0 else f"{(r['psnr'] - H[i-1]['psnr']) / (r['MB'] - H[i-1]['MB']):.3f} dB/MB"
        out.append(f"  {r['MB']:7.3f} MB  {r['psnr']:.3f}  lam {r['lam']:.5f}  {r['lever']:16s} {sl}")
    out.append("  base (lambda-only, K4096R256 q7) points vs joint hull at the same bytes:")
    gains = []
    for r in base:
        y = interp(H, r['MB']); g = '' if y is None else f"joint hull {y:.3f} (+{y - r['psnr']:.3f})"
        if y is not None: gains.append(y - r['psnr'])
        out.append(f"    lam {r['lam']:.5f}  {r['MB']:7.3f} MB  {r['psnr']:.3f}  {g}")
    out.append("  lever prices (dB/MB) per lambda; lambda-chord slope = towards neighbouring lambda:")
    sp = collections.defaultdict(list)
    for i, b in enumerate(base):
        up = f"{(base[i+1]['psnr'] - b['psnr']) / (base[i+1]['MB'] - b['MB']):.3f}" if i + 1 < len(base) else '  -  '
        dn = f"{(b['psnr'] - base[i-1]['psnr']) / (b['MB'] - base[i-1]['MB']):.3f}" if i > 0 else '  -  '
        lv = {r['lever']: r for r in pts if r['lam'] == b['lam'] and not r['base']}
        s = '  '.join(f"{k}:{(lv[k]['psnr'] - b['psnr']) / (lv[k]['MB'] - b['MB']):.3f}" for k in LEVERS)
        for k in LEVERS:
            v = (lv[k]['psnr'] - b['psnr']) / (lv[k]['MB'] - b['MB']); sp[k].append(v); price[k].append(v)
        out.append(f"    lam {b['lam']:.5f} ({b['MB']:.1f} MB): chord up {up} / down {dn} | {s}")
        cells = ' & '.join(f"{lv[k]['MB'] - b['MB']:+.2f} / {lv[k]['psnr'] - b['psnr']:+.3f}" for k in LEVERS)
        lamtxt = {0.00062: '6.2e-4', 0.00185: '1.85e-3', 0.0062: '6.2e-3', 0.0185: '1.85e-2'}[b['lam']]
        tex.append(f"{S}, {lamtxt} & {b['MB']:.2f} / {b['psnr']:.3f} & {cells} \\\\")
    # hull membership of non-base levers
    onhull = [f"{r['lever'].replace(' q7','')}@{r['lam']:g}" for r in H if not r['base']]
    mx = max(gains) if gains else 0.0; mh = max(g for _, g in gh) if gh else 0.0; mhx = [x for x, g in gh if g == mh][0] if gh else 0.0
    out.append(f"  joint hull vs lambda-only hull: max gain {mh:+.3f} dB at {mhx:.2f} MB (gain at base points max {mx:+.3f})")
    summ.append(f"{S} & {mh:+.3f} & {', '.join(onhull) if onhull else '--'} & " + ' & '.join(f"{sum(sp[k])/len(sp[k]):.3f}" for k in LEVERS) + " \\\\")
    out.append(f"  max joint-hull gain over the fixed setting at a base point: {mx:+.3f} dB; non-base hull members: {onhull}")
out.append("\n=== lever prices over 44 (scene, lambda) pairs: mean / min / max dB per MB ===")
for k in LEVERS: v = price[k]; out.append(f"  {k:16s} {sum(v)/len(v):+.3f}  [{min(v):+.3f}, {max(v):+.3f}]")
txt = '\n'.join(out); print(txt); open('fan-analysis-v2.txt', 'w').write(txt + '\n')
open('fan-table-v2.tex', 'w').write('\n'.join(tex) + '\n'); open('fan-summary-v2.tex', 'w').write('\n'.join(summ) + '\n')
print('\nSUMMARY (scene & max gain of joint hull over lambda-only hull & hull members & mean prices q8 / R1 / K1024 / R1024 / K16384R1024):'); print('\n'.join(summ))
