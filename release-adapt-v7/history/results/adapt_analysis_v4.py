"""adapt_analysis_v4 (2026-11-06): the 12 adapt-v4 rows vs the fixed-lambda curves (lambda-k-fan-v2.csv base rows), same method as
adapt_analysis_v2.py; also reads adapt-v3-2026-11-06.txt for the per-run v3 hit.  Columns: hit, probe@12k/@18k, final/probe@18k (growth of the
kept set after the last control point), phase-A endpoint V vs V*, limiter-bound / front-loaded counts, collapses, equal-bytes dPSNR."""
import re, csv, math, collections, sys
def parse(fn):
    rows = []
    for line in open(fn):
        S = line.split('|')[0].strip(); g = lambda pat: re.search(pat, line)
        T = float(g(r'target (\d+) MB').group(1)); lam0 = float(g(r'lam0 ([\d.e+-]+)').group(1))
        qa = [float(x) for x in g(r'lamA q10/q50/q90 ([\d.e+-]+)/([\d.e+-]+)/([\d.e+-]+)').groups()]; lamf = float(g(r'lam_final \(shadow\) ([\d.e+-]+)').group(1))
        fp = float(g(r'final probe ([\d.]+) MB').group(1)); V = int(g(r'trained V (\d+)').group(1)); F = int(g(r'F (\d+)').group(1))
        dl = float(g(r'delivered ([\d.]+) MB').group(1)); hit = float(g(r'hit ([\d.]+)').group(1)); pd = float(g(r'probe/delivered ([\d.]+)').group(1))
        t = g(r'test ([\d.]+) ([\d.]+) ([\d.]+)').groups(); val = float(g(r'val ([\d.]+)').group(1)); ndn = int(g(r'densify events (\d+)').group(1))
        m = g(r'limiter-bound (\d+), front-loaded (\d+), V\* (\d+)'); nb, nf, vs = (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (None, None, None)
        sig = [float(x) for x in g(r'sigma q10/q50/q90 ([\d.]+)/([\d.]+)/([\d.]+)').groups()]
        pr = [(int(a), float(b), float(c), int(d), int(e), f, float(h)) for a, b, c, d, e, f, h in re.findall(r"\((\d+), ([\d.]+), ([\d.]+), (\d+), (\d+), '(\w+)', '([\d.e+-]+)'", line)]
        rows.append(dict(scene=S, target=T, lam0=lam0, lamA_q10=qa[0], lamA_q50=qa[1], lamA_q90=qa[2], lam_final=lamf, final_probe=fp, V=V, F=F, delivered=dl, hit=hit, pd=pd,
                         psnr=float(t[0]), ssim=float(t[1]), lpips=float(t[2]), val=val, prune=pr, n_densify=ndn, n_bound=nb, n_front=nf, v_star=vs, sig50=sig[1]))
    return rows
rows = parse('adapt-v4-2026-11-06.txt'); assert len(rows) == 12
v3 = {(r['scene'], r['target']): r for r in parse('adapt-v3-2026-11-06.txt')}
fan = [r for r in csv.DictReader(open('lambda-k-fan-v2.csv')) if r['base'] == 'True']
curve = collections.defaultdict(list)
for r in fan: curve[r['scene']].append((float(r['MB']), float(r['psnr']), float(r['lam'])))
for S in curve: curve[S].sort()
def interp_curve(S, x):
    c = curve[S]; lx = math.log(x); flag = ''
    if x < c[0][0]: a, b = c[0], c[1]; flag = '*'
    elif x > c[-1][0]: a, b = c[-2], c[-1]; flag = '*'
    else:
        for a, b in zip(c, c[1:]):
            if a[0] <= x <= b[0]: break
    w = (lx - math.log(a[0])) / (math.log(b[0]) - math.log(a[0]))
    return a[1] + w * (b[1] - a[1]), math.exp(math.log(a[2]) + w * (math.log(b[2]) - math.log(a[2]))), flag
rows.sort(key=lambda r: (r['target'], r['scene']))
out = ["adapt-v4 analysis (2026-11-06): 4 scenes x {5,10,20} MB, train_rd_adapt_v4 (probe at final floor + 4x, count-only phase B, limiter-aware geometric phase A), K4096 R256 q7", "",
       f"{'scene':8s} {'T':>2s} {'deliv':>6s} {'hit':>5s} {'v3hit':>5s} | {'V12k':>8s} {'V*':>8s} {'V/V*':>5s} {'bound':>5s} {'front':>5s} {'sig50':>5s} | {'P12':>6s} {'P18':>6s} {'kept18':>6s} {'fin/P18':>7s} {'coll%':>5s} | {'PSNR':>7s} {'curve':>7s} {'dPSNR':>7s} {'v3dP':>7s}"]
tex = []; csvrows = []
for r in rows:
    S, T = r['scene'], r['target']; p = r['prune']; P12, P18 = p[0][1], p[-1][1]; V12 = p[0][3]
    y, lam_eff, flag = interp_curve(S, r['delivered']); d = r['psnr'] - y; o = v3[(S, T)]; y3, _, f3 = interp_curve(S, o['delivered']); d3 = o['psnr'] - y3
    coll = (p[0][3] - p[-1][4]) / p[0][3]; modes = ''.join(m[5][0] for m in p)
    r.update(P12=P12, P18=P18, growth=r['final_probe'] / P18, V12=V12, kept18=p[-1][2], curve_psnr=y, dpsnr=d, lam_eff=lam_eff, flag=flag, collapsed=coll, modes=modes, v3hit=o['hit'], v3dpsnr=d3)
    out.append(f"{S:8s} {T:2.0f} {r['delivered']:6.2f} {r['hit']:5.2f} {o['hit']:5.2f} | {V12:8d} {r['v_star']:8d} {V12/r['v_star']:5.2f} {r['n_bound']:5d} {r['n_front']:5d} {r['sig50']:5.2f} | {P12:6.2f} {P18:6.2f} {r['kept18']:6.2f} {r['growth']:7.2f} {100*coll:5.1f} | {r['psnr']:7.3f} {y:7.3f} {d:+7.3f}{flag} {d3:+7.3f}{f3}")
    tex.append(f"{S}, {T:.0f} & {r['lam0']:.1e} & {r['lamA_q10']:.1e}--{r['lamA_q90']:.1e} & {P12:.1f} & {P18:.1f} & {r['final_probe']:.2f} & {r['delivered']:.2f} ({r['hit']:.2f}) & {r['psnr']:.3f} & {d:+.3f} \\\\")
    csvrows.append({k: (v if not isinstance(v, list) else str(v)) for k, v in r.items()})
hits = [r['hit'] for r in rows]; dpa = [r['dpsnr'] for r in rows]; dps = [r['dpsnr'] for r in rows if not r['flag']]
byT = {T: [r['hit'] for r in rows if r['target'] == T] for T in (5, 10, 20)}
ind = [r for r in rows if r['scene'] in ('room', 'kitchen')]; outd = [r for r in rows if r['scene'] in ('garden', 'bicycle')]
out += ["", f"hit {min(hits):.2f}-{max(hits):.2f} (mean {sum(hits)/12:.2f}); by target " + ', '.join(f"{T} MB {min(h):.2f}-{max(h):.2f}" for T, h in byT.items()) + f"; within 0.9-1.1: {sum(0.9 <= h <= 1.1 for h in hits)}/12 (v3: 3/12, mean 0.87; v2: 5/12, mean 0.86)",
        f"quality vs fixed-lambda curve at equal bytes: {min(dpa):+.3f}..{max(dpa):+.3f} dB (mean {sum(dpa)/12:+.3f}; inside the measured range {min(dps):+.3f}..{max(dps):+.3f}, n={len(dps)}); v3 mean {sum(r['v3dpsnr'] for r in rows)/12:+.3f}  (* = extrapolated)",
        f"probe accuracy (final / probe@18k): outdoor {min(r['growth'] for r in outd):.2f}-{max(r['growth'] for r in outd):.2f} (v3 1.03-1.28) -> solved by the 4x pass; indoor {min(r['growth'] for r in ind):.2f}-{max(r['growth'] for r in ind):.2f} (v3 1.12-1.39) -> still under-reads, growing with the target",
        f"  indoor probes RISE 12k->18k (" + ', '.join(f"{r['scene']}{r['target']:.0f} {r['P12']:.1f}->{r['P18']:.1f}" for r in ind) + f") with V constant: the kept set keeps growing while the opacity floor ramps to 24000 (faces un-stack), a dynamic the 18k probe cannot see",
        f"phase B: modes per round (c=count, n=none) " + ', '.join(f"{r['scene']}{r['target']:.0f}:{r['modes']}" for r in rows) + "; collapsed only where the probe was above target (garden 5, kitchen 5)",
        f"phase A: V@12k / V* = " + ', '.join(f"{r['scene']}{r['target']:.0f} {r['V12']/r['v_star']:.2f}({r['n_bound']}b)" for r in rows),
        f"  20 MB: limiter-bound 8-18 of 17-18 intervals (bicycle 18/18 -> 1.58 M of 2.04 M: the upstream add_percentage cap); bicycle 10: bound 4 only, yet V@12k/V* 0.79: rank-lambda delivered pass/desired 0.66-0.85 every interval (the next interval's ratios are not the current ones)",
        "lam_final / lam_eff: " + ', '.join(f"{r['scene']}{r['target']:.0f}:{r['lam_final']/r['lam_eff']:.2f}" for r in rows)]
open('adapt-v4-analysis.txt', 'w').write('\n'.join(out) + '\n'); open('adapt-v4-table.tex', 'w').write('\n'.join(tex) + '\n')
with open('adapt-v4.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(csvrows[0].keys())); w.writeheader(); w.writerows(csvrows)
print('\n'.join(out))
