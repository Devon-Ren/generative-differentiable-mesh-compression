"""adapt_analysis_v2 (2026-11-06): the 12 adapt-v2 rows (train_rd_adapt_v2: real-encoder probes at 12k/14k/16k/18k + final) vs the fixed-lambda
curves (lambda-k-fan-v2.csv base rows).  Per run: hit, probe/delivered, probe@12k and @18k vs target, final/probe@18k (growth of the kept set after the
last control point), model V vs delivered V, phase-A endpoint vs its aim (target x 1.3 on the corrected constant projection), equal-bytes dPSNR."""
import re, csv, math, collections
rows = []
for line in open('adapt-v2-2026-11-06.txt'):
    S = line.split('|')[0].strip(); g = lambda pat: re.search(pat, line)
    T = float(g(r'target (\d+) MB').group(1)); lam0 = float(g(r'lam0 ([\d.e+-]+)').group(1))
    qa = [float(x) for x in g(r'lamA q10/q50/q90 ([\d.e+-]+)/([\d.e+-]+)/([\d.e+-]+)').groups()]; lamf = float(g(r'lam_final \(shadow\) ([\d.e+-]+)').group(1))
    fp = float(g(r'final probe ([\d.]+) MB').group(1)); V = int(g(r'trained V (\d+)').group(1)); F = int(g(r'F (\d+)').group(1))
    dl = float(g(r'delivered ([\d.]+) MB').group(1)); hit = float(g(r'hit ([\d.]+)').group(1)); pd = float(g(r'probe/delivered ([\d.]+)').group(1))
    t = g(r'test ([\d.]+) ([\d.]+) ([\d.]+)').groups(); val = float(g(r'val ([\d.]+)').group(1)); ndn = int(g(r'densify events (\d+)').group(1))
    pr = [(int(a), float(b), int(c), int(d), float(e), float(f), float(h)) for a, b, c, d, e, f, h in re.findall(r"\((\d+), ([\d.]+), (\d+), (\d+), '([\d.e+-]+)', ([\d.]+), ([\d.]+)\)", line)]
    rows.append(dict(scene=S, target=T, lam0=lam0, lamA_q10=qa[0], lamA_q50=qa[1], lamA_q90=qa[2], lam_final=lamf, final_probe=fp, V=V, F=F, delivered=dl, hit=hit, pd=pd,
                     psnr=float(t[0]), ssim=float(t[1]), lpips=float(t[2]), val=val, prune=pr, n_densify=ndn))
assert len(rows) == 12
fan = [r for r in csv.DictReader(open('lambda-k-fan-v2.csv')) if r['base'] == 'True']
curve = collections.defaultdict(list)
for r in fan: curve[r['scene']].append((float(r['MB']), float(r['psnr']), float(r['lam'])))
for S in curve: curve[S].sort()
def interp_curve(S, x):
    c = curve[S]; lx = math.log(x); flag = ''
    if x < c[0][0]: a, b = c[0], c[1]; flag = ' (extrapolated below lam 1.85e-2)'
    elif x > c[-1][0]: a, b = c[-2], c[-1]; flag = ' (extrapolated above)'
    else:
        for a, b in zip(c, c[1:]):
            if a[0] <= x <= b[0]: break
    w = (lx - math.log(a[0])) / (math.log(b[0]) - math.log(a[0]))
    return a[1] + w * (b[1] - a[1]), math.exp(math.log(a[2]) + w * (math.log(b[2]) - math.log(a[2]))), flag
BV, BF, OH, FPV, OVER = 10.0, 2.0, 0.6, 2.5, 1.3
out = ["adapt-v2 analysis (2026-11-06): 4 scenes x target {5, 10, 20} MB, train_rd_adapt_v2 (real-encoder probes), K4096 R256 q7 delivery", "",
       f"{'scene':8s} {'T':>3s} {'deliv':>6s} {'hit':>5s} {'prb/dl':>6s} | {'P12k':>5s} {'P18k':>5s} {'fin/P18':>7s} {'P12/T':>5s} | {'Vmodel':>8s} {'Vdeliv':>8s} {'kept':>5s} {'projA':>6s} {'aimA':>6s} {'projA/aim':>9s} | {'lamA50':>8s} {'lamF':>8s} {'lam_eff':>8s} | {'PSNR':>6s} {'curve':>6s} {'dPSNR':>6s}"]
tex = []; csvrows = []
for r in rows:
    S, T = r['scene'], r['target']; p = r['prune']; P12, P18 = p[0][1], p[-1][1]; Vm = p[0][2]
    projA = (Vm * (BV - 2 * BF) + FPV * Vm * BF) / 1e6 + OH; aim = T * OVER
    y, lam_eff, flag = interp_curve(S, r['delivered']); d = r['psnr'] - y
    r.update(P12=P12, P18=P18, growth=r['final_probe'] / P18, Vmodel=Vm, kept=r['V'] / Vm, projA=projA, aimA=aim, curve_psnr=y, dpsnr=d, lam_eff=lam_eff, flag=flag, collapsed=sum(a[2] - a[3] for a in p))
    out.append(f"{S:8s} {T:3.0f} {r['delivered']:6.2f} {r['hit']:5.2f} {r['pd']:6.3f} | {P12:5.2f} {P18:5.2f} {r['growth']:7.2f} {P12/T:5.2f} | {Vm:8d} {r['V']:8d} {r['kept']:5.2f} {projA:6.2f} {aim:6.2f} {projA/aim:9.2f} | {r['lamA_q50']:8.2e} {r['lam_final']:8.2e} {lam_eff:8.2e} | {r['psnr']:6.3f} {y:6.3f} {d:+6.3f}{flag}")
    tex.append(f"{S}, {T:.0f} & {r['lam0']:.1e} & {r['lamA_q10']:.1e}--{r['lamA_q90']:.1e} & {P12:.1f} & {P18:.1f} & {r['final_probe']:.2f} & {r['delivered']:.2f} ({r['hit']:.2f}) & {r['psnr']:.3f} & {d:+.3f} \\\\")
    csvrows.append({k: (v if not isinstance(v, list) else str(v)) for k, v in r.items()})
hits = [r['hit'] for r in rows]; dpa = [r['dpsnr'] for r in rows]; dps = [r['dpsnr'] for r in rows if not r['flag']]
byT = {T: [r['hit'] for r in rows if r['target'] == T] for T in (5, 10, 20)}
out += ["", f"hit delivered/target: {min(hits):.2f}-{max(hits):.2f} (mean {sum(hits)/12:.2f}); by target " + ', '.join(f"{T} MB {min(h):.2f}-{max(h):.2f}" for T, h in byT.items()) + f"; within 0.9-1.1: {sum(0.9 <= h <= 1.1 for h in hits)}/12",
        f"byte meter: probe/delivered {min(r['pd'] for r in rows):.3f}-{max(r['pd'] for r in rows):.3f} -> the encoder probe IS the delivered size (v1's 0.77 projection bias gone)",
        f"phase B never collapsed: vertices before == after in {sum(r['collapsed'] == 0 for r in rows)}/12 runs x 4 rounds (every mid-training probe read below target: P12/T {min(r['P12']/r['target'] for r in rows):.2f}-{max(r['P12']/r['target'] for r in rows):.2f}, P18/T {min(r['P18']/r['target'] for r in rows):.2f}-{max(r['P18']/r['target'] for r in rows):.2f})",
        f"  -> the 'shadow lam' column is the phase-A lambda carried over, not a shadow price (no pairs were collapsed)",
        f"growth after the last control point: final probe / probe@18k = {min(r['growth'] for r in rows):.2f}-{max(r['growth'] for r in rows):.2f} (mean {sum(r['growth'] for r in rows)/12:.2f}), larger for larger targets (" + ', '.join(f"{T} MB {sum(r['growth'] for r in rows if r['target']==T)/4:.2f}" for T in (5, 10, 20)) + ")",
        f"  cause: the probe cleans with the final rule (max blending > 0.5 over the train views) but at 12k-18k the opacity floor is still ramping, so faces that will be opaque (kept) at 30000 are dropped now; the model's own V is constant 12k->30k and {min(r['kept'] for r in rows):.2f}-{max(r['kept'] for r in rows):.2f} of it is delivered",
        f"phase A endpoint: constant-projection at the end of growth (V_model x 11 B + 0.6) / aim (target x 1.3) = {min(r['projA']/r['aimA'] for r in rows):.2f}-{max(r['projA']/r['aimA'] for r in rows):.2f} (" + ', '.join(f"{T} MB {sum(r['projA']/r['aimA'] for r in rows if r['target']==T)/4:.2f}" for T in (5, 10, 20)) + ") -> growth stops short of the aim, more so for large targets; lamA q10 = lam0 on room/bicycle 20 (lambda never went below its start) -> the split COUNT, not lambda, limited growth (upstream add_percentage / vertex pruning in phase A): read the densify events",
        f"quality: test PSNR vs the fixed-lambda curve at the delivered bytes {min(dpa):+.3f}..{max(dpa):+.3f} dB (mean {sum(dpa)/12:+.3f}; inside the measured range {min(dps):+.3f}..{max(dps):+.3f}, n={len(dps)}) -> still on the hull",
        "lam_final / lam_eff: " + ', '.join(f"{r['scene']}{r['target']:.0f}:{r['lam_final']/r['lam_eff']:.2f}" for r in rows)]
open('adapt-v2-analysis.txt', 'w').write('\n'.join(out) + '\n'); open('adapt-v2-table.tex', 'w').write('\n'.join(tex) + '\n')
with open('adapt-v2.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(csvrows[0].keys())); w.writeheader(); w.writerows(csvrows)
print('\n'.join(out))
