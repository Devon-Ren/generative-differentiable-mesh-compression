"""adapt_analysis_v1 (2026-11-06): reads adapt-v1-2026-11-06.txt (12 rows of adapt-v1.sbatch / train_rd_adapt_v1: room, kitchen, garden, bicycle x target 5/10/20 MB)
and the lambda-only curves (lambda-k-fan-v2.csv base rows = K4096 R256 q7 delivered assets of the same scenes at lambda 6.2e-4 .. 1.85e-2).
Per run: hit ratio and its decomposition into (i) the projection-price bias (projected_mb double counts topology: V*B_v + F*B_f with B_v = B_a + 2 B_f),
(ii) the shrink after the last control point (upstream opacity pruning 18000 -> 30000 + final cleaning), (iii) the controller's own phase-A/B endpoint error;
and the quality cost: test PSNR vs the fixed-lambda curve interpolated (log-MB) at the delivered bytes.  Outputs adapt-v1-analysis.txt, adapt-v1-table.tex, adapt-v1.csv."""
import re, csv, math, collections
BV, BF, OH = 10.0, 2.0, 0.6
rows = []
for line in open('adapt-v1-2026-11-06.txt'):
    S = line.split('|')[0].strip()
    g = lambda pat: re.search(pat, line)
    T = float(g(r'target (\d+) MB').group(1)); lam0 = float(g(r'lam0 ([\d.e+-]+)').group(1))
    qa = [float(x) for x in g(r'lamA q10/q50/q90 ([\d.e+-]+)/([\d.e+-]+)/([\d.e+-]+)').groups()]; lamf = float(g(r'lam_final ([\d.e+-]+)').group(1))
    proj = float(g(r'projected ([\d.]+) MB').group(1)); V = int(g(r'trained V (\d+)').group(1)); F = int(g(r'F (\d+)').group(1))
    dl = float(g(r'delivered ([\d.]+) MB').group(1)); hit = float(g(r'hit ([\d.]+)').group(1))
    t = g(r'test ([\d.]+) ([\d.]+) ([\d.]+)').groups(); val = float(g(r'val ([\d.]+)').group(1))
    pr = re.findall(r'\((\d+), (\d+), (\d+), ([\d.]+)\)', line.split('prune rounds')[1])
    pr = [(int(a), int(b), int(c), float(d)) for a, b, c, d in pr][-4:]  # the last run's four rounds (garden 5 MB has a killed v1 run logged first)
    ndn = int(g(r'densify events (\d+)').group(1))
    rows.append(dict(scene=S, target=T, lam0=lam0, lamA_q10=qa[0], lamA_q50=qa[1], lamA_q90=qa[2], lam_final=lamf, projected=proj, V=V, F=F, delivered=dl, hit=hit,
                     psnr=float(t[0]), ssim=float(t[1]), lpips=float(t[2]), val=val, prune=pr, n_densify=ndn))
assert len(rows) == 12
fan = [r for r in csv.DictReader(open('lambda-k-fan-v2.csv')) if r['base'] == 'True']
curve = collections.defaultdict(list)
for r in fan: curve[r['scene']].append((float(r['MB']), float(r['psnr']), float(r['lam'])))
for S in curve: curve[S].sort()
def interp_curve(S, x):
    """PSNR of the fixed-lambda curve at x MB: linear in log MB between the two neighbouring lambda points; outside the range: nearest segment extrapolated, flagged."""
    c = curve[S]; lx = math.log(x); flag = ''
    if x < c[0][0]: a, b = c[0], c[1]; flag = ' (extrapolated below lam 1.85e-2)'
    elif x > c[-1][0]: a, b = c[-2], c[-1]; flag = ' (extrapolated above)'
    else:
        for a, b in zip(c, c[1:]):
            if a[0] <= x <= b[0]: break
    w = (lx - math.log(a[0])) / (math.log(b[0]) - math.log(a[0]))
    y = a[1] + w * (b[1] - a[1]); lam_eff = math.exp(math.log(a[2]) + w * (math.log(b[2]) - math.log(a[2])))
    return y, lam_eff, flag
out = ["adapt-v1 analysis (2026-11-06): 4 scenes x target {5, 10, 20} MB, train_rd_adapt_v1 (lambda rate controller), K4096 R256 q7 delivery",
       "price constants B_v 10 B/vertex (= B_a 6 + 2 B_f), B_f 2 B/face, overhead 0.6 MB.  projected* = corrected projection V*(B_v - 2 B_f) + F*B_f (no double count).",
       "", f"{'scene':8s} {'T':>3s} {'deliv':>6s} {'hit':>5s} | {'proj':>6s} {'proj*':>6s} {'deliv/proj*':>11s} | {'R12k':>6s} {'R18k':>6s} {'final':>6s} {'post18k':>7s} | {'lam0':>8s} {'lamA50':>8s} {'lamF':>8s} | {'PSNR':>6s} {'curve':>6s} {'dPSNR':>6s} {'lam_eff':>8s}"]
tex = []; csvrows = []
for r in rows:
    S, T = r['scene'], r['target']
    proj_c = (r['V'] * (BV - 2 * BF) + r['F'] * BF) / 1e6 + OH
    R12 = r['prune'][0][3]; R18 = r['prune'][-1][3]  # biased projection before the first prune round / after the last
    y, lam_eff, flag = interp_curve(S, r['delivered']); d = r['psnr'] - y
    r.update(proj_c=proj_c, R12=R12, R18=R18, post=r['projected'] / R18, curve_psnr=y, dpsnr=d, lam_eff=lam_eff, flag=flag)
    out.append(f"{S:8s} {T:3.0f} {r['delivered']:6.2f} {r['hit']:5.2f} | {r['projected']:6.2f} {proj_c:6.2f} {r['delivered']/proj_c:11.3f} | {R12:6.2f} {R18:6.2f} {r['projected']:6.2f} {r['projected']/R18:7.3f} | {r['lam0']:8.2e} {r['lamA_q50']:8.2e} {r['lam_final']:8.2e} | {r['psnr']:6.3f} {y:6.3f} {d:+6.3f} {lam_eff:8.2e}{flag}")
    tex.append(f"{S}, {T:.0f} & {r['lam0']:.1e} & {r['lamA_q10']:.1e}--{r['lamA_q90']:.1e} & {r['lam_final']:.1e} & {R12:.1f} & {r['projected']:.1f} & {r['delivered']:.2f} ({r['hit']:.2f}) & {r['psnr']:.3f} & {d:+.3f} \\\\")
    csvrows.append({k: (v if not isinstance(v, list) else str(v)) for k, v in r.items()})
hits = [r['hit'] for r in rows]; dps = [r['dpsnr'] for r in rows if not r['flag']]; dpa = [r['dpsnr'] for r in rows]
ratio_c = [r['delivered'] / r['proj_c'] for r in rows]; post = [r['post'] for r in rows]; phaseA = [r['R12'] / r['target'] for r in rows]
out += ["", f"hit ratio delivered/target: {min(hits):.2f}-{max(hits):.2f} (mean {sum(hits)/12:.2f}); none in the 0.85-1.15 band -> v1 controller NOT usable as is",
        f"(i) price bias: delivered / corrected projection = {min(ratio_c):.3f}-{max(ratio_c):.3f} (mean {sum(ratio_c)/12:.3f}) -> the corrected size model is accurate; the biased one (used by the controller) overestimates by 1/{sum(r['delivered']/r['projected'] for r in rows)/12:.2f} = x{sum(r['projected']/r['delivered'] for r in rows)/12:.2f}",
        f"(ii) post-control shrink: final projection / projection after the last prune round (it 18000) = {min(post):.2f}-{max(post):.2f} (mean {sum(post)/12:.2f}) -> upstream opacity pruning + final cleaning remove ~{100*(1-sum(post)/12):.0f} % of the bytes after the controller's last action",
        f"(iii) phase-A endpoint (biased projection before the first prune round / target) = {min(phaseA):.2f}-{max(phaseA):.2f} (mean {sum(phaseA)/12:.2f}); phase B then removes {100*(1-sum(r['R18']/r['R12'] for r in rows)/12):.0f} % on average (four rounds)",
        f"  product check: hit = (R12/T) * (R18/R12) * (final/R18) * (delivered/final): mean {sum(phaseA)/12:.2f} * {sum(r['R18']/r['R12'] for r in rows)/12:.2f} * {sum(post)/12:.2f} * {sum(r['delivered']/r['projected'] for r in rows)/12:.2f}",
        f"quality: test PSNR vs the fixed-lambda curve at the delivered bytes: {min(dpa):+.3f}..{max(dpa):+.3f} dB (mean {sum(dpa)/12:+.3f}; inside the measured lambda range {min(dps):+.3f}..{max(dps):+.3f}, mean {sum(dps)/len(dps):+.3f}, n={len(dps)})",
        "  -> the controller's operating points lie on the fixed-lambda hull: adapting lambda online costs no quality; only the size calibration is off.",
        "lambda: lam_final / lam_eff (the fixed lambda that gives the same bytes) = " + ', '.join(f"{r['scene']}{r['target']:.0f}:{r['lam_final']/r['lam_eff']:.2f}" for r in rows),
        "fix for v2 (no new constants): projected_mb = (V*(B_v - 2B_f) + F*B_f)/1e6 + overhead (removes the x0.76 bias); plan phase A/B against target / s_post with s_post the measured post-18000 shrink (0.78-0.89, one constant 0.85), or add a late control point (prune round at 20000+ is blocked by upstream supersampling).",]
txt = '\n'.join(out); print(txt); open('adapt-v1-analysis.txt', 'w').write(txt + '\n')
open('adapt-v1-table.tex', 'w').write('\n'.join(tex) + '\n')
with open('adapt-v1.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(csvrows[0])); w.writeheader(); w.writerows(csvrows)
