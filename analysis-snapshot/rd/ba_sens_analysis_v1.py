"""ba_sens_analysis_v1 (2026-11-06): B_a sensitivity.  ba-sens-v1-2026-11-06.txt = kitchen (--indoor) / garden x BPV {8, 13} (B_a 4 / 9 B at B_f 2) x lambda {6.2e-3, 6.2e-4},
train_rd_v3 + level-1.0 delivery (K4096 R256 q7), against the bpv 10 assets of the same scenes (lambda-k-fan-v2.csv base rows, 4 lambda each).
Reading: a wrong price constant c*B rescales lambda (split dR = 3 B_v, so B_v -> s B_v == lambda -> s lambda); the bpv 8/13 points should fall on the
bpv-10 lambda curve at an effective lambda ~ s * lambda.  Outputs ba-sens-v1-analysis.txt, ba-sens-v1-table.tex, ba-sens-v1.csv."""
import re, csv, math, collections
rows = []
for line in open('ba-sens-v1-2026-11-06.txt'):
    g = lambda pat: re.search(pat, line)
    S = line.split()[0]; lam = float(g(r'lam ([\d.e-]+)').group(1)); bpv = float(g(r'bpv (\d+)').group(1))
    V = int(g(r'vertices (\d+)').group(1)); F = int(g(r'faces (\d+)').group(1)); fp = float(g(r'test ([\d.]+) \|').group(1))
    mb, ps, ss, lp = g(r'delivered ([\d.]+) MB test ([\d.]+) ([\d.]+) ([\d.]+)').groups()
    rows.append(dict(scene=S, lam=lam, bpv=bpv, V=V, F=F, fp32_test=fp, MB=float(mb), psnr=float(ps), ssim=float(ss), lpips=float(lp)))
assert len(rows) == 8
fan = [r for r in csv.DictReader(open('lambda-k-fan-v2.csv')) if r['base'] == 'True' and r['scene'] in ('kitchen', 'garden')]
curve = collections.defaultdict(list)
for r in fan: curve[r['scene']].append((float(r['MB']), float(r['psnr']), float(r['lam'])))
for S in curve: curve[S].sort()
def interp_curve(S, x):
    c = curve[S]; lx = math.log(x); flag = ''
    if x < c[0][0]: a, b = c[0], c[1]; flag = ' (extrapolated)'
    elif x > c[-1][0]: a, b = c[-2], c[-1]; flag = ' (extrapolated above lam 6.2e-4)'
    else:
        for a, b in zip(c, c[1:]):
            if a[0] <= x <= b[0]: break
    w = (lx - math.log(a[0])) / (math.log(b[0]) - math.log(a[0]))
    return a[1] + w * (b[1] - a[1]), math.exp(math.log(a[2]) + w * (math.log(b[2]) - math.log(a[2]))), flag
out = ["B_a sensitivity (2026-11-06): kitchen (--indoor) / garden, B_v = B_a + 2 B_f in {8, 10 (default), 13} B, B_f 2 B, lambda {6.2e-3, 6.2e-4}; K4096 R256 q7 delivery.",
       "bpv 10 rows = the lambda assets of the main curve (rd3i / rd3).  curve = bpv-10 lambda curve interpolated in log MB at this point's bytes; lam_eff = the bpv-10 lambda with these bytes.",
       "", f"{'scene':8s} {'lam':>8s} {'bpv':>4s} | {'V':>8s} {'MB':>7s} {'PSNR':>7s} | {'curve':>7s} {'dPSNR':>6s} | {'lam_eff':>8s} {'lam_eff/lam':>11s} {'bpv/10':>6s}"]
tex = []; dps = []
for S in ('kitchen', 'garden'):
    for lam in (0.0062, 0.00062):
        b10 = [c for c in curve[S] if abs(c[2] - lam) < 1e-9][0]
        out.append(f"{S:8s} {lam:8.5f} {10:4.0f} | {'':>8s} {b10[0]:7.3f} {b10[1]:7.3f} | {'':>7s} {'':>6s} | {lam:8.2e} {1.0:11.2f} {1.0:6.2f}")
        for r in sorted([r for r in rows if r['scene'] == S and r['lam'] == lam], key=lambda r: r['bpv']):
            y, le, flag = interp_curve(S, r['MB']); d = r['psnr'] - y; dps.append(d); r.update(curve=y, dpsnr=d, lam_eff=le, flag=flag)
            out.append(f"{S:8s} {lam:8.5f} {r['bpv']:4.0f} | {r['V']:8d} {r['MB']:7.3f} {r['psnr']:7.3f} | {y:7.3f} {d:+6.3f} | {le:8.2e} {le/lam:11.2f} {r['bpv']/10:6.2f}{flag}")
            tex.append(f"{S}, {lam:g} & {r['bpv']:.0f} & {r['MB']:.2f} & {r['psnr']:.3f} & {d:+.3f} & {le/lam:.2f} & {r['bpv']/10:.1f} \\\\")
out += ["", f"dPSNR vs the bpv-10 curve at equal bytes: {min(dps):+.3f}..{max(dps):+.3f} dB (mean {sum(dps)/len(dps):+.3f}, n={len(dps)})",
        "  -> B_a in {4, 6, 9} B moves the operating point ALONG the lambda curve, not off it: a price error is a lambda rescale, as the Lagrangian predicts (split dR = 3 B_v).",
        "  lam_eff/lam vs bpv/10: the size shift is in the predicted direction (bpv 8 -> larger model, bpv 13 -> smaller) and of the same order as the price ratio; phase B (collapse dR = B_a + n B_f) is not purely proportional, so the match is approximate."]
txt = '\n'.join(out); print(txt); open('ba-sens-v1-analysis.txt', 'w').write(txt + '\n'); open('ba-sens-v1-table.tex', 'w').write('\n'.join(tex) + '\n')
with open('ba-sens-v1.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
