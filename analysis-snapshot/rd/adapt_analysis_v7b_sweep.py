#!/usr/bin/env python3
"""adapt-v7b scene sweep analysis (2026-11-07, node-batch9-v2): 11 Mip-360 / T&T scenes x {5, 10, 20} MB + 8 NeRF-Synthetic scenes x {2.5, 4} MB.

Inputs (analysis/rd/):
  adapt-v7-2026-11-07.txt              33 Mip-360 / T&T rows (adapt-v7b.sbatch, VERSION v7-r2)
  adapt-v7-blender-2026-11-07.txt      11 Blender rows (adapt-v7b-blender.sbatch; lego/chair from batch8-v2; 5 x 2.5 MB jobs failed at the smoke stage)
  rd-hull6-v2-table.md                 fixed-lambda hull on Mip-360 / T&T (lam 6.2e-4, 1.85e-3, 6.2e-3, 1.85e-2; v3 delivery, 11/01)
  redeliver-v3-blender-2026-11-07.txt  Blender fixed-lambda refs (lam 6.2e-4 / 6.2e-3) re-delivered with val_select_v3
Outputs: adapt-v7b-sweep-analysis.txt, adapt-v7b-sweep.csv, adapt-v7b-sweep-table.tex
"""
import ast
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MIP = os.path.join(HERE, 'adapt-v7-2026-11-07.txt')
BLD = os.path.join(HERE, 'adapt-v7-blender-2026-11-07.txt')
HULL = os.path.join(HERE, 'rd-hull6-v2-table.md')
REFS = os.path.join(HERE, 'redeliver-v3-blender-2026-11-07.txt')
OUT = os.path.join(HERE, 'adapt-v7b-sweep-analysis.txt')
CSV = os.path.join(HERE, 'adapt-v7b-sweep.csv')
TEX = os.path.join(HERE, 'adapt-v7b-sweep-table.tex')
ORDER_MIP = ['room', 'kitchen', 'counter', 'bonsai', 'garden', 'bicycle', 'stump', 'flowers', 'treehill', 'truck', 'train']
ORDER_BLD = ['lego', 'chair', 'drums', 'ficus', 'hotdog', 'materials', 'mic', 'ship']
PRIOR_BV = 11.0  # phase-A size prior: 10 B attributes-per-vertex + 2.5 faces x 2 B ... written as b_v + fpv*b_f = 8 + 2.5*... see train_rd_adapt docstring: 11 B per vertex + 0.6 MB
PRIOR_OVH = 0.6


def g(p, s, cast=float, d=None):
    m = re.search(p, s)
    return cast(m.group(1)) if m else d


def parse_rows(path, blender):
    runs = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = dict(scene=line.split('|')[0].strip(), T=g(r'target ([0-9.]+) MB', line), deliv=g(r'delivered ([0-9.]+) MB', line),
                 hit=g(r'hit ([0-9.]+) x', line), pd=g(r'probe/delivered ([0-9.]+)', line), probe=g(r'final probe ([0-9.]+)', line),
                 capped=g(r'growth-capped (True|False|None)', line, str) == 'True', last_est=g(r'last landing est ([0-9.]+)\s*MB', line),
                 bound=g(r'limiter-bound (\d+)', line, int), Vstar=g(r'V\* (\d+)', line, int), V=g(r'trained V (\d+)', line, int),
                 F=g(r'trained V \d+ F (\d+)', line, int), active=g(r'prune\s*rounds \d+ active (\d+)', line, int, 0),
                 collapsed=g(r'collapsed (\d+)', line, int, 0), psnr=g(r'test ([0-9.]+)', line), ssim=g(r'test [0-9.]+ ([0-9.]+)', line),
                 lpips=g(r'test [0-9.]+ [0-9.]+ ([0-9.]+)', line), val=g(r'\| val ([0-9.]+)', line), best=g(r'best iter (\d+)', line, int, 30000))
        m = re.search(r'V (\d+) -> (\d+)\)', line)
        r['V12'], r['V19'] = int(m.group(1)), int(m.group(2))
        lists = re.findall(r'\[\(.*?\)\]', line)
        r['extra'] = ast.literal_eval(lists[0]) if len(lists) == 2 else []
        r['rounds'] = ast.literal_eval(lists[-1])
        r['P12'] = r['rounds'][0][1]
        r['e2'] = r['rounds'][1][2]
        r['b_v12'] = r['rounds'][0][8]
        r['b_f12'] = r['rounds'][0][9]
        r['blender'] = blender
        runs[(r['scene'], r['T'])] = r
    return runs


def parse_hull(path):
    """per scene: list of (MB, PSNR) for the four lambda points of the v3 delivery (level 1.0 excluded: different method)."""
    hull = {}
    for line in open(path):
        if not line.startswith('| ') or line.startswith('| scene') or line.startswith('|---'):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) != 7 or cells[0] not in ORDER_MIP:
            continue
        pts = []
        for c in cells[3:7]:
            m = re.match(r'([0-9.]+) MB \d+× ([0-9.]+)', c)
            pts.append((float(m.group(1)), float(m.group(2))))
        fp = re.match(r'([0-9.]+) MB / ([0-9.]+)', cells[1])
        hull[cells[0]] = dict(pts=pts, fp32=(float(fp.group(1)), float(fp.group(2))))
    assert len(hull) == 11, sorted(hull)
    return hull


def parse_refs(path):
    refs = {}
    for line in open(path):
        m = re.match(r'([a-z]+)-rd3b4-lam([0-9.]+)-mu1.0 .*-> v3 ([0-9.]+) MB V \d+ F \d+ test ([0-9.]+)', line)
        if m:
            refs.setdefault(m.group(1), []).append((float(m.group(3)), float(m.group(4))))
    return refs


def upper_hull(pts):
    """upper concave envelope in (ln MB, PSNR), sorted by MB."""
    P = sorted(set(pts))
    if len(P) <= 2:
        return P
    X = [(math.log(m), p) for m, p in P]
    H = []
    for i, (x, y) in enumerate(X):
        while len(H) >= 2:
            (x1, y1), (x2, y2) = H[-2], H[-1]
            if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= 0:  # H[-1] is below the chord H[-2] -> new: drop it
                H.pop()
            else:
                break
        H.append((x, y))
    return [(math.exp(x), y) for x, y in H]


def interp(pts, mb):
    """PSNR on the upper hull at mb (log-linear); '*' when extrapolated from the nearest segment / '=' single point."""
    H = upper_hull(pts)
    if len(H) == 1:
        return H[0][1], '='
    xs = [math.log(m) for m, _ in H]
    x = math.log(mb)
    if x <= xs[0]:
        i = 0
    elif x >= xs[-1]:
        i = len(H) - 2
    else:
        i = max(j for j in range(len(H) - 1) if xs[j] <= x)
    (x0, y0), (x1, y1) = (xs[i], H[i][1]), (xs[i + 1], H[i + 1][1])
    y = y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return y, ('*' if (x < xs[0] or x > xs[-1]) else ' ')


def main():
    mip = parse_rows(MIP, False)
    bld = parse_rows(BLD, True)
    hull = parse_hull(HULL)
    refs = parse_refs(REFS)
    out, P = [], None
    P = out.append
    P('adapt-v7b scene sweep (2026-11-07, node-batch9-v2, 42 of 47 jobs exit 0): Mip-360 / T&T 11 scenes x {5, 10, 20} MB (adapt-v7b.sbatch: 8 prune rounds 12k-19k),')
    P('  NeRF-Synthetic {lego, chair, drums} x 2.5 MB and 8 scenes x 4 MB (adapt-v7b-blender.sbatch: 16 rounds 12k-19.5k; ficus/hotdog/materials/mic/ship 2.5 MB failed at the smoke stage).')
    P('fixed-lambda reference: Mip/T&T = rd-hull6-v2 (lam 6.2e-4 / 1.85e-3 / 6.2e-3 / 1.85e-2, upper hull in log bytes); Blender = lam 6.2e-4 / 6.2e-3 re-delivered with val_select_v3.')
    P('')
    P('1. landing, Mip-360 / T&T (phase B collapsed nothing in all 33 runs: the extrapolated landing never exceeded the target)')
    P('scene       T |  deliv   hit  P/dl | capped bound | V12/V*  P12/T  e2/T | lastEst err | B/V@12k (prior %.0f) | test PSNR   hull  dPSNR' % PRIOR_BV)
    csv = ['dataset,scene,target_mb,delivered_mb,hit,probe_mb,probe_over_delivered,growth_capped,limiter_bound,V12_over_Vstar,P12_over_T,last_est_err_pct,bytes_per_vertex_12k,active_rounds,collapsed,test_psnr,test_ssim,test_lpips,hull_psnr,dpsnr,extrapolated']
    rows_tex = []
    stats = {}
    for T in (5.0, 10.0, 20.0):
        for s in ORDER_MIP:
            r = mip[(s, T)]
            hp, star = interp(hull[s]['pts'], r['deliv'])
            d = r['psnr'] - hp
            err = (r['last_est'] / r['probe'] - 1.0) * 100 if r['last_est'] else float('nan')
            bv = r['P12'] * 1e6 / r['V12']
            r.update(hull_psnr=hp, dpsnr=d, star=star, err=err, bv12=bv)
            P('%-9s %4g |  %6.3f  %.3f  %.3f | %-5s  %2d/17 |  %.2f    %.2f   %.2f |   %+5.1f %%   |   %5.2f             | %7.3f  %7.3f  %+.3f%s'
              % (s, T, r['deliv'], r['hit'], r['pd'], r['capped'], r['bound'], r['V12'] / r['Vstar'], r['P12'] / T, r['e2'] / T, err, bv, r['psnr'], hp, d, star))
            csv.append('mip360-tandt,%s,%g,%.3f,%.3f,%.3f,%.3f,%s,%d,%.3f,%.3f,%.1f,%.2f,%d,%d,%.3f,%.4f,%.4f,%.3f,%.3f,%s'
                       % (s, T, r['deliv'], r['hit'], r['probe'], r['pd'], r['capped'], r['bound'], r['V12'] / r['Vstar'], r['P12'] / T, err, bv, r['active'], r['collapsed'],
                          r['psnr'], r['ssim'], r['lpips'], hp, d, star.strip() == '*'))
        hits = [mip[(s, T)]['hit'] for s in ORDER_MIP]
        ds = [mip[(s, T)]['dpsnr'] for s in ORDER_MIP]
        caps = [s for s in ORDER_MIP if mip[(s, T)]['capped']]
        stats[T] = dict(hits=hits, ds=ds, caps=caps)
        P('  %g MB: hit %.2f-%.2f (mean %.3f), within +-10 %%: %d/11, growth-capped flag: %d (%s); equal-bytes dPSNR vs hull %+.2f..%+.2f (mean %+.3f)'
          % (T, min(hits), max(hits), sum(hits) / 11, sum(1 for h in hits if 0.9 <= h <= 1.1), len(caps), ' '.join(caps), min(ds), max(ds), sum(ds) / 11))
    P('columns: P/dl = final probe / delivered; capped = growth_capped flag; bound = limiter-bound densify intervals; V12/V* = vertices at 12k over the phase-A plan;')
    P('  P12/T = first probe over target; e2/T = extrapolated landing at the 2nd round over target; lastEst err = last-round landing estimate vs the final probe;')
    P('  B/V@12k = probe bytes per vertex at 12k (file bytes incl. overhead / V); hull = fixed-lambda upper hull at the delivered bytes (* extrapolated beyond lam 1.85e-2 / 6.2e-4).')
    P('')
    P('2. landing, NeRF-Synthetic')
    P('scene       T |  deliv   hit  P/dl | capped bound | V12/V*  P12/T | lastEst err | active collapsed | best   val    test  | chord  dPSNR')
    for T in (2.5, 4.0):
        for s in ORDER_BLD:
            if (s, T) not in bld:
                continue
            r = bld[(s, T)]
            if s in refs and len(refs[s]) >= 1:
                hp, star = interp(refs[s], r['deliv'])
            else:
                hp, star = float('nan'), '?'
            d = r['psnr'] - hp
            err = (r['last_est'] / r['probe'] - 1.0) * 100 if r['last_est'] else float('nan')
            r.update(hull_psnr=hp, dpsnr=d, star=star, err=err, bv12=r['P12'] * 1e6 / r['V12'])
            P('%-9s %4g |  %6.3f  %.3f  %.3f | %-5s  %2d/17 |  %.2f    %.2f  |   %+5.1f %%   |  %2d    %6d    | %5d  %.3f  %.3f | %.3f %+.3f%s'
              % (s, T, r['deliv'], r['hit'], r['pd'], r['capped'], r['bound'], r['V12'] / r['Vstar'], r['P12'] / T, err, r['active'], r['collapsed'],
                 r['best'], r['val'] or float('nan'), r['psnr'], hp, d, star))
            csv.append('nerf-synthetic,%s,%g,%.3f,%.3f,%.3f,%.3f,%s,%d,%.3f,%.3f,%.1f,%.2f,%d,%d,%.3f,%.4f,%.4f,%.3f,%.3f,%s'
                       % (s, T, r['deliv'], r['hit'], r['probe'], r['pd'], r['capped'], r['bound'], r['V12'] / r['Vstar'], r['P12'] / T, err, r['bv12'], r['active'], r['collapsed'],
                          r['psnr'], r['ssim'], r['lpips'], hp, d, star.strip() == '*'))
    P('  chord = log-linear between the lam 6.2e-3 and 6.2e-4 references (* extrapolated; all 8 scenes have both references; only the lego FP32 original was re-delivered (the other originals have no checkpoints));')
    P('  ficus / mic 4 MB: the target exceeds the FP32 model size class (lam 6.2e-4 reference 1.88 / 2.60 MB) -> growth-capped is the right answer.')
    P('')
    P('3. why Mip-360 / T&T land under target (phase A, open loop until the first probe at 12k; densification ends at 10k; phase B can only remove)')
    for T in (5.0, 10.0, 20.0):
        rs = [mip[(s, T)] for s in ORDER_MIP]
        bvs = [r['bv12'] for r in rs]
        vv = [r['V12'] / r['Vstar'] for r in rs]
        p12 = [r['P12'] / T for r in rs]
        grow = [r['probe'] / r['P12'] for r in rs]
        P('  %2g MB: B/V@12k %.1f-%.1f (prior %.0f -> V* low by %.0f-%.0f %%); V12/V* %.2f-%.2f (limiter-bound %s intervals); P12/T %.2f-%.2f; final probe / P12 %.2f-%.2f (late growth); hit %.2f-%.2f'
          % (T, min(bvs), max(bvs), PRIOR_BV, 100 * (1 - min(bvs) / PRIOR_BV), 100 * (1 - max(bvs) / PRIOR_BV), min(vv), max(vv),
             '%d-%d' % (min(r['bound'] for r in rs), max(r['bound'] for r in rs)), min(p12), max(p12), min(grow), max(grow), min(r['hit'] for r in rs), max(r['hit'] for r in rs)))
    P('  -> the vertex plan V* = (T - 0.6 MB) / 11 B is 10-30 % too small because the delivered bytes per vertex fall with model size (12.8 -> 7.7 B/V from 5 to 20 MB on the')
    P('     same scenes), phase A reaches only 83-97 % of even that plan (rank yield / limiter), and nothing after 12k can add vertices. The v4 run (phase-A aim x1.15) hid')
    P('     part of this with a constant; v7b removed the constant and the shortfall shows in full. Fix for v8 = measure B/V with the real encoder during phase A (probe at')
    P('     the densify intervals from ~6k, 30-60 s each) and re-plan V* from the measurement -- no constant, same machinery as phase B.')
    P('')
    P('4. quality: equal-bytes dPSNR vs the fixed-lambda hull')
    allm = [mip[(s, T)]['dpsnr'] for T in (5.0, 10.0, 20.0) for s in ORDER_MIP]
    P('  Mip/T&T 33 runs: %+.2f..%+.2f, mean %+.3f; by target 5 / 10 / 20 MB mean %+.3f / %+.3f / %+.3f; runs below the hull by > 0.1 dB: %s'
      % (min(allm), max(allm), sum(allm) / 33, sum(stats[5.0]['ds']) / 11, sum(stats[10.0]['ds']) / 11, sum(stats[20.0]['ds']) / 11,
         ' '.join('%s%g(%+.2f)' % (s, T, mip[(s, T)]['dpsnr']) for T in (5.0, 10.0, 20.0) for s in ORDER_MIP if mip[(s, T)]['dpsnr'] < -0.1) or 'none'))
    bl = [(k, bld[k]['dpsnr']) for k in bld if not math.isnan(bld[k]['dpsnr'])]
    P('  Blender %d runs: %s' % (len(bl), ' '.join('%s%g(%+.2f%s)' % (k[0], k[1], d, bld[k]['star'].strip()) for k, d in sorted(bl, key=lambda kv: (kv[0][1], ORDER_BLD.index(kv[0][0]))))))
    P('  Note: these are v7b models whose phase B never collapsed anything on Mip/T&T (no lambda-prune): the v4 Mip-360 analysis measured the lambda-prune at +0.07 dB equal-bytes,')
    P('  and batch8-v2 showed -0.3/-0.5 dB on the gate-closed Blender 4 MB runs vs v6; the lambda-prune-as-regulariser ablation is still open.')
    open(OUT, 'w').write('\n'.join(out) + '\n')
    open(CSV, 'w').write('\n'.join(csv) + '\n')
    # LaTeX table: scene x target: delivered MB (hit), dPSNR
    tex = []
    tex.append('% adapt-v7b scene sweep 2026-11-07 (node-batch9-v2); generated by adapt_analysis_v7b_sweep.py')
    tex.append('\\begin{tabular}{l' + 'rr' * 3 + '}')
    tex.append('\\toprule')
    tex.append('& \\multicolumn{2}{c}{target 5 MB} & \\multicolumn{2}{c}{target 10 MB} & \\multicolumn{2}{c}{target 20 MB} \\\\')
    tex.append('scene & MB (hit) & $\\Delta$PSNR & MB (hit) & $\\Delta$PSNR & MB (hit) & $\\Delta$PSNR \\\\')
    tex.append('\\midrule')
    for s in ORDER_MIP:
        cells = []
        for T in (5.0, 10.0, 20.0):
            r = mip[(s, T)]
            cells.append('%.1f (%.2f)%s & %+.2f' % (r['deliv'], r['hit'], '$^\\dagger$' if r['capped'] else '', r['dpsnr']))
        tex.append('%s & %s \\\\' % (s, ' & '.join(cells)))
    tex.append('\\midrule')
    cells = []
    for T in (5.0, 10.0, 20.0):
        cells.append('mean hit %.2f & %+.2f' % (sum(stats[T]['hits']) / 11, sum(stats[T]['ds']) / 11))
    tex.append('mean & %s \\\\' % ' & '.join(cells))
    tex.append('\\bottomrule')
    tex.append('\\end{tabular}')
    tex.append('% $\\dagger$ growth-capped flag (upstream add_percentage limiter bound the growth); $\\Delta$PSNR = test PSNR minus the fixed-$\\lambda$ hull at the delivered bytes.')
    tex.append('')
    tex.append('\\begin{tabular}{lrrrr}')
    tex.append('\\toprule')
    tex.append('& \\multicolumn{2}{c}{target 2.5 MB} & \\multicolumn{2}{c}{target 4 MB} \\\\')
    tex.append('scene & MB (hit) & $\\Delta$PSNR & MB (hit) & $\\Delta$PSNR \\\\')
    tex.append('\\midrule')
    for s in ORDER_BLD:
        cells = []
        for T in (2.5, 4.0):
            if (s, T) in bld:
                r = bld[(s, T)]
                cells.append('%.2f (%.2f)%s & %+.2f%s' % (r['deliv'], r['hit'], '$^\\dagger$' if r['capped'] else '', r['dpsnr'], '$^*$' if r['star'].strip() else ''))
            else:
                cells.append('-- & --')
        tex.append('%s & %s \\\\' % (s, ' & '.join(cells)))
    tex.append('\\bottomrule')
    tex.append('\\end{tabular}')
    open(TEX, 'w').write('\n'.join(tex) + '\n')
    print('\n'.join(out))
    print('\nwritten', OUT, CSV, TEX)


if __name__ == '__main__':
    main()
