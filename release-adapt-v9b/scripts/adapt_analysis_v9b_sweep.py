#!/usr/bin/env python3
"""adapt-v9b sweep analysis (2026-11-08, node-batch14-v1: 6 test jobs + gate + 43-job sweep, 47 of 49 jobs exit 0; mic / ship 2.5 MB failed at the smoke stage).
Reuses the row / hull parsers of analysis-snapshot/rd/adapt_analysis_v7b_sweep.py.  Inputs: release-adapt-v9b/results/adapt-v9b-2026-11-08.txt (33 Mip/T&T rows),
adapt-v9b-blender-2026-11-08.txt (14 rows), the fixed-lambda hull (rd-hull6-v2-table.md), the Blender refs (redeliver-v3-blender-2026-11-07.txt), and the v7b sweep csv
for the side-by-side.  Outputs next to the inputs: adapt-v9b-sweep-analysis.txt, adapt-v9b-sweep.csv, adapt-v9b-sweep-table.tex."""
import importlib.util, math, os, sys
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); A = os.path.join(os.path.dirname(R), 'analysis-snapshot', 'rd')
spec = importlib.util.spec_from_file_location('v7b', os.path.join(A, 'adapt_analysis_v7b_sweep.py')); v7b = importlib.util.module_from_spec(spec); spec.loader.exec_module(v7b)
RES = os.path.join(R, 'results'); OM, OB = v7b.ORDER_MIP, v7b.ORDER_BLD
mip = v7b.parse_rows(os.path.join(RES, 'adapt-v9b-2026-11-08.txt'), False); bld = v7b.parse_rows(os.path.join(RES, 'adapt-v9b-blender-2026-11-08.txt'), True)
hull = v7b.parse_hull(os.path.join(A, 'rd-hull6-v2-table.md')); refs = v7b.parse_refs(os.path.join(A, 'redeliver-v3-blender-2026-11-07.txt'))
old = {}
for line in open(os.path.join(A, 'adapt-v7b-sweep.csv')).read().splitlines()[1:]:
    c = line.split(','); old[(c[1], float(c[2]))] = dict(hit=float(c[4]), psnr=float(c[15]), dpsnr=float(c[19]))
INDOOR = {'room', 'kitchen', 'counter', 'bonsai'}
out = []; P = out.append
csv = ['dataset,scene,target_mb,delivered_mb,hit,probe_mb,probe_over_delivered,growth_capped,limiter_bound,V12_over_Vstar,P12_over_T,last_est_err_pct,active_rounds,collapsed,first_active_round,test_psnr,test_ssim,test_lpips,hull_psnr,dpsnr,extrapolated,v7b_hit,v7b_dpsnr']
P('adapt-v9b scene sweep (2026-11-08, node-batch14-v1): v9 + continuous phase-A price + phase A aims at 1.25 x target (Mip/T&T) / 1.1 x (Blender) + 15 prune rounds 12500-19500.')
P('  Mip-360 / T&T 11 scenes x {5, 10, 20} MB (33 rows); NeRF-Synthetic 6 x 2.5 MB + 8 x 4 MB (14 rows; mic / ship 2.5 MB failed at the smoke stage: K-means on the degenerate 1550-it smoke mesh).')
P('  fixed-lambda reference: Mip/T&T = rd-hull6-v2 upper hull in log bytes; Blender = lam 6.2e-4 / 6.2e-3 re-delivered with val_select_v3 (chord).  v7b columns from adapt-v7b-sweep.csv.')
P('')
P('1. landing, Mip-360 / T&T')
P('scene       T |  deliv   hit  P/dl | active coll  1st | V12/V*  P12.5/T | lastEst err | test PSNR   hull  dPSNR | v7b hit  v7b dPSNR')
stats = {}
for T in (5.0, 10.0, 20.0):
    for s in OM:
        r = mip[(s, T)]; hp, star = v7b.interp(hull[s]['pts'], r['deliv']); d = r['psnr'] - hp
        err = (r['last_est'] / r['probe'] - 1.0) * 100 if r['last_est'] else float('nan'); fa = next((x[0] for x in r['rounds'] if x[10] > 0), None)
        r.update(hull_psnr=hp, dpsnr=d, star=star, err=err, fa=fa); o = old[(s, T)]
        P('%-9s %4g |  %6.3f  %.3f  %.3f |   %2d  %6d  %5s |  %.2f    %.3f  |   %+5.1f %%   | %7.3f  %7.3f  %+.3f%s |  %.3f    %+.3f'
          % (s, T, r['deliv'], r['hit'], r['pd'], r['active'], r['collapsed'], fa if fa else '-', r['V12'] / r['Vstar'], r['P12'] / T, err, r['psnr'], hp, d, star, o['hit'], o['dpsnr']))
        csv.append('mip360-tandt,%s,%g,%.3f,%.3f,%.3f,%.3f,%s,%d,%.3f,%.3f,%.1f,%d,%d,%s,%.3f,%.4f,%.4f,%.3f,%.3f,%s,%.3f,%.3f'
                   % (s, T, r['deliv'], r['hit'], r['probe'], r['pd'], r['capped'], r['bound'], r['V12'] / r['Vstar'], r['P12'] / T, err, r['active'], r['collapsed'], fa, r['psnr'], r['ssim'], r['lpips'], hp, d, star.strip() == '*', o['hit'], o['dpsnr']))
    hits = [mip[(s, T)]['hit'] for s in OM]; ds = [mip[(s, T)]['dpsnr'] for s in OM]; oh = [old[(s, T)]['hit'] for s in OM]; od = [old[(s, T)]['dpsnr'] for s in OM]
    stats[T] = (hits, ds)
    P('  %g MB: hit %.3f-%.3f (mean %.3f; v7b %.3f), within +-5 %%: %d/11, within +-10 %%: %d/11 (v7b %d/11); phase B active on %d/11 rows; equal-bytes dPSNR vs hull %+.2f..%+.2f (mean %+.3f; v7b %+.3f)'
      % (T, min(hits), max(hits), sum(hits) / 11, sum(oh) / 11, sum(0.95 <= h <= 1.05 for h in hits), sum(0.9 <= h <= 1.1 for h in hits), sum(0.9 <= h <= 1.1 for h in oh), sum(mip[(s, T)]['active'] > 0 for s in OM), min(ds), max(ds), sum(ds) / 11, sum(od) / 11))
allh = [h for T in stats for h in stats[T][0]]; alld = [d for T in stats for d in stats[T][1]]
P('  all 33: mean hit %.3f, within +-5 %% %d/33, within +-10 %% %d/33; mean equal-bytes dPSNR %+.3f (v7b: %.3f, %d/33, %d/33, %+.3f)' % (sum(allh) / 33, sum(0.95 <= h <= 1.05 for h in allh), sum(0.9 <= h <= 1.1 for h in allh), sum(alld) / 33,
      sum(old[(s, T)]['hit'] for T in (5., 10., 20.) for s in OM) / 33, sum(0.95 <= old[(s, T)]['hit'] <= 1.05 for T in (5., 10., 20.) for s in OM), sum(0.9 <= old[(s, T)]['hit'] <= 1.1 for T in (5., 10., 20.) for s in OM), sum(old[(s, T)]['dpsnr'] for T in (5., 10., 20.) for s in OM) / 33))
P('columns: P/dl = final probe / delivered; active / coll = phase-B rounds that collapsed / pairs; 1st = first active round; V12/V* = vertices after the Delaunay over the plan; P12.5/T = first probe over target;')
P('  lastEst err = last-round landing estimate vs the final probe; hull = fixed-lambda upper hull at the delivered bytes (* extrapolated beyond lam 1.85e-2 / 6.2e-4).')
P('')
P('2. landing, NeRF-Synthetic')
P('scene       T |  deliv   hit  P/dl | capped bound | V12/V*  P12.5/T | lastEst err | active collapsed | best   val    test  | chord  dPSNR | v7b hit')
for T in (2.5, 4.0):
    for s in OB:
        if (s, T) not in bld: P('%-9s %4g |  (no row: smoke-stage failure, resubmit with SMOKE=0)' % (s, T)); continue
        r = bld[(s, T)]; hp, star = v7b.interp(refs[s], r['deliv']) if s in refs else (float('nan'), '?'); d = r['psnr'] - hp
        err = (r['last_est'] / r['probe'] - 1.0) * 100 if r['last_est'] else float('nan'); fa = next((x[0] for x in r['rounds'] if x[10] > 0), None); o = old.get((s, T), {})
        r.update(hull_psnr=hp, dpsnr=d, star=star, err=err, fa=fa)
        P('%-9s %4g |  %6.3f  %.3f  %.3f | %-5s  %2d/17 |  %.2f    %.3f   |   %+5.1f %%   |  %2d    %6d    | %5d  %.3f  %.3f | %.3f %+.3f%s |  %s'
          % (s, T, r['deliv'], r['hit'], r['pd'], r['capped'], r['bound'], r['V12'] / r['Vstar'], r['P12'] / T, err, r['active'], r['collapsed'], r['best'], r['val'] or float('nan'), r['psnr'], hp, d, star, ('%.3f' % o['hit']) if o else '-'))
        csv.append('nerf-synthetic,%s,%g,%.3f,%.3f,%.3f,%.3f,%s,%d,%.3f,%.3f,%.1f,%d,%d,%s,%.3f,%.4f,%.4f,%.3f,%.3f,%s,%s,%s'
                   % (s, T, r['deliv'], r['hit'], r['probe'], r['pd'], r['capped'], r['bound'], r['V12'] / r['Vstar'], r['P12'] / T, err, r['active'], r['collapsed'], fa, r['psnr'], r['ssim'], r['lpips'], hp, d, star.strip() == '*', o.get('hit', ''), o.get('dpsnr', '')))
b25 = [bld[(s, 2.5)] for s in OB if (s, 2.5) in bld and s != 'ficus']; b4 = [bld[(s, 4.0)] for s in OB if (s, 4.0) in bld and s not in ('ficus', 'mic')]
P('  2.5 MB without ficus (%d rows): hit %.3f-%.3f (mean %.3f), dPSNR vs chord %+.3f..%+.3f (mean %+.3f); 4 MB without ficus / mic (%d rows): hit %.3f-%.3f (mean %.3f), dPSNR %+.3f..%+.3f (mean %+.3f)'
  % (len(b25), min(r['hit'] for r in b25), max(r['hit'] for r in b25), sum(r['hit'] for r in b25) / len(b25), min(r['dpsnr'] for r in b25), max(r['dpsnr'] for r in b25), sum(r['dpsnr'] for r in b25) / len(b25),
     len(b4), min(r['hit'] for r in b4), max(r['hit'] for r in b4), sum(r['hit'] for r in b4) / len(b4), min(r['dpsnr'] for r in b4), max(r['dpsnr'] for r in b4), sum(r['dpsnr'] for r in b4) / len(b4)))
P('  ficus 2.5 / 4 and mic 4: lambda at lam_min (1e-4) from the first intervals and the upstream add_percentage limiter binding at 9500: the scene has no residual left to buy at any price (model-size class limit, as in v7b); growth_capped stays False because the limiter bound only the last 1-2 intervals (flag definition: majority of intervals).')
P('')
P('3. the two residual mechanisms (read from the logs)')
ind = [(s, T) for T in (5., 10., 20.) for s in OM if s in INDOOR]; outd = [(s, T) for T in (5., 10., 20.) for s in OM if s not in INDOOR]
P('  a. over-target indoor rows (%s): the first probe reads %.2f-%.2f x target and the landing estimate stays UNDER target for 3-9 rounds (kitchen 10: 9.43 / 8.51 / 9.24 / 9.79 MB at 12.5-14k, final 11.34),'
  % (' '.join('%s%g' % k for k in ind if mip[k]['hit'] > 1.05), min(mip[k]['P12'] / k[1] for k in ind), max(mip[k]['P12'] / k[1] for k in ind)))
P('     because the kept set keeps growing under the opacity ramp (un-stacking: indoor final / first probe = %.2f-%.2f); the gate opens late and 7-11 rounds collapse %d-%d pairs (0.8-1 %% of V per round, the foldover / independent-set limit, far below the 15 %% cap) without reaching the target.'
  % (min(mip[k]['probe'] / mip[k]['P12'] for k in ind), max(mip[k]['probe'] / mip[k]['P12'] for k in ind), min(mip[k]['collapsed'] for k in ind if mip[k]['hit'] > 1.05), max(mip[k]['collapsed'] for k in ind if mip[k]['hit'] > 1.05)))
P('     -> the lever is trimming authority / an earlier gate, not the aim: more rounds (every 250 it from 12500: 30 rounds, +8 min/run, sbatch flag) or a prior on the un-stacking growth.')
P('  b. under-target outdoor 20 MB rows (%s): V12/V* %.2f-%.2f (the last interval admitted 88-92 %% of the asked splits: rank-yield error with no interval left to correct), first probe %.2f-%.2f x target, phase B never opened;'
  % (' '.join('%s%g' % k for k in outd if mip[k]['hit'] < 0.95), min(mip[k]['V12'] / mip[k]['Vstar'] for k in outd if k[1] == 20.), max(mip[k]['V12'] / mip[k]['Vstar'] for k in outd if k[1] == 20.), min(mip[k]['P12'] / 20 for k in outd if k[1] == 20.), max(mip[k]['P12'] / 20 for k in outd if k[1] == 20.)))
P('     the implied landing of these scenes (kappa 0.91-0.92 x topology 3.4-3.8 B/V, kappa-analysis-v1) was 0.95-0.97 with the 1.25 aim, so the margin is thin for the outdoor class at 20 MB and the yield shortfall takes the rest.')
P('')
P('4. quality: equal-bytes dPSNR vs the fixed-lambda hull, all %d Mip/T&T rows: %+.3f..%+.3f, mean %+.3f (v7b mean %+.3f); rows below the hull by > 0.1 dB: %s'
  % (33, min(alld), max(alld), sum(alld) / 33, sum(old[(s, T)]['dpsnr'] for T in (5., 10., 20.) for s in OM) / 33, ' '.join('%s%g(%+.2f)' % (s, T, mip[(s, T)]['dpsnr']) for T in (5., 10., 20.) for s in OM if mip[(s, T)]['dpsnr'] < -0.1) or 'none'))
P('   phase B active (26 rows) vs inactive (7): mean dPSNR %+.3f vs %+.3f' % (sum(mip[k]['dpsnr'] for k in mip if mip[k]['active'] > 0) / sum(1 for k in mip if mip[k]['active'] > 0), sum(mip[k]['dpsnr'] for k in mip if mip[k]['active'] == 0) / max(1, sum(1 for k in mip if mip[k]['active'] == 0))))
open(os.path.join(RES, 'adapt-v9b-sweep-analysis.txt'), 'w').write('\n'.join(out) + '\n'); open(os.path.join(RES, 'adapt-v9b-sweep.csv'), 'w').write('\n'.join(csv) + '\n')
tex = ['% adapt-v9b sweep: delivered MB (hit) and equal-bytes dPSNR vs the fixed-lambda hull; generated by adapt_analysis_v9b_sweep.py', '\\begin{tabular}{@{}l' + 'rr' * 3 + '@{}}', '\\toprule',
       'scene & \\multicolumn{2}{c}{5 MB} & \\multicolumn{2}{c}{10 MB} & \\multicolumn{2}{c}{20 MB} \\\\', ' & MB (hit) & $\\Delta$PSNR & MB (hit) & $\\Delta$PSNR & MB (hit) & $\\Delta$PSNR \\\\', '\\midrule']
for s in OM:
    tex.append(s + ''.join(' & %.2f (%.2f) & %+.2f%s' % (mip[(s, T)]['deliv'], mip[(s, T)]['hit'], mip[(s, T)]['dpsnr'], '$^*$' if mip[(s, T)]['star'].strip() == '*' else '') for T in (5., 10., 20.)) + ' \\\\')
tex += ['\\midrule', 'mean' + ''.join(' & %.2f (%.3f) & %+.3f' % (sum(mip[(s, T)]['deliv'] for s in OM) / 11, sum(stats[T][0]) / 11, sum(stats[T][1]) / 11) for T in (5., 10., 20.)) + ' \\\\', '\\bottomrule', '\\end{tabular}']
open(os.path.join(RES, 'adapt-v9b-sweep-table.tex'), 'w').write('\n'.join(tex) + '\n')
print('\n'.join(out))
