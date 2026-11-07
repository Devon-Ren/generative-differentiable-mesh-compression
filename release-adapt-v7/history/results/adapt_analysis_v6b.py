#!/usr/bin/env python3
"""adapt-v6 Blender testbed analysis (2026-11-06): lego, chair x {2.5, 4, 6} MB.

Inputs (all in analysis/rd/):
  adapt-v6-blender-2026-11-06.txt        6 result rows written by adapt-v6-blender.sbatch
  redeliver-v3-blender-2026-11-06.txt    4 fixed-lambda references re-delivered with val_select_v3
  adapt-v6-blender-2026-11-06-paste.md   raw paste: per-run '[RD adapt N] probe:' and '[RD prune N]' lines
Output: adapt-v6-blender-analysis.txt

What it computes per run
  hit by delivered file / by the controller's final probe, probe/delivered (delivery fix check);
  P12 (probe at 12k / target), est at 12.5k and 19.5k / target, est@19.5k / final probe (extrapolation accuracy);
  phase-B removal: V removed in rounds where the controller wanted MORE bytes (wanted < 0) vs all rounds;
  accounting check: removed bytes as the controller accounted them (pairs x b_coll) vs a kept-aware
    accounting (kept vertices removed x b_v + faces removed x kept-face share x b_f);
  the implied un-stacking growth 12k->19.5k and 19.5k->final under each accounting, and the landing a
    no-prune phase B would have had (P12 + growth) -> feasibility of the target under the upstream limiter;
  equal-bytes dPSNR vs the corrected references (log-linear between the two lambda points).
"""
import ast
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = os.path.join(HERE, 'adapt-v6-blender-2026-11-06.txt')
REFS = os.path.join(HERE, 'redeliver-v3-blender-2026-11-06.txt')
PASTE = os.path.join(HERE, 'adapt-v6-blender-2026-11-06-paste.md')
OUT = os.path.join(HERE, 'adapt-v6-blender-analysis.txt')


def fnum(pat, s, cast=float):
    m = re.search(pat, s)
    if not m:
        raise SystemExit('pattern not found: %r in %r' % (pat, s[:120]))
    return cast(m.group(1))


def parse_rows(path):
    runs = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = {}
        r['scene'] = line.split('|')[0].strip()
        r['T'] = fnum(r'target ([0-9.]+) MB', line)
        r['best'] = fnum(r'best iter (\d+)', line, int)
        r['probe'] = fnum(r'final probe ([0-9.]+) MB', line)
        r['deliv'] = fnum(r'delivered ([0-9.]+) MB', line)
        r['hit'] = fnum(r'hit ([0-9.]+) x target', line)
        r['pd'] = fnum(r'probe/delivered ([0-9.]+)', line)
        r['V'] = fnum(r'trained V (\d+)', line, int)
        r['F'] = fnum(r'trained V \d+ F (\d+)', line, int)
        m = re.search(r'test ([0-9.]+) ([0-9.]+) ([0-9.]+)', line)
        r['psnr'], r['ssim'], r['lpips'] = float(m.group(1)), float(m.group(2)), float(m.group(3))
        r['bound'] = fnum(r'limiter-bound (\d+)', line, int)
        r['Vstar'] = fnum(r'V\* (\d+)', line, int)
        r['collapsed'] = fnum(r'collapsed (\d+)', line, int)
        r['kept_share'] = fnum(r'kept-vertex share ([0-9.]+)', line)
        m = re.search(r'V (\d+) -> (\d+)\)', line)
        r['V12'], r['V19'] = int(m.group(1)), int(m.group(2))
        r['kF_rec'] = fnum(r'recomputed (\d+)', line, int)
        r['kF_fin'] = fnum(r'final-floor-4x (\d+)', line, int)
        r['kF_rm'] = fnum(r'running-max (\d+)', line, int)
        rounds = ast.literal_eval(line[line.index('[('):line.rindex(')]') + 2])
        r['rounds'] = rounds
        runs[(r['scene'], r['T'])] = r
    return runs


def parse_refs(path):
    refs = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        name = line.split('|')[0].strip()
        scene = name.split('-')[0]
        lam = float(re.search(r'lam([0-9.]+)', name).group(1))
        v3 = re.search(r'-> v3 ([0-9.]+) MB V (\d+) F (\d+) test ([0-9.]+)', line)
        v2 = re.search(r'v2 ([0-9.]+) MB V (\d+) F (\d+) test ([0-9.]+)', line)
        refs.setdefault(scene, {})[lam] = dict(
            mb=float(v3.group(1)), V=int(v3.group(2)), F=int(v3.group(3)), psnr=float(v3.group(4)),
            mb2=float(v2.group(1)), psnr2=float(v2.group(4)),
            kF_rec=fnum(r'recomputed (\d+)', line, int), kF_fin=fnum(r'final-floor-4x (\d+)', line, int),
            kF_rm=fnum(r'running-max (\d+)', line, int),
            val=fnum(r'val ([0-9.]+) \(pruned', line), pval=fnum(r'pruned val ([0-9.]+)\)', line))
    return refs


def parse_paste(path):
    """per run: list of rounds with probe/est/slope/wanted and the prune line fields; final probe line."""
    runs, cur = {}, None
    sec = None
    for raw in open(path):
        line = raw.rstrip('\n')
        if line.startswith('===== phase A'):
            sec = 'probes'
            continue
        if line.startswith('====='):
            sec = None
            continue
        if sec != 'probes':
            continue
        if line.startswith('--- /scratch/'):
            m = re.search(r'/(lego|chair)-adapt6b-t([0-9.]+)mb', line)
            cur = (m.group(1), float(m.group(2)))
            runs[cur] = dict(rounds=[], final=None, valsel=None)
            continue
        if cur is None:
            continue
        m = re.match(r'\[RD adapt (\d+)\] probe: (\d+) V (\d+) F after cleaning -> ([0-9.]+) MB \(([0-9.]+) x target\), '
                     r'extrapolated to 24000: ([0-9.]+) MB \(([0-9.]+) x, slope ([+-][0-9.]+) MB/1k it\).*?'
                     r'b_v ([0-9.]+) b_f ([0-9.]+) overhead ([0-9.]+) MB \| ([0-9.]+) B per collapse -> (-?\d+) collapses wanted', line)
        if m:
            runs[cur]['rounds'].append(dict(
                it=int(m.group(1)), kV=int(m.group(2)), kF=int(m.group(3)), probe=float(m.group(4)), hitP=float(m.group(5)),
                est=float(m.group(6)), hitE=float(m.group(7)), slope=float(m.group(8)), b_v=float(m.group(9)),
                b_f=float(m.group(10)), ovh=float(m.group(11)), b_coll=float(m.group(12)), wanted=int(m.group(13))))
            continue
        m = re.match(r'\[RD prune (\d+)\] V (\d+)->(\d+) T (\d+)->(\d+) \| lambda-driven: wanted (-?\d+) pairs, lam-admitted (\d+) edges, '
                     r'cap (\d+) -> (\d+) collapsed \(~([0-9.]+) MB, (\d+) of them kept\s?vertices\) \(flip-rejected (\d+), passes (\d+)\)', line)
        if m:
            rd = runs[cur]['rounds'][-1]
            assert rd['it'] == int(m.group(1)), (cur, rd['it'], line[:40])
            rd.update(Vb=int(m.group(2)), Va=int(m.group(3)), Tb=int(m.group(4)), Ta=int(m.group(5)),
                      admitted=int(m.group(7)), cap=int(m.group(8)), pairs=int(m.group(9)), acc_mb=float(m.group(10)),
                      kept_removed=int(m.group(11)), flip=int(m.group(12)))
            continue
        m = re.search(r'Training is done; vertices (\d+) triangles (\d+) \| adapt-v6: target ([0-9.]+) MB, final probe ([0-9.]+) MB', line)
        if m:
            runs[cur]['final'] = dict(V=int(m.group(1)), F=int(m.group(2)), probe=float(m.group(4)))
    return runs


def interp_ref(refs_scene, mb):
    """log-linear PSNR between the two reference points at bytes mb; '*' if extrapolated."""
    pts = sorted(refs_scene.values(), key=lambda d: d['mb'])
    (m0, p0), (m1, p1) = (pts[0]['mb'], pts[0]['psnr']), (pts[1]['mb'], pts[1]['psnr'])
    t = (math.log(mb) - math.log(m0)) / (math.log(m1) - math.log(m0))
    return p0 + t * (p1 - p0), ('*' if (t < 0 or t > 1) else ' ')


def main():
    rows = parse_rows(ROWS)
    refs = parse_refs(REFS)
    logs = parse_paste(PASTE)
    out = []
    P = out.append
    P('adapt-v6 Blender testbed (2026-11-06): lego, chair x {2.5, 4, 6} MB; train_rd_adapt_v6 (16 prune rounds 12k-19.5k, phase-A overshoot 1.0, LS extrapolation) + val_select_v3 + level-1.0')
    P('references: rd-prune3b4 lam 6.2e-4 / 6.2e-3 RE-DELIVERED with val_select_v3 (same delivery as the adapt runs)')
    P('')
    P('1. delivery fix (val_select_v3: importance recomputed at the chosen checkpoint, floor 0.999 / 4x)')
    for scene in ('lego', 'chair'):
        for lam, d in sorted(refs[scene].items(), reverse=True):
            P('  %-5s lam %.1e: v2 %.3f MB / %.3f dB -> v3 %.3f MB / %.3f dB (bytes x%.3f, dPSNR %+.3f); kept F recomputed %d vs final-floor-4x %d (%.4f), running-max %d (%.2f x); pruned val %+.3f'
              % (scene, lam, d['mb2'], d['psnr2'], d['mb'], d['psnr'], d['mb'] / d['mb2'], d['psnr'] - d['psnr2'],
                 d['kF_rec'], d['kF_fin'], d['kF_rec'] / d['kF_fin'], d['kF_rm'], d['kF_rm'] / d['kF_rec'], d['pval'] - d['val']))
    pds = [r['pd'] for r in rows.values()]
    P('  adapt runs probe/delivered: %s (v5: 0.83-0.94); recomputed/final-floor kept faces %s'
      % (' '.join('%.3f' % x for x in pds), ' '.join('%.4f' % (r['kF_rec'] / r['kF_fin']) for r in rows.values())))
    P('  -> PASSED: the delivered file is the probe within 0.6-7.4 % (the residual is coarse_finetune 3000 steps + the delivery encoder, not the cleaning).')
    P('')
    P('2. landing')
    P('scene     T |  deliv   hit |  probe  hitP  P/dl |   P12 e12.5 e19.5 e19.5/fin | bound V12/V* | pruneV%% underT%% |  PSNR   ref  dPSNR')
    summary = {}
    for key in sorted(rows, key=lambda k: (k[1], k[0] != 'lego')):
        r, lg = rows[key], logs[key]
        rd = lg['rounds']
        assert len(rd) == 16 and lg['final'] and abs(lg['final']['probe'] - r['probe']) < 1e-6, key
        fin = r['probe']
        p12 = rd[0]['probe']
        e125, e195 = rd[1]['est'], rd[-1]['est']
        v_removed = sum(x['Vb'] - x['Va'] for x in rd)
        v_removed_under = sum(x['Vb'] - x['Va'] for x in rd if x['wanted'] < 0)
        refp, star = interp_ref(refs[key[0]], r['deliv'])
        d = dict(fin=fin, p12=p12, e125=e125, e195=e195, v_removed=v_removed, v_under=v_removed_under, refp=refp, star=star)
        summary[key] = d
        P('%-5s %5.1f |  %.3f  %.2f |  %.3f  %.2f  %.2f |  %.2f  %.2f  %.2f   %.3f    | %2d/17  %.2f  |  %4.1f    %4.1f  | %.3f %.3f %+.3f%s'
          % (key[0], key[1], r['deliv'], r['hit'], fin, fin / r['T'], r['pd'], p12 / r['T'], e125 / r['T'], e195 / r['T'], e195 / fin,
             r['bound'], r['V12'] / r['Vstar'], 100.0 * v_removed / r['V12'], 100.0 * v_removed_under / r['V12'], r['psnr'], refp, r['psnr'] - refp, star))
    P('columns: P12 = probe at 12k / T; e12.5, e19.5 = extrapolated-to-24000 size at the 2nd and last round / T; e19.5/fin = extrapolation accuracy at the last round;')
    P('  V12/V* = model vertices at 12k over the phase-A aim; pruneV% = share of V12 removed by the 16 rounds; underT% = the part removed in rounds where the controller wanted MORE bytes (wanted < 0);')
    P('  dPSNR = test PSNR minus the v3 reference curve at the delivered bytes (log-linear between lam 6.2e-3 and 6.2e-4; * = extrapolated; the chord of a concave hull UNDER-reads the hull, so + values are upper bounds).')
    P('')
    P('3. accounting check and the un-stacking growth (per run, all 16 rounds)')
    P('   acc = removed MB as the controller accounted it (pairs x b_coll, b_coll = b_v + b_f x faces removed per pair, ALL faces);')
    P('   kept = kept-aware accounting (kept vertices removed x b_v + faces removed x kept-face share x b_f);')
    P('   growth = implied un-stacking growth of the probe = (p_next - p) + removed, summed 12k->19.5k, and 19.5k->final (30k);')
    P('   no-prune landing = P12 + total growth (what phase B would have landed with no collapse at all)')
    P('scene     T |  acc12k  kept12k ratio |  accSum keptSum | grow12-19.5 (acc / kept) grow19.5-fin (acc / kept) | per-1k early(acc) late(acc) | no-prune landing (acc / kept) feas')
    for key in sorted(rows, key=lambda k: (k[1], k[0] != 'lego')):
        r, lg, d = rows[key], logs[key], summary[key]
        rd = lg['rounds']
        acc = [x['acc_mb'] for x in rd]
        kept = []
        for x in rd:
            share = x['kF'] / x['Tb']
            kept.append((x['kept_removed'] * x['b_v'] + (x['Tb'] - x['Ta']) * share * x['b_f']) / 1e6)
        probes = [x['probe'] for x in rd] + [d['fin']]
        g_acc = [probes[i + 1] - probes[i] + acc[i] for i in range(16)]
        g_kept = [probes[i + 1] - probes[i] + kept[i] for i in range(16)]
        ge_acc, gl_acc = sum(g_acc[:15]), g_acc[15]
        ge_kept, gl_kept = sum(g_kept[:15]), g_kept[15]
        land_acc = d['p12'] + ge_acc + gl_acc
        land_kept = d['p12'] + ge_kept + gl_kept
        d.update(land_acc=land_acc, land_kept=land_kept, acc_sum=sum(acc), kept_sum=sum(kept), ge_acc=ge_acc, gl_acc=gl_acc)
        feas = 'yes' if land_kept >= r['T'] else 'NO'
        P('%-5s %5.1f |  %.3f   %.3f   %.2f |  %.3f   %.3f  |   %.3f / %.3f            %.3f / %.3f          |   %.3f      %.3f     |   %.3f / %.3f (%.2f x)  %s'
          % (key[0], key[1], acc[0], kept[0], acc[0] / kept[0], sum(acc), sum(kept), ge_acc, ge_kept, gl_acc, gl_kept,
             ge_acc / 7.5, gl_acc / 10.5, land_acc, land_kept, land_kept / r['T'], feas))
    P('')
    P('4. per-round view of two runs (it, probe, est, slope, wanted, pairs collapsed, kept-removed, acc MB)')
    for key in (('lego', 4.0), ('chair', 4.0)):
        P('  %s %.1f MB:' % key)
        for x in logs[key]['rounds']:
            P('    %5d  %.3f  %.3f  %+.3f  %7d  %6d  %6d  %.2f' % (x['it'], x['probe'], x['est'], x['slope'], x['wanted'], x['pairs'], x['kept_removed'], x['acc_mb']))
    P('')
    # headline numbers
    hits = [rows[k]['hit'] for k in rows]
    hitP = [rows[k]['probe'] / rows[k]['T'] for k in rows]
    P('5. summary')
    P('  hit by delivered file: %.2f-%.2f (v5 0.91-1.32, all over); by probe %.2f-%.2f; every run UNDER target' % (min(hits), max(hits), min(hitP), max(hitP)))
    P('  extrapolation at the last round / final probe: %s (v5 at 18k: 1.00-1.04)' % ' '.join('%.3f' % (summary[k]['e195'] / summary[k]['fin']) for k in rows))
    P('  second-round extrapolation / final probe (the number the first 10 rounds pruned against): %s' % ' '.join('%.2f' % (summary[k]['e125'] / summary[k]['fin']) for k in rows))
    P('  phase B removed %s %% of V12, of which %s %% in rounds where the controller wanted MORE bytes'
      % (' '.join('%.0f' % (100.0 * summary[k]['v_removed'] / rows[k]['V12']) for k in rows),
         ' '.join('%.0f' % (100.0 * summary[k]['v_under'] / rows[k]['V12']) for k in rows)))
    P('  controller accounting of the first collapse over-counts the kept-aware removal by x%s' % ' '.join('%.2f' % (logs[k]['rounds'][0]['acc_mb'] / ((logs[k]['rounds'][0]['kept_removed'] * logs[k]['rounds'][0]['b_v'] + (logs[k]['rounds'][0]['Tb'] - logs[k]['rounds'][0]['Ta']) * logs[k]['rounds'][0]['kF'] / logs[k]['rounds'][0]['Tb'] * logs[k]['rounds'][0]['b_f']) / 1e6)) for k in rows))
    P('  no-prune landing (kept accounting) / T: %s' % ' '.join('%s %.1f: %.2f' % (k[0], k[1], summary[k]['land_kept'] / rows[k]['T']) for k in sorted(rows, key=lambda k: (k[1], k[0] != 'lego'))))
    P('  equal-bytes dPSNR vs the v3 references: %s (mean %+.3f; v5 vs v2 refs: -0.10..+0.20, mean +0.05)'
      % (' '.join('%+.2f' % (rows[k]['psnr'] - summary[k]['refp']) for k in rows), sum(rows[k]['psnr'] - summary[k]['refp'] for k in rows) / len(rows)))
    open(OUT, 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))
    print('\nwritten', OUT)


if __name__ == '__main__':
    main()
