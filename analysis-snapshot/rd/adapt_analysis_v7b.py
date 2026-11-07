#!/usr/bin/env python3
"""adapt-v7b Blender testbed analysis (2026-11-06, node-batch8-v2): lego, chair x {2.5, 4} MB.

Inputs (all in analysis/rd/):
  adapt-v7b-blender-2026-11-06.txt        4 result rows written by adapt-v7b-blender.sbatch (VERSION v7b-r2)
  adapt-v6-blender-2026-11-06.txt         6 v6 rows (same scenes, 2.5/4/6 MB) for the PSNR comparison
  redeliver-v3-blender-2026-11-06.txt     4 fixed-lambda references re-delivered with val_select_v3
  adapt-v7b-blender-2026-11-06-paste.md   raw paste: per-run '[RD adapt N] probe:' / '[RD prune N]' lines
Output: adapt-v7b-blender-analysis.txt

Pass bar set before the run: (a) feasible hits 0.90-1.10 and probe/delivered ~1.0; (b) chair 4 growth-capped True;
(c) last-round landing estimate within +-5 % of the final probe; (d) test PSNR >= v6 at equal bytes.
"""
import ast
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = os.path.join(HERE, 'adapt-v7b-blender-2026-11-06.txt')
ROWS6 = os.path.join(HERE, 'adapt-v6-blender-2026-11-06.txt')
REFS = os.path.join(HERE, 'redeliver-v3-blender-2026-11-06.txt')
PASTE = os.path.join(HERE, 'adapt-v7b-blender-2026-11-06-paste.md')
OUT = os.path.join(HERE, 'adapt-v7b-blender-analysis.txt')


def fnum(pat, s, cast=float):
    m = re.search(pat, s)
    if not m:
        raise SystemExit('pattern not found: %r in %r' % (pat, s[:120]))
    return cast(m.group(1))


def parse_common(line):
    r = {}
    r['scene'] = line.split('|')[0].strip()
    r['T'] = fnum(r'target ([0-9.]+) MB', line)
    r['best'] = fnum(r'best iter (\d+)', line, int)
    r['val'] = fnum(r'best iter \d+ val ([0-9.]+)', line)
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
    m = re.search(r'V (\d+) -> (\d+)\)', line)
    r['V12'], r['V19'] = int(m.group(1)), int(m.group(2))
    r['kF_rec'] = fnum(r'recomputed (\d+)', line, int)
    r['kF_fin'] = fnum(r'final-floor-4x (\d+)', line, int)
    return r


def parse_rows_v7(path):
    runs = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = parse_common(line)
        m = re.search(r'growth-capped (True|False)\s*\(last landing est ([0-9.]+) MB\)', line)
        r['capped'], r['last_est'] = m.group(1) == 'True', float(m.group(2))
        r['active'] = fnum(r'prune\s*rounds \d+ active (\d+)', line, int)
        lists = re.findall(r'\[\(.*?\)\]', line)
        assert len(lists) == 2, (r['scene'], r['T'], len(lists))
        r['extra'] = ast.literal_eval(lists[0])
        r['rounds'] = ast.literal_eval(lists[1])
        assert len(r['rounds']) == 16, (r['scene'], r['T'], len(r['rounds']))
        runs[(r['scene'], r['T'])] = r
    return runs


def parse_rows_v6(path):
    runs = {}
    for line in open(path):
        line = line.strip()
        if line:
            r = parse_common(line)
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
        refs.setdefault(scene, {})[lam] = dict(mb=float(v3.group(1)), psnr=float(v3.group(4)), best=fnum(r'best iter (\d+)', line, int),
                                               val=fnum(r'val ([0-9.]+) \(pruned', line))
    return refs


def parse_paste(path):
    """per run: rounds with probe/est/wanted and, for active rounds, wanted/cap/collapsed; held rounds; final line."""
    runs, cur = {}, None
    for raw in open(path):
        line = raw.rstrip('\n')
        m = re.match(r'--- /scratch/zr2245/(lego|chair)-adapt7b-t([0-9.]+)mb', line)
        if m:
            cur = (m.group(1), float(m.group(2)))
            runs[cur] = dict(rounds=[], final=None)
            continue
        if cur is None:
            continue
        m = re.match(r'\[RD adapt (\d+)\] probe: (\d+) V (\d+) F after cleaning -> ([0-9.]+) MB \(([0-9.]+) x target\), '
                     r'extrapolated to 24000: ([0-9.]+) MB \(([0-9.]+) x, slope ([+-][0-9.]+) MB/1k it\).*?'
                     r'([0-9.]+) B per collapse -> (-?\d+) collapses wanted', line)
        if m:
            runs[cur]['rounds'].append(dict(it=int(m.group(1)), kV=int(m.group(2)), kF=int(m.group(3)), probe=float(m.group(4)),
                                            est=float(m.group(6)), slope=float(m.group(8)), b_coll=float(m.group(9)), wanted=int(m.group(10)),
                                            mode='none', pairs=0, cap=0, plan=0, excess=0, admitted=0, flip=0, passes=0))
            continue
        m = re.match(r'\[RD prune (\d+)\] landing ([0-9.]+) MB under target [0-9.]+ MB: no collapse this round \(lambda-prune held; (\d+) edges below lam ([0-9.e+-]+)\)', line)
        if m:
            rd = runs[cur]['rounds'][-1]
            assert rd['it'] == int(m.group(1))
            rd.update(mode='none', admitted=int(m.group(3)), lam=float(m.group(4)))
            continue
        m = re.match(r'\[RD prune (\d+)\] V (\d+)->(\d+) T (\d+)->(\d+) \| (capped|count|lambda)-driven: wanted (-?\d+) pairs \(per-round plan (-?\d+), whole excess (-?\d+)\), '
                     r'lam-admitted (\d+) edges, cap (\d+) -> (\d+)\s*collapsed \(([0-9.]+) MB exact: (\d+) kept vertices \+ (\d+) kept faces; ([0-9.]+) B per pair\) '
                     r'\(flip-rejected (\d+), passes (\d+)\) \| lam ([0-9.e+-]+)', line)
        if m:
            rd = runs[cur]['rounds'][-1]
            assert rd['it'] == int(m.group(1)), (cur, rd['it'], line[:40])
            rd.update(Vb=int(m.group(2)), Va=int(m.group(3)), Tb=int(m.group(4)), Ta=int(m.group(5)), mode=m.group(6), wanted_p=int(m.group(7)),
                      plan=int(m.group(8)), excess=int(m.group(9)), admitted=int(m.group(10)), cap=int(m.group(11)), pairs=int(m.group(12)),
                      removed_mb=float(m.group(13)), kept_removed=int(m.group(14)), kept_faces=int(m.group(15)), b_pair=float(m.group(16)),
                      flip=int(m.group(17)), passes=int(m.group(18)), lam=float(m.group(19)))
            continue
        m = re.search(r'Training is done; vertices (\d+) triangles (\d+) \| adapt-v7b: target ([0-9.]+) MB, final probe ([0-9.]+) MB .*growth-capped (True|False) \(limiter bound (\d+) of (\d+) intervals, last landing estimate ([0-9.]+)\)', line)
        if m:
            runs[cur]['final'] = dict(V=int(m.group(1)), F=int(m.group(2)), probe=float(m.group(4)), capped=m.group(5) == 'True',
                                      bound=int(m.group(6)), nint=int(m.group(7)), last_est=float(m.group(8)))
    return runs


def chord(refs_scene, mb):
    """log-linear PSNR between the two reference points at bytes mb; '*' if extrapolated; also the slope per ln(bytes)."""
    pts = sorted(refs_scene.values(), key=lambda d: d['mb'])
    (m0, p0), (m1, p1) = (pts[0]['mb'], pts[0]['psnr']), (pts[1]['mb'], pts[1]['psnr'])
    slope = (p1 - p0) / (math.log(m1) - math.log(m0))
    t = (math.log(mb) - math.log(m0)) / (math.log(m1) - math.log(m0))
    return p0 + t * (p1 - p0), ('*' if (t < 0 or t > 1) else ' '), slope


def main():
    rows = parse_rows_v7(ROWS)
    rows6 = parse_rows_v6(ROWS6)
    refs = parse_refs(REFS)
    logs = parse_paste(PASTE)
    order = sorted(rows, key=lambda k: (k[1], k[0] != 'lego'))
    out = []
    P = out.append
    P('adapt-v7b Blender testbed (2026-11-06, node-batch8-v2, jobs 22079433-36): lego, chair x {2.5, 4} MB; train_rd_adapt_v7b (v6 + exact kept-aware accounting,')
    P('  prune gated on the extrapolated landing and capped at the excess, growth-capped flag, diagnostic probes 22k/26k) + val_select_v3 + level-1.0')
    P('references: rd-prune3b4 lam 6.2e-4 / 6.2e-3 re-delivered with val_select_v3; v6 rows from node-batch7 (same scenes, same delivery)')
    P('')
    P('1. landing (pass bar a-c)')
    P('scene     T |  deliv   hit |  probe  hitP  P/dl | capped bound | lastEst  err% |   P12 e12.5 | active collapsed  V12->V19.5 (-%) | best   val    test   val-test')
    for key in order:
        r, lg = rows[key], logs[key]
        assert lg['final'] and abs(lg['final']['probe'] - r['probe']) < 2e-3 and lg['final']['capped'] == r['capped'], key
        rd = r['rounds']
        err = 100.0 * (r['last_est'] / r['probe'] - 1.0)
        P('%-5s %5.1f |  %.3f  %.3f |  %.3f  %.3f  %.3f | %-5s  %2d/17 |  %.3f  %+5.1f |  %.2f  %.2f  |  %2d     %6d   %6d->%6d (%4.1f) | %5d  %.3f  %.3f  %.3f'
          % (key[0], key[1], r['deliv'], r['hit'], r['probe'], r['probe'] / r['T'], r['pd'], r['capped'], r['bound'], r['last_est'], err,
             rd[0][1] / r['T'], rd[1][2] / r['T'], r['active'], r['collapsed'], r['V12'], r['V19'], 100.0 * (1 - r['V19'] / r['V12']),
             r['best'], r['val'], r['psnr'], r['val'] - r['psnr']))
    P('columns: hitP = final probe / T; P/dl = probe / delivered file; capped = growth_capped flag (last est < T and limiter bound > half the intervals);')
    P('  lastEst = landing estimate at the last round (19.5k), err = lastEst / final probe - 1; P12, e12.5 = probe at 12k and estimate at 12.5k over T;')
    P('  active = rounds that collapsed anything; val = unpruned val PSNR at the chosen checkpoint; val-test = generalisation gap.')
    P('')
    P('2. quality (pass bar d): test PSNR vs the fixed-lambda chord and vs the v6 run at the same target')
    P('scene     T |  bytes   PSNR | chord   dChord | v6 bytes v6 PSNR | bytes x  dPSNR raw  dPSNR eq-bytes | v6 pruned-V% v6 best')
    for key in order:
        r, r6 = rows[key], rows6[key]
        refp, star, slope = chord(refs[key[0]], r['deliv'])
        eq = r['psnr'] - (r6['psnr'] + slope * math.log(r['deliv'] / r6['deliv']))
        P('%-5s %5.1f |  %.3f  %.3f | %.3f  %+.3f%s | %.3f    %.3f   |  %.3f    %+.3f      %+.3f         |   %4.1f        %5d'
          % (key[0], key[1], r['deliv'], r['psnr'], refp, r['psnr'] - refp, star, r6['deliv'], r6['psnr'], r['deliv'] / r6['deliv'],
             r['psnr'] - r6['psnr'], eq, 100.0 * (1 - r6['V19'] / r6['V12']), r6['best']))
    P('columns: chord = log-linear PSNR between the two re-delivered references at the delivered bytes (* = extrapolated below 2.391 / 2.763 MB);')
    P('  dPSNR eq-bytes = v7b PSNR minus the v6 PSNR moved to the v7b bytes along the chord slope (lego %.3f, chair %.3f dB per ln(bytes)).'
      % (chord(refs['lego'], 3.0)[2], chord(refs['chair'], 3.0)[2]))
    P('  val-test gaps for context: refs lego 6.2e-3 %.3f (2.76 MB), 6.2e-4 %.3f (7.52 MB); chair 6.2e-3 %.3f (2.39 MB), 6.2e-4 %.3f (4.66 MB); v6 chair 4/6: %.3f / %.3f'
      % (refs['lego'][0.0062]['val'] - refs['lego'][0.0062]['psnr'], refs['lego'][0.00062]['val'] - refs['lego'][0.00062]['psnr'],
         refs['chair'][0.0062]['val'] - refs['chair'][0.0062]['psnr'], refs['chair'][0.00062]['val'] - refs['chair'][0.00062]['psnr'],
         rows6[('chair', 4.0)]['val'] - rows6[('chair', 4.0)]['psnr'], rows6[('chair', 6.0)]['val'] - rows6[('chair', 6.0)]['psnr']))
    P('')
    P('3. phase B per round (it, probe, est, wanted, mode, cap, collapsed, collapsed/cap, kept V removed, passes)')
    for key in order:
        lg = logs[key]
        P('  %s %.1f MB:' % key)
        for x in lg['rounds']:
            if x['mode'] == 'none':
                P('    %5d  %.3f  %.3f  %7d  %-6s  held (%d edges below lam %.2e)' % (x['it'], x['probe'], x['est'], x['wanted'], x['mode'], x['admitted'], x['lam']))
            else:
                P('    %5d  %.3f  %.3f  %7d  %-6s  %6d  %6d  %.2f  %6d  %2d' % (x['it'], x['probe'], x['est'], x['wanted'], x['mode'], x['cap'], x['pairs'], x['pairs'] / max(x['cap'], 1), x['kept_removed'], x['passes']))
    P('')
    P('4. collapse throughput (independent-set matching + foldover filter deliver only a fraction of the capped count per round)')
    for key in order:
        act = [x for x in logs[key]['rounds'] if x['mode'] != 'none']
        if not act:
            P('  %-5s %.1f: no active round (gate closed in all 16)' % key)
            continue
        ratios = [x['pairs'] / max(x['cap'], 1) for x in act]
        P('  %-5s %.1f: %d active rounds, collapsed %d of cap-sum %d (%.2f); per-round collapsed/cap %.2f-%.2f; last round wanted %d -> collapsed %d'
          % (key[0], key[1], len(act), sum(x['pairs'] for x in act), sum(x['cap'] for x in act), sum(x['pairs'] for x in act) / sum(x['cap'] for x in act),
             min(ratios), max(ratios), act[-1]['cap'], act[-1]['pairs']))
    P('')
    P('5. summary')
    feas = [k for k in order if not rows[k]['capped'] or rows[k]['hit'] >= 0.95]
    P('  (a) hits: %s; feasible runs (flag False or hit >= 0.95) all within 0.90-1.10: %s; probe/delivered %s'
      % (' '.join('%s %.1f %.3f' % (k[0], k[1], rows[k]['hit']) for k in order), all(0.9 <= rows[k]['hit'] <= 1.1 for k in feas),
         ' '.join('%.3f' % rows[k]['pd'] for k in order)))
    P('  (b) growth-capped: %s (chair 4 True as expected; lego 4 True at a 1 %% shortfall and lands 0.994 -> in tables treat capped = flag AND hit < 0.95)'
      % ' '.join('%s %.1f %s' % (k[0], k[1], rows[k]['capped']) for k in order))
    P('  (c) last-round estimate error: %s %% (bar +-5 %%)' % ' '.join('%+.1f' % (100.0 * (rows[k]['last_est'] / rows[k]['probe'] - 1.0)) for k in order))
    eqs = {}
    for k in order:
        _, _, slope = chord(refs[k[0]], rows[k]['deliv'])
        eqs[k] = rows[k]['psnr'] - (rows6[k]['psnr'] + slope * math.log(rows[k]['deliv'] / rows6[k]['deliv']))
    P('  (d) equal-bytes dPSNR vs v6: %s -> PASS on the two runs where phase B collapsed (2.5 MB), FAIL on the two gate-closed runs (4 MB):'
      % ' '.join('%s %.1f %+.2f' % (k[0], k[1], eqs[k]) for k in order))
    P('      the 4 MB runs collapsed nothing (est < T all 16 rounds) and are pure phase-A models chosen at 26000; v6 removed 16-18 % of V12 by lambda-prune')
    P('      and landed under target with HIGHER test PSNR: the gradual lambda-prune buys test PSNR on these sparse-view scenes (val-test gap 0.22-0.25 pruned')
    P('      vs 0.84-0.88 unpruned on chair). The v7 gate is right for landing and wrong for RD on non-binding targets; v6 was the reverse.')
    open(OUT, 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))
    print('\nwritten', OUT)


if __name__ == '__main__':
    main()
