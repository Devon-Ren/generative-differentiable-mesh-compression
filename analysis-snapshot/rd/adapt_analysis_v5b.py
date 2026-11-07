"""adapt_analysis_v5b (2026-11-06): the 6 adapt-v5 Blender testbed rows (lego, chair x {2.5, 4, 6} MB) vs the two fixed-lambda
reference points per scene (rd-prune3b4, lam 6.2e-4 / 6.2e-3, same protocol).  Hit by the DELIVERED file and by the PROBE meter
(the delivery goes through val_select_v2, whose importance mask is the training-time running max: it keeps more faces than the
final cleaning, see analysis text), probe growth after phase A, phase-B authority, feasibility flag, equal-bytes dPSNR.
Inputs (this directory): adapt-v5-blender-2026-11-06.txt, -ref.txt, -paste.md (probe lines per run)."""
import re, math, collections
def parse_rows(fn):
    rows = []
    for line in open(fn):
        if not line.strip(): continue
        S = line.split('|')[0].strip(); g = lambda pat: re.search(pat, line)
        T = float(g(r'target ([\d.]+) MB').group(1)); best = int(g(r'best iter (\d+)').group(1))
        fp = float(g(r'final probe ([\d.]+) MB').group(1)); V = int(g(r'trained V (\d+)').group(1)); F = int(g(r'F (\d+)').group(1))
        dl = float(g(r'delivered ([\d.]+) MB').group(1)); hit = float(g(r'hit ([\d.]+)').group(1)); pd = float(g(r'probe/delivered ([\d.]+)').group(1))
        t = g(r'test ([\d.]+) ([\d.]+) ([\d.]+)').groups(); ndn = int(g(r'densify events (\d+)').group(1))
        m = g(r'limiter-bound (\d+), front-loaded (\d+), V\* (\d+)'); nb, nf, vs = int(m.group(1)), int(m.group(2)), int(m.group(3))
        pr = [(int(a), float(b), float(c), float(k), int(d), int(e), f, float(h)) for a, b, c, k, d, e, f, h in
              re.findall(r"\((\d+), ([\d.]+), ([\d.]+), ([\d.]+), (\d+), (\d+), '(\w+)', '([\d.e+-]+)'", line)]
        if ndn > 17: ndn, nb, nf = ndn // 2, nb // 2, nf // 2   # chair 2.5: the cancelled first attempt appended its 17 events to the same rd_log
        rows.append(dict(scene=S, target=T, best=best, final_probe=fp, V=V, F=F, delivered=dl, hit=hit, pd=pd, psnr=float(t[0]), prune=pr, n_densify=ndn, n_bound=nb, v_star=vs))
    return rows
def parse_refs(fn):
    ref = collections.defaultdict(list)
    for line in open(fn):
        m = re.match(r'(\w+) lam ([\d.e-]+) .*?delivered ([\d.]+) MB test ([\d.]+)', line)
        if m: ref[m.group(1)].append((float(m.group(3)), float(m.group(4)), float(m.group(2))))
    for S in ref: ref[S].sort()
    return ref
def parse_probes(fn):
    """per run: {iteration: (kept V, kept F, probe MB)} from the '[RD adapt N] probe:' lines under '--- /scratch/zr2245/<scene>-adapt5b-t<T>mb'."""
    out = {}; cur = None
    for line in open(fn):
        m = re.match(r'--- /scratch/zr2245/(\w+)-adapt5b-t([\d.]+)mb', line)
        if m: cur = (m.group(1), float(m.group(2))); out[cur] = {}; continue
        m = re.match(r'\[RD adapt (\d+)\] probe: (\d+) V (\d+) F after cleaning -> ([\d.]+) MB', line)
        if m and cur: out[cur][int(m.group(1))] = (int(m.group(2)), int(m.group(3)), float(m.group(4)))
    return out
def interp(c, x):
    (x0, y0, l0), (x1, y1, l1) = c[0], c[-1]; flag = '' if x0 <= x <= x1 else '*'
    w = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return y0 + w * (y1 - y0), flag
rows = parse_rows('adapt-v5-blender-2026-11-06.txt'); assert len(rows) == 6
ref = parse_refs('adapt-v5-blender-2026-11-06-ref.txt'); assert set(ref) == {'lego', 'chair'} and all(len(v) == 2 for v in ref.values())
probes = parse_probes('adapt-v5-blender-2026-11-06-paste.md')
rows.sort(key=lambda r: (r['scene'] != 'lego', r['target']))
out = ["adapt-v5 Blender testbed (2026-11-06): lego, chair x {2.5, 4, 6} MB; train_rd_adapt_v5 + val_select_v2 + level-1.0; refs = rd-prune3b4 lam 6.2e-4 / 6.2e-3", "",
       f"{'scene':6s} {'T':>4s} | {'deliv':>6s} {'hit':>5s} | {'probe':>6s} {'hitP':>5s} {'P/dl':>5s} | {'P12':>5s} {'P18':>5s} {'est18':>5s} {'fin/P12':>7s} {'est/fin':>7s} | {'V12':>7s} {'V18+':>7s} {'coll%':>5s} {'kV12':>7s} {'kV18':>7s} {'kF12':>7s} {'kF18':>7s} | {'bound':>5s} {'V12/V*':>6s} {'feas':>4s} | {'PSNR':>6s} {'ref':>6s} {'dPSNR':>6s}"]
dps = []
for r in rows:
    S, T, p = r['scene'], r['target'], r['prune']; P12, P18, est18 = p[0][1], p[-1][1], p[-1][2]; V12, V18a = p[0][4], p[-1][5]
    pk = probes.get((S, T), {}); kv12, kf12 = pk.get(12000, (None, None, None))[:2]; kv18, kf18 = pk.get(18000, (None, None, None))[:2]
    feas = not (est18 < T and r['n_bound'] * 2 >= r['n_densify'])
    y, flag = interp(ref[S], r['delivered']); d = r['psnr'] - y; dps.append(d)
    r.update(P12=P12, P18=P18, est18=est18, V12=V12, growth=r['final_probe'] / P12, acc=est18 / r['final_probe'], coll=(V12 - V18a) / V12, kv12=kv12, kv18=kv18, kf12=kf12, kf18=kf18, feas=feas, dpsnr=d, flag=flag, hitP=r['final_probe'] / T)
    out.append(f"{S:6s} {T:4.1f} | {r['delivered']:6.3f} {r['hit']:5.2f} | {r['final_probe']:6.3f} {r['hitP']:5.2f} {r['pd']:5.2f} | {P12:5.2f} {P18:5.2f} {est18:5.2f} {r['growth']:7.2f} {r['acc']:7.2f} | {V12:7d} {V18a:7d} {100*r['coll']:5.1f} {kv12 or 0:7d} {kv18 or 0:7d} {kf12 or 0:7d} {kf18 or 0:7d} | {r['n_bound']:2d}/{r['n_densify']:<2d} {V12/r['v_star']:6.2f} {'yes' if feas else 'NO':>4s} | {r['psnr']:6.3f} {y:6.3f} {d:+6.3f}{flag}")
hits = [r['hit'] for r in rows]; hp = [r['hitP'] for r in rows if r['feas']]
out += ["", "columns: deliv/hit = delivered file (val_select_v2 + level-1.0); probe/hitP = the controller's own final probe (upstream cleaning at 30000, real encoder); P/dl = probe/delivered;",
        "P12/P18 = probe at the first/last prune round, est18 = probe extrapolated to 24000 at the last round; fin/P12 = growth of the delivered set after phase A; est/fin = extrapolation accuracy;",
        "V12 = model vertices at the first round, V18+ = after the last round's collapses, coll% = share removed by phase B; kV/kF = kept vertices/faces of the probe at 12k/18k; bound = limiter-bound densify intervals;",
        "feas = NO when the extrapolated probe at the last round is still under target AND the limiter bound a majority of the intervals (growth-capped target); dPSNR = test PSNR minus the reference curve at the delivered bytes (log-linear between the two reference points, * = extrapolated).",
        "",
        f"hit by delivered file: {min(hits):.2f}-{max(hits):.2f}; by the probe meter (feasible runs): {min(hp):.2f}-{max(hp):.2f}; probe/delivered {min(r['pd'] for r in rows):.2f}-{max(r['pd'] for r in rows):.2f} (Mip-360 adapt-v4: 0.998-1.000)",
        f"  -> the delivered file carries {min(1/r['pd'] for r in rows)-1:.0%}-{max(1/r['pd'] for r in rows)-1:.0%} more bytes than the probe: val_select_v2 prunes with the checkpoint's importance_score, which upstream keeps as a RUNNING MAX of max_blending since the Delaunay rebuild",
        "     (reset only at iteration 29900 = iterations - #train views); the final cleaning re-renders at the final floor.  Faces visible once under the soft floor stay 'important' -> 7-20 % more faces delivered than the probe's cleaning keeps.  lego 2.5: delivered F 666976 vs final model F 561481, V 154537 vs 154281 (vertices unchanged: the surplus is faces).",
        f"extrapolation (est@18k / final probe): {min(r['acc'] for r in rows):.2f}-{max(r['acc'] for r in rows):.2f} (slight over-estimate: the trend adds back n_pairs x b_coll per round, but part of the collapsed vertices were not in the kept set)",
        f"growth of the delivered set after phase A (final / P12): {min(r['growth'] for r in rows):.2f}-{max(r['growth'] for r in rows):.2f}; kept faces 12k->18k " + ', '.join(f"{r['scene']}{r['target']:g} {r['kf12']}->{r['kf18']} ({r['kf18']/r['kf12']:.2f}x)" for r in rows if r['kf12']) ,
        "  kept vertices hold or RISE (" + ', '.join(f"{r['scene']}{r['target']:g} {r['kv12']}->{r['kv18']}" for r in rows if r['kv12']) + ") although phase B removed " + ', '.join(f"{100*r['coll']:.0f} %" for r in rows) + " of V: by 18k nearly every model vertex is kept (lego 2.5: 154199 of 154866 = 99.6 %; final model V 154281), the collapses took vertices that were not (yet) kept, and the kept F/V keeps rising (lego 2.5: 2.63 at 12k -> 3.64 final).",
        "  => the delivered size is b_v x V_model + b_f x F_kept + overhead with F_kept/V still growing after the last round; the probe's kept V/F at 12k-18k under-read the final, which the trend extrapolation covers (est/fin 1.00-1.04).",
        "  phase B's lever (one vertex + ~2 faces per collapse, 12-16 B) is right in kind but its authority is ~3 % of V per round: of ~22k cap-admitted pairs 12-16k are foldover-rejected, 2-6k collapse.  4 rounds -> 7-10 % of V; the un-stacking adds 9-40 % bytes.",
        "phase A: V12/V* " + ', '.join(f"{r['scene']}{r['target']:g} {r['V12']/r['v_star']:.2f}({r['n_bound']}b)" for r in rows) + "; the prior size model (B/v 10, B/face 2, faces/vertex 2.5 -> 11 B per vertex + 0.6 MB) is below the measured per-vertex rate of the kept set (12.9 B/kept V at 12k, 8.2 + 1.8 x 3.6-4.3 = 14.7-16 B at the end), so V* is ~35-45 % too high on Blender; the limiter (13/17 bound) stopped most runs short of it.",
        "  chair 6: growth-capped (13/17 bound, V 303652 of V* 572727, est@18k 4.63 < 6): the target is above what the upstream add_percentage reaches by 10000 -> reported, not a controller error.",
        f"quality at equal bytes vs the fixed-lambda references: {min(dps):+.3f}..{max(dps):+.3f} dB (mean {sum(dps)/len(dps):+.3f}); both sides delivered through the same val_select_v2, so the comparison is fair, the absolute bytes of both are inflated.",
        "net: overshoot 1.15 x (aim) x ~0.92 (prior/limiter) x 1.09-1.42 (un-stacking) / 1.07-1.10 (collapses) -> 1.09-1.17 by the probe meter, x 1.07-1.20 by the stale importance -> 1.18-1.32 delivered."]
open('adapt-v5-blender-analysis.txt', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
