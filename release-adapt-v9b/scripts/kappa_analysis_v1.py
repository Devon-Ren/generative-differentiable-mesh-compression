#!/usr/bin/env python3
"""kappa_analysis_v1.py (2026-11-07): the observability gap of phase A, measured on every finished adapt run on disk (CPU, json only).
For each run dir with original/rd_summary.json + original/rd_log.jsonl it reads the FINAL byte probe (the real encoder on the cleaned
30k model) and the phase-B / phase-A probes and reports the three quantities the soup cannot show before the 11k Delaunay rebuild:
  kappa_end  = delivered vertices / model vertices before the final cleaning (last prune round's vertices_after)
  topo_v     = delivered topology bytes per delivered vertex = b_f x faces/vertex (prior in the plan: 2 B x 2.5 = 5 B/V)
  b_v soup   = attribute bytes per kept vertex read on the soup at 9500 (v8/v9 only) vs b_v on the delivered mesh
plus the plan's implied landing f = kappa x (b_v_end + topo_v) / (b_v_plan + 5) with b_v_plan = soup b_v (v8/v9) or 12k mesh b_v (v7b,
flagged '~'), the observed hit = final probe / target, the kept share at the first probe, and the horizon error (last landing estimate / final).
Per dataset class it prints mean / min / max and 1 / mean(f) = the phase-A aim factor that would have landed on target.
Usage: kappa_analysis_v1.py <run dir glob> [<glob> ...]      e.g.  '/scratch/zr2245/*-adapt7-t*mb' '/scratch/zr2245/*-adapt9-t*mb'"""
import glob, json, os, re, sys

TOPO = ('top_stream', '_triangle_indices', 'local_faces', 'local_rotations'); BANK = ('app_bank', 'app_residual_bank')
INDOOR = {'room', 'kitchen', 'counter', 'bonsai'}; OUTDOOR = {'garden', 'bicycle', 'stump', 'flowers', 'treehill'}; TANDT = {'truck', 'train'}
BLENDER = {'lego', 'chair', 'drums', 'ficus', 'hotdog', 'materials', 'mic', 'ship'}


def cls(scene):
    return 'mip-indoor' if scene in INDOOR else 'mip-outdoor' if scene in OUTDOOR else 'tandt' if scene in TANDT else 'blender' if scene in BLENDER else 'other'


def streams(p):
    pb = p['payload_bytes']; topo = sum(b for k, b in pb.items() if k in TOPO); banks = sum(b for k, b in pb.items() if k in BANK)
    header = max(0, int(p['file_bytes']) - sum(pb.values())); V, F = int(p['vertices']), int(p['faces'])
    return dict(topo=topo, b_f=topo / max(F, 1), fpv=F / max(V, 1), topo_v=topo / max(V, 1), b_v=(p['file_bytes'] - topo - banks - header) / max(V, 1), overhead=(banks + header) / 1e6, V=V, F=F)


rows = []
for pat in sys.argv[1:]:
    for d in sorted(glob.glob(pat)):
        rs_p, lg_p = os.path.join(d, 'original', 'rd_summary.json'), os.path.join(d, 'original', 'rd_log.jsonl')
        if not (os.path.exists(rs_p) and os.path.exists(lg_p)): continue
        m = re.match(r'^(\w+?)-adapt(\d+b?)-t([0-9.]+)mb$', os.path.basename(d))
        if not m: continue
        scene, ver, T = m.group(1), m.group(2), float(m.group(3))
        try:
            rs = json.load(open(rs_p)); ev = [json.loads(l) for l in open(lg_p)]
        except Exception as e:
            print(f'SKIP {d}: {e}'); continue
        probes = rs.get('probes') or []
        if not probes: print(f'SKIP {d}: no probes'); continue
        fin = max(probes, key=lambda p: p.get('iteration', 0)); first = min(probes, key=lambda p: p.get('iteration', 10 ** 9))
        pr = [e for e in ev if e.get('event') == 'prune']
        V_model = pr[-1]['vertices_after'] if pr else fin.get('vertices_model')
        sf, s0 = streams(fin), streams(first)
        kappa = fin['vertices'] / max(V_model, 1)
        pa = rs.get('probes_a') or []
        bv_soup = pa[-1].get('b_v') if pa else None
        bv_plan = bv_soup if bv_soup is not None else s0['b_v']
        f = kappa * (sf['b_v'] + sf['topo_v']) / (bv_plan + 5.0)
        hit = rs['probe_final_mb'] / rs['target_mb']
        horizon = (rs['last_probe_est_mb'] / rs['probe_final_mb']) if rs.get('last_probe_est_mb') else None
        rows.append(dict(scene=scene, ver=ver, T=T, cls=cls(scene), kappa=kappa, topo_v=sf['topo_v'], fpv=sf['fpv'], b_f=sf['b_f'], b_v_end=sf['b_v'], b_v_12k=s0['b_v'], bv_soup=bv_soup, f=f, hit=hit,
                         kept0=first.get('kept_frac'), it0=first.get('iteration'), horizon=horizon, over=rs.get('phase_a_overshoot'), collapsed=sum(e.get('n_pairs', 0) for e in pr), V_model=V_model, V_del=fin['vertices'], ovh=sf['overhead']))
if not rows: sys.exit('no runs found for ' + ' '.join(sys.argv[1:]))
rows.sort(key=lambda r: (r['cls'], r['scene'], r['T'], r['ver']))
print(f"{'scene':9s} {'ver':4s} {'T':>5s} | {'kappa':>5s} {'topo_v':>6s} {'fpv':>4s} {'b_f':>4s} | {'b_v_end':>7s} {'b_v_12k':>7s} {'b_v_soup':>8s} | {'f':>5s} {'hit':>5s} {'1/f':>5s} | {'kept@1st':>8s} {'horizon':>7s} {'over':>4s} {'coll':>6s} {'V_model':>8s} {'V_deliv':>8s} {'ovh':>5s}")
for r in rows:
    print(f"{r['scene']:9s} {r['ver']:4s} {r['T']:5g} | {r['kappa']:5.3f} {r['topo_v']:6.2f} {r['fpv']:4.2f} {r['b_f']:4.2f} | {r['b_v_end']:7.2f} {r['b_v_12k']:7.2f} {('%8.2f' % r['bv_soup']) if r['bv_soup'] is not None else '       ~'} | {r['f']:5.3f} {r['hit']:5.3f} {1 / r['f']:5.3f} | "
          f"{r['kept0']:5.3f}@{r['it0'] // 1000:>2d}k {('%7.3f' % r['horizon']) if r['horizon'] else '      -'} {r['over']:4.2f} {r['collapsed']:6d} {r['V_model']:8d} {r['V_del']:8d} {r['ovh']:5.2f}")
print('\nf = kappa x (b_v_end + topo_v) / (b_v_plan + 5): the landing the v9 plan implies (b_v_plan = soup b_v at 9500, or ~ the 12k mesh b_v for v7b runs);'
      ' hit = final probe / target (v7b/v9 rows: observed; the two agree where phase B collapsed nothing).  1/f = aim factor that lands on target.')
print('\nclass summary (mean [min..max]) over runs:')
for c in ('mip-indoor', 'mip-outdoor', 'tandt', 'blender', 'other'):
    rr = [r for r in rows if r['cls'] == c]
    if not rr: continue
    def stat(k): a = [r[k] for r in rr if r[k] is not None]; return f"{sum(a) / len(a):.3f} [{min(a):.3f}..{max(a):.3f}]" if a else '-'
    print(f"  {c:11s} n={len(rr):2d} | kappa {stat('kappa')} | topo_v {stat('topo_v')} | fpv {stat('fpv')} | b_f {stat('b_f')} | f {stat('f')} | hit {stat('hit')} | aim factor 1/mean(f) = {1 / (sum(r['f'] for r in rr) / len(rr)):.3f} | horizon {stat('horizon')}")
by_ver = {}
for r in rows:
    if r['ver'] in ('5', '6', '7', '7b'): continue
    by_ver.setdefault(r['ver'], []).append(r)
for v, rr in sorted(by_ver.items()):
    s = [r for r in rr if r['bv_soup'] is not None]
    if s: print(f"  soup b_v over-read (v{v}, n={len(s)}): soup / mesh-at-12k {sum(r['bv_soup'] / r['b_v_12k'] for r in s) / len(s):.3f}, soup / delivered {sum(r['bv_soup'] / r['b_v_end'] for r in s) / len(s):.3f}")
