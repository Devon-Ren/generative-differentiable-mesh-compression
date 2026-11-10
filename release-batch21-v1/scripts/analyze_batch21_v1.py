"""analyze_batch21_v1.py (2026-11-10): summarise the batch21 collect file into the per-claim tables.
usage: python analyze_batch21_v1.py <collect.txt> > results.md"""
import math, re, sys
T = open(sys.argv[1]).read()
def sec(name):
    m = re.search(r'^===== ' + re.escape(name) + r'.*?\n(.*?)(?=^===== |\Z)', T, re.S | re.M); return m.group(1) if m else ''
def interp(pts, mb):
    pts = sorted(pts); xs = [math.log(a) for a, _ in pts]; ys = [b for _, b in pts]; x = math.log(mb)
    for i in range(len(pts) - 1):
        if xs[i] <= x <= xs[i + 1]: return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])
    i, j = ((0, 1) if x < xs[0] else (len(pts) - 2, len(pts) - 1)); return ys[i] + (ys[j] - ys[i]) * (x - xs[i]) / (xs[j] - xs[i])
# ---- reference hulls (delivered MB, test PSNR), K4096 R256 q7
HULL = {'room': [(22.1, 28.25), (12.4, 28.02), (5.4, 27.30), (2.9, 26.38)], 'counter': [(20.7, 25.95), (13.1, 25.84), (6.4, 25.32), (3.3, 24.54)],
        'kitchen': [(24.5, 27.09), (16.5, 26.89), (8.0, 25.97), (3.8, 24.62)], 'bonsai': [(24.3, 27.89), (13.2, 27.41), (6.1, 26.36), (3.5, 25.36)],
        'garden': [(36.4, 24.73), (24.6, 24.50), (10.2, 23.65), (5.0, 22.78)], 'bicycle': [(31.6, 22.99), (27.0, 23.02), (11.9, 22.69), (5.2, 22.08)],
        'stump': [(27.6, 24.81), (21.4, 24.71), (8.9, 24.12), (3.9, 23.21)], 'flowers': [(25.7, 19.02), (24.2, 19.07), (12.0, 19.21), (5.6, 18.84)],
        'treehill': [(27.0, 20.31), (27.4, 20.45), (11.9, 20.38), (5.1, 20.30)], 'truck': [(27.3, 21.78), (19.3, 22.20), (9.4, 22.37), (4.4, 21.64)],
        'train': [(30.1, 18.62), (25.4, 18.71), (13.4, 18.75), (6.4, 18.62)],
        'drjohnson': [(30.69, 26.449), (16.397, 26.44), (6.889, 25.652), (3.301, 24.654)], 'playroom': [(29.891, 27.645), (14.367, 27.287), (5.762, 26.423), (2.922, 25.259)]}
BL_OLD = {'lego': [(7.517, 33.516), (2.763, 32.583)], 'chair': [(4.660, 32.212), (2.391, 32.166)], 'drums': [(4.439, 25.519), (2.862, 25.494)], 'ficus': [(1.882, 32.109), (1.435, 31.975)],
          'hotdog': [(3.689, 35.825), (2.028, 35.439)], 'materials': [(3.814, 27.773), (2.154, 27.757)], 'mic': [(2.601, 33.410), (1.705, 32.920)], 'ship': [(5.658, 29.619), (3.361, 29.305)]}
for m in re.finditer(r'^(\w+)-rd3b4-lam([\d.]+)-mu1\.0 \|.*?-> v3 ([\d.]+) MB V (\d+) F \d+ test ([\d.]+)', sec('redeliver-v3-blender'), re.M):
    BL_OLD.setdefault(m[1], []).append((float(m[3]), float(m[5])))
HULL.update(BL_OLD)
BL = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
out = []; P = out.append
# ---- G0
def g0(section, tagA, tagB, title):
    rows = {}
    for m in re.finditer(r'^(\w+) budget (\d+) (' + tagA + '|' + tagB + r') \| vertices (\d+) faces (\d+) .*?\| test ([\d.]+) \| delivered ([\d.]+) MB test ([\d.]+)', section, re.M):
        rows.setdefault((m[1], int(m[2])), {})[m[3]] = dict(V=int(m[4]), F=int(m[5]), pre=float(m[6]), mb=float(m[7]), psnr=float(m[8]))
    P(f'\n## {title}\n'); P('| scene | budget (V) | train-at-budget MB / PSNR (faces) | post-hoc collapse MB / PSNR (faces) | Δ budget − collapse | λ-grown at equal bytes | Δ budget − λ |'); P('|---|---|---|---|---|---|---|')
    d1 = []; d2 = []
    for (s, b), d in sorted(rows.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        if tagA not in d or tagB not in d: P(f'| {s} | {b} | incomplete | | | | |'); continue
        a, c = d[tagA], d[tagB]; h = interp(HULL[s], a['mb']); x = a['psnr'] - c['psnr']; y = a['psnr'] - h; d1.append(x); d2.append(y)
        P(f"| {s} | {b} | {a['mb']:.2f} / {a['psnr']:.2f} ({a['F']/1e3:.0f}k) | {c['mb']:.2f} / {c['psnr']:.2f} ({c['F']/1e3:.0f}k) | {x:+.2f} | {h:.2f} | {y:+.2f} |")
    if d1: P(f'\n**train-at-budget − post-hoc collapse (equal V): {min(d1):+.2f} to {max(d1):+.2f} dB, mean {sum(d1)/len(d1):+.2f}, {sum(v>0 for v in d1)} of {len(d1)}; vs λ-grown at equal bytes: {min(d2):+.2f} to {max(d2):+.2f}, mean {sum(d2)/len(d2):+.2f}, λ wins {sum(v<0 for v in d2)} of {len(d2)}.**')
    return d1, d2
db1, db2 = g0(sec('g0-refresh Deep Blending'), 'TRAIN-AT-BUDGET', 'POST-HOC-COLLAPSE', 'G0 on Deep Blending (2 scenes × 3 budgets)')
bl1, bl2 = g0(sec('g0-refresh Blender'), 'TRAIN-AT-BUDGET-BLENDER', 'POST-HOC-COLLAPSE-BLENDER', 'G0 on NeRF-Synthetic (8 scenes × 4 budgets, Blender protocol)')
# ---- new lambda points
P('\n## NeRF-Synthetic fixed-λ hull: now four points per scene (val_select_v3 re-delivery, -best3)\n'); P('| scene | λ 6.2e-4 | λ 1.85e-3 | λ 6.2e-3 | λ 1.85e-2 |'); P('|---|---|---|---|---|')
for s in BL:
    pts = sorted(HULL[s], reverse=True); P(f'| {s} | ' + ' | '.join(f'{a:.2f} MB / {b:.2f}' for a, b in pts) + ' |')
# ---- controller
def ctrl(section, title, chord):
    P(f'\n## {title}\n'); P('| scene | target | delivered | hit | test PSNR | fixed-λ at equal bytes | Δ |'); P('|---|---|---|---|---|---|---|')
    hits = []; dps = []; capped = []
    for m in re.finditer(r'^(\w+) \| target ([\d.]+) MB \|.*?lam_final \(shadow\) ([\d.e+-]+) \|.*?delivered ([\d.]+) MB \(hit ([\d.]+) x target.*?\| test ([\d.]+)', section, re.M):
        s, t, lf, mb, hit, ps = m[1], float(m[2]), float(m[3]), float(m[4]), float(m[5]), float(m[6]); h = interp(HULL[s], mb); d = ps - h
        cap = hit < 0.9 and lf <= 1.1e-4
        (capped if cap else hits).append(hit); dps.append(d)
        P(f'| {s} | {t:g} | {mb:.2f} | {hit:.3f}{" (capacity limit: λ at floor)" if cap else ""} | {ps:.2f} | {h:.2f} | {d:+.2f} |')
    if hits: P(f'\n**hit {min(hits):.3f}–{max(hits):.3f}, mean {sum(hits)/len(hits):.3f}; within ±10 %: {sum(abs(h-1)<=0.10 for h in hits)}/{len(hits)}; within ±5 %: {sum(abs(h-1)<=0.05 for h in hits)}/{len(hits)}; equal-bytes ΔPSNR vs {chord}: {min(dps):+.2f} to {max(dps):+.2f}, mean {sum(dps)/len(dps):+.2f}' + (f'; {len(capped)} rows at the capacity limit (λ at its floor from the first intervals; reported, excluded from the hit statistics)' if capped else '') + '.**')
    return hits, dps, capped
A = sec('adapt-v9b (DB rows + 7.5 / 15 MB rows)')
db_rows = '\n'.join(l for l in A.splitlines() if l.startswith(('drjohnson', 'playroom')))
mip_rows = '\n'.join(l for l in A.splitlines() if not l.startswith(('drjohnson', 'playroom')))
ctrl(db_rows, 'Controller v9b (headline version) on Deep Blending, 5 / 10 / 20 MB', 'the DB fixed-λ hull')
ctrl(mip_rows, 'Controller v9b on Mip-360 / T&T at the two new targets, 7.5 and 15 MB', 'the rd-hull6-v2 hull')
ctrl(sec('adapt-v9b-blender (1.5 / 3 / 6 MB rows)'), 'Controller v9b on NeRF-Synthetic at the three new targets, 1.5 / 3 / 6 MB', 'the 4-point chord')
# ---- fan
P('\n## λ × dictionary fan + DC q8 on Deep Blending and NeRF-Synthetic\n'); P('| scene | λ | base MB / PSNR | DC q8 Δ (ΔMB) | best other K×R (Δ, ΔMB) | joint hull − λ-only hull at base bytes |'); P('|---|---|---|---|---|---|')
F = sec('lambda-k-fan (DB + Blender rows)'); fan = {}
for m in re.finditer(r'^(\w+) \| lam ([\d.]+) \| (.+?) \| +([\d.]+) MB \| test ([\d.]+) [\d.]+ [\d.]+ \| dPSNR ([+-][\d.]+) \| dMB ([+-][\d.]+)', F, re.M):
    fan.setdefault((m[1], float(m[2])), {})[m[3].strip()] = (float(m[4]), float(m[5]), float(m[6]), float(m[7]))
gains = {}
for (s, L), d in sorted(fan.items()):
    base = d.get('K4096 R256 q7 (base)');
    if not base: continue
    q8 = d.get('K4096 R256 q8'); others = {k: v for k, v in d.items() if k.startswith('K') and 'base' not in k and k != 'K4096 R256 q8'}
    best = max(others.items(), key=lambda kv: kv[1][2]) if others else None
    # joint hull over all variants of all lambdas of this scene: gain at this base's bytes
    allpts = [(v[0], v[1]) for (s2, L2), d2 in fan.items() if s2 == s for v in d2.values()]
    # upper envelope in log-bytes: keep points not dominated
    env = [p for p in sorted(allpts) if not any(q[0] <= p[0] and q[1] > p[1] for q in allpts)]
    basepts = [d2['K4096 R256 q7 (base)'][:2] for (s2, L2), d2 in fan.items() if s2 == s and 'K4096 R256 q7 (base)' in d2]
    g = (interp(env, base[0]) - interp(basepts, base[0])) if len(env) >= 2 and len(basepts) >= 2 else 0.0; gains.setdefault(s, []).append(g)  # joint hull minus the lambda-only hull at this base's bytes
    P(f"| {s} | {L:g} | {base[0]:.2f} / {base[1]:.2f} | {q8[2]:+.2f} ({q8[3]:+.2f}) | {best[0]} ({best[1][2]:+.2f}, {best[1][3]:+.2f}) | {max(g,0):+.2f} |" if q8 and best else f"| {s} | {L:g} | {base[0]:.2f} / {base[1]:.2f} | incomplete | | |")
if gains: P('\n**max joint-hull gain over the fixed K4096 R256 at a base point, per scene: ' + ', '.join(f'{s} {max(max(v),0):+.2f}' for s, v in sorted(gains.items())) + '.**')
# ---- generic codecs: same ORIGINAL mesh, MPEG SC3DMC/TFAN (Open3DGC) vs our level-1.0 delivery of that mesh (rd-hull6-v2 'level 1.0' column)
LEVEL = {'room': 25.7, 'counter': 23.6, 'kitchen': 24.3, 'bonsai': 28.2, 'garden': 32.5, 'bicycle': 28.4, 'stump': 29.2, 'flowers': 26.9, 'treehill': 28.5, 'truck': 20.0, 'train': 22.3}
P('\n## Generic codecs on the same original mesh: MPEG SC3DMC/TFAN (Open3DGC) vs our codec\n'); P('| scene | MPEG, positions + faces only | MPEG, positions + faces + 8-bit SH (full asset) | ours, full asset (level 1.0) | MPEG full / ours | MPEG geometry alone / ours |'); P('|---|---|---|---|---|---|')
G = sec('generic codecs'); geo = {m[1]: float(m[2]) for m in re.finditer(r'^(\w+) o3dgc_tfan_geo \| ([\d.]+) MB', G, re.M)}; full = {m[1]: float(m[2]) for m in re.finditer(r'^(\w+) o3dgc_tfan_sh8int \| ([\d.]+) MB', G, re.M)}
rat = []
for s in sorted(set(geo) | set(full)):
    o = LEVEL.get(s)
    if o: rat.append(full[s] / o) if s in full else None
    P(f"| {s} | {geo.get(s, float('nan')):.1f} MB | {full.get(s, float('nan')):.1f} MB | {o if o else 'n/a'} MB | {full[s]/o:.1f}× | {geo[s]/o:.2f}× |" if o and s in full and s in geo else f"| {s} | {geo.get(s, float('nan')):.1f} MB | {full.get(s, float('nan')):.1f} MB | n/a | | |")
if rat: P(f'\n**MPEG with 8-bit SH attributes is {min(rat):.1f}–{max(rat):.1f}× our file for the same mesh at the same vertex count (room and garden, measured earlier: 3.5–15× for the generic rows incl. Draco). MPEG\'s geometry alone is already 0.5–0.9× of our whole file including appearance.**')
P('\nDraco / zstd rows (run_generic_v1) exist for room, garden, drjohnson, playroom; the 9 other Mip/T&T scenes failed at the runner\'s scene table (only room/garden are built in) and are re-run in batch22 with RUN/DATA/IMAGES/QP exported.')
print('\n'.join(out))
