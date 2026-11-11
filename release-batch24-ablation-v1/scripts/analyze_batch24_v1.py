#!/usr/bin/env python3
"""analyze_batch24_v1.py (2026-11-11): the training-rule ablation table from the batch24 rows.  Run from the repo root.
  python3 release-batch24-ablation-v1/scripts/analyze_batch24_v1.py <collect.txt> [<collect2.txt> ...]
Rows: 'abl24-<variant>' rows (ablation-b24.txt / -blender.txt), the paper's controller rows (adapt-v9b*.txt, batch21 collect) as 'full'.
Quality at the same file size: each scene's constant-price curve = log-linear interpolation of the scene's two abl24 constlam points
(extrapolated beyond them, marked *).  dPSNR(variant) = PSNR - curve(file size); 'vs full' = dPSNR(variant) - dPSNR(full at its own size)."""
import re, sys, math, statistics as st, collections
MIP = ['room', 'kitchen', 'counter', 'bonsai', 'garden', 'bicycle', 'stump', 'flowers', 'treehill', 'truck', 'train']; BL = ['lego', 'chair', 'drums', 'ficus', 'hotdog', 'materials', 'mic', 'ship']
R = re.compile(r'^(\w+)(?: abl24-(\S+))? \| target ([\d.]+) MB \|.*?lam_final \(shadow\) ([\d.e+-]+) \|.*?delivered ([\d.]+) MB \(hit ([\d.]+).*?\| test ([\d.]+) ([\d.]+) ([\d.]+).*?growth-capped (\w+)')
pre = {}
args = sys.argv[1:]
if '--precodec' in args:
    i = args.index('--precodec'); pf = args[i + 1]; args = args[:i] + args[i + 2:]
    for l in open(pf):
        m = re.match(r'ctrl (\w+) ([\d.]+) precodec_test ([\d.]+)', l)
        if m: pre[(m[1], float(m[2]))] = float(m[3])
sys.argv[1:] = args
rows = collections.defaultdict(dict)   # (scene, target) -> variant -> dict
for f in sys.argv[1:] + ['release-adapt-v9b/results/adapt-v9b-2026-11-08.txt', 'release-adapt-v9b/results/adapt-v9b-blender-2026-11-08.txt', 'release-batch21-v1/results/batch21-collect-2026-11-09-2110-raw.txt']:
    for l in open(f):
        m = R.match(l)
        if not m: continue
        s, var, t, lamf, mb, hit, p, ss, lp, gc = m[1], m[2] or 'full', float(m[3]), float(m[4]), float(m[5]), float(m[6]), float(m[7]), float(m[8]), float(m[9]), m[10]
        if s not in MIP + BL: continue
        capped = (var in ('full', 'nocollapse', 'constants')) and ((hit < 0.9 and lamf <= 1.1e-4) or gc == 'True')
        rows[(s, t)].setdefault(var, dict(mb=mb, hit=hit, psnr=p, ssim=ss, lpips=lp, capped=capped))   # first occurrence wins (the paper's row)
curve = {}  # scene -> sorted [(mb, psnr)] from abl24 constlam rows
for (s, t), d in rows.items():
    for var, r in d.items():
        if var.startswith('constlam-lam'): curve.setdefault(s, []).append((r['mb'], r['psnr']))
for s in curve: curve[s].sort()
def on_curve(s, mb):
    pts = curve.get(s)
    if not pts or len(pts) < 2: return None, True
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    if len(pts) > 2:
        for a, b in zip(pts, pts[1:]):
            if a[0] <= mb <= b[0]: (x0, y0), (x1, y1) = a, b
    f = (math.log(mb) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return y0 + f * (y1 - y0), not (x0 <= mb <= x1)
def fmt(v, n=2, sign=True): return ('%+.*f' if sign else '%.*f') % (n, v)
out = []
for name, scenes, targets in [('Mip-NeRF 360 / T&T', MIP, [10.0, 20.0]), ('NeRF-Synthetic', BL, [2.5, 1.5])]:
    for t in targets:
        capped_sc = [s for s in scenes if (s, t) in rows and any(r.get('capped') for r in rows[(s, t)].values())]
        have = [s for s in scenes if (s, t) in rows and 'full' in rows[(s, t)] and s not in capped_sc]
        variants = ['full', 'nocollapse', 'constants', 'matched']
        out.append(f'\n## {name}, target {t:g} MB ({len(have)} scenes; capacity-limited and excluded: {", ".join(capped_sc) or "none"})\n')
        out.append('| variant | scenes | size error mean | range | within 10 % | dPSNR vs constant-price curve: mean (range) | vs full: mean (range), better |')
        out.append('|---|---|---|---|---|---|---|')
        dfull = {}
        for s in have:
            r = rows[(s, t)]['full']; c, ex = on_curve(s, r['mb']); dfull[s] = (r['psnr'] - c) if c is not None else None
        for var in variants:
            sc = [s for s in have if var in rows[(s, t)]]
            if not sc: out.append(f'| {var} | 0 | not run yet | | | | |'); continue
            errs = [rows[(s, t)][var]['hit'] - 1 for s in sc]
            d = []; rel = []; ext = 0
            for s in sc:
                r = rows[(s, t)][var]; c, ex = on_curve(s, r['mb'])
                if c is None: continue
                ext += ex; d.append(r['psnr'] - c)
                if dfull.get(s) is not None: rel.append(r['psnr'] - c - dfull[s])
            dcol = f'{fmt(st.mean(d))} ({fmt(min(d))} .. {fmt(max(d))}){"*" if ext else ""}' if d else '-'
            rcol = ('0 (reference)' if var == 'full' else (f'{fmt(st.mean(rel))} ({fmt(min(rel))} .. {fmt(max(rel))}), {sum(1 for x in rel if x > 0)}/{len(rel)}' if rel else '-'))
            out.append(f'| {var} | {len(sc)} | {fmt(100 * st.mean(errs), 1)} % | {fmt(100 * min(errs), 1)} .. {fmt(100 * max(errs), 1)} % | {sum(1 for e in errs if abs(e) <= 0.10)}/{len(errs)} | {dcol} | {rcol} |')
        # matched vs nocollapse (same split counts, both without collapse): direct PSNR difference and size
        pairs = [(s, rows[(s, t)]['matched'], rows[(s, t)]['nocollapse']) for s in have if 'matched' in rows[(s, t)] and 'nocollapse' in rows[(s, t)]]
        if pairs:
            dd = [n['psnr'] - m['psnr'] for s, m, n in pairs]; dm = [n['mb'] / m['mb'] - 1 for s, m, n in pairs]
            out.append(f'\nrate-aware ranking vs MeshSplatting importance sampling at the same split counts (both without collapse): nocollapse - matched = {fmt(st.mean(dd))} dB mean ({fmt(min(dd))} .. {fmt(max(dd))}), rate-aware better on {sum(1 for x in dd if x > 0)}/{len(dd)}; file size of the rate-aware mesh / importance mesh - 1 = {fmt(100 * st.mean(dm), 1)} % ({fmt(100 * min(dm), 1)} .. {fmt(100 * max(dm), 1)} %)')
    # constant-price rows: spread of file size across scenes at one price
    out.append(f'\n### constant price, {name}: file size spread across scenes at one price (abl24 constlam rows)\n')
    lam_rows = collections.defaultdict(list)
    for (s, t), d in rows.items():
        if s in scenes:
            for var, r in d.items():
                if var.startswith('constlam-lam'): lam_rows[var].append((s, r['mb'], r['psnr']))
    for var, lst in sorted(lam_rows.items()):
        mbs = [x[1] for x in lst]; out.append(f'- {var}: {len(lst)} scenes, file size {min(mbs):.1f} .. {max(mbs):.1f} MB (max/min {max(mbs)/min(mbs):.1f}x)')
out.append('\n### per-scene rows (file size MB / PSNR / size error; dPSNR vs the scene\'s constant-price curve)\n')
for (s, t) in sorted(rows, key=lambda k: (k[0] not in MIP, MIP.index(k[0]) if k[0] in MIP else BL.index(k[0]), k[1])):
    d = rows[(s, t)]
    if 'full' not in d or not any(v != 'full' and not v.startswith('constlam') for v in d): continue
    cells = []
    for var in ['full', 'nocollapse', 'constants', 'matched']:
        if var in d:
            r = d[var]; c, ex = on_curve(s, r['mb']); cells.append(f"{var}: {r['mb']:.2f} / {r['psnr']:.2f} / {fmt(100*(r['hit']-1),1)} % / {fmt(r['psnr']-c) if c is not None else '-'}{'*' if ex else ''}")
    cl = [f"{v[12:]}: {r['mb']:.2f} / {r['psnr']:.2f}" for v, r in sorted(d.items()) if v.startswith('constlam')]
    out.append(f"- {s} {t:g} MB: " + ' | '.join(cells) + (' | constlam ' + ', '.join(cl) if cl else ''))

# ---------------- vertex-cap rerun at the controller counts: E1 and E2 ----------------
import re as _re
CAP = _re.compile(r'^(\w+) budget (\d+) (TRAIN-AT-BUDGET|POST-HOC-COLLAPSE)(?:-BLENDER)? \| vertices (\d+) faces (\d+) \|.*?\| test ([\d.]+) \| delivered ([\d.]+) MB test ([\d.]+) ([\d.]+) ([\d.]+)')
VR = _re.compile(r'^(\w+) \| target ([\d.]+) MB \|.*?trained V (\d+)')
ctrlV = {}
for f in ['release-adapt-v9b/results/adapt-v9b-2026-11-08.txt', 'release-adapt-v9b/results/adapt-v9b-blender-2026-11-08.txt', 'release-batch21-v1/results/batch21-collect-2026-11-09-2110-raw.txt']:
    for l in open(f):
        m = VR.match(l)
        if m and (m[1], float(m[2])) not in ctrlV: ctrlV[(m[1], float(m[2]))] = int(m[3])
budget_of = {k: int(round(v / 0.9 / 1000.0)) * 1000 for k, v in ctrlV.items()}
cap = {}  # (scene, target) -> {'TRAIN-AT-BUDGET': row, 'POST-HOC-COLLAPSE': row}
for f in sys.argv[1:]:
    for l in open(f):
        m = CAP.match(l)
        if not m: continue
        s_, b, kind = m[1], int(m[2]), m[3]
        for (sc, t), bb in budget_of.items():
            if sc == s_ and bb == b: cap.setdefault((sc, t), {})[kind] = dict(V=int(m[4]), F=int(m[5]), pre=float(m[6]), mb=float(m[7]), psnr=float(m[8]), ssim=float(m[9]), lpips=float(m[10]))
out.append('\n## E1 at the controller counts: training with a vertex cap vs simplification after training (same vertex count, after codec)\n')
out.append('| dataset | pairs | mean gain (dB) | range | trained better | gain at largest / mid / smallest budget |'); out.append('|---|---|---|---|---|---|')
for name, scenes, targets in [('Mip-NeRF 360 / T&T', MIP, [20.0, 10.0, 5.0]), ('NeRF-Synthetic', BL, [4.0, 2.5, 1.5])]:
    g = {t: [] for t in targets}
    for (sc, t), d in cap.items():
        if sc in scenes and t in g and 'TRAIN-AT-BUDGET' in d and 'POST-HOC-COLLAPSE' in d: g[t].append(d['TRAIN-AT-BUDGET']['psnr'] - d['POST-HOC-COLLAPSE']['psnr'])
    allg = [x for t in targets for x in g[t]]
    if allg: out.append(f"| {name} | {len(allg)} | {fmt(st.mean(allg))} | {fmt(min(allg))} .. {fmt(max(allg))} | {sum(1 for x in allg if x > 0)}/{len(allg)} | " + ' / '.join(fmt(st.mean(g[t])) if g[t] else '-' for t in targets) + ' |')
out.append('\n## E2 at the controller counts: BudgetMesh (controller run) vs MeshSplatting with a vertex cap\n')
out.append('Right half (same file size, after codec): the cap mesh encoded with the same codec; the cap curve of each scene (its 3 budgets, log-linear in file size) read at the controller file size. Same vertex count: the cap run whose budget was set from this controller mesh (V within the stated band). No-codec column: pre-codec test PSNR (3000-step fine-tuned FP32 meshes, same evaluator) of the controller mesh minus the cap mesh at the same vertex count (within 15 %).\n')
out.append('| dataset | target | pairs (file size / vertex count within 15 %) | V_cap / V_ctrl range of the vertex-count pairs | same vertex count, NO codec (FP32, fine-tuned): mean gain (range), better | same vertex count, after codec: mean gain (range), better | same file size: mean gain (range), BudgetMesh better |'); out.append('|---|---|---|---|---|---|---|')
for name, scenes, targets in [('Mip-NeRF 360 / T&T', MIP, [20.0, 10.0, 5.0]), ('NeRF-Synthetic', BL, [4.0, 2.5, 1.5])]:
    tot_fs = []; tot_v = []; tot_nc = []
    for t in targets:
        fs = []; vv = []; vr = []; nc = []
        for sc in scenes:
            if (sc, t) not in rows or 'full' not in rows[(sc, t)] or (sc, t) not in cap or 'TRAIN-AT-BUDGET' not in cap[(sc, t)]: continue
            c = rows[(sc, t)]['full']; k = cap[(sc, t)]['TRAIN-AT-BUDGET']
            ratio_v = k['V'] / ctrlV[(sc, t)]
            if abs(ratio_v - 1) <= 0.15:
                vr.append(ratio_v); vv.append(c['psnr'] - k['psnr'])
                if (sc, t) in pre: nc.append(pre[(sc, t)] - k['pre'])
            pts = sorted((cap[(sc, tt)]['TRAIN-AT-BUDGET']['mb'], cap[(sc, tt)]['TRAIN-AT-BUDGET']['psnr']) for tt in targets if (sc, tt) in cap and 'TRAIN-AT-BUDGET' in cap[(sc, tt)])
            if len(pts) >= 2:
                mb = c['mb']; (x0, y0), (x1, y1) = pts[0], pts[-1]
                for a, b in zip(pts, pts[1:]):
                    if a[0] <= mb <= b[0]: (x0, y0), (x1, y1) = a, b
                yc = y0 + (math.log(mb) - math.log(x0)) / (math.log(x1) - math.log(x0)) * (y1 - y0); fs.append(c['psnr'] - yc)
        if vv or fs:
            ncc = f"{fmt(st.mean(nc))} ({fmt(min(nc))} .. {fmt(max(nc))}), {sum(1 for x in nc if x > 0)}/{len(nc)}" if nc else '-'
            out.append(f"| {name} | {t:g} MB | {len(fs)} / {len(vv)} | {min(vr):.2f} .. {max(vr):.2f} | {ncc} | {fmt(st.mean(vv))} ({fmt(min(vv))} .. {fmt(max(vv))}), {sum(1 for x in vv if x > 0)}/{len(vv)} | {fmt(st.mean(fs))} ({fmt(min(fs))} .. {fmt(max(fs))}), {sum(1 for x in fs if x > 0)}/{len(fs)} |")
            tot_fs += fs; tot_v += vv; tot_nc += nc
    if tot_v: out.append(f"| {name} | all | {len(tot_fs)} / {len(tot_v)} | | {(fmt(st.mean(tot_nc)) + ', ' + str(sum(1 for x in tot_nc if x > 0)) + '/' + str(len(tot_nc))) if tot_nc else '-'} | {fmt(st.mean(tot_v))}, {sum(1 for x in tot_v if x > 0)}/{len(tot_v)} | {fmt(st.mean(tot_fs))}, {sum(1 for x in tot_fs if x > 0)}/{len(tot_fs)} |")
out.append('\n### per-scene cap rows (controller: MB / PSNR / V; cap: budget -> V / MB / PSNR / pre-codec test; simplified: PSNR)\n')
for (sc, t) in sorted(cap, key=lambda k: (k[0] not in MIP, (MIP + BL).index(k[0]), -k[1])):
    d = cap[(sc, t)]; c = rows.get((sc, t), {}).get('full'); k = d.get('TRAIN-AT-BUDGET'); ph = d.get('POST-HOC-COLLAPSE')
    out.append(f"- {sc} {t:g} MB: controller {c['mb']:.2f} / {c['psnr']:.2f} / {ctrlV[(sc, t)]}" + (f" | cap {budget_of[(sc, t)]} -> {k['V']} / {k['mb']:.2f} / {k['psnr']:.2f} / pre {k['pre']:.2f}" if k else ' | cap missing') + (f" | simplified {ph['psnr']:.2f}" if ph else ''))
print('\n'.join(out))
