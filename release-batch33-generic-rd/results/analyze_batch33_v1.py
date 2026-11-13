"""analyze_batch33_v1.py <collect>: 11-scene means of the generic-codec operating points (4 meshes x 3 SH precisions), the Pareto front of each
codec, and the comparison with BudgetMesh (size ratio at equal PSNR, PSNR gap at equal size).  Writes generic-rd-means.json next to the collect."""
import re, sys, math, json, statistics as st, collections
rows = collections.defaultdict(dict)
for l in open(sys.argv[1]):
    m = re.match(r'(\w+) GENERIC-RD \| mesh (\w+) \| V (\d+) F (\d+) \| sh(\d) \| zstd ([\d.]+) MB \| draco ([\d.]+) MB \| tfan ([\d.]+) MB \| test ([\d.]+) ([\d.]+) ([\d.]+) \| unquantised ([\d.]+)', l)
    if m: rows[m[1]][(m[2], int(m[5]))] = dict(V=int(m[3]), zstd=float(m[6]), draco=float(m[7]), tfan=float(m[8]), psnr=float(m[9]), ssim=float(m[10]), lpips=float(m[11]), ref=float(m[12]))
scenes = sorted(rows); print(len(scenes), 'scenes,', sum(len(v) for v in rows.values()), 'rows')
pts = {}
for rank in range(4):
    for b in (8, 6, 4):
        sel = []
        for s in scenes:
            meshes = sorted({k[0] for k in rows[s]}, key=lambda mname: -rows[s][(mname, 8)]['V']); sel.append(rows[s][(meshes[rank], b)])
        pts[(rank, b)] = {k: st.mean(x[k] for x in sel) for k in ('V', 'zstd', 'draco', 'tfan', 'psnr', 'ssim', 'lpips', 'ref')}
print('%-22s %9s %7s %7s %7s | %6s %6s %6s (loss vs the mesh before quantisation)' % ('mesh, SH bits', 'vertices', 'TFAN', 'zstd', 'Draco', 'PSNR', 'SSIM', 'LPIPS'))
for (rank, b), p in pts.items(): print('%-22s %9.0f %7.1f %7.1f %7.1f | %6.2f %6.4f %6.4f (%+.2f dB)' % (('FP32 model' if rank == 0 else 'simplified #%d' % rank) + ', %d bit' % b, p['V'], p['tfan'], p['zstd'], p['draco'], p['psnr'], p['ssim'], p['lpips'], p['psnr'] - p['ref']))
def front(codec):
    P = sorted(((p[codec], p['psnr'], k) for k, p in pts.items())); out = []; best = -1
    for mb, ps, k in P:
        if ps > best: out.append((mb, ps, k)); best = ps
    return out
fr = {c: front(c) for c in ('tfan', 'zstd', 'draco')}
for c, f in fr.items(): print(c, 'front:', ', '.join('%.1f MB %.2f dB (%s%d bit)' % (mb, ps, 'full ' if k[0] == 0 else 'simp#%d ' % k[0], k[1]) for mb, ps, k in f))
ours = [(5.0, 22.72), (7.6, 23.12), (10.2, 23.39), (15.1, 23.62), (19.9, 23.75)]
def at(curve, x):
    for (a, pa), (b, pb) in zip(curve, curve[1:]):
        if a <= x <= b: return pa + (pb - pa) * math.log(x / a) / math.log(b / a)
def size_at(curve, y):
    for (a, pa), (b, pb) in zip(curve, curve[1:]):
        if pa <= y <= pb: return a * (b / a) ** ((y - pa) / (pb - pa))
print('\nBudgetMesh against each generic codec (Pareto fronts, 11-scene means)')
for c, f in fr.items():
    cu = [(mb, ps) for mb, ps, _ in f]
    print(' ', c, '| PSNR of the codec at the BudgetMesh sizes:', ', '.join('%.1f MB: %s' % (x, ('%.2f (BudgetMesh %+.2f dB)' % (at(cu, x), y - at(cu, x))) if at(cu, x) is not None else 'below its smallest file (%.1f MB)' % cu[0][0]) for x, y in ours))
    print(' ', ' ' * len(c), '| size the codec needs for the BudgetMesh PSNR:', ', '.join('%.2f dB: %s' % (y, ('%.0f MB (%.1fx)' % (size_at(cu, y), size_at(cu, y) / x)) if size_at(cu, y) else 'n/a') for x, y in ours))
json.dump(dict(points={'%d_%d' % k: v for k, v in pts.items()}, fronts={c: [(mb, ps, list(k)) for mb, ps, k in f] for c, f in fr.items()}), open(sys.argv[1].replace('.txt', '-means.json'), 'w'), indent=1)
