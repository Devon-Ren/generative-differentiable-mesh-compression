"""bd_hold_vs_cap_v1.py: Bjontegaard deltas of the hold controller against MeshSplatting with a vertex cap (both with our codec), per scene.
Piecewise-linear interpolation of PSNR over log10(size) (and of log10(size) over PSNR for BD-rate), integrated over the overlap of the two
curves; the cap curve uses only its upper envelope (points not dominated by a smaller, better point).  Scenes need >= 2 hold points and
an overlap."""
import re, glob, math, collections, statistics as st, numpy as np
MIP = ['room','kitchen','counter','bonsai','garden','bicycle','stump','flowers','treehill','truck','train']; DB = ['drjohnson','playroom']; OBJ = ['lego','chair','drums','ficus','hotdog','materials','mic','ship']
C = re.compile(r'^(\w+) budget (\d+) TRAIN-AT-BUDGET(?:-BLENDER)? \| vertices (\d+) .*?\| delivered ([\d.]+) MB test ([\d.]+)')
cap = collections.defaultdict(dict)
for f in sorted(glob.glob('release-*/results/*.txt')):
    for l in open(f, errors='ignore'):
        m = C.match(l)
        if m: cap[m[1]][int(m[2])] = (float(m[4]), float(m[5]))
H = collections.defaultdict(list)
for l in open('release-batch31-hold/results/batch31-collect-2026-11-13-0126.txt'):
    m = re.match(r'(\w+) abl24-hold \| target ([\d.]+) MB \|.*?delivered ([\d.]+) MB.*?\| test ([\d.]+)', l)
    if m: H[m[1]].append((float(m[3]), float(m[4])))
def envelope(pts):   # keep points that are not dominated (no smaller-or-equal file with higher-or-equal PSNR)
    out = []
    for r, p in sorted(pts):
        if not out or p > out[-1][1]: out.append((r, p))
    return out
def bd(a, b):   # a = test, b = anchor; returns (BD-PSNR dB, BD-rate %)
    a, b = sorted(a), sorted(b); la, pa = np.log10([x[0] for x in a]), np.array([x[1] for x in a]); lb, pb = np.log10([x[0] for x in b]), np.array([x[1] for x in b])
    lo, hi = max(la[0], lb[0]), min(la[-1], lb[-1])
    if hi - lo < 1e-6: return None
    g = np.linspace(lo, hi, 400); dp = float(np.trapezoid(np.interp(g, la, pa) - np.interp(g, lb, pb), g) / (hi - lo))
    plo, phi = max(pa.min(), pb.min()), min(pa.max(), pb.max()); dr = None
    if phi - plo > 1e-6 and np.all(np.diff(pa) > 0) and np.all(np.diff(pb) > 0):
        q = np.linspace(plo, phi, 400); dr = float((10 ** (np.trapezoid(np.interp(q, pa, la) - np.interp(q, pb, lb), q) / (phi - plo)) - 1) * 100)
    return dp, dr
res = collections.defaultdict(list)
for s in MIP + DB + OBJ:
    h = envelope(H[s]); c = envelope(cap[s].values())
    if len(h) < 2 or len(c) < 2: print('%-10s skipped (hold points %d, cap points %d)' % (s, len(h), len(c))); continue
    r = bd(h, c)
    if r is None: print('%-10s no overlap' % s); continue
    g = 'Mip-NeRF 360 / T&T' if s in MIP else ('Deep Blending' if s in DB else 'objects'); res[g].append(r); res['all'].append(r)
    print('%-10s hold pts %d (%.1f..%.1f MB) cap pts %d | BD-PSNR %+.2f dB | BD-rate %s' % (s, len(h), h[0][0], h[-1][0], len(c), r[0], ('%+.1f %%' % r[1]) if r[1] is not None else 'n/a'))
print()
for g, v in res.items():
    dr = [x[1] for x in v if x[1] is not None]
    print('%-20s n %2d | BD-PSNR mean %+.2f dB (better on %d) range %+.2f..%+.2f | BD-rate mean %s (n %d)' % (g, len(v), st.mean(x[0] for x in v), sum(x[0] > 0 for x in v), min(x[0] for x in v), max(x[0] for x in v), ('%+.1f %%' % st.mean(dr)) if dr else 'n/a', len(dr)))
