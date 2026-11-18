"""bd_metrics_v1.py (2026-11-18): Bjontegaard deltas of ByteMesh against MeshSplatting-cap (both BM-Codec) in PSNR, SSIM and LPIPS.
Same points and conventions as bd_hold_vs_cap_v1.py: per scene, the cap curve is its upper envelope in the metric, piecewise-linear over
log10(file size), averaged over the overlap of the two curves; then averaged over scenes.  For LPIPS, lower is better: the delta is
reported as ByteMesh minus cap (negative = ByteMesh better).  Run from the repository root."""
import re, glob, collections, statistics as st, numpy as np
MIP = ['room','kitchen','counter','bonsai','garden','bicycle','stump','flowers','treehill','truck','train']; DB = ['drjohnson','playroom']; OBJ = ['lego','chair','drums','ficus','hotdog','materials','mic','ship']
C = re.compile(r'^(\w+) budget (\d+) TRAIN-AT-BUDGET(?:-BLENDER)? \| vertices (\d+) .*?\| delivered ([\d.]+) MB test ([\d.]+) ([\d.]+) ([\d.]+)')
cap = collections.defaultdict(dict)
for f in sorted(glob.glob('release-*/results/*.txt')):
    for l in open(f, errors='ignore'):
        m = C.match(l)
        if m: cap[m[1]][int(m[2])] = tuple(float(x) for x in m.groups()[3:])
H = collections.defaultdict(list)
for fn in ('release-batch31-hold/results/batch31-collect-2026-11-13-0126.txt', 'release-batch37-more-points/results/batch37-rows-2026-11-14.txt', 'release-batch31-hold/results/batch31-ship-rows-2026-11-14.txt'):
    for l in open(fn):
        m = re.match(r'(\w+) abl24-hold \| target ([\d.]+) MB \|.*?delivered ([\d.]+) MB.*?\| test ([\d.]+) ([\d.]+) ([\d.]+)', l)
        if m: H[m[1]].append(tuple(float(x) for x in m.groups()[2:]))
def env(pts, k, sign):      # pts (MB, psnr, ssim, lpips); keep points not dominated in metric k (sign +1 higher better, -1 lower better)
    out = []
    for p in sorted(pts):
        if not out or sign * p[k] > sign * out[-1][1]: out.append((p[0], p[k]))
    return out
def bdq(a, b):
    la, pa = np.log10([x[0] for x in a]), np.array([x[1] for x in a]); lb, pb = np.log10([x[0] for x in b]), np.array([x[1] for x in b])
    lo, hi = max(la[0], lb[0]), min(la[-1], lb[-1])
    if hi - lo < 1e-6: return None
    g = np.linspace(lo, hi, 400); return float(np.trapezoid(np.interp(g, la, pa) - np.interp(g, lb, pb), g) / (hi - lo))
R = collections.defaultdict(lambda: collections.defaultdict(list)); W = collections.defaultdict(lambda: collections.defaultdict(int))
for s in MIP + DB + OBJ:
    grp = 'Mip-NeRF 360 + T&T (11)' if s in MIP else ('Deep Blending (2)' if s in DB else 'NeRF-Synthetic (8)'); row = []
    for k, name, sign in ((1, 'PSNR', 1), (2, 'SSIM', 1), (3, 'LPIPS', -1)):
        h, c = env(H[s], k, sign), env(cap[s].values(), k, sign)
        d = bdq(h, c) if len(h) >= 2 and len(c) >= 2 else None
        row.append('n/a' if d is None else ('%+.4f' % d if k > 1 else '%+.2f' % d))
        if d is not None:
            for gg in (grp, 'all (21)'): R[gg][name].append(d); W[gg][name] += (sign * d > 0)
    print('%-10s BD-PSNR %s dB | BD-SSIM %s | BD-LPIPS %s' % (s, *row))
print()
for g in ('Mip-NeRF 360 + T&T (11)', 'Deep Blending (2)', 'NeRF-Synthetic (8)', 'all (21)'):
    print('%-24s ' % g + ' | '.join('BD-%s %s (n %d, ByteMesh better on %d)' % (n, ('%+.2f dB' if n == 'PSNR' else '%+.4f') % st.mean(R[g][n]), len(R[g][n]), W[g][n]) for n in ('PSNR', 'SSIM', 'LPIPS')))
