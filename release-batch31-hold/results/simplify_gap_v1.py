"""simplify_gap_v1.py: PSNR gap between a mesh optimized at a size (hold controller) and the FP32 MeshSplatting model simplified
after training (post-hoc edge collapse, same codec) at the same file size, Mip-NeRF 360 + T&T. The hold PSNR is interpolated linearly
over log10(size) between hold points; simplified points outside the hold range are skipped."""
import re, collections, numpy as np, statistics as st
MIP = ['room','kitchen','counter','bonsai','garden','bicycle','stump','flowers','treehill','truck','train']
H = collections.defaultdict(list)
for fn in ('release-batch31-hold/results/batch31-collect-2026-11-13-0126.txt', 'release-batch37-more-points/results/batch37-rows-2026-11-14.txt'):
    for l in open(fn):
        m = re.match(r'(\w+) abl24-hold \| target ([\d.]+) MB \|.*?delivered ([\d.]+) MB.*?\| test ([\d.]+)', l)
        if m: H[m[1]].append((float(m[3]), float(m[4])))
simp = collections.defaultdict(set)
for f in ['release-g0-refresh-v1/results/g0-refresh-2026-11-09.txt', 'release-g0-refresh-v1/results/g0-refresh-indoor-2026-11-09.txt', 'release-batch21-v1/results/batch21-collect-2026-11-09-2110-raw.txt', 'release-batch24-ablation-v1/results/batch24-collect-2026-11-11-0641.txt']:
    for l in open(f):
        m = re.match(r'(\w+) budget (\d+) POST-HOC-COLLAPSE \| .*?\| delivered ([\d.]+) MB test ([\d.]+)', l)
        if m and m[1] in MIP: simp[m[1]].add((float(m[3]), float(m[4])))
out = []
for s in MIP:
    h = sorted(H[s]); lx = np.log10([p[0] for p in h]); y = [p[1] for p in h]
    for mb, ps in sorted(simp[s]):
        if lx[0] - 1e-9 <= np.log10(mb) <= lx[-1] + 1e-9: out.append((s, mb, float(np.interp(np.log10(mb), lx, y)) - ps))
for s, mb, d in out: print('%-9s %6.2f MB  optimized-at-size minus simplified %+.2f dB' % (s, mb, d))
print('all: n %d mean %+.2f range %+.2f..%+.2f' % (len(out), st.mean(o[2] for o in out), min(o[2] for o in out), max(o[2] for o in out)))
for lo, hi in ((0, 8), (8, 12), (12, 40)):
    v = [o for o in out if lo <= o[1] < hi]
    print('%2d-%2d MB: n %2d mean %+.2f range %+.2f..%+.2f' % (lo, hi, len(v), st.mean(o[2] for o in v), min(o[2] for o in v), max(o[2] for o in v)))
