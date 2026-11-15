"""trained_vs_simplified_v1.py: MeshSplatting trained with a vertex cap against the FP32 model simplified after training to the SAME vertex
count (both fine-tuned, both written with our codec), every scene/cap pair that has both rows (TRAIN-AT-BUDGET / POST-HOC-COLLAPSE).
Neither side uses our controller.  PSNR of the decoded files."""
import re, glob, collections, statistics as st
MIP = ['room','kitchen','counter','bonsai','garden','bicycle','stump','flowers','treehill','truck','train']; DB = ['drjohnson','playroom']
A = re.compile(r'^(\w+) budget (\d+) (TRAIN-AT-BUDGET|POST-HOC-COLLAPSE)(?:-BLENDER)? \| vertices (\d+) .*?\| delivered ([\d.]+) MB test ([\d.]+)')
R = collections.defaultdict(dict)
for f in sorted(glob.glob('release-*/results/*.txt')):
    for l in open(f, errors='ignore'):
        m = A.match(l)
        if m: R[(m[1], int(m[2]))][m[3]] = (int(m[4]), float(m[5]), float(m[6]))
P = [(k[0], v['TRAIN-AT-BUDGET'], v['POST-HOC-COLLAPSE']) for k, v in R.items() if len(v) == 2]
def rep(name, sel):
    q = [p for p in P if sel(p[0])]; d = [p[1][2] - p[2][2] for p in q]
    print('%-9s pairs %3d scenes %2d | trained better on %3d | mean %+.2f dB (%+.2f..%+.2f) | vertices %.2f..%.2f M | trained file %.1f..%.1f MB' % (
        name, len(q), len({p[0] for p in q}), sum(x > 0 for x in d), st.mean(d), min(d), max(d), min(p[1][0] for p in q) / 1e6, max(p[1][0] for p in q) / 1e6, min(p[1][1] for p in q), max(p[1][1] for p in q)))
rep('Mip+T&T', lambda s: s in MIP); rep('DB', lambda s: s in DB); rep('captured', lambda s: s in MIP + DB); rep('objects', lambda s: s not in MIP + DB); rep('all', lambda s: True)
for lo, hi in ((0, 8), (8, 12), (12, 40)):
    v = [p[1][2] - p[2][2] for p in P if p[0] in MIP + DB and lo <= p[2][1] < hi]
    print('captured, simplified file %2d-%2d MB: n %2d mean %+.2f (%+.2f..%+.2f)' % (lo, hi, len(v), st.mean(v), min(v), max(v)))
