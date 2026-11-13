import re, sys, math, statistics as st, collections
MIP = ['room','kitchen','counter','bonsai','garden','bicycle','stump','flowers','treehill','truck','train']; DB = ['drjohnson','playroom']; OBJ = ['lego','chair','drums','ficus','hotdog','materials','mic','ship']; ALL = MIP + DB + OBJ
txt = open('release-batch30-overnight/results/batch30-collect-2026-11-12-2007.txt').read(); sec = {m.group(1): m.group(2) for m in re.finditer(r'===== ([^\n]+)\n(.*?)(?=\n===== |\Z)', txt, re.S)}
C = re.compile(r'^(\w+) budget (\d+) TRAIN-AT-BUDGET(?:-BLENDER)? \| vertices (\d+) .*?\| delivered ([\d.]+) MB test ([\d.]+) ([\d.]+) ([\d.]+)')
cap = collections.defaultdict(dict)
srcs = [sec['cap rows']] + [open(f).read() for f in ['release-g0-refresh-v1/results/g0-refresh-2026-11-09.txt', 'release-g0-refresh-v1/results/g0-refresh-indoor-2026-11-09.txt', 'release-batch21-v1/results/batch21-collect-2026-11-09-2110-raw.txt', 'release-batch24-ablation-v1/results/batch24-collect-2026-11-11-0641.txt', 'release-batch28-cap-stability/results/batch28-collect-2026-11-12-rows.txt', 'release-batch31-hold/results/batch31-collect-2026-11-13-0126.txt']]
for src in srcs:
    for l in src.splitlines():
        m = C.match(l)
        if m: cap[m[1]][int(m[2])] = (float(m[4]), float(m[5]), float(m[6]), float(m[7]))
def cap_at(s, mb, k=1):
    pts = sorted(cap[s].values())
    for a, b in zip(pts, pts[1:]):
        if a[0] <= mb <= b[0] and b[0] > a[0]: w = math.log(mb / a[0]) / math.log(b[0] / a[0]); return a[k] + (b[k] - a[k]) * w
    return None
H = {}
for l in open('release-batch31-hold/results/batch31-collect-2026-11-13-0126.txt'):
    m = re.match(r'(\w+) abl24-hold \| target ([\d.]+) MB \|(.*?)trained V (\d+).*?delivered ([\d.]+) MB.*?\| test ([\d.]+) ([\d.]+) ([\d.]+)', l)
    if m: H[(m[1], float(m[2]))] = (float(m[5]), float(m[6]), float(m[7]), float(m[8]))
res = collections.defaultdict(list); out = []
for (s, T), (mb, p, ss, lp) in sorted(H.items(), key=lambda kv: (ALL.index(kv[0][0]), -kv[0][1])):
    c = cap_at(s, mb); g = 'mip' if s in MIP else ('db' if s in DB else 'obj')
    rng = sorted(v[0] for v in cap[s].values())
    if c is None: out.append((s, T, mb, 'cap runs span %.2f..%.2f MB (n %d)' % (rng[0], rng[-1], len(rng)) if rng else 'no cap runs')); continue
    res[(g, T)].append((p - c, s)); res[(g, 'all')].append((p - c, s)); res[('all', 'all')].append((p - c, s))
    print('%-10s T %4g  %7.3f MB  PSNR %+.2f dB  (cap pts n %d)' % (s, T, mb, p - c, len(rng)))
print()
for k, r in res.items(): print(k, 'n', len(r), 'mean %+.2f' % st.mean(x[0] for x in r), 'better on', sum(x[0] > 0 for x in r), 'min %+.2f max %+.2f' % (min(x[0] for x in r), max(x[0] for x in r)))
print('\nnot comparable:'); [print('  ', o) for o in out]
