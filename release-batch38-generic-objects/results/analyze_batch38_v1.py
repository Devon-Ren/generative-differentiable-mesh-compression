"""analyze_batch38_v1.py: generic codecs on the NeRF-Synthetic objects (cap-trained meshes at three caps per object + the default-cap model)
against our codec on the same cap meshes and against BudgetMesh (hold) at 1.5 / 2.5 / 4 MB."""
import re, glob, json, math, collections
OBJ = ['lego','chair','drums','ficus','hotdog','materials','mic','ship']; mean = lambda v: sum(v) / len(v)
R = re.compile(r'(\w+) GENERIC-RD \| mesh (\w+) \| V (\d+) F (\d+) \| sh(\d) \| zstd ([\d.]+) MB \| draco ([\d.]+) MB \| tfan ([\d.]+) MB \| test ([\d.]+) [\d.]+ [\d.]+ \| unquantised ([\d.]+)')
G = collections.defaultdict(dict)
for l in open('release-batch38-generic-objects/results/batch38-collect-2026-11-14-0924.txt').read().splitlines() + open('release-batch38-generic-objects/results/batch38-materials-cap259000-rows.txt').read().splitlines():
    m = R.match(l)
    if m: G[(m[1], m[2])][int(m[5])] = dict(zstd=float(m[6]), draco=float(m[7]), tfan=float(m[8]), psnr=float(m[9]), V=int(m[3]))
C = re.compile(r'^(\w+) budget (\d+) TRAIN-AT-BUDGET-BLENDER \| vertices (\d+) .*?\| delivered ([\d.]+) MB test ([\d.]+)'); cap = collections.defaultdict(dict)
for f in sorted(glob.glob('release-*/results/*.txt')):
    for l in open(f, errors='ignore'):
        m = C.match(l)
        if m and m[1] in OBJ: cap[m[1]].setdefault(int(m[2]), (float(m[4]), float(m[5])))
H = collections.defaultdict(dict)
for l in open('release-batch31-hold/results/batch31-collect-2026-11-13-0126.txt').read().splitlines() + open('release-batch31-hold/results/batch31-ship-rows-2026-11-14.txt').read().splitlines():
    m = re.match(r'(\w+) abl24-hold \| target ([\d.]+) MB \|.*?delivered ([\d.]+) MB.*?\| test ([\d.]+)', l)
    if m and m[1] in OBJ: H[m[1]][float(m[2])] = (float(m[3]), float(m[4]))
LEV = {}
for s in OBJ:
    caps = sorted(int(k[3:]) for (sc, k) in G if sc == s and k.startswith('cap'))
    for name, c in zip(('low', 'mid', 'high'), caps): LEV[(s, name)] = c
FULL = [s for s in OBJ if all((s, n) in LEV and len(G[(s, f'cap{LEV[(s, n)]}')]) == 3 for n in ('low', 'mid', 'high')) and len(G[(s, 'full')]) == 3]
print('caps per object:', {s: [LEV.get((s, n)) for n in ('low', 'mid', 'high')] for s in OBJ}); print('objects with all generic rows:', FULL); print('hold targets per object:', {s: sorted(H[s]) for s in OBJ})
def at_psnr(curve, p):
    c = sorted(curve, key=lambda x: x[1])
    for (a, pa), (b, pb) in zip(c, c[1:]):
        if pa <= p <= pb and pb > pa: return math.exp(math.log(a) + (math.log(b) - math.log(a)) * (p - pa) / (pb - pa))
print('\n== codec alone: same cap-trained mesh, size each generic codec needs for the PSNR of our codec (per object, then mean over objects and levels) ==')
allr = collections.defaultdict(list)
for n in ('low', 'mid', 'high'):
    row = {}
    for k in ('tfan', 'zstd', 'draco'):
        r = []
        for s in OBJ:
            if (s, n) not in LEV: continue
            c = LEV[(s, n)]; mb, p = cap[s][c]; g = at_psnr([(G[(s, f'cap{c}')][b][k], G[(s, f'cap{c}')][b]['psnr']) for b in (8, 6, 4)], p)
            if g: r.append(g / mb); allr[k].append(g / mb)
        row[k] = r
    print('level %-4s | ' % n + ' | '.join('%s x%.2f (%.2f..%.2f, n %d)' % (k, mean(v), min(v), max(v), len(v)) for k, v in row.items() if v))
print('all levels: ' + ' | '.join('%s x%.2f (n %d)' % (k, mean(v), len(v)) for k, v in allr.items()))
SET = [s for s in FULL if len(H[s]) == 3]; print('\nobjects with all generic rows and all three BudgetMesh targets:', SET)
def hull(p):
    P = sorted((math.log(x), y) for x, y in p); Hh = []
    for q in P:
        while len(Hh) >= 2 and (Hh[-1][0] - Hh[-2][0]) * (q[1] - Hh[-2][1]) - (Hh[-1][1] - Hh[-2][1]) * (q[0] - Hh[-2][0]) >= 0: Hh.pop()
        Hh.append(q)
    k = max(range(len(Hh)), key=lambda t: Hh[t][1]); return [(math.exp(a), b) for a, b in Hh[:k + 1]]
pts = collections.defaultdict(list)
for n in ('low', 'mid', 'high', 'full'):
    for b in (8, 6, 4):
        key = lambda s: 'full' if n == 'full' else f'cap{LEV[(s, n)]}'
        for k in ('tfan', 'zstd', 'draco'): pts[k].append((mean([G[(s, key(s))][b][k] for s in SET]), mean([G[(s, key(s))][b]['psnr'] for s in SET])))
co = [(mean([cap[s][LEV[(s, n)]][0] for s in SET]), mean([cap[s][LEV[(s, n)]][1] for s in SET])) for n in ('low', 'mid', 'high')]
bm = [(mean([H[s][t][0] for s in SET]), mean([H[s][t][1] for s in SET])) for t in (1.5, 2.5, 4.0)]; HU = {k: hull(v) for k, v in pts.items()}
print('== mean curves over these %d objects ==' % len(SET)); print('BudgetMesh     :', [(round(a, 2), round(b, 2)) for a, b in bm]); print('cap + our codec:', [(round(a, 2), round(b, 2)) for a, b in co])
for k in ('tfan', 'zstd', 'draco'): print('%-5s hull:' % k, [(round(a, 2), round(b, 2)) for a, b in HU[k]])
for name, cur in (('BudgetMesh', bm), ('cap + our codec', co)):
    for mb, p in cur: print('  %-16s %5.2f MB %.2f dB | ' % (name, mb, p) + ' | '.join('%s %s' % (k, ('%.1f MB (x%.1f)' % (at_psnr(HU[k], p), at_psnr(HU[k], p) / mb)) if at_psnr(HU[k], p) else 'out of range') for k in ('tfan', 'zstd', 'draco')))
json.dump(dict(objects=SET, budgetmesh=bm, cap_ours=co, points=pts, hull=HU), open('release-batch38-generic-objects/results/batch38-means.json', 'w'), indent=1)
