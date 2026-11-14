"""analyze_batch36_v1.py: generic codecs on cap-trained meshes against our codec on the same meshes and against BudgetMesh (hold).
11 Mip-NeRF 360 / T&T scenes.  Mesh levels: cap 600,000 / the scene's cap near 1.08 million / 3,000,000 / default 4,000,000 (FP32 model, batch33)."""
import re, glob, json, math, collections, statistics as st
MIP = ['room','kitchen','counter','bonsai','garden','bicycle','stump','flowers','treehill','truck','train']
MID = {'room': 1078000, 'kitchen': 1086000, 'counter': 1083000, 'bonsai': 1073000, 'garden': 1073000, 'bicycle': 1080000, 'stump': 1069000, 'flowers': 1071000, 'treehill': 1079000, 'truck': 1078000, 'train': 1085000}
LEV = ['cap 0.6M', 'cap 1.08M', 'cap 3M', 'default 4M']
def lev(s, c): return 'cap 0.6M' if c == 600000 else ('cap 3M' if c == 3000000 else ('cap 1.08M' if c == MID[s] else None))
G = collections.defaultdict(dict)   # (scene, level) -> bits -> dict(zstd, draco, tfan, psnr, unq)
R = re.compile(r'(\w+) GENERIC-RD \| mesh (\w+) \| V (\d+) F (\d+) \| sh(\d) \| zstd ([\d.]+) MB \| draco ([\d.]+) MB \| tfan ([\d.]+) MB \| test ([\d.]+) [\d.]+ [\d.]+ \| unquantised ([\d.]+)')
for f in ['release-batch36-generic-on-cap/results/batch36-collect-2026-11-13-2342.txt', 'release-batch33-generic-rd/results/batch33-collect-2026-11-13-0109.txt']:
    for l in open(f):
        m = R.match(l)
        if not m or m[1] not in MIP: continue
        L = 'default 4M' if m[2] == 'full' else (lev(m[1], int(m[2][3:])) if m[2].startswith('cap') else None)
        if L: G[(m[1], L)][int(m[5])] = dict(zstd=float(m[6]), draco=float(m[7]), tfan=float(m[8]), psnr=float(m[9]), unq=float(m[10]), V=int(m[3]))
C = re.compile(r'^(\w+) budget (\d+) TRAIN-AT-BUDGET \| vertices (\d+) .*?\| delivered ([\d.]+) MB test ([\d.]+)')
O = {}   # (scene, level) -> (MB, PSNR) of our codec on the same cap mesh
for f in sorted(glob.glob('release-*/results/*.txt')):
    for l in open(f, errors='ignore'):
        m = C.match(l)
        if m and m[1] in MIP and lev(m[1], int(m[2])): O[(m[1], lev(m[1], int(m[2])))] = (float(m[4]), float(m[5]))
D4 = {}; FPp = {}
for l in open('release-batch22-v1/results/generic-codecs-all-scenes-2026-11-10.md'):
    c = [x.strip() for x in l.split('|')]
    if len(c) > 4 and c[1] in MIP:
        m = re.match(r'([\d.]+) \(([-+\d.]+)\)', c[3]); D4[c[1]] = (float(m[1]), float(m[2]))
H = collections.defaultdict(dict)
for l in open('release-batch31-hold/results/batch31-collect-2026-11-13-0126.txt'):
    m = re.match(r'(\w+) abl24-hold \| target ([\d.]+) MB \|.*?delivered ([\d.]+) MB.*?\| test ([\d.]+)', l)
    if m and m[1] in MIP: H[m[1]][float(m[2])] = (float(m[3]), float(m[4]))
for s in MIP:   # our codec at the default cap: size from the table, PSNR = FP32 model PSNR + the loss listed there (the 'unquantised' of the full mesh is the fine-tuned FP32 model)
    O[(s, 'default 4M')] = (D4[s][0], None)
n = {L: sum((s, L) in G and len(G[(s, L)]) == 3 for s in MIP) for L in LEV}; print('scenes with all three precisions per level:', n, '| our codec rows:', {L: sum((s, L) in O for s in MIP) for L in LEV})
mean = lambda v: sum(v) / len(v)
print('\n== means over the 11 scenes: size in MB (PSNR) ==')
print('%-11s | %-16s | %-44s | %-44s | %-44s' % ('mesh', 'our codec', 'TFAN sh8 / sh6 / sh4', 'zstd sh8 / sh6 / sh4', 'Draco sh8 / sh6 / sh4'))
M = {}; pts = collections.defaultdict(list)
for L in LEV:
    row = {}
    for b in (8, 6, 4):
        for k in ('tfan', 'zstd', 'draco'): row[(k, b)] = mean([G[(s, L)][b][k] for s in MIP])
        row[('psnr', b)] = mean([G[(s, L)][b]['psnr'] for s in MIP])
        for k in ('tfan', 'zstd', 'draco'): pts[k].append((row[(k, b)], row[('psnr', b)]))
    o = [O[(s, L)] for s in MIP]; om = mean([x[0] for x in o]); op = mean([x[1] for x in o]) if all(x[1] is not None for x in o) else None
    M[L] = (om, op, row)
    f = lambda k: ' / '.join('%.1f (%.2f)' % (row[(k, b)], row[('psnr', b)]) for b in (8, 6, 4))
    print('%-11s | %5.1f (%s) | %-44s | %-44s | %-44s' % (L, om, ('%.2f' % op) if op else 'see batch33', f('tfan'), f('zstd'), f('draco')))
def at_psnr(curve, p):   # size of a (MB, PSNR) curve at PSNR p, log-linear in size; None outside
    c = sorted(curve, key=lambda x: x[1])
    for (a, pa), (b, pb) in zip(c, c[1:]):
        if pa <= p <= pb and pb > pa: return math.exp(math.log(a) + (math.log(b) - math.log(a)) * (p - pa) / (pb - pa))
    return None
print('\n== codec alone: same cap-trained mesh, size each generic codec needs for the PSNR of our codec (per scene, then mean) ==')
for L in LEV[:3]:
    out = {}
    for k in ('tfan', 'zstd', 'draco'):
        r = []
        for s in MIP:
            mb, p = O[(s, L)]; g = at_psnr([(G[(s, L)][b][k], G[(s, L)][b]['psnr']) for b in (8, 6, 4)], p)
            if g: r.append(g / mb)
        out[k] = r
    print('%-10s our codec %.1f MB | ' % (L, M[L][0]) + ' | '.join('%s x%.2f (%.2f..%.2f, n %d)' % (k, mean(v), min(v), max(v), len(v)) for k, v in out.items() if v))
def hull(p):
    P = sorted((math.log(x), y) for x, y in p); Hh = []
    for q in P:
        while len(Hh) >= 2 and (Hh[-1][0] - Hh[-2][0]) * (q[1] - Hh[-2][1]) - (Hh[-1][1] - Hh[-2][1]) * (q[0] - Hh[-2][0]) >= 0: Hh.pop()
        Hh.append(q)
    k = max(range(len(Hh)), key=lambda t: Hh[t][1]); return [(math.exp(a), b) for a, b in Hh[:k + 1]]
ours_cap = [(M[L][0], M[L][1]) for L in LEV[:3]]
bm = [(mean([H[s][t][0] for s in MIP]), mean([H[s][t][1] for s in MIP])) for t in (5.0, 10.0, 20.0)]
print('\n== mean curves ==\nBudgetMesh (hold):', [(round(a, 2), round(b, 2)) for a, b in bm]); print('cap + our codec  :', [(round(a, 2), round(b, 2)) for a, b in ours_cap])
HU = {k: hull(v) for k, v in pts.items()}
for k in ('tfan', 'zstd', 'draco'): print('%-5s hull:' % k, [(round(a, 1), round(b, 2)) for a, b in HU[k]])
print('\n== size each generic codec needs (upper hull of its 12 mean points) for the PSNR of ... ==')
for name, cur in (('BudgetMesh', bm), ('cap + our codec', ours_cap)):
    for mb, p in cur:
        print('  %-16s %5.2f MB %.2f dB | ' % (name, mb, p) + ' | '.join('%s %s' % (k, ('%.1f MB (x%.1f)' % (at_psnr(HU[k], p), at_psnr(HU[k], p) / mb)) if at_psnr(HU[k], p) else 'out of range') for k in ('tfan', 'zstd', 'draco')))
json.dump(dict(budgetmesh=bm, cap_ours=ours_cap, points={k: v for k, v in pts.items()}, hull=HU, default_ours_mb=M['default 4M'][0]), open('release-batch36-generic-on-cap/results/batch36-means.json', 'w'), indent=1)
