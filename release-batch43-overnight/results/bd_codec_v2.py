"""bd_codec_v2.py (2026-11-16): the codec comparison in the order the advisor asked for.
 (1) BM-Codec against TFAN / zstd / Draco on the SAME meshes, per scene, Bjontegaard deltas of BM-Codec relative to each generic codec:
       CAP  : MeshSplatting trained with vertex caps (and the default cap), 11 Mip-NeRF 360 / T&T scenes, 2 Deep Blending scenes, 8 objects
       SIMP : the FP32 model simplified after training, 11 Mip-NeRF 360 / T&T scenes
     BM-Codec curve = its file at each mesh level; a generic codec's curve = upper envelope of its files over all levels x {8,6,4}-bit SH.
 (2) ByteMesh (hold) against MeshSplatting with a cap + BM-Codec: the numbers of bd_hold_vs_cap_v1.py, per dataset (printed there).
 (3) raw (FP32, uncompressed) size and PSNR of the meshes at each level: V * 52 * 4 + F * 3 * 4 bytes; PSNR before quantisation.
Piecewise-linear interpolation over log10(size)."""
import re, glob, collections, statistics as st, numpy as np
MIP = ['room','kitchen','counter','bonsai','garden','bicycle','stump','flowers','treehill','truck','train']; DB = ['drjohnson','playroom']; OBJ = ['lego','chair','drums','ficus','hotdog','materials','mic','ship']
GR = re.compile(r'(\w+) GENERIC-RD \| mesh (\w+) \| V (\d+) F (\d+) \| sh(\d) \| zstd ([\d.]+) MB \| draco ([\d.]+) MB \| tfan ([\d.]+) MB \| test ([\d.]+) [\d.]+ [\d.]+ \| unquantised ([\d.]+)')
G = collections.defaultdict(dict)       # (scene, mesh) -> bits -> dict
for f in ['release-batch36-generic-on-cap/results/batch36-collect-2026-11-13-2342.txt', 'release-batch33-generic-rd/results/batch33-collect-2026-11-13-0109.txt', 'release-batch38-generic-objects/results/batch38-collect-2026-11-14-0924.txt',
          'release-batch38-generic-objects/results/batch38-materials-cap259000-rows.txt', 'release-batch43-overnight/results/batch43-db-generic-rows.txt']:
    for l in open(f):
        m = GR.match(l)
        if m: G[(m[1], m[2])][int(m[5])] = dict(zstd=float(m[6]), draco=float(m[7]), tfan=float(m[8]), psnr=float(m[9]), unq=float(m[10]), V=int(m[3]), F=int(m[4]))
A = re.compile(r'^(\w+) budget (\d+) (TRAIN-AT-BUDGET|POST-HOC-COLLAPSE)(?:-BLENDER)? \| vertices (\d+) .*?\| delivered ([\d.]+) MB test ([\d.]+)')
O = collections.defaultdict(dict)       # ('cap'|'simp', scene) -> budget -> (MB, PSNR)
for f in sorted(glob.glob('release-*/results/*.txt')):
    for l in open(f, errors='ignore'):
        m = A.match(l)
        if m: O[('cap' if m[3].startswith('TRAIN') else 'simp', m[1])][int(m[2])] = (float(m[5]), float(m[6]))
FULL = {}                               # our codec on the FP32 model (default cap)
FP = {'room': 28.48, 'counter': 26.48, 'kitchen': 27.46, 'bonsai': 28.28, 'garden': 24.73, 'bicycle': 23.03, 'stump': 24.93, 'flowers': 19.45, 'treehill': 20.39, 'truck': 22.39, 'train': 18.90}
for l in open('release-batch22-v1/results/generic-codecs-all-scenes-2026-11-10.md'):
    c = [x.strip() for x in l.split('|')]
    if len(c) > 4 and c[1] in MIP:
        m = re.match(r'([\d.]+) \(([-+\d.]+)\)', c[3]); FULL[c[1]] = (float(m[1]), FP[c[1]] + float(m[2]))
for f in sorted(glob.glob('release-*/results/*.txt')):
    for l in open(f, errors='ignore'):
        m = re.match(r'^(\w+) ORIGINAL-INDOOR \|.*?delivered ([\d.]+) MB test ([\d.]+)', l)
        if m and m[1] in DB: FULL[m[1]] = (float(m[2]), float(m[3]))
def envelope(pts):
    out = []
    for r, p in sorted(pts):
        if not out or p > out[-1][1]: out.append((r, p))
    return out
def bd(a, b):       # a relative to b: (BD-PSNR, BD-rate %)
    la, pa = np.log10([x[0] for x in a]), np.array([x[1] for x in a]); lb, pb = np.log10([x[0] for x in b]), np.array([x[1] for x in b])
    lo, hi = max(la[0], lb[0]), min(la[-1], lb[-1]); dp = None
    if hi - lo > 1e-6: g = np.linspace(lo, hi, 400); dp = float(np.trapezoid(np.interp(g, la, pa) - np.interp(g, lb, pb), g) / (hi - lo))
    plo, phi = max(pa.min(), pb.min()), min(pa.max(), pb.max()); dr = None
    if phi - plo > 1e-6: q = np.linspace(plo, phi, 400); dr = float((10 ** (np.trapezoid(np.interp(q, pa, la) - np.interp(q, pb, lb), q) / (phi - plo)) - 1) * 100)
    return dp, dr
def levels(kind, s):    # mesh names of this scene in G, with our codec's point
    out = []
    for (sc, mesh), g in G.items():
        if sc != s: continue
        if mesh == 'full': p = FULL.get(s)
        elif mesh.startswith('cap'): p = O[('cap', s)].get(int(mesh[3:])) if kind == 'cap' else None
        else: p = O[('simp', s)].get(int(mesh)) if kind == 'simp' else None
        if mesh == 'full' and s in OBJ: p = None
        if p is None and not (mesh == 'full'): 
            if (kind == 'cap') == mesh.startswith('cap'): pass
            continue
        if p is None: continue
        out.append((mesh, p, g))
    return out
print('== (1) BM-Codec relative to each generic codec, same meshes ==')
SUM = {}
for kind, name, scenes in (('cap', 'CAP  Mip-NeRF 360 / T&T', MIP), ('cap', 'CAP  Deep Blending', DB), ('cap', 'CAP  NeRF-Synthetic', OBJ), ('simp', 'SIMP Mip-NeRF 360 / T&T', MIP)):
    res = collections.defaultdict(list); nl = []
    for s in scenes:
        L = levels(kind, s); nl.append(len(L))
        if len(L) < 2: print('  %s: only %d level(s), skipped' % (s, len(L))); continue
        ours = envelope([p for _, p, _ in L])
        for k in ('tfan', 'zstd', 'draco'):
            gen = envelope([(g[b][k], g[b]['psnr']) for _, _, g in L for b in g])
            if len(ours) >= 2 and len(gen) >= 2: res[k].append(bd(ours, gen))
    print('%-26s scenes %2d, mesh levels per scene %d..%d' % (name, len(scenes), min(nl), max(nl)))
    for k in ('tfan', 'zstd', 'draco'):
        dp = [x[0] for x in res[k] if x[0] is not None]; dr = [x[1] for x in res[k] if x[1] is not None]; SUM[(name, k)] = (st.mean(dp) if dp else None, len(dp), st.mean(dr) if dr else None, len(dr))
        print('   vs %-5s BD-PSNR %s (n %d) | BD-rate %s (n %d, %s)' % (k, ('%+.2f dB' % st.mean(dp)) if dp else 'n/a', len(dp), ('%+.1f %%' % st.mean(dr)) if dr else 'n/a', len(dr), ('%+.1f..%+.1f' % (min(dr), max(dr))) if dr else '-'))
print('\n== (3) raw meshes before compression: mean over the 11 Mip-NeRF 360 / T&T scenes ==')
MID = {'room': 1078000, 'kitchen': 1086000, 'counter': 1083000, 'bonsai': 1073000, 'garden': 1073000, 'bicycle': 1080000, 'stump': 1069000, 'flowers': 1071000, 'treehill': 1079000, 'truck': 1078000, 'train': 1085000}
raw = lambda g: (g['V'] * 208 + g['F'] * 12) / 1e6
for lab, key in (('cap 0.6M', lambda s: 'cap600000'), ('cap 1.08M', lambda s: 'cap%d' % MID[s]), ('cap 3M', lambda s: 'cap3000000'), ('default (FP32 model)', lambda s: 'full')):
    v = [G[(s, key(s))][8] for s in MIP if (s, key(s)) in G]; print('  %-22s n %2d | raw %.0f MB | PSNR before quantisation %.2f dB | vertices %.2f M' % (lab, len(v), st.mean(raw(g) for g in v), st.mean(g['unq'] for g in v), st.mean(g['V'] for g in v) / 1e6))
S = collections.defaultdict(list)
for s in MIP:
    for i, (mesh, g) in enumerate(sorted(((m, g) for (sc, m), g in G.items() if sc == s and m.isdigit()), key=lambda t: -int(t[0]))): S[i].append(g[8])
for i, v in S.items(): print('  simplified level %d       n %2d | raw %.0f MB | PSNR before quantisation %.2f dB | vertices %.2f M' % (i + 1, len(v), st.mean(raw(g) for g in v), st.mean(g['unq'] for g in v), st.mean(g['V'] for g in v) / 1e6))
print('\n== Deep Blending means (2 scenes): our codec and generic codecs on the cap meshes ==')
for mesh in ('cap600000', 'cap1081000', 'cap3000000', 'full'):
    o = [(FULL[s] if mesh == 'full' else O[('cap', s)][int(mesh[3:])]) for s in DB]; g = [G[(s, mesh)] for s in DB]
    print('  %-11s BM-Codec %.1f MB %.2f dB | ' % (mesh, st.mean(x[0] for x in o), st.mean(x[1] for x in o)) + ' | '.join('sh%d: tfan %.1f zstd %.1f draco %.1f MB, %.2f dB' % (b, st.mean(x[b]['tfan'] for x in g), st.mean(x[b]['zstd'] for x in g), st.mean(x[b]['draco'] for x in g), st.mean(x[b]['psnr'] for x in g)) for b in (8, 6, 4)))
