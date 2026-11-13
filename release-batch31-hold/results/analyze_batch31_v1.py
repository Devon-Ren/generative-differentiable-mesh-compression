"""analyze_batch31_v1.py <collect>: the hold controller at every target: size error, PSNR against the current method (same target, and at the
same file size on the current method's own curve), against the final design of batch30 at the middle target; checkpoints of the objects."""
import re, sys, math, statistics as st, collections
MIP = ['room','kitchen','counter','bonsai','garden','bicycle','stump','flowers','treehill','truck','train']; DB = ['drjohnson','playroom']; OBJ = ['lego','chair','drums','ficus','hotdog','materials','mic','ship']; ALL = MIP + DB + OBJ
old = collections.defaultdict(dict)
for f in ['release-adapt-v9b/results/adapt-v9b-2026-11-08.txt', 'release-adapt-v9b/results/adapt-v9b-blender-2026-11-08.txt', 'release-batch21-v1/results/batch21-collect-2026-11-09-2110-raw.txt', 'release-adapt-v9d/results/batch18-deep-blending-2026-11-08-raw.txt']:
    for l in open(f):
        m = re.match(r'(\w+) \| target ([\d.]+) MB \|.*?lam_final \(shadow\) ([\d.e+-]+) \|.*?delivered ([\d.]+) MB \(hit ([\d.]+) x target.*?\| test ([\d.]+) ([\d.]+) ([\d.]+)', l)
        if m: old[m[1]][float(m[2])] = (float(m[4]), float(m[6]), float(m[7]), float(m[8]), float(m[5]) < 0.9 and float(m[3]) <= 1.1e-4)
def old_at(s, mb):
    pts = sorted(v[:2] for v in old[s].values())
    for k, ((a, pa), (b, pb)) in enumerate(zip(pts, pts[1:])):
        if mb <= b or k == len(pts) - 2: return pa + (pb - pa) * math.log(mb / a) / math.log(b / a)
new = collections.defaultdict(dict)
for l in open(sys.argv[1]):
    m = re.match(r'(\w+) abl24-(hold|final) \| target ([\d.]+) MB \|(.*?)trained V (\d+).*?delivered ([\d.]+) MB.*?\| test ([\d.]+) ([\d.]+) ([\d.]+)', l)
    if m: bi = re.search(r'best iter (\d+)', m[4]); new[m[2]][(m[1], float(m[3]))] = (float(m[6]), float(m[7]), float(m[8]), float(m[9]), int(bi.group(1)) if bi else None)
H = new['hold']; print(len(H), 'hold runs,', len(new['final']), 'final-design runs in this file')
def summ(e): a = [abs(x) for x in e]; return 'n %2d | mean abs %4.1f%% | within 2%%: %2d  5%%: %2d  10%%: %2d | range %+5.1f%% .. %+5.1f%%' % (len(a), st.mean(a), sum(x <= 2 for x in a), sum(x <= 5 for x in a), sum(x <= 10 for x in a), min(e), max(e))
print('\n%-10s %5s | %8s %7s %7s %6s %6s | current: %7s %7s | dPSNR same target | dPSNR same size | final design: err, PSNR' % ('scene', 'T', 'MB', 'err', 'PSNR', 'SSIM', 'LPIPS', 'err', 'PSNR'))
res = collections.defaultdict(list)
for s in ALL:
    for T in sorted({t for (x, t) in H if x == s}, reverse=True):
        mb, p, ss, lp, bi = H[(s, T)]; o = old[s].get(T); f = new['final'].get((s, T)); grp = 'Mip-360 / T&T' if s in MIP else ('Deep Blending' if s in DB else 'objects')
        e = 100 * (mb / T - 1); ds = p - old_at(s, mb) if len(old[s]) >= 2 else None; capl = o[4] if o else False
        res[(grp, T)].append((e, ds, (p - o[1]) if o else None, 100 * (o[0] / T - 1) if o else None, capl, s))
        print('%-10s %5g | %8.3f %+6.1f%% %7.3f %6.4f %6.4f | %s | %s | %s | %s%s' % (s, T, mb, e, p, ss, lp, ('%+6.1f%% %7.3f' % (100 * (o[0] / T - 1), o[1])) if o else '      -       -', ('%+.2f' % (p - o[1])) if o else '  -  ', ('%+.2f' % ds) if ds is not None else '  -  ',
              ('%+.1f%% %.3f (hold %+.2f dB)' % (100 * (f[0] / T - 1), f[1], p - f[1])) if f else '-', ('  ckpt %d' % bi) if bi else ''))
print()
for (grp, T), r in sorted(res.items(), key=lambda kv: (['Mip-360 / T&T', 'Deep Blending', 'objects'].index(kv[0][0]), -kv[0][1])):
    ds = [x[1] for x in r if x[1] is not None]; oe = [x[3] for x in r if x[3] is not None]
    print('%-14s T %4g | hold: %s | current: %s | PSNR at the same size: mean %+.2f dB, better on %d of %d' % (grp, T, summ([x[0] for x in r]), summ(oe) if oe else '-', st.mean(ds) if ds else float('nan'), sum(x > 0 for x in ds), len(ds)))
allh = [x for r in res.values() for x in r]; incap = [x for x in allh if x[5] not in ('ficus',) and not (x[5] == 'mic' and False)]
print('\nall hold runs:', summ([x[0] for x in allh])); misses = sorted([(x[5], T, round(x[0], 1)) for (g, T), r in res.items() for x in r if abs(x[0]) > 5], key=lambda t: t[2]); print('runs outside 5%:', misses)
ds = [x[1] for x in allh if x[1] is not None]; print('PSNR at the same size, hold minus current, all runs: mean %+.2f dB, better on %d of %d' % (st.mean(ds), sum(x > 0 for x in ds), len(ds)))
