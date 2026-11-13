"""analyze_batch30_cap_v1.py <collect file>: the final design against the vertex cap: PSNR at the same file size (cap curve of the scene
interpolated at our file size, all cap runs of the scene) and time per training (cap jobs split by their stage time stamps)."""
import re, sys, math, statistics as st, collections, datetime
txt = open(sys.argv[1]).read(); sec = {m.group(1): m.group(2) for m in re.finditer(r'===== ([^\n]+)\n(.*?)(?=\n===== |\Z)', txt, re.S)}
MIP = ['room','kitchen','counter','bonsai','garden','bicycle','stump','flowers','treehill','truck','train']; DB = ['drjohnson','playroom']; OBJ = ['lego','chair','drums','ficus','hotdog','materials','mic','ship']; ALL = MIP + DB + OBJ
T0 = {s: (2.5 if s in OBJ else 10.0) for s in ALL}
C = re.compile(r'^(\w+) budget (\d+) TRAIN-AT-BUDGET(?:-BLENDER)? \| vertices (\d+) .*?\| delivered ([\d.]+) MB test ([\d.]+) ([\d.]+) ([\d.]+)')
cap = collections.defaultdict(dict)
srcs = [sec['cap rows']] + [open(f).read() for f in ['release-g0-refresh-v1/results/g0-refresh-2026-11-09.txt', 'release-g0-refresh-v1/results/g0-refresh-indoor-2026-11-09.txt', 'release-batch21-v1/results/batch21-collect-2026-11-09-2110-raw.txt', 'release-batch24-ablation-v1/results/batch24-collect-2026-11-11-0641.txt', 'release-batch28-cap-stability/results/batch28-collect-2026-11-12-rows.txt']]
for src in srcs:
    for l in src.splitlines():
        m = C.match(l)
        if m: cap[m[1]][int(m[2])] = (float(m[4]), float(m[5]), float(m[6]), float(m[7]))
fin = {}
for l in sec['controller rows'].splitlines():
    m = re.match(r'(\w+) abl24-final \| target ([\d.]+) MB \|.*?delivered ([\d.]+) MB.*?\| test ([\d.]+) ([\d.]+) ([\d.]+)', l)
    if m: fin[m[1]] = (float(m[3]), float(m[4]), float(m[5]), float(m[6]))
def cap_at(s, mb, k=1):
    pts = sorted(cap[s].values()); best = None
    for a, b in zip(pts, pts[1:]):
        if a[0] <= mb <= b[0] and b[0] > a[0]: w = math.log(mb / a[0]) / math.log(b[0] / a[0]); return a[k] + (b[k] - a[k]) * w
    return None
print('final design minus vertex cap at the same file size (cap curve interpolated between the two nearest cap runs of the scene)')
res = collections.defaultdict(list)
for s in ALL:
    if s not in fin: continue
    mb, p, ss, lp = fin[s]; c = cap_at(s, mb); cs = cap_at(s, mb, 2); cl = cap_at(s, mb, 3)
    if c is None: print('  %-10s %.3f MB: outside the range of the cap runs (%s)' % (s, mb, ', '.join('%.2f' % v[0] for v in sorted(cap[s].values())))); continue
    g = 'mip' if s in MIP else ('db' if s in DB else 'obj'); res[g].append((p - c, ss - cs, lp - cl)); print('  %-10s %7.3f MB  PSNR %+.2f dB  SSIM %+.4f  LPIPS %+.4f' % (s, mb, p - c, ss - cs, lp - cl))
for g, name in (('mip', 'Mip-360 / T&T'), ('db', 'Deep Blending'), ('obj', 'objects')):
    r = res[g]
    if r: print('  %-14s n %2d  PSNR mean %+.2f dB (better on %d)  SSIM %+.4f  LPIPS %+.4f' % (name, len(r), st.mean(x[0] for x in r), sum(x[0] > 0 for x in r), st.mean(x[1] for x in r), st.mean(x[2] for x in r)))
# time: cap stage A (train + fine-tune + encode) from the stage time stamps
stamp = re.compile(r'\((\w{3} \w{3} +\d+ +[\d:]+ (?:AM|PM) \w+ \d{4})\)')
def ts(s): return datetime.datetime.strptime(re.sub(r' +', ' ', re.sub(r' [A-Z]{3,4} (\d{4})$', r' \1', s)), '%a %b %d %I:%M:%S %p %Y')
capA = {}; cur = {}
for l in sec.get('cap job stages (time stamps)', '').splitlines():
    m = re.match(r'=== (A|B): (\w+) (?:budget (\d+)|control)', l); t = stamp.search(l)
    if not (m and t): continue
    if m[1] == 'A' and 'train with' in l: cur[m[2]] = ts(t[1])
    if m[1] == 'B' and m[2] in cur: capA[m[2]] = (ts(t[1]) - cur.pop(m[2])).total_seconds() / 60
def secs(t): h, m, s = t.split(':'); return int(h) * 60 + int(m) + int(s) / 60
dur = collections.defaultdict(dict)
for l in sec.get('sacct', '').splitlines():
    p = l.split()
    if len(p) >= 4 and p[2] == 'COMPLETED':
        m = re.match(r'(\w+)-(final|cap1)-b30', p[1])
        if m and secs(p[3]) > 5: dur[m[2]][m[1]] = secs(p[3])
print('\ntime of one training in minutes, same scenes (cap: training + fine-tuning + encoding from the stage time stamps; ours: the whole job)')
for name, grp in (('Mip-360 / T&T', MIP), ('Deep Blending', DB), ('objects', OBJ)):
    g = [s for s in grp if s in capA and s in dur['final']]
    if g: print('  %-14s n %d  cap %.1f  final design %.1f  (ratio %.2f)  scenes: %s' % (name, len(g), st.mean(capA[s] for s in g), st.mean(dur['final'][s] for s in g), st.mean(dur['final'][s] for s in g) / st.mean(capA[s] for s in g), ' '.join(g)))
print('  cap stage A minutes:', {s: round(v, 1) for s, v in capA.items()})
