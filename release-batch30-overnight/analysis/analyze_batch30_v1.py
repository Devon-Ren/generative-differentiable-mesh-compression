"""analyze_batch30_v1.py <collect file>: size error and PSNR of the controller variants (one target per dataset), against the current method
(same target) and the cap baseline; cap retries; job durations."""
import re, sys, math, statistics as st, collections
F = sys.argv[1]; txt = open(F).read()
sec = {m.group(1): m.group(2) for m in re.finditer(r'===== ([^\n]+)\n(.*?)(?=\n===== |\Z)', txt, re.S)}
MIP = ['room','kitchen','counter','bonsai','garden','bicycle','stump','flowers','treehill','truck','train']; DB = ['drjohnson','playroom']; OBJ = ['lego','chair','drums','ficus','hotdog','materials','mic','ship']; ALL = MIP + DB + OBJ
T0 = {s: (2.5 if s in OBJ else 10.0) for s in ALL}
# current method
old = collections.defaultdict(dict)
for f in ['release-adapt-v9b/results/adapt-v9b-2026-11-08.txt', 'release-adapt-v9b/results/adapt-v9b-blender-2026-11-08.txt', 'release-batch21-v1/results/batch21-collect-2026-11-09-2110-raw.txt']:
    for l in open(f):
        m = re.match(r'(\w+) \| target ([\d.]+) MB \|.*?delivered ([\d.]+) MB \(hit ([\d.]+) x target.*?\| test ([\d.]+) ([\d.]+) ([\d.]+)', l)
        if m: old[m[1]][float(m[2])] = (float(m[3]), float(m[5]), float(m[6]), float(m[7]))
def old_at(s, mb):
    pts = sorted(old[s].values())
    for k, ((a, pa, *_), (b, pb, *_)) in enumerate(zip(pts, pts[1:])):
        if mb <= b or k == len(pts) - 2: return pa + (pb - pa) * math.log(mb / a) / math.log(b / a)
new = collections.defaultdict(dict)
for l in sec['controller rows'].splitlines():
    m = re.match(r'(\w+) abl24-(\w+) \| target ([\d.]+) MB \|.*?trained V (\d+).*?delivered ([\d.]+) MB.*?\| test ([\d.]+) ([\d.]+) ([\d.]+)', l)
    if m and float(m[3]) == T0[m[1]]: new[m[2]][m[1]] = (float(m[5]), float(m[6]), float(m[7]), float(m[8]), int(m[4]), re.search(r'best iter (\d+)', l))
def summ(errs): a = [abs(e) for e in errs]; return 'n %2d | mean abs %4.1f%% | within 2%%: %2d  5%%: %2d  10%%: %2d | range %+5.1f%% .. %+5.1f%%' % (len(a), st.mean(a), sum(x <= 2 for x in a), sum(x <= 5 for x in a), sum(x <= 10 for x in a), min(errs), max(errs))
print('## current method at the same targets'); e = {s: 100 * (old[s][T0[s]][0] / T0[s] - 1) for s in ALL}
for name, grp in (('Mip-360 / T&T', MIP), ('Deep Blending', DB), ('objects', OBJ), ('all', ALL)): print('  %-14s' % name, summ([e[s] for s in grp]))
for v in ('final', 'ratio35', 'ratio'):
    if not new[v]: continue
    print('\n## variant %s (%d of 21 scenes)' % (v, len(new[v])))
    print('  %-10s %8s %7s %7s | current: %7s %7s | PSNR new - current at the new file size | SSIM / LPIPS new vs current run' % ('scene', 'MB', 'err', 'PSNR', 'err', 'PSNR'))
    dps = {}
    for s in ALL:
        if s not in new[v]: continue
        mb, p, ss, lp, V, bi = new[v][s]; o = old[s][T0[s]]; dp = p - old_at(s, mb); dps[s] = dp
        print('  %-10s %8.3f %+6.1f%% %7.3f | %+6.1f%% %7.3f | %+.2f dB | %.4f / %.4f vs %.4f / %.4f%s' % (s, mb, 100 * (mb / T0[s] - 1), p, 100 * (o[0] / T0[s] - 1), o[1], dp, ss, lp, o[2], o[3], ('  (checkpoint %s)' % bi.group(1)) if bi else ''))
    for name, grp in (('Mip-360 / T&T', MIP), ('Deep Blending', DB), ('objects', OBJ), ('all', ALL)):
        g = [s for s in grp if s in new[v]]
        if g: print('  %-14s' % name, summ([100 * (new[v][s][0] / T0[s] - 1) for s in g]), '| PSNR at the same size: mean %+.2f dB, better on %d of %d' % (st.mean(dps[s] for s in g), sum(dps[s] > 0 for s in g), len(g)))
# cap baseline
PRI = dict(room=1078000, kitchen=1086000, counter=1083000, bonsai=1073000, garden=1105000, bicycle=1080000, stump=1069000, flowers=1071000, treehill=1079000, truck=1078000, train=1085000, drjohnson=1081000, playroom=1081000, lego=183000, chair=178000, drums=176000, ficus=171000, hotdog=177000, materials=176000, mic=174000, ship=177000)
cap = collections.defaultdict(dict)
for src in [sec['cap rows']] + [open(f).read() for f in ['release-batch28-cap-stability/results/batch28-collect-2026-11-12-rows.txt']]:
    for l in src.splitlines():
        m = re.match(r'(\w+) budget (\d+) TRAIN-AT-BUDGET(?:-BLENDER)? \| vertices (\d+) .*?\| delivered ([\d.]+) MB test ([\d.]+)', l)
        if m: cap[m[1]][int(m[2])] = (float(m[4]), float(m[5]), int(m[3]))
cap['garden'].setdefault(1105000, (10.298, 23.49, 884328)); cap['truck'].setdefault(1078000, (9.95, 22.16, 885368)); cap['playroom'].setdefault(1081000, (10.96, 27.083, 847103))
retry = {}
for l in sec.get('retry launchers', '').splitlines():
    m = re.match(r'PASS2 (\w+) cap1 (\d+) delivered ([\d.]+) MB .*?-> cap2 (\d+)', l)
    if m: retry[m[1]] = int(m[4])
retry.setdefault('kitchen', 900000); retry.setdefault('lego', 154000)
print('\n## cap baseline (cap from the leave-one-out prior; one retry on a miss > 10%)')
print('  %-10s %9s %7s %7s | retry cap %9s %7s | trainings to +-10%% | our final design: err' % ('scene', 'cap', 'err', 'PSNR', '', 'err'))
first, after, ntr = [], [], []
for s in ALL:
    if PRI[s] not in cap[s]: print('  %-10s %9d   (no result)' % (s, PRI[s])); continue
    mb, p, V = cap[s][PRI[s]]; e1 = 100 * (mb / T0[s] - 1); first.append(e1); line = '  %-10s %9d %+6.1f%% %7.3f |' % (s, PRI[s], e1, p); e2 = None
    if abs(e1) > 10 and s in retry and retry[s] in cap[s]: e2 = 100 * (cap[s][retry[s]][0] / T0[s] - 1); line += ' %19d %+6.1f%% |' % (retry[s], e2)
    elif abs(e1) > 10: line += ' %19s %7s |' % (retry.get(s, '-'), 'pending')
    else: line += ' %19s %7s |' % ('', '')
    fin = abs(e1) <= 10 and 1 or (e2 is not None and abs(e2) <= 10 and 2 or None); after.append(e1 if abs(e1) <= 10 else (e2 if e2 is not None else e1)); ntr.append(fin)
    line += ' %s' % (fin if fin else ('>2' if e2 is not None else '?')); f = new['final'].get(s); line += ' | %s' % ('%+.1f%%' % (100 * (f[0] / T0[s] - 1)) if f else '-'); print(line)
if 130000 in cap['lego']: print('  lego third training (cap 130000): %+.1f%%' % (100 * (cap['lego'][130000][0] / 2.5 - 1)))
print('  first training:', summ(first)); print('  after the retry:', summ(after), '| scenes within 10%% after 1 / 2 trainings: %d / %d of %d' % (sum(x == 1 for x in ntr), sum(x in (1, 2) for x in ntr), len(ntr)))
# durations
def secs(t): h, m, s = t.split(':'); return int(h) * 3600 + int(m) * 60 + int(s)
dur = collections.defaultdict(dict)
for l in sec.get('sacct', '').splitlines():
    p = l.split()
    if len(p) >= 4 and p[2] == 'COMPLETED':
        m = re.match(r'(\w+)-(final|ratio35|ratio|cap1|cap2|cap3)-b\d+', p[1])
        if m and secs(p[3]) > 300: dur[m[2]][m[1]] = secs(p[3]) / 60
print('\n## job duration in minutes (whole job: training, fine-tuning, encoding; cap jobs also run a control stage)')
for v in ('final', 'ratio35', 'cap1'):
    for name, grp in (('Mip-360 / T&T', MIP), ('Deep Blending', DB), ('objects', OBJ)):
        g = [dur[v][s] for s in grp if s in dur[v]]
        if g: print('  %-8s %-14s n %2d  mean %5.1f  range %5.1f .. %5.1f' % (v, name, len(g), st.mean(g), min(g), max(g)))
