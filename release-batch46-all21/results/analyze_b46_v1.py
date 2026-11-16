"""analyze_b46_v1.py (2026-11-16): summary of batch46 (run from the repository root).  Input: batch46-collect-2026-11-16.txt and earlier result files."""
import re, io, contextlib, collections, statistics as st
R = 'release-batch46-all21/results/batch46-collect-2026-11-16.txt'; T = open(R).read()
MIP = ['room','kitchen','counter','bonsai','garden','bicycle','stump','flowers','treehill','truck','train']; DB = ['drjohnson','playroom']; OBJ = ['lego','chair','drums','ficus','hotdog','materials','mic','ship']
GROUPS = (('Mip-NeRF 360 + T&T (11)', MIP), ('Deep Blending (2)', DB), ('captured (13)', MIP + DB), ('NeRF-Synthetic (8)', OBJ), ('all (21)', MIP + DB + OBJ))
m = lambda v: st.mean(v)
# ---- time
tm = {}
for s, mb, e, d in re.findall(r'TIME (\w+) ours \| V \d+ F \d+ \| ([\d.]+) MB \| ours encode ([\d.]+) s \| decode ([\d.]+) s', T): tm[s] = dict(mb=float(mb), ent=float(e), dec=float(d))
for s, ze, zd, de, dd in re.findall(r'TIME (\w+) generic .*?zstd encode ([\d.]+) s decode ([\d.]+) s \| draco encode ([\d.]+) s decode ([\d.]+) s', T): tm[s].update(ze=float(ze), zd=float(zd), de=float(de), dd=float(dd))
for s, e, d in re.findall(r'TFANTIME (\w+) \| whole call [\d.]+ s \| encode ([\d.]+) s \| decode ([\d.]+) s', T): tm[s].update(te=float(e), td=float(d))
vq = {}
for f in ('release-batch44-tfan-time/results/vq-time-2026-11-16.txt', R):
    for s, a, b, c in re.findall(r'VQTIME (\w+)\s+k-means\s+(\d+) s \| assign\s+(\d+) s \| refine\s+(\d+) s', open(f).read()): vq[s] = (int(a), int(b), int(c))
print('== coding time (s), same node for all 21 scenes; vector quantization time from the delivery logs (GPU)')
for n, S in GROUPS:
    k, a, r = (m([vq[s][i] for s in S]) for i in range(3)); ent = m([tm[s]['ent'] for s in S])
    print('%-24s BM file %.2f MB | k-means %.1f assign %.1f refine %.1f entropy %.2f -> encode %.1f (no refinement %.1f) decode %.2f | zstd %.1f / %.2f | draco %.1f / %.2f | tfan %.2f / %.2f' % (
        n, m([tm[s]['mb'] for s in S]), k, a, r, ent, k + a + r + ent, k + a + ent, m([tm[s]['dec'] for s in S]), m([tm[s]['ze'] for s in S]), m([tm[s]['zd'] for s in S]), m([tm[s]['de'] for s in S]), m([tm[s]['dd'] for s in S]), m([tm[s]['te'] for s in S]), m([tm[s]['td'] for s in S])))
# ---- refinement
nr = {}; vp = {}
for f in ('release-batch45-vq-noref/results/vq-noref-2026-11-16.txt', R):
    for s, b, mb1, p1, mb0, p0 in re.findall(r'VQNOREF (\w+)\s+\| before coding test ([\d.]+) [\d.]+ [\d.]+ \| with refinement ([\d.]+) MB test ([\d.]+) [\d.]+ [\d.]+ \| without refinement ([\d.]+) MB test ([\d.]+)', open(f).read()): nr[s] = tuple(map(float, (b, mb1, p1, mb0, p0)))
for s, b, k, r in re.findall(r'VQPSNR (\w+)\s+before ([\d.]+) \| k-means only ([\d.]+) \([-\d.]+\) \| refined ([\d.]+)', open('release-batch44-tfan-time/results/vq-psnr-2026-11-16.txt').read()): vp[s] = (float(b), float(k), float(r))
for s, b, k, r in re.findall(r'VQTIME (\w+)\s+k-means.*?validation PSNR before ([\d.]+), k-means only ([\d.]+), refined ([\d.]+)', T): vp[s] = (float(b), float(k), float(r))
print('== refinement of the codewords (test views): before coding | with refinement | without ; validation loss k-means only -> refined')
for n, S in GROUPS:
    print('%-24s before %.3f | with %.2f MB %.3f (loss %.3f) | without %.2f MB %.3f (loss %.3f) | refinement gain %.3f, better on %d of %d | validation loss %.2f -> %.2f' % (n, m([nr[s][0] for s in S]), m([nr[s][1] for s in S]), m([nr[s][2] for s in S]),
        m([nr[s][0] - nr[s][2] for s in S]), m([nr[s][3] for s in S]), m([nr[s][4] for s in S]), m([nr[s][0] - nr[s][4] for s in S]), m([nr[s][2] - nr[s][4] for s in S]), sum(nr[s][2] > nr[s][4] for s in S), len(S), m([vp[s][0] - vp[s][1] for s in S]), m([vp[s][0] - vp[s][2] for s in S])))
# ---- codec design choices
ch = collections.defaultdict(dict); old = open('release-batch40-dc-predict/results/codec-choices-2026-11-15.md').read(); od = open('release-batch40-dc-predict/results/dc-predict-2026-11-15.md').read()
for l in old.split('## SH indices')[0].splitlines():
    c = [x.strip() for x in l.split('|')]
    if len(c) > 8 and c[1] in MIP: ch[c[1]].update(ship=float(c[2]), prev=float(c[3]), first=float(c[4]), mean=float(c[5]), para=float(c[6]), xfile=float(c[8].split()[0]))
for l in old.split('## SH indices')[1].splitlines():
    c = [x.strip() for x in l.split('|')]
    if len(c) > 7 and c[1] in MIP: ch[c[1]].update(ms=float(c[2]), mn=float(c[3]), rs=float(c[4]), rn=float(c[5]), ifile=float(c[7].split()[0]))
for l in od.splitlines():
    c = [x.strip() for x in l.split('|')]
    if len(c) > 7 and c[1] in MIP: ch[c[1]].update(file=float(c[2]), dz=float(c[3].split(',')[1]), df=float(c[4]), dm=float(c[5]), dfile=float(c[7].split()[0]))
for s, v, f, dz, df, dm, dfile in re.findall(r'DCPRED (\w+) \| V (\d+) \| file ([\d.]+) MB \| DC zstd [\d.]+ MB \(([\d.]+) b/v\) \| first ([\d.]+) b/v \| mean ([\d.]+) b/v \| best mean: DC [-\d.]+ %, file ([-\d.]+) %', T): ch[s].update(V=int(v), file=float(f), dz=float(dz), df=float(df), dm=float(dm), dfile=float(dfile))
for s, sh, pv, fi, me, pa, xf in re.findall(r'XYZPRED (\w+) \| q\d+ \| shipped [\d.]+ MB \(([\d.]+) b/v\) \| prev ([\d.]+) b/v \| first ([\d.]+) b/v \| mean ([\d.]+) b/v .*?\| para ([\d.]+) b/v .*?file ([-\d.]+) % vs first', T): ch[s].update(ship=float(sh), prev=float(pv), first=float(fi), mean=float(me), para=float(pa), xfile=float(xf))
for s, ms, mn, rs, rn, isv, ifl in re.findall(r'IDXCTX (\w+) \| main K \d+: shipped ([\d.]+) b/v, no context ([\d.]+) b/v, fixed 12 b \| residual R \d+: shipped ([\d.]+) b/v, no context ([\d.]+) b/v, fixed 8 b \| context saves\s*([-\d.]+) % of the index streams, ([-\d.]+) % of the file', T): ch[s].update(ms=float(ms), mn=float(mn), rs=float(rs), rn=float(rn), ifile=float(ifl))
for s in MIP: ch[s]['V'] = None
print('== codec design choices (bits per vertex; file change in %, per scene then averaged).  ifile > 0: the neighbour candidates make the file smaller')
for n, S in GROUPS:
    g = lambda k: m([ch[s][k] for s in S]); bpv = lambda s: ch[s]['file'] * 8e6 / ch[s]['V'] if ch[s]['V'] else None
    # file change of a position predictor relative to 'first' (test coder), from the b/v difference and the file size
    print('%-24s positions: shipped %.2f | first %.2f prev %.2f para %.2f mean %.2f | mean vs first: file %.2f %% ; order mean<first<prev<para on %d of %d' % (n, g('ship'), g('first'), g('prev'), g('para'), g('mean'), g('xfile'), sum(ch[s]['mean'] < ch[s]['first'] < ch[s]['prev'] < ch[s]['para'] for s in S), len(S)))
    print('%-24s   first<prev on %d, first<para on %d of %d' % ('', sum(ch[s]['first'] < ch[s]['prev'] for s in S), sum(ch[s]['first'] < ch[s]['para'] for s in S), len(S)))
    print('%-24s DC: zstd %.2f | first %.2f | mean %.2f | file %.2f %% (%.2f to %.2f)' % ('', g('dz'), g('df'), g('dm'), g('dfile'), max(ch[s]['dfile'] for s in S), min(ch[s]['dfile'] for s in S)))
    print('%-24s SH indices: with candidates %.2f (main %.2f + residual %.2f) | no neighbour information %.2f (%.2f + %.2f) | fixed 20 | candidates change the file by %+.2f %% (positive = smaller); help on %d of %d ; main index alone helps on %d, residual on %d' % ('',
          g('ms') + g('rs'), g('ms'), g('rs'), g('mn') + g('rn'), g('mn'), g('rn'), g('ifile'), sum(ch[s]['ifile'] > 0 for s in S), len(S), sum(ch[s]['ms'] < ch[s]['mn'] for s in S), sum(ch[s]['rs'] < ch[s]['rn'] for s in S)))
# file-size changes of the table rows for the new scenes (V known): prev / para / fixed relative to the file
for n, S in (('Deep Blending (2)', DB), ('NeRF-Synthetic (8)', OBJ)):
    f = lambda s, d: d * ch[s]['V'] / 8e6 / ch[s]['file'] * 100
    print('%-24s file change: prev %+.2f %% | para %+.2f %% | mean %+.2f %% | no neighbour info %+.2f %% | fixed length %+.2f %%' % (n, m([f(s, ch[s]['prev'] - ch[s]['first']) for s in S]), m([f(s, ch[s]['para'] - ch[s]['first']) for s in S]), m([f(s, ch[s]['mean'] - ch[s]['first']) for s in S]),
          m([f(s, ch[s]['mn'] + ch[s]['rn'] - ch[s]['ms'] - ch[s]['rs']) for s in S]), m([f(s, 20 - ch[s]['ms'] - ch[s]['rs']) for s in S])))
# ---- alpha
print('== growth aim, target 10 MB')
for s, a, mb, p in re.findall(r'(\w+) abl24-holda(\d+) \| target 10 MB .*?delivered ([\d.]+) MB .*?test ([\d.]+)', T): print('  %-8s alpha %.2f | %.3f MB | %.3f dB' % (s, int(a) / 100, float(mb), float(p)))
