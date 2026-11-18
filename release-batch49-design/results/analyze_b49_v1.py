"""analyze_b49_v1.py (2026-11-18): summary of batch49.  Run from the repository root."""
import re, statistics as st, collections, numpy as np, importlib.util
T = open('release-batch49-design/results/batch49-collect-2026-11-18.txt').read()
MIP = ['room','kitchen','counter','bonsai','garden','bicycle','stump','flowers','treehill','truck','train']; DB = ['drjohnson','playroom']; OBJ = ['lego','chair','drums','ficus','hotdog','materials','mic','ship']
G = (('Mip-NeRF 360 + T&T (11)', MIP), ('Deep Blending (2)', DB), ('captured (13)', MIP + DB), ('NeRF-Synthetic (8)', OBJ), ('all (21)', MIP + DB + OBJ))
# ---- VQ + mesh codecs
V = {}
for m in re.finditer(r'VQMESH (\w+) \| V (\d+) F \d+ \| BM-Codec ([\d.]+) MB \| VQ \+ Draco ([\d.]+) MB .*?VQ \+ TFAN ([\d.]+) MB .*?VQ \+ zstd ([\d.]+) MB .*?decoded V (\d+)', T):
    V[m[1]] = dict(n=int(m[2]), bm=float(m[3]), draco=float(m[4]), tfan=float(m[5]), zstd=float(m[6]), dv=int(m[7]))
print('== same vector quantization, different lossless coding: mean file size (MB) and mean per-scene increase over BM-Codec')
for n, S in G:
    f = lambda k: st.mean(V[s][k] for s in S); inc = lambda k: [100 * (V[s][k] / V[s]['bm'] - 1) for s in S]
    print('%-24s BM-Codec %.2f | VQ + TFAN %.2f (%+.1f %%, %+.1f..%+.1f) | VQ + zstd %.2f (%+.1f %%) | VQ + Draco %.2f (%+.1f %%; Draco decodes %.1fx the vertices)' % (
        n, f('bm'), f('tfan'), st.mean(inc('tfan')), min(inc('tfan')), max(inc('tfan')), f('zstd'), st.mean(inc('zstd')), f('draco'), st.mean(inc('draco')), st.mean(V[s]['dv'] / V[s]['n'] for s in S)))
# ---- position bits
Q = {m[1]: (int(m[2]), int(m[3]), float(m[4])) for m in re.finditer(r'QBITS (\w+)\s+\|.*?-> (\d+) bits \| used (\d+) bits \| step\s*used / median edge ([\d.]+)', T)}
print('== position bits: rule (step <= 0.1 x median edge) vs used')
for n, S in G: print('%-24s rule = used on %d of %d | used - rule: %s | step used / median edge %.3f..%.3f' % (n, sum(Q[s][0] == Q[s][1] for s in S), len(S), ' '.join(str(Q[s][1] - Q[s][0]) for s in S), min(Q[s][2] for s in S), max(Q[s][2] for s in S)))
# ---- probe time
P = [(float(a), float(b)) for a, b in re.findall(r'PROBETIME \S+(?: \d+ MB)?\s*\| probes \d+ \| probe time\s+(\d+) s \| training\s+(\d+) s', T)]
print('== probe time: %d runs | probe %.0f s of %.0f s training (%.1f %%, lower bound) | per run %.1f..%.1f %%' % (len(P), st.mean(a for a, _ in P), st.mean(b for _, b in P), st.mean(100 * a / b for a, b in P), min(100 * a / b for a, b in P), max(100 * a / b for a, b in P)))
# ---- Phase 1 only vs ByteMesh vs MeshSplatting-cap (BM-Codec everywhere)
P1 = collections.defaultdict(list)
for m in re.finditer(r'(\w+) abl24-p1only \| target ([\d.]+) MB .*?delivered ([\d.]+) MB .*?test ([\d.]+) ([\d.]+) ([\d.]+)', T): P1[m[1]].append((float(m[3]), float(m[4]), float(m[5]), float(m[6]), float(m[2])))
spec = importlib.util.spec_from_file_location('bm', 'release-batch49-design/results/bd_metrics_v1.py'); src = open(spec.origin).read(); ns = {}
exec(compile(src.split("R = collections.defaultdict")[0], 'bm', 'exec'), ns); H, cap, env, bdq = ns['H'], ns['cap'], ns['env'], ns['bdq']
print('== Phase 1 only (no collapse rounds, growth aims at the target) | size / target, and BD against ByteMesh and against MeshSplatting-cap')
rows = collections.defaultdict(list)
for s in ['garden', 'room', 'bicycle', 'kitchen', 'truck']:
    p = sorted(P1[s]); hits = ['%.2f' % (x[0] / x[4]) for x in p]; out = []
    for k, name, sign in ((1, 'PSNR', 1), (2, 'SSIM', 1), (3, 'LPIPS', -1)):
        a = env([x[:4] for x in p], k, sign); h = env(H[s], k, sign); c = env(cap[s].values(), k, sign)
        d1 = bdq(a, h); d2 = bdq(a, c); rows[name + ' vs ByteMesh'].append(d1); rows[name + ' vs cap'].append(d2)
        out.append('%s vs ByteMesh %s, vs cap %s' % (name, ('%+.3f' if k == 1 else '%+.4f') % d1 if d1 is not None else 'n/a', ('%+.3f' if k == 1 else '%+.4f') % d2 if d2 is not None else 'n/a'))
    print('%-8s size/target %s | %s' % (s, ' '.join(hits), ' | '.join(out)))
print('mean     ' + ' | '.join('%s %s' % (k, ('%+.3f' if 'PSNR' in k else '%+.4f') % st.mean(v)) for k, v in rows.items() if None not in v))
