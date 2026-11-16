"""bd_collapse_b46_v1.py (2026-11-16): BD-rate of the codecs on the meshes from collapse after training, now also Deep Blending and NeRF-Synthetic.
Same conventions as bd_codec_v3.py (batch43); the batch46 rows are added to the table of the baseline codecs.  Run from the repository root."""
import io, contextlib, statistics as st, re
src = open('release-batch43-overnight/results/bd_codec_v2.py').read(); cut = src.index("print('== (1) BM-Codec relative to each generic codec")
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src[:cut], 'v2', 'exec'))
for l in open('release-batch46-all21/results/batch46-collect-2026-11-16.txt'):
    m = GR.match(l)
    if m: G[(m[1], m[2])][int(m[5])] = dict(zstd=float(m[6]), draco=float(m[7]), tfan=float(m[8]), psnr=float(m[9]), unq=float(m[10]), V=int(m[3]), F=int(m[4]))
rate = lambda a, b: (bd(a, b) or (None, None))[1]
for name, scenes in (('Mip-NeRF 360 / T&T', MIP), ('Deep Blending', DB), ('NeRF-Synthetic', OBJ), ('all', MIP + DB + OBJ)):
    R = collections.defaultdict(list); U = collections.defaultdict(list); nl = []
    for s in scenes:
        L = levels('simp', s); nl.append(len(L))
        if len(L) < 2: print('   %s: %d level(s) with both BM-Codec and baseline rows, skipped' % (s, len(L))); continue
        ours = envelope([p for _, p, _ in L]); rawc = envelope([((g[8]['V'] * 208 + g[8]['F'] * 12) / 1e6, g[8]['unq']) for _, _, g in L])
        cur = {k: envelope([(g[b][k], g[b]['psnr']) for _, _, g in L for b in g]) for k in ('tfan', 'zstd', 'draco')}
        for k, c in cur.items():
            v = rate(ours, c) if len(c) >= 2 and len(ours) >= 2 else None
            if v is not None: R[k].append(v)
        for k, c in list(cur.items()) + [('bm', ours)]:
            v = rate(c, rawc) if len(c) >= 2 and len(rawc) >= 2 else None
            if v is not None: U[k].append(v)
    print('collapse after training, %s | levels per scene %s' % (name, nl))
    print('   BM-Codec relative to : ' + ' | '.join('%s %+.1f %% (n %d)' % (k, st.mean(R[k]), len(R[k])) for k in ('tfan', 'zstd', 'draco') if R[k]))
    print('   relative to no codec : ' + ' | '.join('%s %+.1f %% (n %d)' % (k, st.mean(U[k]), len(U[k])) for k in ('bm', 'tfan', 'zstd', 'draco') if U[k]))
