"""bd_codec_v3.py (2026-11-16): BD-rate of each baseline codec, and of the uncoded meshes (32-bit floats), RELATIVE TO BM-Codec on the same
meshes (anchor = BM-Codec; positive = the other needs more bytes for the same PSNR).  Per scene, then averaged.  Also the opposite
direction against the uncoded meshes (anchor = no codec).  Data and conventions: bd_codec_v2.py."""
import io, contextlib, statistics as st
src = open('release-batch43-overnight/results/bd_codec_v2.py').read(); cut = src.index("print('== (1) BM-Codec relative to each generic codec")
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src[:cut], 'v2', 'exec'))
def rate(a, b):     # BD-rate of a relative to b (%), over the common PSNR range
    r = bd(a, b); return None if r is None else r[1]
for kind, name, scenes in (('cap', 'MeshSplatting-cap, Mip-NeRF 360 / T&T', MIP), ('cap', 'MeshSplatting-cap, Deep Blending', DB), ('cap', 'MeshSplatting-cap, NeRF-Synthetic', OBJ), ('simp', 'collapse after training, Mip-NeRF 360 / T&T', MIP)):
    R = collections.defaultdict(list); U = collections.defaultdict(list)
    for s in scenes:
        L = levels(kind, s)
        if len(L) < 2: continue
        ours = envelope([p for _, p, _ in L]); rawc = envelope([((g[8]['V'] * 208 + g[8]['F'] * 12) / 1e6, g[8]['unq']) for _, _, g in L])
        cur = {k: envelope([(g[b][k], g[b]['psnr']) for _, _, g in L for b in g]) for k in ('tfan', 'zstd', 'draco')}; cur['none'] = rawc
        for k, c in cur.items():
            if len(c) >= 2 and len(ours) >= 2:
                v = rate(c, ours)
                if v is not None: R[k].append(v)
        for k, c in list(cur.items())[:3] + [('bm', ours)]:
            if len(c) >= 2 and len(rawc) >= 2:
                v = rate(c, rawc)
                if v is not None: U[k].append(v)
    print(name)
    print('   relative to BM-Codec : ' + ' | '.join('%s %+.0f %% (x%.1f, n %d)' % (k, st.mean(R[k]), 1 + st.mean(R[k]) / 100, len(R[k])) for k in ('tfan', 'zstd', 'draco', 'none') if R[k]))
    print('   relative to no codec : ' + ' | '.join('%s %+.1f %% (n %d)' % (k, st.mean(U[k]), len(U[k])) for k in ('bm', 'tfan', 'zstd', 'draco') if U[k]))
