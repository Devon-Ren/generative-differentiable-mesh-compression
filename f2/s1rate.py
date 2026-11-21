"""F2 stage-1 rate: hierarchical coding of inverse depth rho and colour on the icosphere (L8, ~655k vertices)."""
import numpy as np, sys, json
sys.path.insert(0, '.')
from f2lib import *; from sph import *
import s1
from s1 import X_inside, inside_views, eval_views
S = GTScene('../gt/Sponza.gltf'); c = np.array([0, 1.6, 0.])
s1.rng = np.random.default_rng(123)
TV = {r: inside_views(S, c, r, 12) for r in [0, 0.5]}
z = np.load('sponza_L8.npz'); rho, col = z['rho'], z['col']
out = []
for L in [6, 7, 8]:
    lev = icosphere_levels(L); n = len(lev[-1][0]); F = lev[-1][1]; D = lev[-1][0]
    r_, c_ = rho[:n], col[:n]   # nested: the first n vertices of L8 are the level-L vertices (direction-identical)
    # YCoCg
    Y = np.stack([0.25 * c_[:, 0] + 0.5 * c_[:, 1] + 0.25 * c_[:, 2], 0.5 * c_[:, 0] - 0.5 * c_[:, 2], -0.25 * c_[:, 0] + 0.5 * c_[:, 1] - 0.25 * c_[:, 2]], 1)
    for sr, sc in [(2e-4, 2 / 255), (1e-3, 4 / 255), (4e-3, 8 / 255)]:
        rr, br = hier_code(lev, r_, sr)
        yy, bc = hier_code(lev, Y, sc)
        cc = np.stack([yy[:, 0] + yy[:, 1] - yy[:, 2], yy[:, 0] + yy[:, 2], yy[:, 0] - yy[:, 1] - yy[:, 2]], 1)
        M = SphereMap(X_inside(c, D, np.maximum(rr[:, 0], 0)), F, cc.clip(0, 1))
        e0 = eval_views(S, M, TV[0]); e5 = eval_views(S, M, TV[0.5])
        row = dict(L=L, V=n, step_rho=sr, step_col=round(sc * 255), MB_geo=br / 8e6, MB_col=bc / 8e6, MB=(br + bc) / 8e6,
                   psnr_r0=e0['psnr'], bad_r0=e0['bad'], psnr_r05=e5['psnr'], bad_r05=e5['bad'])
        out.append(row)
        print(f"L{L} V={n:6d} q_rho={sr:.0e} q_col={row['step_col']}/255: geo {row['MB_geo']:.3f} MB + col {row['MB_col']:.3f} MB = {row['MB']:.3f} MB | r0 psnr {e0['psnr']:.2f} bad {e0['bad']*100:.1f}% | r0.5 psnr {e5['psnr']:.2f} bad {e5['bad']*100:.1f}%", flush=True)
json.dump(out, open('s1rate_results.json', 'w'), indent=1)
