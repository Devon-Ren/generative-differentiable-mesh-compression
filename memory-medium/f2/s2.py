"""F2 stage 2: unfold a genus-0 sphere map by multi-view depth fitting (visibility = free-space evidence) + edge-split refinement.
Connectivity changes only by edge splits => the surface stays a (possibly self-intersecting) sphere."""
import numpy as np, sys, time, json
sys.path.insert(0, '.')
from f2lib import *; from sph import *
from s1 import build_inside, X_inside, inside_views, eval_views  # reuses stage-1 code (runs nothing at import? guarded below)
rng = np.random.default_rng(1)
SKYR = 60.0

def edges_of(F):
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    Eu, inv = np.unique(E, axis=0, return_inverse=True)
    return Eu, inv.reshape(3, -1).T

def split_edges(X, F, C, mask_edge_fn, max_new):
    """one pass of independent edge bisections (keeps the mesh closed, no T-junctions)."""
    Eu, fe = edges_of(F)
    cand = np.where(mask_edge_fn(X, Eu))[0]
    if len(cand) == 0: return X, F, C, 0
    L = np.linalg.norm(X[Eu[cand, 0]] - X[Eu[cand, 1]], axis=1)
    cand = cand[np.argsort(-L)][:max_new * 3]
    # faces of each edge
    ef = [[] for _ in range(0)]
    owner = np.full((len(Eu), 2), -1)
    for k in range(3):
        e = fe[:, k]; first = owner[e, 0] < 0
        owner[e[first], 0] = np.where(first)[0]
        sec = ~first; owner[e[sec], 1] = np.where(sec)[0]
    used = np.zeros(len(F), bool); chosen = []
    for e in cand:
        f0, f1 = owner[e]
        if f1 < 0 or used[f0] or used[f1]: continue
        used[f0] = used[f1] = True; chosen.append(e)
        if len(chosen) >= max_new: break
    chosen = np.array(chosen, int)
    a, b = Eu[chosen, 0], Eu[chosen, 1]
    m = len(X) + np.arange(len(chosen))
    X = np.concatenate([X, 0.5 * (X[a] + X[b])]); C = np.concatenate([C, 0.5 * (C[a] + C[b])])
    F = F.copy(); newF = []
    for i, e in enumerate(chosen):
        for f in owner[e]:
            tri = F[f]
            # rotate so that the split edge is (tri[0], tri[1])
            for r in range(3):
                t = np.roll(tri, -r)
                if {t[0], t[1]} == {a[i], b[i]}: break
            F[f] = [t[0], m[i], t[2]]; newF.append([m[i], t[1], t[2]])
    return X, np.concatenate([F, np.array(newF, int)]), C, len(chosen)

def train_cams(S, c, r, n):
    cams = []
    for pos, yaw, pitch in inside_views(S, c, r, n):
        cams.append((pos, yaw, pitch))
    return cams

def fit(S, c, region_r, L0=6, iters=40, n_cams=48, W=96, H=72, ang=0.012, vmax=600000, log=print):
    lev, D, F, rho, col = build_inside(S, c, L0)
    rho_c = np.minimum(1 / np.maximum(rho, 1e-12), SKYR)
    X = c + D * rho_c[:, None]; C = col.copy()
    # fixed training rays (GT cached): camera positions sampled over the whole region (radii uniform in [0, region_r])
    rays = []
    for i in range(n_cams):
        rr = region_r * np.sqrt(rng.uniform())
        for pos, yaw, pitch in inside_views(S, c, rr, 1):
            d = cam_rays(pos, yaw, pitch, W, H, fov=90, ss=1)
            tg, cg, _ = S.cast(pos, d); tg = np.minimum(tg, SKYR)
            rays.append((pos, d, tg, cg))
    for it in range(iters):
        M = SphereMap(X, F, C)
        acc = np.zeros((len(X), 3)); wsum = np.zeros(len(X)); cacc = np.zeros((len(X), 3)); nbad = 0; ntot = 0
        for pos, d, tg, cg in rays:
            loc, ir, itri = M.rt.intersects_location(np.broadcast_to(pos, d.shape), d, multiple_hits=False)
            ts = np.einsum('ij,ij->i', loc - pos, d[ir])
            err = np.abs(1 / ts - 1 / tg[ir]) * tg[ir]
            nbad += int((err > 0.05).sum()); ntot += len(ir)
            bc = trimesh.triangles.points_to_barycentric(X[F[itri]], loc)
            delta = d[ir] * (tg[ir] - ts)[:, None]
            w = bc.clip(0, 1)
            for k in range(3):
                np.add.at(acc, F[itri, k], w[:, k:k + 1] * delta)
                np.add.at(wsum, F[itri, k], w[:, k])
                np.add.at(cacc, F[itri, k], w[:, k:k + 1] * cg[ir])
        has = wsum > 1e-6
        X[has] += 0.6 * acc[has] / wsum[has, None]
        C[has] = 0.5 * C[has] + 0.5 * cacc[has] / wsum[has, None]
        # tangential relaxation (keeps shape, improves triangle quality): move towards 1-ring mean, remove normal part
        Eu, _ = edges_of(F)
        nb = np.zeros((len(X), 3)); deg = np.zeros(len(X))
        np.add.at(nb, Eu[:, 0], X[Eu[:, 1]]); np.add.at(nb, Eu[:, 1], X[Eu[:, 0]])
        np.add.at(deg, Eu[:, 0], 1); np.add.at(deg, Eu[:, 1], 1)
        lap = nb / deg[:, None] - X
        vn = trimesh.Trimesh(X, F, process=False).vertex_normals
        lap_t = lap - (lap * vn).sum(1, keepdims=True) * vn
        X += 0.3 * lap_t + 0.05 * (lap - lap_t) * (~has)[:, None]   # unobserved vertices: also mild smoothing (minimal-surface fill)
        # refine: split edges that look long from the region (angular size > ang)
        n_new = 0
        if len(X) < vmax and it < iters - 5:
            def crit(X, Eu):
                mid = 0.5 * (X[Eu[:, 0]] + X[Eu[:, 1]]); dist = np.maximum(np.linalg.norm(mid - c, axis=1) - region_r * 0.5, 0.5)
                return np.linalg.norm(X[Eu[:, 0]] - X[Eu[:, 1]], axis=1) / dist > ang
            for _ in range(2):
                X, F, C, k = split_edges(X, F, C, crit, max_new=min(60000, vmax - len(X)))
                n_new += k
        log(f'  it {it:2d} V {len(X):7d} F {len(F):7d} train bad-depth {nbad / max(ntot, 1) * 100:5.2f}%  split {n_new}')
    return X, F, C

if __name__ == '__main__':
    S = GTScene('../gt/Sponza.gltf'); c = np.array([0, 1.6, 0.])
    rng0 = np.random.default_rng(0)
    import s1; s1.rng = np.random.default_rng(123)
    radii = [0, 1, 2, 4]
    TV = {r: inside_views(S, c, r, 12) for r in radii}
    out = {}
    for region in [2.0, 4.0]:
        t0 = time.time(); print(f'== region radius {region} m', flush=True)
        X, F, C = fit(S, c, region, log=lambda s: print(s, flush=True))
        M = SphereMap(X, F, C)
        m = trimesh.Trimesh(X, F, process=False)
        print(f'  final V {len(X)} F {len(F)} euler {m.euler_number} (sphere=2)  time {time.time() - t0:.0f}s', flush=True)
        for r in radii:
            e = eval_views(S, M, TV[r]); out[f'region{region} r{r}'] = e
            print(f'  test r={r}: psnr {e["psnr"]:.2f} bad-depth {e["bad"] * 100:.1f}% holes {e["hole"] * 100:.2f}%', flush=True)
        np.savez(f'sponza_s2_region{region}.npz', X=X, F=F, C=C)
    json.dump(out, open('s2_results.json', 'w'), indent=1)
