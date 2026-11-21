"""F2 stage 3: the free-space principle constructively. Carve Omega = space seen from the walk region (voxels),
take its boundary by marching cubes, measure (a) how well dOmega reproduces test views, (b) its topology (components, genus)."""
import numpy as np, sys, time, json
sys.path.insert(0, '.')
from f2lib import *; from sph import *
import s1
from s1 import inside_views, eval_views
from skimage import measure
import trimesh

def carve(S, c, region_r, vox, n_pos=160, Ldir=5, margin=1.5):
    lo = S.mesh.bounds[0] - 2 * vox; hi = S.mesh.bounds[1] + 2 * vox
    hi[1] += 1.0  # room for the sky lid
    shape = np.ceil((hi - lo) / vox).astype(int)
    free = np.zeros(shape, bool)
    D = icosphere_levels(Ldir)[-1][0]
    rng = np.random.default_rng(7)
    for i in range(n_pos):
        rr = region_r * np.sqrt(rng.uniform())
        pos = inside_views(S, c, rr, 1)[0][0] if rr > 0 else c
        t, _, _ = S.cast(pos, D)
        # sky rays: carve until the grid boundary
        tmax = np.where(np.isfinite(t), t - margin * vox, 100.0)
        steps = np.arange(0, 100.0, 0.5 * vox)
        for a in range(0, len(D), 2000):
            d = D[a:a + 2000]; tm = tmax[a:a + 2000]
            P = pos + d[:, None, :] * steps[None, :, None]          # (n, s, 3)
            ok = steps[None, :] < tm[:, None]
            ijk = np.floor((P - lo) / vox).astype(int)
            inb = ok & np.all((ijk >= 0) & (ijk < shape), axis=2)
            q = ijk[inb]
            free[q[:, 0], q[:, 1], q[:, 2]] = True
    return free, lo

def boundary_mesh(free, lo, vox):
    f = np.pad(free, 1).astype(np.float32)
    from scipy.ndimage import gaussian_filter
    f = gaussian_filter(f, 0.6)
    v, F, _, _ = measure.marching_cubes(f, 0.5)
    v = (v - 1) * vox + lo
    m = trimesh.Trimesh(v, F, process=True)
    return m

if __name__ == '__main__':
    S = GTScene('../gt/Sponza.gltf'); c = np.array([0, 1.6, 0.])
    s1.rng = np.random.default_rng(123)
    radii = [0, 1, 2, 4]
    TV = {r: inside_views(S, c, r, 12) for r in radii}
    ytop = S.mesh.bounds[1][1]
    out = {}
    for region, vox in [(2.0, 0.08), (4.0, 0.08), (4.0, 0.05)]:
        t0 = time.time()
        free, lo = carve(S, c, region, vox)
        m = boundary_mesh(free, lo, vox)
        comps = m.split(only_watertight=False)
        big = sorted(comps, key=lambda x: -len(x.faces))
        g_main = (2 - big[0].euler_number) / 2
        print(f'== region {region} m, voxel {vox} m: free voxels {free.sum()}  dOmega V {len(m.vertices)} F {len(m.faces)}'
              f'  components {len(comps)}  main comp watertight {big[0].is_watertight} genus {g_main:.0f}  ({time.time() - t0:.0f}s)', flush=True)
        # colour: nearest GT surface point colour via casting from the vertex along its inward... simpler: sample GT colour by casting from c
        # (colour is not under test here; geometry is). vertex colour = GT colour at closest point along -normal
        n = m.vertex_normals
        o = m.vertices + n * 0.02
        tt, rgb, _ = S.cast(o, -n)  # normals of the free-space indicator point to decreasing 'free' i.e. into the solid? check both
        tt2, rgb2, _ = S.cast(m.vertices - n * 0.02, n)
        C = np.where((tt <= tt2)[:, None], rgb, rgb2)
        # sky lid: vertices above the GT top -> sky colour
        sky = m.vertices[:, 1] > ytop
        C[sky] = SKY
        X = m.vertices.copy(); X[sky] = c + (X[sky] - c) / np.linalg.norm(X[sky] - c, axis=1, keepdims=True) * 1e4  # lid -> infinity
        M = SphereMap(X, m.faces, C)
        for r in radii:
            e = eval_views(S, M, TV[r]); out[f'carve region{region} vox{vox} r{r}'] = e
            print(f'  test r={r}: psnr {e["psnr"]:.2f} bad-depth {e["bad"] * 100:.1f}% holes {e["hole"] * 100:.2f}%', flush=True)
        out[f'carve region{region} vox{vox} topo'] = dict(V=len(m.vertices), F=len(m.faces), comps=len(comps), genus_main=g_main,
                                                          comp_sizes=[len(x.faces) for x in big[:10]])
        m.export(f'sponza_dOmega_r{region}_v{vox}.ply')
    json.dump(out, open('s3_results.json', 'w'), indent=1)
