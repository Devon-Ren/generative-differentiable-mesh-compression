"""F2 stage 3b: voxel-centric free-space carving (no ray gaps). voxel is free iff it lies in front of the
(min-filtered) panoramic depth seen from some region position."""
import numpy as np, sys, time, json
sys.path.insert(0, '.')
from f2lib import *
import s1
from s1 import inside_views, eval_views
from s3 import boundary_mesh
from scipy.ndimage import minimum_filter
import trimesh

def pano_depth(S, pos, W=2048, H=1024):
    th = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi; ph = (np.arange(H) + 0.5) / H * np.pi - np.pi / 2
    T, P = np.meshgrid(th, ph)
    d = np.stack([np.cos(P) * np.sin(T), np.sin(P), np.cos(P) * np.cos(T)], -1).reshape(-1, 3)
    loc, ir, _ = S.rt.intersects_location(np.broadcast_to(pos, d.shape), d, multiple_hits=False)
    t = np.full(len(d), 1e9); t[ir] = np.linalg.norm(loc - pos, axis=1)
    return minimum_filter(t.reshape(H, W), size=3, mode='wrap')

def carve_vox(S, c, region_r, vox, n_pos=160, seed=7):
    lo = S.mesh.bounds[0] - 2 * vox; hi = S.mesh.bounds[1] + 2 * vox; hi[1] += 1.0
    shape = np.ceil((hi - lo) / vox).astype(int)
    idx = np.indices(shape).reshape(3, -1).T
    P = (lo + (idx + 0.5) * vox).astype(np.float32)
    free = np.zeros(len(P), bool)
    rng = np.random.default_rng(seed); s1.rng = np.random.default_rng(seed)
    for i in range(n_pos):
        rr = region_r * np.sqrt(rng.uniform())
        pos = c if i == 0 else inside_views(S, c, rr, 1)[0][0]
        Dm = pano_depth(S, pos); H, W = Dm.shape
        for a in range(0, len(P), 4_000_000):
            q = P[a:a + 4_000_000] - pos.astype(np.float32)
            dist = np.linalg.norm(q, axis=1)
            th = np.arctan2(q[:, 0], q[:, 2]); ph = np.arcsin(np.clip(q[:, 1] / np.maximum(dist, 1e-6), -1, 1))
            u = ((th + np.pi) / (2 * np.pi) * W).astype(int) % W; v = np.clip(((ph + np.pi / 2) / np.pi * H).astype(int), 0, H - 1)
            free[a:a + 4_000_000] |= dist < Dm[v, u] - 0.6 * vox
    return free.reshape(shape), lo

if __name__ == '__main__':
    region, vox = float(sys.argv[1]), float(sys.argv[2])
    S = GTScene('../gt/Sponza.gltf'); c = np.array([0, 1.6, 0.])
    s1.rng = np.random.default_rng(123)
    radii = [0, 1, 2, 4]
    TV = {r: inside_views(S, c, r, 12) for r in radii}
    ytop = S.mesh.bounds[1][1]
    t0 = time.time()
    free, lo = carve_vox(S, c, region, vox)
    m = boundary_mesh(free, lo, vox)
    comps = m.split(only_watertight=False); big = sorted(comps, key=lambda x: -len(x.faces))
    chis = [x.euler_number for x in big]
    print(f'== region {region} m voxel {vox} m: dOmega V {len(m.vertices)} F {len(m.faces)} components {len(comps)} '
          f'(largest 5 face counts {[len(x.faces) for x in big[:5]]}) main watertight {big[0].is_watertight} genus {(2 - chis[0]) / 2:.0f} ({time.time() - t0:.0f}s)', flush=True)
    n = m.vertex_normals
    tt, rgb, _ = S.cast(m.vertices + n * 0.02, -n); tt2, rgb2, _ = S.cast(m.vertices - n * 0.02, n)
    C = np.where((tt <= tt2)[:, None], rgb, rgb2)
    sky = m.vertices[:, 1] > ytop; C[sky] = SKY
    X = m.vertices.copy(); X[sky] = c + (X[sky] - c) / np.linalg.norm(X[sky] - c, axis=1, keepdims=True) * 1e4
    M = SphereMap(X, m.faces, C); out = {}
    for r in radii:
        e = eval_views(S, M, TV[r]); out[f'r{r}'] = e
        print(f'  test r={r}: psnr {e["psnr"]:.2f} bad-depth {e["bad"] * 100:.1f}% holes {e["hole"] * 100:.2f}%', flush=True)
    out['topo'] = dict(V=len(m.vertices), F=len(m.faces), comps=len(comps), genus_main=(2 - chis[0]) / 2, comp_faces=[len(x.faces) for x in big[:20]])
    json.dump(out, open(f's3b_region{region}_vox{vox}.json', 'w'), indent=1)
    m.export(f'sponza_dOmega_b_r{region}_v{vox}.ply')
