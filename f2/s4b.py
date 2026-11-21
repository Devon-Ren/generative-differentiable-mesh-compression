"""F2 stage 4: free-space carving in VIEW-ADAPTIVE coordinates around the capture centre c:
(azimuth, elevation, log distance) + an 'infinity' slab. Angular resolution like the radial sphere,
relative depth resolution uniform (log bins), sky = infinity, topology handled by carving."""
import numpy as np, sys, time, json
sys.path.insert(0, '.')
from f2lib import *
import s1
from s1 import inside_views, eval_views
from s3b import pano_depth
from skimage import measure
from scipy.ndimage import gaussian_filter
import trimesh

def grid_world(c, NA, NE, NS, dmin, dmax):
    az = (np.arange(NA) + 0.5) / NA * 2 * np.pi - np.pi
    el = (np.arange(NE) + 0.5) / NE * np.pi - np.pi / 2
    s = np.log(dmin) + (np.arange(NS) + 0.5) / NS * (np.log(dmax) - np.log(dmin))
    return az, el, s

def to_world(c, az, el, d):
    return c + np.stack([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)], -1) * d[..., None]

def carve_adaptive(S, c, region_r, NA, NE, NS, dmin=0.25, dmax=80.0, n_pos=160, seed=7, margin_bins=0.6):
    az, el, s = grid_world(c, NA, NE, NS, dmin, dmax)
    rel = np.exp((np.log(dmax) - np.log(dmin)) / NS * margin_bins) - 1     # relative margin = 0.6 bin
    A, E = np.meshgrid(az, el, indexing='ij')
    dirs = np.stack([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)], -1).reshape(-1, 3).astype(np.float32)
    ds = (np.log(dmax) - np.log(dmin)) / NS
    free = np.full((NA * NE, NS + 1), -2.0, np.float32)   # signed field; last slab = infinity
    rng = np.random.default_rng(seed); s1.rng = np.random.default_rng(seed)
    dist = np.exp(s).astype(np.float32)
    for i in range(n_pos):
        rr = region_r * np.sqrt(rng.uniform())
        pos = c if i == 0 else inside_views(S, c, rr, 1)[0][0]
        Dm = pano_depth(S, pos); H, W = Dm.shape
        off = (pos - c).astype(np.float32)
        for k in range(NS + 1):
            if k < NS:
                P = dirs * dist[k] - off                         # cell centre relative to pos
                dd = np.linalg.norm(P, axis=1)
                # conservative for oblique observers: the cell's tangential extent projected on the observer ray
                cosb = np.abs((P * dirs).sum(1)) / np.maximum(dd, 1e-6)
                ext_t = dist[k] * (2 * np.pi / NA)
                dd = dd + 0.5 * ext_t * np.sqrt(np.clip(1 - cosb ** 2, 0, 1))
            else:
                P = dirs; dd = np.full(len(P), np.inf, np.float32)   # point at infinity in direction dirs
            th = np.arctan2(P[:, 0], P[:, 2]); ph = np.arcsin(np.clip(P[:, 1] / np.maximum(np.linalg.norm(P, axis=1), 1e-6), -1, 1))
            u = ((th + np.pi) / (2 * np.pi) * W).astype(int) % W; v = np.clip(((ph + np.pi / 2) / np.pi * H).astype(int), 0, H - 1)
            dep = Dm[v, u]
            if k < NS: free[:, k] = np.maximum(free[:, k], np.clip((np.log(dep) - np.log(np.maximum(dd, 1e-6))) / ds, -2, 2))
            else: free[:, k] = np.maximum(free[:, k], np.where(dep > 1e8, 2.0, -2.0))
    return free.reshape(NA, NE, NS + 1), (az, el, s)

SMOOTH = 0.0
MIN_SOLID = 8

def boundary_adaptive(free, c, axes, dmin, dmax):
    az, el, s = axes; NA, NE, NS1 = free.shape; NS = NS1 - 1
    f = gaussian_filter(free.astype(np.float32), SMOOTH, mode=('wrap', 'nearest', 'nearest')) if SMOOTH > 0 else free.astype(np.float32)
    # Omega = the free component that contains the capture centre (definition); tiny solid islands = noise
    from scipy.ndimage import label
    lab, n = label(f > 0)
    near = lab[:, :, 0]; ids, cnt = np.unique(near[near > 0], return_counts=True)
    if len(ids): f = np.where((lab == ids[np.argmax(cnt)]) | (f <= 0), f, -0.5)
    sl, ns = label(f <= 0); sz = np.bincount(sl.ravel())
    small = sz < MIN_SOLID; small[0] = False
    f = np.where(small[sl], 0.5, f)
    f = np.pad(f, ((0, 0), (1, 1), (0, 0)), mode='edge')                       # poles: no artificial wall
    f = np.concatenate([np.full_like(f[:, :, :1], 2.0), f, np.full_like(f[:, :, :1], -2.0)], 2)   # near c: free; beyond infinity: solid
    f = np.concatenate([f, f[:1]], 0)                                            # periodic azimuth (one copy column)
    v, F, _, _ = measure.marching_cubes(f, 0.0)
    ia = np.mod(v[:, 0], NA); ie = v[:, 1] - 1; isl = v[:, 2] - 1
    A = (ia + 0.5) / NA * 2 * np.pi - np.pi; E = np.clip((ie + 0.5) / NE * np.pi - np.pi / 2, -np.pi / 2, np.pi / 2)
    ls = np.log(dmin) + (isl + 0.5) / NS * (np.log(dmax) - np.log(dmin))
    d = np.where(isl >= NS - 0.5, 1e4, np.exp(np.clip(ls, np.log(dmin) - 1, np.log(dmax))))
    dirs = np.stack([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)], -1)
    key = np.stack([np.round(ia * 1e3), np.round(ie * 1e3), np.round(isl * 1e3)], 1).astype(np.int64)   # merge seam only; poles stay open (tiny)
    _, uid, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    X = c + dirs[uid] * d[uid, None]
    F2 = inv[F]
    F2 = F2[(F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 2] != F2[:, 0])]
    m = trimesh.Trimesh(X, F2, process=False); m.remove_unreferenced_vertices()
    return m, d

def genus_with_boundary(m):
    E = np.sort(m.edges, axis=1); u, cnt = np.unique(E, axis=0, return_counts=True)
    be = u[cnt == 1]
    g = trimesh.graph.nx.Graph(); g.add_edges_from(map(tuple, be))
    b = trimesh.graph.nx.number_connected_components(g) if len(be) else 0
    chi = m.euler_number
    return (2 - chi - b) / 2, b, int((cnt > 2).sum())

if __name__ == '__main__':
    region, NA, NE, NS = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    S = GTScene('../gt/Sponza.gltf'); c = np.array([0, 1.6, 0.])
    s1.rng = np.random.default_rng(123)
    radii = [0, 0.5, 1, 2, 4]
    TV = {r: inside_views(S, c, r, 12) for r in radii}
    t0 = time.time()
    free, axes = carve_adaptive(S, c, region, NA, NE, NS)
    m, d = boundary_adaptive(free, c, axes, 0.25, 80.0)
    comps = m.split(only_watertight=False); big = sorted(comps, key=lambda x: -len(x.faces))
    print(f'== adaptive region {region} grid {NA}x{NE}x{NS}+inf: dOmega V {len(m.vertices)} F {len(m.faces)} comps {len(comps)} '
          f'(largest {[len(x.faces) for x in big[:4]]}) genus(main, boundary-corrected) {genus_with_boundary(big[0])} ({time.time() - t0:.0f}s)', flush=True)
    n = m.vertex_normals
    far = np.linalg.norm(m.vertices - c, axis=1) > 1e3
    tt, rgb, _ = S.cast(m.vertices + n * 0.02, -n); tt2, rgb2, _ = S.cast(m.vertices - n * 0.02, n)
    C = np.where((tt <= tt2)[:, None], rgb, rgb2); C[far] = SKY
    M = SphereMap(m.vertices, m.faces, C); out = {}
    for r in radii:
        e = eval_views(S, M, TV[r]); out[f'r{r}'] = e
        print(f'  test r={r}: psnr {e["psnr"]:.2f} bad-depth {e["bad"] * 100:.1f}% holes {e["hole"] * 100:.2f}%', flush=True)
    out['topo'] = dict(V=len(m.vertices), F=len(m.faces), comps=len(comps), genus_main=genus_with_boundary(big[0])[0],
                       comp_faces=[len(x.faces) for x in big[:20]])
    json.dump(out, open(f's4b_region{region}_{NA}x{NE}x{NS}.json', 'w'), indent=1)
    m.export(f'sponza_s4b_r{region}_{NA}x{NE}x{NS}.ply')
