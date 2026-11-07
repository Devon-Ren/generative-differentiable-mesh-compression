"""surf_extract_v2.py (VERSION surf_extract_v2 2026-11-05): P2b step 1 -- a MANIFOLD surface from a trained MeshSplatting soup, no training.
GL-free rewrite of surf_extract_v1 (open3d needs libEGL on the HPC nodes): TSDF fusion in torch, marching cubes from scikit-image,
quadric decimation from fast_simplification, Taubin smoothing + manifold statistics in numpy/scipy.

The soup is used only as a geometry finder: its MEDIAN depth (upstream 'surf_depth' = allmap[5], the depth where accumulated alpha
crosses 0.5; 2DGS-style, unbiased at silhouettes unlike 'expected_depth') is rendered at every training view, pixels with
accumulated alpha ('rend_alpha') below --alpha_min are dropped, and the maps are fused into a dense TSDF volume over the (robust) soup bbox, a mesh is extracted by marching cubes, small components are dropped,
and the mesh is decimated (quadric) to the requested face counts and lightly smoothed (Taubin).  Outputs per target T:
<out>/mesh-T<T>.npz (vertices float32, faces int32, model space), <out>/mesh-T<T>.ply, plus mesh-raw.npz/.ply, two depth PNGs,
summary.json, and one 'SURF ...' line per mesh (V, F, edge-manifold, vertex-manifold, watertight, components).

Startup self-check (fails before the long part): the first depth map is back-projected with the intrinsics/extrinsics used for
fusion and must land on the soup (median nearest-vertex distance < 1 % of the bbox diagonal); both camera conventions
(OpenCV and y/z-flipped) are tried and the winner must beat the other 3x.

Needs: scikit-image, fast-simplification, scipy (pip install scikit-image fast-simplification).  Run (GPU node):
  python surf_extract_v2.py --lab LAB --dataset DATA --model MODEL_DIR(--iteration 30000) --out OUT --targets 50000 100000 200000 [--res 512 --taubin 5]
  python surf_extract_v2.py --selftest     (CPU: camera round trip + full pipeline on a synthetic sphere, no upstream needed)
"""
import argparse, json, math, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

VERSION = 'surf_extract_v2 2026-11-05'


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


# ----------------------------------------------------------------------------- cameras (same as v1)
def intrinsics(W, H, fovx, fovy):
    fx = W / (2.0 * math.tan(fovx / 2.0)); fy = H / (2.0 * math.tan(fovy / 2.0))
    return fx, fy, W / 2.0, H / 2.0


def backproject(depth, fx, fy, cx, cy, E):
    """depth (H,W) numpy, E = camera-from-world 4x4 (OpenCV: x right, y down, z forward) -> world points (N,3) at valid pixels."""
    v, u = np.nonzero(depth > 0)
    z = depth[v, u]
    x = (u + 0.5 - cx) / fx * z; y = (v + 0.5 - cy) / fy * z
    pc = np.stack([x, y, z, np.ones_like(z)], 1)
    pw = pc @ np.linalg.inv(E).T
    return pw[:, :3]


def nn_median(points, cloud, n=20000, seed=0):
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    p = points[rng.choice(len(points), min(n, len(points)), replace=False)]
    d, _ = cKDTree(cloud).query(p, k=1)
    return float(np.median(d))


def choose_extrinsic(cam, depth, cloud, diag):
    """Try OpenCV W2V and the y/z-flipped (OpenGL-style) variant; return (E, stats) or raise."""
    W2V = cam.world_view_transform.detach().cpu().numpy().T.astype(np.float64)      # upstream stores the transpose (row-vector convention)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    fx, fy, cx, cy = intrinsics(int(cam.image_width), int(cam.image_height), float(cam.FoVx), float(cam.FoVy))
    cands = {'opencv': W2V, 'flip_yz': flip @ W2V}
    med = {k: nn_median(backproject(depth, fx, fy, cx, cy, E), cloud) / diag for k, E in cands.items()}
    best = min(med, key=med.get); other = max(med, key=med.get)
    log('camera convention check (median NN distance / bbox diag): ' + ', '.join(f'{k} {v:.4f}' for k, v in med.items()))
    if med[best] > 0.01 or med[other] < 3 * med[best]:
        raise SystemExit(f'FAILED: back-projected depth does not land on the soup (best {best} {med[best]:.4f}, other {med[other]:.4f}); camera convention or depth key wrong')
    return cands[best], dict(convention=best, nn_median_frac=med[best], other_frac=med[other])


# ----------------------------------------------------------------------------- TSDF (torch, dense)
class TSDF:
    """Dense truncated signed distance volume. grid[ix, iy, iz] at world origin + i * vox; tsdf in [-1, 1] (units of trunc)."""

    def __init__(self, lo, hi, vox, trunc, device, max_voxels=120_000_000):
        lo = np.asarray(lo, np.float64); hi = np.asarray(hi, np.float64)
        dims = np.ceil((hi - lo) / vox).astype(int) + 1
        while int(np.prod(dims)) > max_voxels:
            vox *= 1.1; dims = np.ceil((hi - lo) / vox).astype(int) + 1
        self.vox, self.trunc, self.origin, self.dims, self.device = float(vox), float(trunc), lo, dims, device
        self.tsdf = torch.ones(tuple(int(d) for d in dims), dtype=torch.float32, device=device)
        self.weight = torch.zeros_like(self.tsdf)
        ix, iy, iz = torch.meshgrid(*[torch.arange(int(d), device=device, dtype=torch.float32) for d in dims], indexing='ij')
        self.coords = torch.stack([ix, iy, iz], -1).reshape(-1, 3) * self.vox + torch.tensor(lo, dtype=torch.float32, device=device)

    def integrate(self, depth, fx, fy, cx, cy, E, dmax, chunk=16_000_000):
        """depth (H,W) float tensor (0 = invalid), E camera-from-world 4x4 numpy (OpenCV). Standard running-average TSDF update."""
        H, W = depth.shape
        depth = depth.to(self.device)
        R = torch.tensor(E[:3, :3], dtype=torch.float32, device=self.device); t = torch.tensor(E[:3, 3], dtype=torch.float32, device=self.device)
        tsdf = self.tsdf.view(-1); weight = self.weight.view(-1)
        n_upd = 0
        for s in range(0, self.coords.shape[0], chunk):
            P = self.coords[s:s + chunk]
            pc = P @ R.T + t
            z = pc[:, 2]
            u = (pc[:, 0] / z.clamp_min(1e-9) * fx + cx).floor().long(); v = (pc[:, 1] / z.clamp_min(1e-9) * fy + cy).floor().long()
            inside = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            d = torch.zeros_like(z)
            d[inside] = depth[v[inside], u[inside]]
            ok = inside & (d > 0) & (d < dmax)
            sdf = (d - z) / self.trunc
            ok &= sdf > -1.0                                              # voxels deep behind the observed surface are not touched
            idx = torch.nonzero(ok, as_tuple=False).squeeze(1) + s
            if idx.numel() == 0: continue
            sd = sdf[ok].clamp(-1.0, 1.0)
            w_old = weight[idx]
            tsdf[idx] = (tsdf[idx] * w_old + sd) / (w_old + 1.0)
            weight[idx] = w_old + 1.0
            n_upd += int(idx.numel())
        return n_upd

    def extract(self):
        from skimage import measure
        tsdf = self.tsdf.cpu().numpy(); w = self.weight.cpu().numpy()
        obs = w > 0
        if tsdf[obs].min() >= 0 or tsdf[obs].max() <= 0:
            raise SystemExit('FAILED: TSDF has no zero crossing (all observed voxels on one side)')
        # skimage tests the mask at ONE corner of each cube (the far one, checked empirically) -> a cube must only be processed
        # when all its corners were observed, otherwise the untouched +1 voxels beyond the band behind the surface create a
        # spurious inner shell.  Erode the observed mask by a 3x3x3 box: valid at a voxel iff its whole 26-neighbourhood is observed,
        # which covers either corner convention (the band is 2 * trunc_vox voxels wide, so losing one voxel per side is harmless).
        from scipy.ndimage import binary_erosion
        cube = binary_erosion(obs, structure=np.ones((3, 3, 3), bool), border_value=0)
        if not cube.any(): raise SystemExit('FAILED: no fully observed TSDF cube')
        verts, faces, _, _ = measure.marching_cubes(tsdf, level=0.0, spacing=(self.vox,) * 3, mask=cube)
        V = verts.astype(np.float64) + self.origin[None, :]
        return V, faces.astype(np.int64)


# ----------------------------------------------------------------------------- mesh utilities (numpy/scipy)
def clean(V, F):
    """drop degenerate (repeated-index or zero-area) and duplicated faces, then unreferenced vertices; returns (V, F) with F int64."""
    F = np.asarray(F, np.int64)
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    F = F[ok]
    if len(F):
        a = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        F = F[np.linalg.norm(a, axis=1) > 1e-20]
    if len(F):
        key = np.ascontiguousarray(np.sort(F, 1)).view(np.dtype((np.void, 24))).reshape(-1)
        _, first = np.unique(key, return_index=True)
        F = F[np.sort(first)]
    used = np.unique(F)
    remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    return np.ascontiguousarray(V[used]), remap[F]


def edge_keys(F, n):
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0)
    E = np.sort(E, 1)
    return E[:, 0] * n + E[:, 1]


def edges_of(F, n=None):
    """unique undirected edges (sorted pairs) and the number of faces on each"""
    n = int(F.max()) + 1 if n is None else int(n)
    uk, cnt = np.unique(edge_keys(F, n), return_counts=True)
    return np.stack([uk // n, uk % n], 1), cnt


def components(V, F):
    """connected components over vertex adjacency -> (labels per face, faces per component)"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    E, _ = edges_of(F, len(V))
    A = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(V), len(V)))
    n, lab = connected_components(A, directed=False)
    fl = lab[F[:, 0]]
    return fl, np.bincount(fl, minlength=n)


def drop_small_components(V, F, min_faces):
    fl, sizes = components(V, F)
    keep = sizes[fl] >= min_faces
    V2, F2 = clean(V, F[keep])
    return V2, F2, int((sizes >= min_faces).sum()), int((sizes < min_faces).sum())


def vertex_link_components(V, F):
    """number of connected components of each vertex's link (edges opposite the vertex in its incident faces)"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(V)
    # link edge of vertex a in face (a,b,c) = (b,c); graph nodes = (vertex, neighbour) pairs encoded as a * n + neighbour
    a = F.reshape(-1); b = F[:, [1, 2, 0]].reshape(-1); c = F[:, [2, 0, 1]].reshape(-1)
    keys = np.concatenate([a * n + b, a * n + c])
    uniq, inv = np.unique(keys, return_inverse=True)
    inv = inv.reshape(-1); m = len(a)
    A = coo_matrix((np.ones(m), (inv[:m], inv[m:])), shape=(len(uniq), len(uniq)))
    _, lab = connected_components(A, directed=False)
    vl = np.unique((uniq // n) * len(uniq) + lab)                     # distinct (vertex, label) pairs
    return np.bincount(vl // len(uniq), minlength=n)


def mesh_stats(V, F):
    E, cnt = edges_of(F, len(V))
    nb = int((cnt == 1).sum()); nnm = int((cnt > 2).sum())
    edge_manifold = nnm == 0
    closed = nb == 0
    lc = vertex_link_components(V, F)
    vertex_manifold = edge_manifold and bool((lc[np.bincount(F.reshape(-1), minlength=len(V)) > 0] == 1).all())
    return dict(vertices=int(len(V)), faces=int(len(F)), edges=int(len(E)), boundary_edges=nb, nonmanifold_edges=nnm,
                edge_manifold=bool(edge_manifold), edge_manifold_closed=bool(edge_manifold and closed), vertex_manifold=bool(vertex_manifold),
                watertight=bool(edge_manifold and closed and vertex_manifold))


def surf_line(tag, st):
    return (f"SURF {tag} | V {st['vertices']} F {st['faces']} | edge-manifold {st['edge_manifold']} (closed {st['edge_manifold_closed']}) vertex-manifold {st['vertex_manifold']} "
            f"watertight {st['watertight']} | non-manifold edges {st['nonmanifold_edges']} boundary edges {st['boundary_edges']}"
            + (f" | components {st['components']} (dropped {st['dropped_components']})" if 'components' in st else ''))


def taubin(V, F, iters, lam=0.5, mu=-0.53):
    """Taubin smoothing with the uniform Laplacian (shrink-free pair of steps)."""
    from scipy.sparse import coo_matrix
    E, _ = edges_of(F, len(V))
    i = np.concatenate([E[:, 0], E[:, 1]]); j = np.concatenate([E[:, 1], E[:, 0]])
    A = coo_matrix((np.ones(len(i)), (i, j)), shape=(len(V), len(V))).tocsr()
    deg = np.asarray(A.sum(1)).reshape(-1); deg[deg == 0] = 1
    V = V.copy()
    for _ in range(iters):
        for s in (lam, mu):
            Lv = (A @ V) / deg[:, None] - V
            V += s * Lv
    return V


def decimate(V, F, T):
    import fast_simplification
    if T >= len(F): return V.copy(), F.copy()
    P, Fd = fast_simplification.simplify(np.ascontiguousarray(V, np.float64), np.ascontiguousarray(F, np.int32), target_count=int(T), agg=7)
    return np.asarray(P, np.float64), np.asarray(Fd, np.int64)


def remove_nonmanifold_edges(V, F):
    """drop every face touching an edge shared by > 2 faces (same effect as open3d remove_non_manifold_edges); iterate until clean"""
    for _ in range(10):
        n = len(V)
        k = edge_keys(F, n)
        uk, cnt = np.unique(k, return_counts=True)
        bad = uk[cnt > 2]
        if len(bad) == 0: break
        touch = np.isin(k, bad).reshape(3, -1).any(0)
        V, F = clean(V, F[~touch])
    return V, F


def write_ply(path, V, F):
    V = np.asarray(V, np.float32); F = np.asarray(F, np.int32)
    with open(path, 'wb') as fh:
        fh.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n'
                  f'element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n').encode())
        fh.write(V.astype('<f4').tobytes())
        rec = np.empty(len(F), dtype=[('n', 'u1'), ('i', '<i4', (3,))]); rec['n'] = 3; rec['i'] = F
        fh.write(rec.tobytes())


def save_mesh(path_base, V, F):
    V32 = np.asarray(V, np.float32); F32 = np.asarray(F, np.int32)
    np.savez(path_base.with_suffix('.npz'), vertices=V32, faces=F32)
    write_ply(path_base.with_suffix('.ply'), V32, F32)
    return V32, F32


# ----------------------------------------------------------------------------- pipeline (shared by main and selftest)
def fuse_and_extract(views, depth_fn, cloud, a, out, device):
    """views: list of dict(fx,fy,cx,cy,W,H,E,name); depth_fn(i) -> (H,W) float32 numpy; cloud: soup vertices (N,3) float64."""
    lo, hi = np.percentile(cloud, 0.5, 0), np.percentile(cloud, 99.5, 0)
    diag = float(np.linalg.norm(hi - lo))
    vox = diag / a.res; trunc = a.trunc_vox * vox
    margin = 0.03 * diag + trunc
    vol = TSDF(lo - margin, hi + margin, vox, trunc, device)
    log(f'TSDF grid {tuple(int(d) for d in vol.dims)} = {int(np.prod(vol.dims)) / 1e6:.1f} M voxels, voxel {vol.vox:.5f}, trunc {trunc:.5f}, robust bbox diag {diag:.3f}')
    t0 = time.time(); n_px = 0; dmax = None
    for i, vw in enumerate(views):
        d = depth_fn(i)
        valid = d > 0
        if dmax is None:
            if valid.mean() < 0.02: raise SystemExit(f'FAILED: first depth map has only {100 * valid.mean():.2f} % valid pixels')
            dmax = float(np.percentile(d[valid], 99.9)) * 1.5
        n_px += int(valid.sum())
        vol.integrate(torch.from_numpy(d), vw['fx'], vw['fy'], vw['cx'], vw['cy'], vw['E'], dmax)
        if (i + 1) % 25 == 0 or i + 1 == len(views): log(f'  integrated {i + 1}/{len(views)} views ({time.time() - t0:.0f} s)')
    log(f'TSDF integration done: {len(views)} views, {n_px / 1e6:.1f} M valid depth px, observed voxels {int((vol.weight > 0).sum())}, {time.time() - t0:.0f} s')
    V, F = vol.extract()
    V, F = clean(V, F)
    if len(F) == 0: raise SystemExit('FAILED: marching cubes produced no triangles')
    V, F, n_keep, n_drop = drop_small_components(V, F, a.min_component_faces)
    st_raw = mesh_stats(V, F); st_raw.update(components=n_keep, dropped_components=n_drop)
    print(surf_line('raw', st_raw), flush=True)
    save_mesh(out / 'mesh-raw', V, F)
    summary = dict(version=VERSION, views=len(views), res=a.res, voxel=vol.vox, trunc_vox=a.trunc_vox, grid=[int(x) for x in vol.dims], bbox_diag=diag, raw=st_raw, meshes={})
    for T in a.targets:
        t1 = time.time()
        Vd, Fd = decimate(V, F, int(T))
        Vd, Fd = clean(Vd, Fd)
        if a.taubin > 0: Vd = taubin(Vd, Fd, a.taubin)
        Vd, Fd = remove_nonmanifold_edges(Vd, Fd)
        st = mesh_stats(Vd, Fd); st['elapsed_s'] = round(time.time() - t1, 1)
        V32, F32 = save_mesh(out / f'mesh-T{T}', Vd, Fd)
        if F32.shape[0] == 0 or int(F32.max()) >= V32.shape[0]: raise SystemExit(f'FAILED: mesh T{T} invalid ({V32.shape[0]} v, {F32.shape[0]} f)')
        st['nn_to_soup_frac'] = nn_median(Vd, cloud) / diag
        print(surf_line(f'T{T}', st) + f" | median dist to soup {100 * st['nn_to_soup_frac']:.3f} % diag | {st['elapsed_s']} s", flush=True)
        summary['meshes'][str(T)] = st
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lab'); p.add_argument('--dataset'); p.add_argument('--model'); p.add_argument('--images', default='train')
    p.add_argument('--iteration', type=int, default=30000); p.add_argument('--out')
    p.add_argument('--targets', nargs='+', type=int, default=[50000, 100000, 200000], help='quadric-decimation face targets')
    p.add_argument('--res', type=int, default=512, help='TSDF voxels across the (robust) soup bbox diagonal')
    p.add_argument('--trunc_vox', type=float, default=4.0, help='sdf truncation in voxels')
    p.add_argument('--depth_key', default='surf_depth', help="render_pkg key of the depth map (surf_depth = median depth; expected_depth = alpha-weighted mean)")
    p.add_argument('--alpha_key', default='rend_alpha', help='render_pkg key of accumulated alpha ("" = do not mask)'); p.add_argument('--alpha_min', type=float, default=0.5)
    p.add_argument('--taubin', type=int, default=5)
    p.add_argument('--min_component_faces', type=int, default=200); p.add_argument('--max_views', type=int, default=0)
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    if a.selftest: selftest(); return
    for k in ('lab', 'dataset', 'model', 'out'):
        if getattr(a, k) is None: p.error(f'--{k} required')
    if not torch.cuda.is_available(): sys.exit('FAILED: CUDA required')
    try:
        import skimage, fast_simplification, scipy  # noqa: F401
    except Exception as e:
        sys.exit(f'FAILED: import ({e}); pip install scikit-image fast-simplification scipy in the venv (login node, has network)')
    up = Path(a.lab).resolve() / 'upstream'
    if not (up / 'train.py').exists(): sys.exit(f'FAILED: no upstream/train.py under {a.lab}')
    sys.path.insert(0, str(up / 'submodules/effrdel/src')); sys.path.insert(0, str(up))
    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render
    from PIL import Image
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = SimpleNamespace(source_path=str(Path(a.dataset).resolve()), model_path=str(Path(a.model).resolve()), images=a.images, resolution=1, eval=True,
                           white_background=False, data_device='cpu', sh_degree=3)
    mesh = TriangleModel(3); scene = Scene(data, mesh, None, None, load_iteration=a.iteration, shuffle=False)
    cams = sorted(scene.getTrainCameras(), key=lambda c: (c.image_name, c.uid))
    if a.max_views > 0: cams = cams[:a.max_views]
    cloud = mesh.vertices.detach().cpu().numpy().astype(np.float64)
    lo, hi = cloud.min(0), cloud.max(0); diag = float(np.linalg.norm(hi - lo))
    log(f'{VERSION} | soup {cloud.shape[0]} v / {int(mesh._triangle_indices.shape[0])} f | full bbox diag {diag:.3f} | train views {len(cams)} | TSDF res {a.res}')
    bg = torch.zeros(3, device='cuda'); pipe = SimpleNamespace(debug=False, convert_SHs_python=False, depth_ratio=1.)

    alpha_state = {}

    def depth_of(c):
        pkg = render(c, mesh, pipe, bg)
        H, W = int(c.image_height), int(c.image_width)
        if a.depth_key not in pkg or pkg[a.depth_key] is None:
            sys.exit(f"FAILED: render_pkg has no '{a.depth_key}'; keys = {sorted(k for k, v in pkg.items() if v is not None)}  (pass --depth_key)")
        d = pkg[a.depth_key].detach().float()
        if d.numel() != H * W: sys.exit(f"FAILED: depth has shape {tuple(pkg[a.depth_key].shape)}, expected H x W = {H} x {W}")
        d = torch.nan_to_num(d.reshape(H, W), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        if a.alpha_key:
            al = pkg.get(a.alpha_key)
            if al is None or al.numel() != H * W:
                if not alpha_state: log(f"WARNING: no '{a.alpha_key}' (H x W) in render_pkg; depth not alpha-masked"); alpha_state['warned'] = True
            else:
                al = torch.nan_to_num(al.detach().float().reshape(H, W), nan=0.0)
                if not alpha_state:
                    log(f"first view alpha: {100 * float((al >= a.alpha_min).float().mean()):.1f} % px >= {a.alpha_min}, {100 * float((al > 0.01).float().mean()):.1f} % px > 0.01"); alpha_state['ok'] = True
                d = torch.where(al >= a.alpha_min, d, torch.zeros_like(d))
        return d.cpu().numpy().astype(np.float32)

    with torch.no_grad():
        d0 = depth_of(cams[0])
    valid = d0 > 0
    if valid.mean() < 0.02: sys.exit(f'FAILED: first depth map has only {100 * valid.mean():.2f} % valid pixels')
    E0, conv = choose_extrinsic(cams[0], d0, cloud, diag)
    log(f"first view {cams[0].image_name}: valid px {100 * valid.mean():.1f} %, depth median {np.median(d0[valid]):.3f} max {d0.max():.3f} | convention {conv['convention']}")
    Image.fromarray((np.clip(d0 / max(d0.max(), 1e-6), 0, 1) * 255).astype(np.uint8)).save(out / 'depth-view0.png')
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    views = []
    for c in cams:
        W, H = int(c.image_width), int(c.image_height)
        fx, fy, cx, cy = intrinsics(W, H, float(c.FoVx), float(c.FoVy))
        W2V = c.world_view_transform.detach().cpu().numpy().T.astype(np.float64)
        views.append(dict(fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H, E=W2V if conv['convention'] == 'opencv' else flip @ W2V, name=c.image_name))

    def depth_fn(i):
        if i == 0: return d0
        with torch.no_grad():
            d = depth_of(cams[i])
        if i == 1: Image.fromarray((np.clip(d / max(d.max(), 1e-6), 0, 1) * 255).astype(np.uint8)).save(out / 'depth-view1.png')
        return d

    summary = fuse_and_extract(views, depth_fn, cloud, a, out, torch.device('cuda'))
    summary.update(model=str(a.model), iteration=a.iteration, depth_key=a.depth_key, alpha_key=a.alpha_key, alpha_min=a.alpha_min, convention=conv, soup_vertices=int(cloud.shape[0]))
    with open(out / 'summary.json', 'w') as fh: json.dump(summary, fh, indent=1)
    log('done')


# ----------------------------------------------------------------------------- selftest
def look_at(eye, target=np.zeros(3), up=np.array([0.0, 0.0, 1.0])):
    """OpenCV camera-from-world: z forward (towards target), y down."""
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    d = np.cross(f, r)                                            # y down = f x r
    R = np.stack([r, d, f], 0); E = np.eye(4); E[:3, :3] = R; E[:3, 3] = -R @ eye
    return E


def sphere_depth(E, fx, fy, cx, cy, W, H, radius=1.0):
    """analytic depth of a unit sphere at the origin seen by camera E (OpenCV)"""
    v, u = np.mgrid[0:H, 0:W]
    dirs = np.stack([(u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy, np.ones((H, W))], -1).reshape(-1, 3)
    Rinv = E[:3, :3].T; o = -Rinv @ E[:3, 3]
    dw = dirs @ Rinv.T
    aq = (dw * dw).sum(1); b = 2 * (dw @ o); c = o @ o - radius ** 2
    disc = b ** 2 - 4 * aq * c
    t = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * aq), 0.0)
    z = np.where(t > 0, t * dirs[:, 2], 0.0)                       # dirs has z = 1 so depth = t
    return z.reshape(H, W).astype(np.float32)


def selftest():
    import tempfile
    rng = np.random.default_rng(0)
    # (1) camera round trip, as in v1
    W, H = 160, 120; fovx = math.radians(60); fovy = 2 * math.atan(math.tan(fovx / 2) * H / W)
    fx, fy, cx, cy = intrinsics(W, H, fovx, fovy)
    assert abs(fx - fy) < 1e-9
    th = rng.uniform(0, math.pi, 4000); ph = rng.uniform(0, 2 * math.pi, 4000)
    P = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
    E = np.eye(4); E[:3, 3] = [0.0, 0.0, 4.0]
    pc = P + E[:3, 3]
    u = (pc[:, 0] / pc[:, 2] * fx + cx); v = (pc[:, 1] / pc[:, 2] * fy + cy)
    depth = np.zeros((H, W), np.float32)
    ui, vi = np.floor(u).astype(int), np.floor(v).astype(int)
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    order = np.argsort(-pc[ok, 2])
    depth[vi[ok][order], ui[ok][order]] = pc[ok, 2][order]
    d = nn_median(backproject(depth, fx, fy, cx, cy, E), P)
    assert d < 0.02, d
    assert nn_median(backproject(depth, fx, fy, cx, cy, np.diag([1.0, -1.0, -1.0, 1.0]) @ E), P) > 10 * d
    # (2) full pipeline on a synthetic unit sphere: 30 analytic depth maps -> TSDF -> MC -> decimate -> Taubin
    W, H = 96, 96; fovx = fovy = math.radians(50)
    fx, fy, cx, cy = intrinsics(W, H, fovx, fovy)
    views = []
    for k in range(30):
        z = 1 - 2 * (k + 0.5) / 30; r = math.sqrt(1 - z * z); phi = k * 2.399963
        eye = 4.0 * np.array([r * math.cos(phi), r * math.sin(phi), z])
        views.append(dict(fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H, E=look_at(eye), name=f'v{k}'))
    cloud = P.astype(np.float64)
    a = SimpleNamespace(res=48, trunc_vox=4.0, targets=[600], taubin=3, min_component_faces=50)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        s = fuse_and_extract(views, lambda i: sphere_depth(views[i]['E'], fx, fy, cx, cy, W, H), cloud, a, out, torch.device('cpu'))
        raw = np.load(out / 'mesh-raw.npz'); m = np.load(out / 'mesh-T600.npz')
        assert (out / 'mesh-T600.ply').stat().st_size > 0
    vox = s['voxel']
    r_raw = np.linalg.norm(raw['vertices'], axis=1); r_dec = np.linalg.norm(m['vertices'], axis=1)
    assert s['raw']['watertight'], s['raw']
    assert s['raw']['components'] == 1, s['raw']
    assert abs(np.median(r_raw) - 1.0) < 2 * vox, (np.median(r_raw), vox)
    assert np.abs(r_raw - 1.0).max() < 4 * vox, (np.abs(r_raw - 1.0).max(), vox)
    assert 400 <= s['meshes']['600']['faces'] <= 700, s['meshes']['600']
    assert s['meshes']['600']['edge_manifold'], s['meshes']['600']
    assert abs(np.median(r_dec) - 1.0) < 4 * vox, np.median(r_dec)
    print('selftest PASS', VERSION, f"(round-trip median NN {d:.4f}; sphere raw V {s['raw']['vertices']} F {s['raw']['faces']} watertight, "
          f"radius median {np.median(r_raw):.4f} (voxel {vox:.4f}); T600 -> F {s['meshes']['600']['faces']} edge-manifold)")


if __name__ == '__main__':
    main()
