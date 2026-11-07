"""surf_extract_v1.py (VERSION surf_extract_v1 2026-11-05): P2b step 1 -- a MANIFOLD surface from a trained MeshSplatting soup, no training.

The soup is used only as a geometry finder: its expected depth is rendered at every training view with the upstream renderer,
fused into a TSDF volume (Open3D), a mesh is extracted by marching cubes, small components are dropped, and the mesh is
decimated (quadric) to the requested face counts and lightly smoothed (Taubin).  Outputs per target T: <out>/mesh-T<T>.npz
(vertices float32, faces int32, model space), <out>/mesh-T<T>.ply, plus mesh-raw.npz/.ply, two depth sanity PNGs, summary.json,
and one 'SURF ...' line per mesh (V, F, edge-manifold, vertex-manifold, watertight, components).

Startup self-check (fails before the long part): the first depth map is back-projected with the intrinsics/extrinsics used for
fusion and must land on the soup (median nearest-vertex distance < 1 % of the bbox diagonal); both camera conventions
(OpenCV and y/z-flipped) are tried and the winner must beat the other 3x.

Needs: open3d (pip install open3d, pure wheel), scipy.  Run (GPU node):
  python surf_extract_v1.py --lab LAB --dataset DATA --model MODEL_DIR(--iteration 30000) --out OUT --targets 50000 100000 200000 [--res 512 --taubin 5]
  python surf_extract_v1.py --selftest     (CPU: camera matrix + back-projection round trip, no upstream / open3d needed)
"""
import argparse, json, math, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

VERSION = 'surf_extract_v1 2026-11-05'


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def intrinsics(W, H, fovx, fovy):
    fx = W / (2.0 * math.tan(fovx / 2.0)); fy = H / (2.0 * math.tan(fovy / 2.0))
    return fx, fy, W / 2.0, H / 2.0


def backproject(depth, fx, fy, cx, cy, E):
    """depth (H,W) numpy, E = camera-from-world 4x4 (OpenCV: x right, y down, z forward) -> world points (N,3) at valid pixels."""
    H, W = depth.shape
    v, u = np.nonzero(depth > 0)
    z = depth[v, u]
    x = (u + 0.5 - cx) / fx * z; y = (v + 0.5 - cy) / fy * z
    pc = np.stack([x, y, z, np.ones_like(z)], 1)                 # camera coords, homogeneous
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


def mesh_stats(m, o3d):
    return dict(vertices=int(np.asarray(m.vertices).shape[0]), faces=int(np.asarray(m.triangles).shape[0]),
                edge_manifold=bool(m.is_edge_manifold(allow_boundary_edges=True)), edge_manifold_closed=bool(m.is_edge_manifold(allow_boundary_edges=False)),
                vertex_manifold=bool(m.is_vertex_manifold()), watertight=bool(m.is_watertight()),
                nonmanifold_edges=int(len(m.get_non_manifold_edges(allow_boundary_edges=True))))


def surf_line(tag, st):
    return (f"SURF {tag} | V {st['vertices']} F {st['faces']} | edge-manifold {st['edge_manifold']} (closed {st['edge_manifold_closed']}) vertex-manifold {st['vertex_manifold']} "
            f"watertight {st['watertight']} | non-manifold edges {st['nonmanifold_edges']}" + (f" | components {st['components']} (dropped {st['dropped_components']})" if 'components' in st else ''))


def save_mesh(m, path_base, o3d):
    V = np.asarray(m.vertices, dtype=np.float32); F = np.asarray(m.triangles, dtype=np.int32)
    np.savez(path_base.with_suffix('.npz'), vertices=V, faces=F)
    o3d.io.write_triangle_mesh(str(path_base.with_suffix('.ply')), m, write_ascii=False)
    return V, F


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lab'); p.add_argument('--dataset'); p.add_argument('--model'); p.add_argument('--images', default='train')
    p.add_argument('--iteration', type=int, default=30000); p.add_argument('--out')
    p.add_argument('--targets', nargs='+', type=int, default=[50000, 100000, 200000], help='quadric-decimation face targets')
    p.add_argument('--res', type=int, default=512, help='TSDF voxels across the soup bbox diagonal')
    p.add_argument('--trunc_vox', type=float, default=4.0, help='sdf truncation in voxels')
    p.add_argument('--depth_key', default='expected_depth'); p.add_argument('--taubin', type=int, default=5)
    p.add_argument('--min_component_faces', type=int, default=200); p.add_argument('--max_views', type=int, default=0)
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    if a.selftest: selftest(); return
    for k in ('lab', 'dataset', 'model', 'out'):
        if getattr(a, k) is None: p.error(f'--{k} required')
    if not torch.cuda.is_available(): sys.exit('FAILED: CUDA required')
    try:
        import open3d as o3d
    except Exception as e:
        sys.exit(f'FAILED: open3d import ({e}); pip install open3d in the venv (login node, has network)')
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
    log(f'{VERSION} | soup {cloud.shape[0]} v / {int(mesh._triangle_indices.shape[0])} f | bbox diag {diag:.3f} | train views {len(cams)} | TSDF res {a.res} (voxel {diag / a.res:.5f})')
    bg = torch.zeros(3, device='cuda'); pipe = SimpleNamespace(debug=False, convert_SHs_python=False, depth_ratio=1.)

    def depth_of(c):
        pkg = render(c, mesh, pipe, bg)
        if a.depth_key not in pkg:
            sys.exit(f"FAILED: render_pkg has no '{a.depth_key}'; keys = {sorted(pkg.keys())}  (pass --depth_key)")
        d = pkg[a.depth_key].detach().float()
        d = d.reshape(int(c.image_height), int(c.image_width)) if d.numel() == int(c.image_height) * int(c.image_width) else None
        if d is None: sys.exit(f"FAILED: depth has shape {tuple(pkg[a.depth_key].shape)}, expected H x W")
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        return d.cpu().numpy().astype(np.float32)

    with torch.no_grad():
        d0 = depth_of(cams[0])
    valid = d0 > 0
    if valid.mean() < 0.02: sys.exit(f'FAILED: first depth map has only {100 * valid.mean():.2f} % valid pixels')
    E0, conv = choose_extrinsic(cams[0], d0, cloud, diag)
    log(f"first view {cams[0].image_name}: valid px {100 * valid.mean():.1f} %, depth median {np.median(d0[valid]):.3f} max {d0.max():.3f} | convention {conv['convention']}")
    dmax = float(np.percentile(d0[valid], 99.9)) * 1.5
    Image.fromarray((np.clip(d0 / max(d0.max(), 1e-6), 0, 1) * 255).astype(np.uint8)).save(out / 'depth-view0.png')

    vox = diag / a.res
    vol = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=vox, sdf_trunc=a.trunc_vox * vox, color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    t0 = time.time(); n_px = 0
    with torch.no_grad():
        for i, c in enumerate(cams):
            d = d0 if i == 0 else depth_of(c)
            W, H = int(c.image_width), int(c.image_height)
            fx, fy, cx, cy = intrinsics(W, H, float(c.FoVx), float(c.FoVy))
            W2V = c.world_view_transform.detach().cpu().numpy().T.astype(np.float64)
            E = W2V if conv['convention'] == 'opencv' else flip @ W2V
            intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            color = o3d.geometry.Image(np.zeros((H, W, 3), dtype=np.uint8))
            depth = o3d.geometry.Image(np.ascontiguousarray(d))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1.0, depth_trunc=dmax, convert_rgb_to_intensity=False)
            vol.integrate(rgbd, intr, E)
            n_px += int((d > 0).sum())
            if i == 1: Image.fromarray((np.clip(d / max(d.max(), 1e-6), 0, 1) * 255).astype(np.uint8)).save(out / 'depth-view1.png')
            if (i + 1) % 25 == 0: log(f'  integrated {i + 1}/{len(cams)} views ({time.time() - t0:.0f} s)')
    log(f'TSDF integration done: {len(cams)} views, {n_px / 1e6:.1f} M valid depth px, {time.time() - t0:.0f} s')

    raw = vol.extract_triangle_mesh()
    raw.remove_degenerate_triangles(); raw.remove_duplicated_triangles(); raw.remove_duplicated_vertices(); raw.remove_unreferenced_vertices()
    if len(raw.triangles) == 0: sys.exit('FAILED: marching cubes produced no triangles')
    cl, n_tri, _ = raw.cluster_connected_triangles()
    cl = np.asarray(cl); n_tri = np.asarray(n_tri)
    keep = n_tri[cl] >= a.min_component_faces
    raw.remove_triangles_by_mask(~keep); raw.remove_unreferenced_vertices()
    st_raw = mesh_stats(raw, o3d); st_raw.update(components=int((n_tri >= a.min_component_faces).sum()), dropped_components=int((n_tri < a.min_component_faces).sum()))
    print(surf_line('raw', st_raw), flush=True)
    save_mesh(raw, out / 'mesh-raw', o3d)
    summary = dict(version=VERSION, model=str(a.model), iteration=a.iteration, views=len(cams), res=a.res, voxel=vox, trunc_vox=a.trunc_vox, depth_key=a.depth_key,
                   convention=conv, bbox_diag=diag, raw=st_raw, meshes={})
    for T in a.targets:
        t1 = time.time()
        m = raw.simplify_quadric_decimation(target_number_of_triangles=int(T)) if T < len(raw.triangles) else o3d.geometry.TriangleMesh(raw)
        m.remove_degenerate_triangles(); m.remove_duplicated_triangles(); m.remove_duplicated_vertices(); m.remove_unreferenced_vertices()
        if a.taubin > 0:
            m = m.filter_smooth_taubin(number_of_iterations=a.taubin)
        m.remove_non_manifold_edges(); m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()
        st = mesh_stats(m, o3d); st['elapsed_s'] = round(time.time() - t1, 1)
        V, F = save_mesh(m, out / f'mesh-T{T}', o3d)
        if F.shape[0] == 0 or int(F.max()) >= V.shape[0]: sys.exit(f'FAILED: mesh T{T} invalid ({V.shape[0]} v, {F.shape[0]} f)')
        st['nn_to_soup_frac'] = nn_median(V.astype(np.float64), cloud) / diag
        print(surf_line(f'T{T}', st) + f" | median dist to soup {100 * st['nn_to_soup_frac']:.3f} % diag | {st['elapsed_s']} s", flush=True)
        summary['meshes'][str(T)] = st
    with open(out / 'summary.json', 'w') as fh: json.dump(summary, fh, indent=1)
    log('done')


def selftest():
    # synthetic camera: points on a sphere, OpenCV extrinsic, intrinsics from FoV -> project -> back-project must round-trip
    rng = np.random.default_rng(0)
    W, H = 160, 120; fovx = math.radians(60); fovy = 2 * math.atan(math.tan(fovx / 2) * H / W)
    fx, fy, cx, cy = intrinsics(W, H, fovx, fovy)
    assert abs(fx - fy) < 1e-9
    th = rng.uniform(0, math.pi, 4000); ph = rng.uniform(0, 2 * math.pi, 4000)
    P = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
    R = np.eye(3); t = np.array([0.0, 0.0, 4.0]); E = np.eye(4); E[:3, :3] = R; E[:3, 3] = t      # camera 4 units back, looking +z
    pc = P @ R.T + t
    u = (pc[:, 0] / pc[:, 2] * fx + cx); v = (pc[:, 1] / pc[:, 2] * fy + cy)
    depth = np.zeros((H, W), np.float32)
    ui, vi = np.floor(u).astype(int), np.floor(v).astype(int)
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    # z-buffer: keep the nearest depth per pixel
    order = np.argsort(-pc[ok, 2])
    depth[vi[ok][order], ui[ok][order]] = pc[ok, 2][order]
    back = backproject(depth, fx, fy, cx, cy, E)
    d = nn_median(back, P)
    assert d < 0.02, d                                   # half-pixel quantisation only
    assert nn_median(backproject(depth, fx, fy, cx, cy, np.diag([1.0, -1.0, -1.0, 1.0]) @ E), P) > 10 * d
    print('selftest PASS', VERSION, f'(round-trip median NN {d:.4f})')


if __name__ == '__main__':
    main()
