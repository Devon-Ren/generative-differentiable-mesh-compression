"""texture_refit_v2.py (VERSION texture_refit_v2 2026-11-05) = texture_refit_v1 + an EXTERNAL mesh (P2b): --mesh mesh.npz (vertices, faces from
surf_extract_v1) replaces the checkpoint geometry; the soup is then used only for the cameras and for the colour initialisation (nearest soup
vertex SH / DC via a KD-tree); --subdivide n applies n midpoint subdivisions (x4 faces each) before fitting (the "pile small faces" control).
--variants accepts the v1 names; on an external mesh vsh3_init = nearest-vertex SH projection without refit.  Original v1 header follows.

texture_refit_v1.py (2026-10-30): probe P2 -- does colour need vertex density?  GPU, read-only on the model.

Takes a trained MeshSplatting checkpoint, FREEZES its geometry (vertices + faces) and refits the appearance on the
training views with several colour parameterisations, all rendered by the SAME opaque rasterizer (nvdiffrast, hard
triangles, z-buffer, black background), then reports train / test PSNR+SSIM per parameterisation:
  vsh3_init  per-vertex SH degree 3 as trained (no refit): opaque-render baseline of the checkpoint itself
  vsh3       per-vertex SH degree 3, refit                 (48 params / vertex)
  vdc        per-vertex RGB, refit                          (3 / vertex)
  frgb       per-face constant RGB                          (3 / face)
  tbB        per-face texture, B x B texel block per face   (3 B^2 / face allocated, 3 B(B+1)/2 used by the triangle)
  tbBsh1     tbB + per-face SH degree-1 view term           (+9 / face)
Reading: on a fixed mesh the curve PSNR vs colour parameters tells whether the colour field is limited by the vertex
density (then tb4 >> vsh3 at equal parameters) or not; across two meshes of one scene (lam 6.2e-3 ~ 0.44 M v vs
lam 1.85e-3 ~ 1.2 M v on Room) the comparison (coarse mesh + tb4) vs (fine mesh + vsh3) at a similar parameter count
says whether the extra vertices of the fine mesh were buying colour.  With --plane_labels (planarity_points_v1 output)
the test error is also split into pixels on planar faces vs the rest.
Only the renderer is new; vertices, faces and the training/test split are the checkpoint's and upstream's.

Needs: nvdiffrast (pip install git+https://github.com/NVlabs/nvdiffrast, compiled at first import, needs nvcc).
Run:  python texture_refit_v1.py --lab /scratch/zr2245/meshsplatting-original-lossless-v1 \
        --model /scratch/zr2245/room-rd3-lam0.0062-mu1.0/original --dataset /scratch/zr2245/datasets/m360/room \
        --images images_2 --label "room v3-lam0.0062" --out x.json [--plane_labels room-v3-lam0.0062-plane-labels.npy]
      python texture_refit_v1.py --selftest          (CPU: atlas / SH / psnr checks, no nvdiffrast needed)
"""
import argparse, json, math, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

VERSION = 'texture_refit_v2 2026-11-05'
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def psnr_from_mse(mse):
    return float(10.0 * math.log10(1.0 / max(mse, 1e-12)))


def eval_sh(deg, sh, dirs):
    """3DGS eval_sh: sh (..., C, (deg+1)^2), dirs (..., 3) unit -> (..., C).  Same constants as utils/sh_utils.py."""
    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3]
        if deg > 1:
            xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
            result = (result + C2[0] * xy * sh[..., 4] + C2[1] * yz * sh[..., 5] + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                      + C2[3] * xz * sh[..., 7] + C2[4] * (xx - yy) * sh[..., 8])
            if deg > 2:
                result = (result + C3[0] * y * (3 * xx - yy) * sh[..., 9] + C3[1] * xy * z * sh[..., 10] + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                          + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                          + C3[5] * z * (xx - yy) * sh[..., 14] + C3[6] * x * (xx - 3 * yy) * sh[..., 15])
    return result


def face_atlas(n_faces, block):
    """One block x block texel square per face in a square atlas; the face's triangle maps to the lower-left half of
    its block with a half-texel inset so bilinear sampling never leaves the block.  Returns (atlas side in texels,
    uv per face corner (F, 3, 2) in [0, 1])."""
    nblk = int(math.ceil(math.sqrt(n_faces)))
    side = nblk * block
    f = np.arange(n_faces)
    bi, bj = f // nblk, f % nblk                      # block row, block column
    u0 = (bj * block + 0.5) / side; u1 = (bj * block + block - 0.5) / side
    v0 = (bi * block + 0.5) / side; v1 = (bi * block + block - 0.5) / side
    uv = np.stack([np.stack([u0, v0], 1), np.stack([u1, v0], 1), np.stack([u0, v1], 1)], 1).astype(np.float32)
    return side, uv


def count_params(kind, n_v, n_f, block=0):
    if kind == 'vsh3': return dict(params=48 * n_v, used=48 * n_v)
    if kind == 'vdc': return dict(params=3 * n_v, used=3 * n_v)
    if kind == 'frgb': return dict(params=3 * n_f, used=3 * n_f)
    if kind == 'tb': return dict(params=3 * block * block * n_f, used=3 * (block * (block + 1) // 2) * n_f)
    if kind == 'tbsh1': return dict(params=3 * block * block * n_f + 9 * n_f, used=3 * (block * (block + 1) // 2) * n_f + 9 * n_f)
    raise ValueError(kind)


def parse_variant(name):
    if name in ('vsh3_init', 'vsh3', 'vdc', 'frgb'): return name, 0
    if name.startswith('tb'):
        core = name[2:]; sh1 = core.endswith('sh1')
        if sh1: core = core[:-3]
        b = int(core)
        if b < 1 or b > 16: raise ValueError(name)
        return ('tbsh1' if sh1 else 'tb'), b
    raise ValueError(f'unknown variant {name}')


def midpoint_subdivide(V, tri):
    """One midpoint subdivision: every face -> 4, one new vertex per unique edge.  V (N,3) float, tri (F,3) int -> (V2, tri2 int32)."""
    f = tri.long(); N = V.shape[0]
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
    e_s, _ = torch.sort(e, dim=1)
    key = e_s[:, 0] * N + e_s[:, 1]
    uk, inv = torch.unique(key, return_inverse=True)
    mid = (V[uk // N] + V[uk % N]) * 0.5
    V2 = torch.cat([V, mid], 0)
    F_ = f.shape[0]
    m01, m12, m20 = N + inv[:F_], N + inv[F_:2 * F_], N + inv[2 * F_:]
    a, b, c = f[:, 0], f[:, 1], f[:, 2]
    tri2 = torch.cat([torch.stack([a, m01, m20], 1), torch.stack([b, m12, m01], 1), torch.stack([c, m20, m12], 1), torch.stack([m01, m12, m20], 1)], 0)
    return V2, tri2.to(torch.int32)


def nearest_init(V_ext, V_soup, dc, rest):
    """Colour initialisation of an external mesh from the soup: SH of the nearest soup vertex (KD-tree on CPU)."""
    from scipy.spatial import cKDTree
    d, idx = cKDTree(V_soup.detach().cpu().numpy()).query(V_ext.detach().cpu().numpy(), k=1)
    idx = torch.as_tensor(idx, device=dc.device, dtype=torch.long)
    return dc[idx].contiguous(), rest[idx].contiguous(), float(np.median(d))


# ----------------------------------------------------------------------------------------------------------------------
class Renderer:
    """Opaque rasterization of a fixed mesh with nvdiffrast.  pos_w (N,3) cuda, tri (F,3) int32 cuda."""

    def __init__(self, pos_w, tri):
        import nvdiffrast.torch as dr
        self.dr = dr; self.ctx = dr.RasterizeCudaContext()
        self.pos_h = torch.cat([pos_w, torch.ones_like(pos_w[:, :1])], 1).contiguous()
        self.tri = tri.contiguous(); self.n_f = int(tri.shape[0])
        self.tri_soup = torch.arange(3 * self.n_f, device=tri.device, dtype=torch.int32).reshape(self.n_f, 3).contiguous()
        self.sx, self.sy = 1.0, 1.0

    def rasterize(self, cam):
        P = cam.full_proj_transform.cuda().float()
        clip = (self.pos_h @ P)
        if self.sx != 1.0 or self.sy != 1.0:
            clip = clip * torch.tensor([self.sx, self.sy, 1.0, 1.0], device=clip.device)
        H, W = int(cam.image_height), int(cam.image_width)
        rast, _ = self.dr.rasterize(self.ctx, clip[None].contiguous(), self.tri, resolution=[H, W])
        return rast                                             # (1,H,W,4): u, v, z/w, tri_id+1

    def interp(self, attr, rast, soup=False):
        """attr (N,C) or (3F,C) -> (C,H,W), zero where no triangle."""
        out, _ = self.dr.interpolate(attr[None].contiguous(), rast, self.tri_soup if soup else self.tri)
        mask = (rast[..., 3:4] > 0).float()
        return (out * mask)[0].permute(2, 0, 1)

    def texture(self, tex, uv_soup, rast):
        """tex (S,S,3), uv_soup (3F,2) -> (3,H,W)."""
        uv, _ = self.dr.interpolate(uv_soup[None].contiguous(), rast, self.tri_soup)
        col = self.dr.texture(tex[None].contiguous(), uv.contiguous(), filter_mode='linear', boundary_mode='clamp')
        mask = (rast[..., 3:4] > 0).float()
        return (col * mask)[0].permute(2, 0, 1)


class Appearance:
    """One colour parameterisation on the fixed mesh.  forward(cam, rast) -> (3,H,W) in [0,1] (clamped)."""

    def __init__(self, kind, block, R, V, tri, dc, rest, dev):
        self.kind, self.block, self.R = kind, block, R
        n_v, n_f = V.shape[0], tri.shape[0]
        self.n_v, self.n_f = n_v, n_f
        V, dc, rest = V.detach(), dc.detach(), rest.detach()
        self.V = V; self.tri_l = tri.long()
        self.params = []
        vdc_rgb = (C0 * dc[:, 0, :] + 0.5)                                              # (N,3) as rendered at DC
        if kind in ('vsh3', 'vsh3_init'):
            self.sh = torch.cat([dc, rest], 1).permute(0, 2, 1).contiguous().clone()   # (N,3,16)
            if kind == 'vsh3': self.sh.requires_grad_(True); self.params = [self.sh]
        elif kind == 'vdc':
            self.rgb = vdc_rgb.clone().requires_grad_(True); self.params = [self.rgb]
        elif kind == 'frgb':
            self.rgb = vdc_rgb[self.tri_l].mean(1).contiguous().clone().requires_grad_(True); self.params = [self.rgb]
        elif kind in ('tb', 'tbsh1'):
            side, uv = face_atlas(n_f, block)
            self.side = side
            self.uv = torch.as_tensor(uv.reshape(3 * n_f, 2), device=dev)
            nblk = side // block
            face_col = vdc_rgb[self.tri_l].mean(1)                                      # (F,3) init = face mean colour
            f = torch.arange(n_f, device=dev); bi, bj = f // nblk, f % nblk
            blk = torch.zeros(nblk, nblk, 3, device=dev); blk[bi, bj] = face_col
            tex = blk.repeat_interleave(block, 0).repeat_interleave(block, 1).contiguous()
            self.tex = tex.requires_grad_(True); self.params = [self.tex]
            if kind == 'tbsh1':
                self.fsh = torch.zeros(n_f, 3, 3, device=dev).requires_grad_(True); self.params.append(self.fsh)
                self.centroid = V[self.tri_l].mean(1)
        else:
            raise ValueError(kind)
        self.counts = count_params('vsh3' if kind == 'vsh3_init' else kind, n_v, n_f, block)

    def forward(self, cam, rast):
        R = self.R
        if self.kind in ('vsh3', 'vsh3_init'):
            d = F.normalize(self.V - cam.camera_center.cuda().float(), dim=-1)
            col = eval_sh(3, self.sh, d) + 0.5
            return R.interp(col, rast).clamp(0, 1)
        if self.kind == 'vdc':
            return R.interp(self.rgb, rast).clamp(0, 1)
        if self.kind == 'frgb':
            return R.interp(self.rgb.repeat_interleave(3, 0), rast, soup=True).clamp(0, 1)
        img = R.texture(self.tex, self.uv, rast)
        if self.kind == 'tbsh1':
            d = F.normalize(self.centroid - cam.camera_center.cuda().float(), dim=-1)
            x, y, z = d[:, 0:1], d[:, 1:2], d[:, 2:3]
            view = -C1 * y * self.fsh[:, :, 0] + C1 * z * self.fsh[:, :, 1] - C1 * x * self.fsh[:, :, 2]   # (F,3)
            img = img + R.interp(view.repeat_interleave(3, 0), rast, soup=True)
        return img.clamp(0, 1)


def evaluate(app, R, cams, ssim_fn, planar_face=None, limit=None):
    """Mean per-image PSNR / SSIM over cams; with planar_face (F,) bool also the aggregate MSE split by planar pixels."""
    ps, ss = [], []; sse_p = sse_o = 0.0; n_p = n_o = 0
    cams = cams if limit is None else cams[:limit]
    with torch.no_grad():
        for c in cams:
            rast = R.rasterize(c); img = app.forward(c, rast); gt = c.original_image[:3].cuda().clamp(0, 1)
            err = (img - gt).square()
            ps.append(psnr_from_mse(float(err.mean()))); ss.append(float(ssim_fn(img, gt)))
            if planar_face is not None:
                tid = rast[0, ..., 3].long() - 1
                pm = (tid >= 0) & planar_face[tid.clamp(min=0)]
                e = err.mean(0)
                sse_p += float(e[pm].sum()); n_p += int(pm.sum()); sse_o += float(e[~pm].sum()); n_o += int((~pm).sum())
    out = dict(psnr=float(np.mean(ps)), ssim=float(np.mean(ss)), views=len(cams))
    if planar_face is not None and n_p > 0:
        out.update(planar_pix_share=n_p / max(n_p + n_o, 1), psnr_planar=psnr_from_mse(sse_p / n_p), psnr_other=psnr_from_mse(sse_o / max(n_o, 1)))
    return out


def fit(app, R, train, ssim_fn, steps, lr, lam_dssim, rng, log_every=500):
    if not app.params: return dict(steps=0)
    opt = torch.optim.Adam(app.params, lr=lr, eps=1e-15)
    t0 = time.time(); last = None
    for it in range(1, steps + 1):
        c = train[int(rng.integers(len(train)))]
        rast = R.rasterize(c); img = app.forward(c, rast); gt = c.original_image[:3].cuda().clamp(0, 1)
        l1 = (img - gt).abs().mean(); loss = (1 - lam_dssim) * l1 + lam_dssim * (1 - ssim_fn(img, gt))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        last = float(loss)
        if it % log_every == 0 or it == steps: log(f'  {app.kind}{app.block or ""} step {it}/{steps} loss {last:.4f} ({time.time() - t0:.0f} s)')
    return dict(steps=steps, final_loss=last, fit_s=round(time.time() - t0, 1))


def summary_line(label, variant, res, n_v, n_f):
    tr, te = res['train'], res['test']
    s = (f"TEXREFIT {label} {variant:9s} | V {n_v} F {n_f} | params {res['params'] / 1e6:.1f}M (used {res['used'] / 1e6:.1f}M, ~{res['used'] / 1e6:.0f} MB @8bit)"
         f" | train {tr['psnr']:.2f} | test {te['psnr']:.3f} / ssim {te['ssim']:.4f}")
    if 'psnr_planar' in te:
        s += f" | planar px {100 * te['planar_pix_share']:.0f} %: planar {te['psnr_planar']:.2f} other {te['psnr_other']:.2f}"
    return s + f" | fit {res.get('fit_s', 0)} s"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lab'); p.add_argument('--model'); p.add_argument('--dataset'); p.add_argument('--images', default='images_2')
    p.add_argument('--iteration', type=int, default=30000); p.add_argument('--label', default=''); p.add_argument('--out', default=None)
    p.add_argument('--variants', nargs='+', default=['vsh3_init', 'vsh3', 'vdc', 'frgb', 'tb2', 'tb4', 'tb4sh1'])
    p.add_argument('--steps', type=int, default=3000); p.add_argument('--lr', type=float, default=0.01); p.add_argument('--lam_dssim', type=float, default=0.2)
    p.add_argument('--lr_sh', type=float, default=0.002, help='v2: learning rate for the per-vertex SH variants (v1 saw lr 0.01 diverge on SH rest)')
    p.add_argument('--plane_labels', default=None); p.add_argument('--test_limit', type=int, default=None); p.add_argument('--train_eval_views', type=int, default=20)
    p.add_argument('--seed', type=int, default=17); p.add_argument('--selftest', action='store_true')
    p.add_argument('--mesh', default=None, help='v2: external mesh .npz (vertices float32 (N,3), faces int32 (F,3), model space) replacing the checkpoint geometry')
    p.add_argument('--subdivide', type=int, default=0, help='v2: midpoint subdivisions applied to the fitted mesh (x4 faces each)')
    a = p.parse_args()
    if a.selftest: selftest(); return
    for k in ('lab', 'model', 'dataset'):
        if getattr(a, k) is None: p.error(f'--{k} required')
    if not torch.cuda.is_available(): sys.exit('FAILED: CUDA required')
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = 'cuda'
    upstream = Path(a.lab).resolve() / 'upstream'
    if not (upstream / 'train.py').exists(): sys.exit(f'FAILED: no upstream/train.py under {a.lab}')
    sys.path.insert(0, str(upstream / 'submodules/effrdel/src')); sys.path.insert(0, str(upstream))
    from scene import Scene
    from scene.triangle_model import TriangleModel
    from utils.loss_utils import ssim as ssim_fn
    try:
        import nvdiffrast.torch  # noqa: F401
    except Exception as e:
        sys.exit(f'FAILED: nvdiffrast import ({e}); pip install git+https://github.com/NVlabs/nvdiffrast in the venv')

    sd_path = Path(a.model) / 'point_cloud' / f'iteration_{a.iteration}' / 'point_cloud_state_dict.pt'
    if not sd_path.exists(): sys.exit(f'FAILED: {sd_path} not found')
    sd = torch.load(sd_path, map_location='cpu', weights_only=False)
    V = sd['triangles_points'].detach().float().to(dev); tri = sd['_triangle_indices'].detach().to(torch.int32).to(dev)
    dc = sd['features_dc'].detach().float().to(dev); rest = sd['features_rest'].detach().float().to(dev)   # saved Parameters carry requires_grad
    n_v, n_f = int(V.shape[0]), int(tri.shape[0])
    if dc.shape[1] != 1 or rest.shape[1] != 15: sys.exit(f'FAILED: unexpected SH shapes dc {tuple(dc.shape)} rest {tuple(rest.shape)}')
    if int(tri.max()) >= n_v: sys.exit('FAILED: face index out of range')
    n_v_soup = n_v
    log(f'{VERSION} | {a.label} | soup {n_v} v / {n_f} f from {sd_path}')
    V_soup = V
    if a.mesh:
        mz = np.load(a.mesh)
        V = torch.as_tensor(np.asarray(mz['vertices'], dtype=np.float32), device=dev); tri = torch.as_tensor(np.asarray(mz['faces'], dtype=np.int32), device=dev)
        if V.ndim != 2 or V.shape[1] != 3 or tri.ndim != 2 or tri.shape[1] != 3 or tri.numel() == 0: sys.exit(f'FAILED: bad mesh arrays in {a.mesh}: {tuple(V.shape)} {tuple(tri.shape)}')
        if int(tri.max()) >= V.shape[0] or int(tri.min()) < 0: sys.exit('FAILED: external mesh face index out of range')
        if a.plane_labels: sys.exit('FAILED: --plane_labels is only defined on the checkpoint geometry, not with --mesh')
        lo_s, hi_s = V_soup.min(0).values, V_soup.max(0).values; lo_m, hi_m = V.min(0).values, V.max(0).values
        if float((lo_m - lo_s).abs().max()) > 0.5 * float((hi_s - lo_s).norm()) or float((hi_m - hi_s).abs().max()) > 0.5 * float((hi_s - lo_s).norm()):
            sys.exit(f'FAILED: external mesh bbox {lo_m.tolist()}..{hi_m.tolist()} far from soup bbox {lo_s.tolist()}..{hi_s.tolist()} (model space mismatch)')
        log(f'external mesh {a.mesh}: {int(V.shape[0])} v / {int(tri.shape[0])} f')
    for _ in range(a.subdivide):
        V, tri = midpoint_subdivide(V, tri)
    if a.subdivide: log(f'after {a.subdivide} midpoint subdivision(s): {int(V.shape[0])} v / {int(tri.shape[0])} f')
    if a.mesh or a.subdivide:
        dc, rest, dmed = nearest_init(V, V_soup, dc, rest)
        log(f'colour init from nearest soup vertex: median distance {dmed:.5f}')
    n_v, n_f = int(V.shape[0]), int(tri.shape[0])

    data = SimpleNamespace(source_path=str(Path(a.dataset).resolve()), model_path=str(Path(a.model).resolve()), images=a.images,
                           resolution=1, eval=True, white_background=False, data_device='cpu', sh_degree=3)
    mesh = TriangleModel(3); scene = Scene(data, mesh, None, None, load_iteration=a.iteration, shuffle=False)
    if int(mesh.vertices.shape[0]) != n_v_soup: sys.exit(f'FAILED: Scene loaded {int(mesh.vertices.shape[0])} vertices, state dict has {n_v_soup}')
    train = sorted(scene.getTrainCameras(), key=lambda c: (c.image_name, c.uid)); test = sorted(scene.getTestCameras(), key=lambda c: (c.image_name, c.uid))
    del mesh, scene; torch.cuda.empty_cache()
    c0 = train[0]; H, W = int(c0.image_height), int(c0.image_width)
    if max(H, W) > 2048: sys.exit(f'FAILED: image {W}x{H} exceeds the 2048 limit of the nvdiffrast CUDA rasterizer; use a smaller --images')
    log(f'train {len(train)} test {len(test)} views at {W}x{H}')

    planar_face = None
    if a.plane_labels:
        lab = np.load(a.plane_labels)
        if len(lab) != n_v: sys.exit(f'FAILED: plane labels {len(lab)} != vertices {n_v}')
        lt = torch.as_tensor(lab, device=dev).long()[tri.long()]
        planar_face = (lt.min(1).values >= 0) & (lt[:, 0] == lt[:, 1]) & (lt[:, 0] == lt[:, 2])
        log(f'planar faces (all 3 vertices in one plane label): {int(planar_face.sum())} / {n_f} = {100 * float(planar_face.float().mean()):.1f} %')

    R = Renderer(V, tri)
    # ---- clip-space axis convention: the DC-coloured opaque render must match the photos; try the 4 sign choices
    probe = Appearance('vdc', 0, R, V, tri, dc, rest, dev)
    scores = {}
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            R.sx, R.sy = sx, sy
            scores[(sx, sy)] = evaluate(probe, R, train, ssim_fn, limit=3)['psnr']
    best = max(scores, key=scores.get); ranked = sorted(scores.values(), reverse=True)
    log('axis check (DC colours, 3 train views): ' + ', '.join(f'sx {k[0]:+.0f} sy {k[1]:+.0f} -> {v:.2f} dB' for k, v in scores.items()))
    if ranked[0] - ranked[1] < 3.0 or ranked[0] < 12.0: sys.exit('FAILED: clip-space convention ambiguous (margins above)')
    R.sx, R.sy = best
    del probe

    results = dict(version=VERSION, label=a.label, model=str(sd_path), mesh=a.mesh, subdivide=a.subdivide, vertices=n_v, faces=n_f, image=[W, H], train_views=len(train), test_views=len(test),
                   steps=a.steps, lr=a.lr, lr_sh=a.lr_sh, lam_dssim=a.lam_dssim, axis=dict(sx=best[0], sy=best[1]), variants={})
    train_eval = [train[i] for i in rng.choice(len(train), min(a.train_eval_views, len(train)), replace=False)]
    for name in a.variants:
        kind, block = parse_variant(name)
        t0 = time.time(); log(f'== {name} ==')
        app = Appearance(kind, block, R, V, tri, dc, rest, dev)
        res = dict(app.counts)
        res.update(fit(app, R, train, ssim_fn, 0 if kind == 'vsh3_init' else a.steps, a.lr_sh if kind == 'vsh3' else a.lr, a.lam_dssim, rng))
        res['train'] = evaluate(app, R, train_eval, ssim_fn); res['test'] = evaluate(app, R, test, ssim_fn, planar_face, a.test_limit)
        res['elapsed_s'] = round(time.time() - t0, 1); res['peak_gpu_gb'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        results['variants'][name] = res
        print(summary_line(a.label, name, res, n_v, n_f), flush=True)
        del app; torch.cuda.empty_cache()
        if a.out:
            with open(a.out, 'w') as fh: json.dump(results, fh, indent=1)
    results['complete'] = True
    if a.out:
        with open(a.out, 'w') as fh: json.dump(results, fh, indent=1)
    log('done')


def selftest():
    # atlas: every corner of every face lies inside its own block with >= 0.5 texel margin; blocks are distinct
    for n_f, b in ((7, 1), (100, 2), (1000, 4), (12345, 8)):
        side, uv = face_atlas(n_f, b)
        nblk = side // b
        tex = uv * side                                                     # texel coordinates of the corners
        f = np.arange(n_f); bi, bj = f // nblk, f % nblk
        lo_u, lo_v = bj[:, None] * b, bi[:, None] * b
        assert np.all(tex[..., 0] >= lo_u + 0.5 - 1e-4) and np.all(tex[..., 0] <= lo_u + b - 0.5 + 1e-4), (n_f, b)
        assert np.all(tex[..., 1] >= lo_v + 0.5 - 1e-4) and np.all(tex[..., 1] <= lo_v + b - 0.5 + 1e-4), (n_f, b)
        blocks = set(zip((tex[:, 0, 0] // b).astype(int), (tex[:, 0, 1] // b).astype(int)))
        assert len(blocks) == n_f, 'blocks not distinct'
        assert side <= b * (int(math.sqrt(n_f)) + 1)
    # sh: degree 0 only reproduces the DC colour; a known degree-1 coefficient gives the expected sign
    sh = torch.zeros(5, 3, 16); sh[:, :, 0] = 1.0
    assert torch.allclose(eval_sh(3, sh, F.normalize(torch.randn(5, 3), dim=-1)), torch.full((5, 3), C0))
    sh[:, :, 2] = 1.0
    assert torch.allclose(eval_sh(1, sh, torch.tensor([[0.0, 0.0, 1.0]] * 5)), torch.full((5, 3), C0 + C1))
    # counts: tb4 on a soup with F = 3V has 48F = 144V allocated, 90V used; vsh3 48V
    c = count_params('tb', 100, 300, 4); assert c['params'] == 48 * 300 and c['used'] == 30 * 300, c
    assert count_params('vsh3', 100, 300)['params'] == 4800
    assert parse_variant('tb4sh1') == ('tbsh1', 4) and parse_variant('tb2') == ('tb', 2) and parse_variant('vdc') == ('vdc', 0)
    assert abs(psnr_from_mse(1e-3) - 30.0) < 1e-9
    # the Appearance init path without a renderer (CPU): per-face texture init equals the face mean colour everywhere in its block
    V = torch.rand(10, 3); tri = torch.tensor([[0, 1, 2], [2, 3, 4], [5, 6, 7], [7, 8, 9], [1, 5, 9]], dtype=torch.int32)
    dc = torch.rand(10, 1, 3).requires_grad_(True) * 1.0; rest = torch.zeros(10, 15, 3)   # non-leaf input, as a saved Parameter would be
    app = Appearance('tb', 4, None, V, tri, dc, rest, 'cpu')
    nblk = app.side // 4; face_col = (C0 * dc[:, 0, :] + 0.5)[tri.long()].mean(1)
    for fi in range(5):
        bi, bj = fi // nblk, fi % nblk
        blk = app.tex[bi * 4:(bi + 1) * 4, bj * 4:(bj + 1) * 4]
        assert torch.allclose(blk, face_col[fi].expand(4, 4, 3)), fi
    assert app.counts['params'] == 48 * 5 and app.tex.is_leaf and app.tex.requires_grad
    app2 = Appearance('tbsh1', 2, None, V, tri, dc, rest, 'cpu'); assert len(app2.params) == 2 and app2.counts['params'] == 12 * 5 + 45
    app3 = Appearance('vsh3_init', 0, None, V, tri, dc, rest, 'cpu'); assert app3.params == [] and app3.sh.shape == (10, 3, 16)
    # v2: midpoint subdivision on a tetrahedron (4 faces, 6 edges) -> 16 faces, 4 + 6 vertices, every edge still shared by 2 faces
    Vt = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]]); Ft = torch.tensor([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=torch.int32)
    V2, F2 = midpoint_subdivide(Vt, Ft)
    assert V2.shape == (10, 3) and F2.shape == (16, 3) and F2.dtype == torch.int32 and int(F2.max()) == 9
    e = torch.cat([F2[:, [0, 1]], F2[:, [1, 2]], F2[:, [2, 0]]], 0).long(); e, _ = torch.sort(e, 1); _, cnt = torch.unique(e[:, 0] * 10 + e[:, 1], return_counts=True)
    assert bool((cnt == 2).all()), 'subdivided tetrahedron is not closed'
    assert torch.allclose(V2[4:].mean(0), Vt.mean(0) * 0 + V2[4:].mean(0))
    V3, F3 = midpoint_subdivide(V2, F2); assert F3.shape[0] == 64 and V3.shape[0] == 10 + 24
    # v2: nearest init picks the soup vertex colour
    dcs = torch.arange(10, dtype=torch.float32).reshape(10, 1, 1).expand(10, 1, 3).contiguous(); rests = torch.zeros(10, 15, 3)
    d2, r2, dm = nearest_init(V[:3] + 1e-3, V, dcs, rests); assert torch.equal(d2[:, 0, 0], torch.tensor([0.0, 1.0, 2.0])) and dm < 0.01
    print('selftest PASS', VERSION)


if __name__ == '__main__':
    main()
