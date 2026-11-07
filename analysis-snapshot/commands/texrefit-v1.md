# texrefit-v1: 探针 P2 —— 颜色需要顶点密度吗（GPU，mesh 冻结），2026-10-30

**P1b 的结论**：顶点级看，室内 46–75 % 的顶点落在 ≥ 1000 点的大平面上（kitchen 75、room 54、counter 50、bonsai 46 %），但面级只有 10–12.7 % 是平面内部顶点。差的这 5–7 倍就是瓦片褶皱：顶点在编码褶皱和颜色，不是形状。
**P2 问的问题**：把 mesh（顶点 + 面）冻住，只换颜色的参数化，用**同一个 opaque rasterizer**（nvdiffrast，硬三角形、z-buffer、黑背景）在训练视角重新拟合，比 test PSNR/SSIM：
- `vsh3_init` 每顶点 SH3 原样（不拟合）= 这个 checkpoint 在 opaque 渲染下的基线；`vsh3` 同参数化重新拟合（48 参数/顶点）；`vdc` 每顶点 RGB（3/顶点）
- `frgb` 每面常量 RGB（3/面）；`tb2` / `tb4` 每面 2×2 / 4×4 texel 的纹理块（Ptex / mesh-colors 的最简版，无 UV；三角形只用块的下半，参数按"分配 / 实际用到"两种口径报）；`tb4sh1` = tb4 + 每面 SH1 视角项（+9/面）
- 有 P1b 的平面标签时，test 误差再拆成"三顶点同一平面标签的面"上的像素 vs 其它像素

**读法**（Room：λ6.2e-3 0.44 M 顶点 ≈ 1.3 M 面，λ1.85e-3 1.19 M 顶点）：
1. 同一 mesh 上 PSNR vs 颜色参数量的曲线：若 `tb4` 明显高于 `vsh3`（参数量相近：tb4 用到 ≈ 90 V，vsh3 = 48 V），说明颜色场被顶点密度卡住了，纹理能解开；若 `frgb`/`tb2` 已接近 `vsh3`，说明颜色本来就不需要那么多自由度（和 codebook 的冗余结论一致）。
2. 跨 mesh：**(粗 mesh λ6.2e-3 + tb4) vs (细 mesh λ1.85e-3 + vsh3)**，两者颜色参数量都 ≈ 60 M。粗 + 纹理 ≥ 细 + SH3 → 细 mesh 多出的 2.7x 顶点是在买颜色，+++ 的 4.1（颜色交给纹理，顶点密度只跟形状）成立；粗 + 纹理仍明显低 → 多出的顶点在买几何（褶皱本身在渲染上有用），重点回到 4.2。
3. 平面像素 vs 其它像素：纹理的增益若集中在平面像素上，就是 P1b 平面的直接后果。
4. `vsh3_init` vs 论文表里的 soft 渲染 PSNR（Room λ6.2e-3 27.38）之差 = opaque 渲染本身的损失，所有变体共享，不影响 1–3 的比较。

**成本**：每个 checkpoint 载入 1–2 min + 6 个拟合各 1–3 min（3000 步，每步一张 images_2 全图）≈ 12–18 min；room + kitchen × 2 λ ≈ 1 GPU-h，32 GB 够。nvdiffrast 第一次 import 会用 nvcc 编译 2–5 min（只一次）。

## §1 安装 nvdiffrast + 脚本 + CPU selftest（有网的 shell：login 节点或 VS Code 的 cpu 节点；整块粘贴，不会退出终端）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/diag $AN/commands $AN/texrefit-v1
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export TORCH_CUDA_ARCH_LIST="8.9" CUDA_HOME=/scratch/zr2245/conda-envs/pmsh-cu126   # L40S = sm_89; lets the CUDA extension build on a node without a GPU
python -c "import nvdiffrast, ninja; print('nvdiffrast', nvdiffrast.__version__, 'already installed')" 2>/dev/null \
  || { pip install ninja > $AN/texrefit-v1/pip-ninja.log 2>&1; mkdir -p $AN/texrefit-v1; pip install --no-build-isolation "https://github.com/NVlabs/nvdiffrast/archive/refs/heads/main.zip" > $AN/texrefit-v1/pip-nvdiffrast.log 2>&1; echo "pip exit $?"; }
python -c "import nvdiffrast, ninja; print('nvdiffrast', nvdiffrast.__version__, 'OK')" || { echo "FAILED: nvdiffrast install, log tail:"; tail -n 30 $AN/texrefit-v1/pip-nvdiffrast.log; }
cat > $AN/diag/texture_refit_v1.py <<'PYEOF_TEXREFIT'
"""texture_refit_v1.py (2026-10-30): probe P2 -- does colour need vertex density?  GPU, read-only on the model.

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

VERSION = 'texture_refit_v1 2026-10-30b'
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
    p.add_argument('--plane_labels', default=None); p.add_argument('--test_limit', type=int, default=None); p.add_argument('--train_eval_views', type=int, default=20)
    p.add_argument('--seed', type=int, default=17); p.add_argument('--selftest', action='store_true')
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
    log(f'{VERSION} | {a.label} | {n_v} v / {n_f} f from {sd_path}')

    data = SimpleNamespace(source_path=str(Path(a.dataset).resolve()), model_path=str(Path(a.model).resolve()), images=a.images,
                           resolution=1, eval=True, white_background=False, data_device='cpu', sh_degree=3)
    mesh = TriangleModel(3); scene = Scene(data, mesh, None, None, load_iteration=a.iteration, shuffle=False)
    if int(mesh.vertices.shape[0]) != n_v: sys.exit(f'FAILED: Scene loaded {int(mesh.vertices.shape[0])} vertices, state dict has {n_v}')
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

    results = dict(version=VERSION, label=a.label, model=str(sd_path), vertices=n_v, faces=n_f, image=[W, H], train_views=len(train), test_views=len(test),
                   steps=a.steps, lr=a.lr, lam_dssim=a.lam_dssim, axis=dict(sx=best[0], sy=best[1]), variants={})
    train_eval = [train[i] for i in rng.choice(len(train), min(a.train_eval_views, len(train)), replace=False)]
    for name in a.variants:
        kind, block = parse_variant(name)
        t0 = time.time(); log(f'== {name} ==')
        app = Appearance(kind, block, R, V, tri, dc, rest, dev)
        res = dict(app.counts)
        res.update(fit(app, R, train, ssim_fn, 0 if kind == 'vsh3_init' else a.steps, a.lr, a.lam_dssim, rng))
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
    print('selftest PASS', VERSION)


if __name__ == '__main__':
    main()
PYEOF_TEXREFIT
cat > $AN/commands/run_texrefit_v1.sh <<'SHEOF_TEXRUN'
#!/usr/bin/env bash
# run_texrefit_v1.sh (2026-10-30): probe P2 (diag/texture_refit_v1.py) on the frozen v3 meshes lam 6.2e-3 and lam 1.85e-3
# of each scene in SCENES (default "room kitchen"); per-vertex plane labels from P1b are passed when present.
# Restartable: a checkpoint whose json says "complete" is skipped.  Rows -> $AN/texrefit-v1.txt (TEXREFIT ...).
# Run on a GPU node:  bash run_texrefit_v1.sh      (env: SCENES, STEPS, VARIANTS, LAMS)
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; OUT=$AN/texrefit-v1; TAB=$AN/texrefit-v1.txt
PY=$AN/diag/texture_refit_v1.py; M360=/scratch/zr2245/datasets/m360; mkdir -p $OUT
SCENES=${SCENES:-"room kitchen"}; LAMS=${LAMS:-"0.0062 0.00185"}; STEPS=${STEPS:-3000}
VARIANTS=${VARIANTS:-"vsh3_init vsh3 vdc frgb tb2 tb4 tb4sh1"}
images_of () { case $1 in room|kitchen|counter|bonsai) echo images_2;; *) echo images_4;; esac; }
n_ok=0; n_fail=0; n_skip=0
for S in $SCENES; do
  for LAM in $LAMS; do
    T=v3-lam$LAM; MODEL=/scratch/zr2245/$S-rd3-lam$LAM-mu1.0/original
    J=$OUT/$S-$T.json; L=$OUT/$S-$T.log; PL=$AN/planarity-v1/$S-$T-plane-labels.npy
    if [ ! -f $MODEL/point_cloud/iteration_30000/point_cloud_state_dict.pt ]; then echo "missing: $S $T ($MODEL)"; n_skip=$((n_skip+1)); continue; fi
    if [ -s $J ] && grep -q '"complete": true' $J; then echo "exists: $S $T"; n_skip=$((n_skip+1)); continue; fi
    PLARG=""; [ -f $PL ] && PLARG="--plane_labels $PL" || echo "no plane labels for $S $T (planar split skipped)"
    echo "=== $S $T ($(date +%H:%M:%S)) ==="
    if python $PY --lab $LAB --model $MODEL --dataset $M360/$S --images $(images_of $S) --label "$S $T" --out $J --steps $STEPS --variants $VARIANTS $PLARG > $L 2>&1; then
      grep -h "^TEXREFIT" $L | tee -a $TAB; n_ok=$((n_ok+1))
    else echo "FAILED: $S $T"; tail -n 15 $L; n_fail=$((n_fail+1)); fi
  done
done
echo "===== $TAB ====="; cat $TAB 2>/dev/null
echo "run_texrefit_v1 done: ok $n_ok failed $n_fail skipped $n_skip  $(date)"
SHEOF_TEXRUN
cat > $AN/commands/texrefit-v1.sbatch <<'SBEOF_TEXRUN'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=texrefit1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/zr2245/analysis/texrefit1-%j.out
#SBATCH --error=/scratch/zr2245/analysis/texrefit1-%j.err
# texrefit-v1.sbatch (2026-10-30): probe P2 on room + kitchen (lam 6.2e-3 and 1.85e-3), ~1 GPU-h.  Same runner as the
# interactive path: bash /scratch/zr2245/analysis/commands/run_texrefit_v1.sh
echo "job $SLURM_JOB_ID on $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash /scratch/zr2245/analysis/commands/run_texrefit_v1.sh
SBEOF_TEXRUN
python -m py_compile $AN/diag/texture_refit_v1.py && python $AN/diag/texture_refit_v1.py --selftest || echo "FAILED: selftest"
```

## §2 GPU smoke（GPU 节点，Room λ6.2e-3，2 个变体 × 30 步，3 张 test 图；第一次会编译 nvdiffrast 的 CUDA 插件）

```bash
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
AN=/scratch/zr2245/analysis; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; mkdir -p $AN/texrefit-v1
python $AN/diag/texture_refit_v1.py --lab $LAB --model /scratch/zr2245/room-rd3-lam0.0062-mu1.0/original --dataset /scratch/zr2245/datasets/m360/room \
  --images images_2 --label "room smoke" --out $AN/texrefit-v1/smoke.json --variants vsh3_init tb2 --steps 30 --test_limit 3 \
  --plane_labels $AN/planarity-v1/room-v3-lam0.0062-plane-labels.npy 2>&1 | tee $AN/texrefit-v1/smoke.log | grep -v "^\s*$" | tail -n 25
```

看三行：`axis check ... -> xx dB`（四个符号组合里最好的应领先 ≥ 3 dB，脚本自己选）、`planar faces ... %`、两行 `TEXREFIT room smoke ...`。出现 `FAILED:` 就把 smoke.log 的尾巴贴给我。

## §3 正式跑（room + kitchen × λ6.2e-3 / λ1.85e-3，≈ 1 GPU-h）

在你手里的 GPU 节点上直接跑：

```bash
AN=/scratch/zr2245/analysis
bash $AN/commands/run_texrefit_v1.sh 2>&1 | tee -a $AN/texrefit-v1-run.log
```

或者交给 Slurm（general，L40S，32 GB，2 h）：

```bash
AN=/scratch/zr2245/analysis
sbatch $AN/commands/texrefit-v1.sbatch && squeue -u $USER
```

可重启：json 里有 `"complete": true` 的 checkpoint 会跳过。只跑一个场景或改步数：`SCENES="room" STEPS=2000 bash $AN/commands/run_texrefit_v1.sh`。

## §4 看结果

```bash
AN=/scratch/zr2245/analysis; cat $AN/texrefit-v1.txt; echo; grep -h "axis check\|planar faces" $AN/texrefit-v1/*.log; tail -2 $AN/texrefit-v1-run.log
```

把 `texrefit-v1.txt` 整个贴给我。
