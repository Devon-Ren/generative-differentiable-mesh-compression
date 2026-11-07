"""envmap_fit_v1.py (2026-10-19): what does a background (environment) model buy a trained MeshSplatting model?  GPU, read-only.

MeshSplatting composites C + T * black: every pixel the opaque mesh does not cover shows black.  The error-anatomy run
(Room original) put 9 % of the test squared error on the 0.4 % of pixels with accumulated alpha < 0.99.  This script
fits, POST HOC and with the mesh frozen, a small learnable equirectangular environment map E(dir) and composites
    out = render + (1 - alpha) * E(dir)
on the training views, then reports test / validation PSNR before and after, the map's bytes, and which pixels changed.
Only pixels with alpha < --alpha_thr are touched, so PSNR after is computed exactly from the cached per-view SSE.
It is an upper-bound-style probe for the 'add an env map to the codec' decision (a few KB); if it pays, the map goes
into training (phase B, alpha detached) as a follow-up.

Self-check: the pixel -> world-ray convention is verified against projected vertices (all axis/transpose variants
tried, the one with < 0.5 deg mean error is used; anything else aborts).

Run (Room ~3 min, garden ~5 min on an L40S):
  python envmap_fit_v1.py --lab /scratch/zr2245/meshsplatting-original-lossless-v1 \
      --model /scratch/zr2245/room-fixed-sharing-0N4OSU/run/original --dataset /scratch/zr2245/datasets/m360/room \
      --images images_2 --out /scratch/zr2245/analysis/envmap/room-original
"""
import argparse, json, math, sys, time, zlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def psnr(mse):
    return float(-10 * math.log10(max(mse, 1e-12)))


def pixel_dirs(c, H, W, sx, sy, transpose, dev):
    """World-space unit ray directions for pixel centres of an H x W image of camera c."""
    py, px = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')
    ndc_x = (2 * px + 1) / W - 1; ndc_y = (2 * py + 1) / H - 1            # inverse of the rasterizer's ndc2Pix
    d = torch.stack([sx * ndc_x * math.tan(c.FoVx * .5), sy * ndc_y * math.tan(c.FoVy * .5), torch.ones_like(px)], -1).reshape(-1, 3)
    R = c.world_view_transform[:3, :3].cuda().float()
    d = d @ (R.T if transpose else R)
    return F.normalize(d, dim=-1).reshape(H, W, 3)


def equirect_uv(d):
    """unit dirs [N,3] -> grid_sample coords in [-1,1]: u = azimuth, v = elevation."""
    u = torch.atan2(d[:, 0], d[:, 2]) / math.pi                             # [-1,1]
    v = torch.asin(d[:, 1].clamp(-1, 1)) / (math.pi / 2)                    # [-1,1]
    return torch.stack([u, v], -1)


class EnvMap(torch.nn.Module):
    def __init__(self, h, w, init):
        super().__init__(); self.map = torch.nn.Parameter(init.view(3, 1, 1).repeat(1, h, w).clone())
    def forward(self, d):
        m = torch.cat([self.map, self.map[:, :, :1]], 2)                     # wrap azimuth
        g = equirect_uv(d).view(1, 1, -1, 2)
        return F.grid_sample(m[None], g, mode='bilinear', padding_mode='border', align_corners=True)[0, :, 0, :].T  # [N,3]
    def tv(self):
        return (self.map[:, 1:, :] - self.map[:, :-1, :]).abs().mean() + (self.map[:, :, 1:] - self.map[:, :, :-1]).abs().mean()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lab', required=True); p.add_argument('--model', required=True); p.add_argument('--dataset', required=True)
    p.add_argument('--images', default='images_2'); p.add_argument('--out', required=True)
    p.add_argument('--supersample', type=int, default=4); p.add_argument('--iteration', type=int, default=30000)
    p.add_argument('--alpha_thr', type=float, default=0.999); p.add_argument('--map_h', type=int, default=64); p.add_argument('--map_w', type=int, default=128)
    p.add_argument('--steps', type=int, default=1500); p.add_argument('--lr', type=float, default=0.02); p.add_argument('--batch', type=int, default=1_000_000)
    p.add_argument('--tv', type=float, default=1e-3); p.add_argument('--max_cached', type=int, default=60_000_000)
    p.add_argument('--seed', type=int, default=17); p.add_argument('--png', action='store_true')
    a = p.parse_args()
    if not torch.cuda.is_available(): p.error('CUDA required')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False); (out / 'config.json').write_text(json.dumps(vars(a), indent=2))
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    upstream = Path(a.lab).resolve() / 'upstream'
    if not (upstream / 'train.py').exists(): sys.exit(f'no upstream/train.py under {a.lab}')
    sys.path.insert(0, str(upstream / 'submodules/effrdel/src')); sys.path.insert(0, str(upstream))
    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render, compute_image_2d_pytorch_exact, transform_point_4x3

    data = SimpleNamespace(source_path=str(Path(a.dataset).resolve()), model_path=str(Path(a.model).resolve()), images=a.images,
                           resolution=1, eval=True, white_background=False, data_device='cpu', sh_degree=3)
    mesh = TriangleModel(3); scene = Scene(data, mesh, None, None, load_iteration=a.iteration, shuffle=False)
    train_all = sorted(scene.getTrainCameras(), key=lambda c: (c.image_name, c.uid))
    validation = train_all[::8]; train = [c for i, c in enumerate(train_all) if i % 8]
    test = sorted(scene.getTestCameras(), key=lambda c: (c.image_name, c.uid))
    nf = len(mesh._triangle_indices); dev = 'cuda'
    for name in ('image_size', 'importance_score', 'pixel_count'):
        t = getattr(mesh, name, None)
        if not isinstance(t, torch.Tensor) or t.numel() != nf: setattr(mesh, name, torch.zeros(nf, device=dev))
    mesh.scaling = a.supersample
    bg = torch.zeros(3, device=dev); pipe = SimpleNamespace(debug=False, convert_SHs_python=False, depth_ratio=1.)
    log(f'model {len(mesh.vertices)} v / {nf} f; train {len(train)} val {len(validation)} test {len(test)}; alpha_thr {a.alpha_thr}')

    # ---- ray convention self-check against projected vertices
    c0 = train[0]; H, W = int(c0.image_height), int(c0.image_width)
    with torch.no_grad():
        vs = mesh.vertices[torch.as_tensor(rng.integers(0, len(mesh.vertices), 4000), device=dev)]
        cam = transform_point_4x3(vs, c0.world_view_transform.cuda()); pix = compute_image_2d_pytorch_exact(vs, c0.full_proj_transform.cuda(), W, H)
        ok = (cam[:, 2] > 0.2) & (pix[:, 0] >= 0) & (pix[:, 0] < W - 1) & (pix[:, 1] >= 0) & (pix[:, 1] < H - 1)
        vs, pix = vs[ok], pix[ok]
        if len(vs) < 100: sys.exit('self-check: too few projected vertices')
        true_d = F.normalize(vs - c0.camera_center.cuda().float(), dim=-1)
        best = None
        for sx in (1, -1):
            for sy in (1, -1):
                for tr in (False, True):
                    D = pixel_dirs(c0, H, W, sx, sy, tr, dev)
                    d = D[pix[:, 1].round().long().clamp(0, H - 1), pix[:, 0].round().long().clamp(0, W - 1)]
                    ang = torch.rad2deg(torch.acos((d * true_d).sum(-1).clamp(-1, 1))).mean().item()
                    if best is None or ang < best[0]: best = (ang, sx, sy, tr)
        log(f'ray convention: mean angle error {best[0]:.3f} deg with sx {best[1]} sy {best[2]} transpose {best[3]}')
        if best[0] > 0.5: sys.exit('self-check FAILED: no pixel->ray convention matches projected vertices')
        _, SX, SY, TR = best

    # ---- cache the pixels an env map can change
    def cache(cams, tag):
        rows = []; dirs = []; ims = []; als = []; gts = []; total = 0
        for i, c in enumerate(cams):
            with torch.no_grad():
                pkg = render(c, mesh, pipe, bg); im = pkg['render'].clamp(0, 1); gt = c.original_image[:3].cuda().clamp(0, 1)
                al = pkg['rend_alpha'][0].clamp(0, 1); H, W = im.shape[1:]
                err = (im - gt).square().mean(0); sel = al < a.alpha_thr
                rows.append(dict(camera=c.image_name, pixels=H * W, sse=float(err.sum()), sel_pix=int(sel.sum()), sel_sse=float(err[sel].sum())))
                if sel.any():
                    D = pixel_dirs(c, H, W, SX, SY, TR, dev)
                    dirs.append(D[sel].half()); ims.append(im.permute(1, 2, 0)[sel].half()); als.append(al[sel].half()); gts.append(gt.permute(1, 2, 0)[sel].half())
                    total += int(sel.sum())
            if (i + 1) % 20 == 0 or i + 1 == len(cams): log(f'{tag}: cached {i + 1}/{len(cams)} views, {total} pixels')
        if total == 0: return rows, None
        D = torch.cat(dirs); I = torch.cat(ims); A = torch.cat(als); G = torch.cat(gts)
        # view index per pixel so test/val PSNR can be recomputed per view
        V = torch.cat([torch.full((r['sel_pix'],), k, device=dev, dtype=torch.int32) for k, r in enumerate(rows) if r['sel_pix'] > 0])
        return rows, (D, I, A, G, V)

    tr_rows, TR_ = cache(train, 'train'); va_rows, VA_ = cache(validation, 'validation'); te_rows, TE_ = cache(test, 'test')
    def split_stats(rows):
        N = sum(r['pixels'] for r in rows); S = sum(r['sse'] for r in rows); s = sum(r['sel_sse'] for r in rows); n = sum(r['sel_pix'] for r in rows)
        return dict(views=len(rows), psnr_mean=float(np.mean([psnr(r['sse'] / r['pixels']) for r in rows])), sel_pixel_share=n / N, sel_sse_share=s / S if S else 0.,
                    psnr_if_sel_zero=float(np.mean([psnr((r['sse'] - r['sel_sse']) / r['pixels']) for r in rows])))
    before = {k: split_stats(v) for k, v in (('train', tr_rows), ('validation', va_rows), ('test', te_rows))}
    for k, v in before.items(): log(f'{k}: PSNR {v["psnr_mean"]:.3f}, alpha<thr pixels {100 * v["sel_pixel_share"]:.2f} % carry {100 * v["sel_sse_share"]:.1f} % SSE, upper bound {v["psnr_if_sel_zero"]:.3f}')
    if TR_ is None: sys.exit('no training pixel has alpha below the threshold: nothing to fit')
    D, I, A, G, V = TR_; train_subsampled = False
    if len(D) > a.max_cached:
        keep = torch.as_tensor(rng.choice(len(D), a.max_cached, replace=False), device=dev); D, I, A, G, V = D[keep], I[keep], A[keep], G[keep], V[keep]
        train_subsampled = True; log(f'subsampled training cache to {a.max_cached} pixels (train PSNR-after will not be reported)')

    # ---- fit
    init = (G.float() - I.float()).clamp(0, 1).mean(0) if len(G) else torch.full((3,), .5, device=dev)
    env = EnvMap(a.map_h, a.map_w, init).to(dev); opt = torch.optim.Adam(env.parameters(), lr=a.lr)
    def composite(E, I_, A_):
        return I_.float() + (1 - A_.float())[:, None] * E
    def eval_split(C):
        if C is None: return None
        D_, I_, A_, G_, V_ = C
        with torch.no_grad():
            per = torch.zeros(int(V_.max()) + 1, device=dev)
            for s in range(0, len(D_), 4_000_000):
                e = (composite(env(D_[s:s + 4_000_000].float()), I_[s:s + 4_000_000], A_[s:s + 4_000_000]).clamp(0, 1) - G_[s:s + 4_000_000].float()).square().mean(1)
                per.index_add_(0, V_[s:s + 4_000_000].long(), e)
        return per.cpu().numpy()
    def split_after(rows, C):
        per = eval_split(C); ps = []
        for k, r in enumerate(rows):
            sel_after = float(per[k]) if per is not None and k < len(per) and r['sel_pix'] > 0 else r['sel_sse']
            ps.append(psnr((r['sse'] - r['sel_sse'] + sel_after) / r['pixels']))
        return dict(psnr_mean=float(np.mean(ps)), per_view=[dict(camera=r['camera'], psnr_after=p_, psnr_before=psnr(r['sse'] / r['pixels'])) for r, p_ in zip(rows, ps)])
    n = len(D); history = []
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, n, (min(a.batch, n),), device=dev)
        pred = composite(env(D[idx].float()), I[idx], A[idx])
        loss = (pred - G[idx].float()).abs().mean() + a.tv * env.tv()
        if not torch.isfinite(loss): sys.exit(f'nonfinite loss at step {step}')
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 250 == 0 or step == a.steps:
            va = split_after(va_rows, VA_)['psnr_mean']; history.append(dict(step=step, loss=float(loss), val_psnr=va))
            log(f'step {step} loss {float(loss):.5f} validation PSNR {va:.3f} (before {before["validation"]["psnr_mean"]:.3f})')
    after = {k: split_after(r, C) for k, r, C in (('train', tr_rows, None if train_subsampled else TR_), ('validation', va_rows, VA_), ('test', te_rows, TE_))}
    m16 = env.map.detach().cpu().half().numpy().tobytes(); raw = len(m16); z = len(zlib.compress(m16, 9))
    try:
        import zstandard; z = len(zstandard.ZstdCompressor(level=19).compress(m16)); zname = 'zstd'
    except Exception: zname = 'zlib'
    summary = dict(model=a.model, vertices=len(mesh.vertices), faces=nf, ray_convention=dict(sx=SX, sy=SY, transpose=TR), map=[a.map_h, a.map_w],
                   map_bytes_fp16=raw, map_bytes_compressed=z, compressor=zname, before=before,
                   after={k: dict(psnr_mean=v['psnr_mean']) for k, v in after.items()}, history=history,
                   per_view_test=after['test']['per_view'], per_view_validation=after['validation']['per_view'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=1)); torch.save(dict(map=env.map.detach().cpu(), convention=dict(sx=SX, sy=SY, transpose=TR)), out / 'envmap.pt')
    lines = [f'# env-map probe: {a.model}', '', f'vertices {len(mesh.vertices)}  faces {nf}  map {a.map_h}x{a.map_w} fp16 = {raw} B ({zname} {z} B)', '',
             '| split | views | PSNR before | alpha<thr pixels | their SSE share | upper bound (sel=0) | PSNR after env map | gain |', '|---|---|---|---|---|---|---|---|']
    for k in ('test', 'validation', 'train'):
        b, af = before[k], after[k]
        lines.append(f'| {k} | {b["views"]} | {b["psnr_mean"]:.3f} | {100 * b["sel_pixel_share"]:.2f} % | {100 * b["sel_sse_share"]:.1f} % | {b["psnr_if_sel_zero"]:.3f} | {af["psnr_mean"]:.3f} | {af["psnr_mean"] - b["psnr_mean"]:+.3f} |')
    worst = sorted(after['test']['per_view'], key=lambda r: r['psnr_before'])[:3]
    lines += ['', 'worst 3 test views (before -> after): ' + ', '.join(f'{r["camera"]} {r["psnr_before"]:.2f} -> {r["psnr_after"]:.2f}' for r in worst)]
    (out / 'envmap.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))
    if a.png:
        from PIL import Image
        m = env.map.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy(); Image.fromarray((m * 255).round().astype(np.uint8)).resize((a.map_w * 4, a.map_h * 4), Image.NEAREST).save(out / 'envmap.png')
        c = next(c for c in test if c.image_name == worst[0]['camera'])
        with torch.no_grad():
            pkg = render(c, mesh, pipe, bg); im = pkg['render'].clamp(0, 1); al = pkg['rend_alpha'][0].clamp(0, 1); Hh, Ww = im.shape[1:]
            E = env(pixel_dirs(c, Hh, Ww, SX, SY, TR, dev).reshape(-1, 3)).T.reshape(3, Hh, Ww)
            comp = (im + (1 - al)[None] * E).clamp(0, 1)
            for tag, x in (('before', im), ('after', comp), ('gt', c.original_image[:3].cuda().clamp(0, 1))):
                Image.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(out / f'worst-{tag}.png')


if __name__ == '__main__':
    main()
