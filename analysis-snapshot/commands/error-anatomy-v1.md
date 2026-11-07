# error-anatomy-v1（2026-10-19）：MeshSplatting 的 PSNR 丢在哪？Room 两个模型，约 10 分钟 GPU，在你手里的 GPU 节点上直接跑（不用 sbatch）

一次粘贴：写 `$AN/diag/error_anatomy_v1.py`，对 Room 原始 dense 模型和 λ=6.2e-3 的 v3 模型各跑一遍（test 全部 + train[::8] 验证集），输出 `anatomy.md`（一张表）和最差 3 个 test 视角的 error map / bucket 图。失败会直接报错退出。

```bash
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
AN=/scratch/zr2245/analysis; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; DATA=/scratch/zr2245/datasets/m360/room; IMAGES=images_2
mkdir -p $AN/diag $AN/anatomy
cp /dev/stdin $AN/diag/error_anatomy_v1.py <<'PYEOF'
"""error_anatomy_v1.py (2026-10-19): where does a MeshSplatting model lose PSNR?  GPU, read-only on inputs.

Renders every test view (and the train[::8] validation subset, same split as all Stage-2 runs) of a trained model with
the ORIGINAL upstream renderer at the upstream evaluation setting (4x supersampling), and splits the squared error of
each view into exclusive pixel buckets:

  background   accumulated alpha < 0.99 after the opaque schedule (holes / sky / far field the mesh never covered)
  silhouette   a 4x4 supersample block in which the dominant triangle id changes AND depth jumps > 5 %  (object edges)
  seam         block in which the id changes WITHOUT a depth jump  (tessellation seams: near-soup mesh, Gouraud colour)
  interior-hf  interior pixel whose ground truth is high-frequency (|Laplacian| above the scene's 75th percentile)
  interior-lf  the rest

For each bucket: pixel share, share of the summed squared error, and the PSNR the model WOULD reach if that bucket's
error were zero (an upper bound on what fixing that failure mode can buy).  Two more upper bounds per view:
  affine       per-view 3x4 affine colour correction fitted by least squares (what a per-image appearance / exposure
               embedding could buy at most)
  lowpass s    PSNR after Gaussian blur (sigma s px) of render and GT: if the gap to 3DGS survives the blur, the loss is
               not in fine detail
And the error as a function of the dominant triangle's screen footprint (pixels at output resolution), to separate
'triangles too big for the texture' from 'geometry / edge' failures.

Output: <out>/anatomy.json (per view + means), <out>/anatomy.md (one table), optional --png error maps of the worst
3 test views.  Run (Room, ~5 min on an L40S):
  python error_anatomy_v1.py --lab /scratch/zr2245/meshsplatting-original-lossless-v1 \
      --model /scratch/zr2245/room-fixed-sharing-0N4OSU/run/original --dataset /scratch/zr2245/datasets/m360/room \
      --images images_2 --out /scratch/zr2245/analysis/anatomy/room-original --png
"""
import argparse, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def psnr_from_mse(m):
    return float(-10 * np.log10(max(m, 1e-12)))


def gaussian_blur(x, sigma):
    r = int(3 * sigma); k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-k.square() / (2 * sigma * sigma)); k = k / k.sum()
    c = x.shape[0]; x = x[None]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='reflect'), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode='reflect'), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return x[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lab', required=True); p.add_argument('--model', required=True, help='run dir holding point_cloud/iteration_30000')
    p.add_argument('--dataset', required=True); p.add_argument('--images', default='images_2')
    p.add_argument('--out', required=True); p.add_argument('--supersample', type=int, default=4)
    p.add_argument('--iteration', type=int, default=30000)
    p.add_argument('--depth_jump', type=float, default=0.05); p.add_argument('--hf_quantile', type=float, default=0.75)
    p.add_argument('--png', action='store_true'); p.add_argument('--max_views', type=int, default=0, help='0 = all')
    a = p.parse_args()
    if not torch.cuda.is_available(): p.error('CUDA required')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False); (out / 'config.json').write_text(json.dumps(vars(a), indent=2))
    upstream = Path(a.lab).resolve() / 'upstream'
    if not (upstream / 'train.py').exists(): sys.exit(f'no upstream/train.py under {a.lab}')
    sys.path.insert(0, str(upstream / 'submodules/effrdel/src')); sys.path.insert(0, str(upstream))
    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render
    from utils.loss_utils import ssim

    data = SimpleNamespace(source_path=str(Path(a.dataset).resolve()), model_path=str(Path(a.model).resolve()), images=a.images,
                           resolution=1, eval=True, white_background=False, data_device='cpu', sh_degree=3)
    mesh = TriangleModel(3); scene = Scene(data, mesh, None, None, load_iteration=a.iteration, shuffle=False)
    train_all = sorted(scene.getTrainCameras(), key=lambda c: (c.image_name, c.uid))
    validation = train_all[::8]; test = sorted(scene.getTestCameras(), key=lambda c: (c.image_name, c.uid))
    if a.max_views: validation, test = validation[:a.max_views], test[:a.max_views]
    nf = len(mesh._triangle_indices); dev = 'cuda'
    for name in ('image_size', 'importance_score', 'pixel_count'):
        t = getattr(mesh, name, None)
        if not isinstance(t, torch.Tensor) or t.numel() != nf: setattr(mesh, name, torch.zeros(nf, device=dev))
    mesh.scaling = a.supersample
    log(f'model: {len(mesh.vertices)} vertices, {nf} faces, sigma {float(mesh.get_sigma):.2e}, opacity_floor {mesh.opacity_floor}; '
        f'val {len(validation)} / test {len(test)} views; supersample {a.supersample}')
    bg = torch.zeros(3, device=dev); pipe = SimpleNamespace(debug=False, convert_SHs_python=False, depth_ratio=1.)
    S = a.supersample
    bucket_names = ['background', 'silhouette', 'seam', 'interior_hf', 'interior_lf']
    area_edges = [0, 1, 4, 16, 64, float('inf')]

    @torch.no_grad()
    def analyse(c, png_prefix=None):
        pkg = render(c, mesh, pipe, bg)
        im = pkg['render'].clamp(0, 1); gt = c.original_image[:3].cuda().clamp(0, 1)
        H, W = im.shape[1:]; N = H * W
        err = (im - gt).square().mean(0)                                             # [H,W] per-pixel MSE over channels
        sse = float(err.sum()); mse = sse / N
        alpha = pkg['rend_alpha'][0]                                                   # [H,W] area-downsampled
        ids = pkg['rend_ids'][0]; dfull = pkg['depth_full'][0]                         # [S*H, S*W]
        # id change / depth jump at full resolution, then any-in-block -> output resolution
        idx_ = (ids[:, 1:] != ids[:, :-1]); idy = (ids[1:, :] != ids[:-1, :])
        dref = dfull.abs().clamp_min(1e-6)
        djx = ((dfull[:, 1:] - dfull[:, :-1]).abs() / torch.maximum(dref[:, 1:], dref[:, :-1])) > a.depth_jump
        djy = ((dfull[1:, :] - dfull[:-1, :]).abs() / torch.maximum(dref[1:, :], dref[:-1, :])) > a.depth_jump
        sil = torch.zeros_like(ids, dtype=torch.bool); seam = torch.zeros_like(ids, dtype=torch.bool)
        sil[:, 1:] |= idx_ & djx; sil[1:, :] |= idy & djy
        seam[:, 1:] |= idx_ & ~djx; seam[1:, :] |= idy & ~djy
        pool = lambda m: F.max_pool2d(m.float()[None, None], S)[0, 0] > 0
        sil_m = pool(sil); seam_m = pool(seam) & ~sil_m
        bg_m = alpha < 0.99
        gray = gt.mean(0, keepdim=True)
        lap = F.conv2d(F.pad(gray[None], (1, 1, 1, 1), mode='reflect'), torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=dev, dtype=gray.dtype).view(1, 1, 3, 3))[0, 0].abs()
        lap = gaussian_blur(lap[None], 1.0)[0]
        interior = ~(bg_m | sil_m | seam_m)
        thr = torch.quantile(lap[interior].flatten()[:: max(1, int(interior.sum()) // 2_000_000)], a.hf_quantile) if interior.any() else lap.max()
        hf_m = interior & (lap > thr); lf_m = interior & ~hf_m
        masks = dict(background=bg_m, silhouette=sil_m & ~bg_m, seam=seam_m & ~bg_m, interior_hf=hf_m, interior_lf=lf_m)
        row = dict(camera=c.image_name, psnr=psnr_from_mse(mse), ssim=float(ssim(im, gt)), pixels=N)
        for k, m in masks.items():
            s = float(err[m].sum())
            row[f'{k}_pix'] = float(m.float().mean()); row[f'{k}_sse'] = s / sse if sse > 0 else 0.
            row[f'{k}_psnr_if_zero'] = psnr_from_mse((sse - s) / N)
        # triangle footprint of the dominant triangle (output pixels), from the supersampled pixel counts
        cnt = pkg['triangle_was_rendered'].float() / (S * S)
        id_lr = ids[S // 2::S, S // 2::S].long().clamp(0, nf - 1)[:H, :W]
        foot = cnt[id_lr]
        for lo, hi in zip(area_edges[:-1], area_edges[1:]):
            m = (foot >= lo) & (foot < hi) & interior
            row[f'foot_{lo}_{hi}_pix'] = float(m.float().mean()); row[f'foot_{lo}_{hi}_mse'] = float(err[m].mean()) if m.any() else float('nan')
        # per-view affine colour correction upper bound
        X = torch.cat([im.reshape(3, -1), torch.ones(1, N, device=dev)], 0).T; Y = gt.reshape(3, -1).T
        sel = torch.arange(0, N, max(1, N // 500_000), device=dev)
        A = torch.linalg.lstsq(X[sel].double(), Y[sel].double()).solution.float()
        row['psnr_affine'] = psnr_from_mse(float((X @ A - Y).square().mean()))
        row['affine_gain'] = row['psnr_affine'] - row['psnr']
        for s in (2.0, 4.0):
            row[f'psnr_lowpass_{int(s)}'] = psnr_from_mse(float((gaussian_blur(im, s) - gaussian_blur(gt, s)).square().mean()))
        if png_prefix:
            from PIL import Image
            def save(x, name): Image.fromarray((x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(f'{png_prefix}-{name}.png')
            save(im, 'render'); save(gt, 'gt'); save((err.sqrt() * 4).clamp(0, 1)[None].repeat(3, 1, 1), 'abs-error-x4')
            lab = torch.zeros(3, H, W, device=dev); col = dict(background=(0, 0, 0), silhouette=(1, 0, 0), seam=(1, 0.6, 0), interior_hf=(0, 0.5, 1), interior_lf=(0.8, 0.8, 0.8))
            for k, m in masks.items():
                for ch in range(3): lab[ch][m] = col[k][ch]
            save(lab, 'buckets')
        return row

    results = {}
    for split, cams in (('test', test), ('validation', validation)):
        rows = []
        for i, c in enumerate(cams):
            rows.append(analyse(c)); log(f'{split} {i + 1}/{len(cams)} {c.image_name} psnr {rows[-1]["psnr"]:.3f} '
                                      f'bg {rows[-1]["background_sse"]:.2f} sil {rows[-1]["silhouette_sse"]:.2f} seam {rows[-1]["seam_sse"]:.2f} '
                                      f'hf {rows[-1]["interior_hf_sse"]:.2f} lf {rows[-1]["interior_lf_sse"]:.2f} affine +{rows[-1]["affine_gain"]:.2f}')
        keys = [k for k in rows[0] if k != 'camera']
        mean = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
        # bucket PSNR-if-zero on the split's pooled error (mean MSE), which is what the headline PSNR is not, so give both
        results[split] = dict(mean=mean, per_view=rows)
    if a.png:
        worst = sorted(results['test']['per_view'], key=lambda r: r['psnr'])[:3]
        (out / 'png').mkdir(exist_ok=True)
        for r in worst:
            c = next(c for c in test if c.image_name == r['camera']); analyse(c, str(out / 'png' / r['camera']))
    (out / 'anatomy.json').write_text(json.dumps(dict(model=a.model, dataset=a.dataset, images=a.images, vertices=len(mesh.vertices), faces=nf,
                                                       supersample=S, results=results), indent=1))
    t, v = results['test']['mean'], results['validation']['mean']
    lines = [f'# error anatomy: {a.model}', '', f'vertices {len(mesh.vertices)}  faces {nf}  supersample {S}', '',
             f'| split | views | PSNR | SSIM | affine UB | lowpass2 | lowpass4 |', '|---|---|---|---|---|---|---|']
    for name, m, n in (('test', t, len(test)), ('validation (train[::8])', v, len(validation))):
        lines.append(f'| {name} | {n} | {m["psnr"]:.3f} | {m["ssim"]:.4f} | {m["psnr_affine"]:.3f} (+{m["affine_gain"]:.2f}) | {m["psnr_lowpass_2"]:.3f} | {m["psnr_lowpass_4"]:.3f} |')
    lines += ['', f'train/test gap (validation - test): {v["psnr"] - t["psnr"]:.3f} dB', '',
              '| bucket (test mean) | pixel share | SSE share | PSNR if bucket error were zero |', '|---|---|---|---|']
    for k in bucket_names:
        lines.append(f'| {k} | {100 * t[f"{k}_pix"]:.1f} % | {100 * t[f"{k}_sse"]:.1f} % | {t[f"{k}_psnr_if_zero"]:.3f} (+{t[f"{k}_psnr_if_zero"] - t["psnr"]:.2f}) |')
    lines += ['', '| dominant-triangle footprint (output px, interior only) | pixel share | MSE |', '|---|---|---|']
    for lo, hi in zip(area_edges[:-1], area_edges[1:]):
        lines.append(f'| [{lo}, {hi}) | {100 * t[f"foot_{lo}_{hi}_pix"]:.1f} % | {t[f"foot_{lo}_{hi}_mse"]:.5f} |')
    (out / 'anatomy.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    main()
PYEOF
python -m py_compile $AN/diag/error_anatomy_v1.py && echo "error_anatomy_v1 compiled"
set -e
for TAG_MODEL in "room-original:/scratch/zr2245/room-fixed-sharing-0N4OSU/run/original" "room-rd3-lam0.0062:/scratch/zr2245/room-rd3-lam0.0062-mu1.0/original"; do
  TAG=${TAG_MODEL%%:*}; MODEL=${TAG_MODEL#*:}
  [ -d "$MODEL/point_cloud/iteration_30000" ] || { echo "MISSING $MODEL/point_cloud/iteration_30000"; exit 1; }
  OUT=$AN/anatomy/$TAG-$(date +%m%d%H%M)
  python $AN/diag/error_anatomy_v1.py --lab $LAB --model $MODEL --dataset $DATA --images $IMAGES --out $OUT --png 2>&1 | tee $AN/anatomy/$TAG.log
  echo "== $TAG -> $OUT/anatomy.md"
done
cat $AN/anatomy/room-original-*/anatomy.md $AN/anatomy/room-rd3-lam0.0062-*/anatomy.md
```

把最后 `cat` 打印的两张表贴回来即可（PNG 不用传）。

**怎么读**：
- `train/test gap`：验证集（训练视角子集）比 test 高多少。Room λ=0 的 sweep 日志已经显示 30.87 vs 28.37（2.5 dB），而 226k 顶点的 λ=3e-2 只差 1.4 dB，所以要看它随顶点数的变化。
- `PSNR if bucket error were zero`：把某一类像素的误差全部清零能换回多少 dB，是修那一类问题的上界。哪个 bucket 的上界 ≥ 1 dB，哪个才值得做实验；都 < 0.5 dB 就说明损失是弥散的，没有单一"错配"。
- `affine UB`：每张图一个 3×4 颜色仿射校正的上界（曝光 / appearance embedding 能买到的上限）。
- `lowpass`：模糊后 PSNR 若仍比 3DGS 低很多，说明差距不在高频细节，而在低频（曝光、背景、大块几何错）。
- footprint 表：误差随主导三角形的屏幕面积变化；大三角形 MSE 高 = 纹理分辨率被顶点密度绑死；小三角形 MSE 高 = 几何/边缘问题。
