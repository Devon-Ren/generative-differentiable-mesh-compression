"""Does vertex_weight (opacity) carry any visible information?  GPU, read-only on inputs.

Renders the Room test views with the official upstream renderer four times:
  original    : the trained per-vertex weights (baseline)
  floor       : every vertex at the opacity floor      (weight -> -inf, opacity = 0.999)
  one         : every vertex fully opaque              (weight -> +inf, opacity = 1.0)
  median      : every vertex at the median trained weight
and reports PSNR/SSIM/LPIPS per variant plus the max |pixel difference| against 'original'.
If all three constants match 'original' to within noise, the 2.1 MB vertex_weight stream
can be dropped from the bitstream (decoder substitutes the constant).

By default colours come from the independent SH of the checkpoint (official CUDA SH path).
With --asset <residual_k4096_r1024_f1/asset.mslb> (original vertex order, no quantization)
colours come from the shared dictionaries instead, through the same python-SH override path
run_experiment.py uses (gate: max error vs official forward <= 1e-4 on one view).

Run in the original-lossless-v1 .venv on a GPU node, from the directory holding this file:
  python vertex_weight_check.py --lab /scratch/zr2245/meshsplatting-original-lossless-v1 \
      --model /scratch/zr2245/room-fixed-sharing-0N4OSU/run/original \
      --dataset /scratch/zr2245/datasets/m360/room --images images_2 \
      --codec-dir /scratch/zr2245/directional-sharing-residual-v1 \
      --asset /scratch/zr2245/room-delivery-quant-0i2F5s/shared-attempt-L4zQIX/run/residual_k4096_r1024_f1/asset.mslb \
      --out vertex-weight-check.json
"""
import argparse, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lab', required=True, help='original-lossless-v1 directory (has upstream/)')
    p.add_argument('--model', required=True, help='original model directory (parent of point_cloud/)')
    p.add_argument('--dataset', required=True); p.add_argument('--images', default='images_2')
    p.add_argument('--iteration', type=int, default=30000)
    p.add_argument('--asset', help='shared asset.mslb in ORIGINAL vertex order (optional)')
    p.add_argument('--codec-dir', help='directory holding appearance.py + codec_benchmark.py (needed with --asset)')
    p.add_argument('--supersample', type=int, default=4, help='4 = what the sharing experiments used')
    p.add_argument('--split', choices=['test', 'validation'], default='test')
    p.add_argument('--png', help='optional directory for original/floor renders of the first view')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if not torch.cuda.is_available(): p.error('CUDA required')
    upstream = Path(a.lab).resolve() / 'upstream'
    sys.path.insert(0, str(upstream / 'submodules/effrdel/src')); sys.path.insert(0, str(upstream))
    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render
    from utils.sh_utils import eval_sh
    from utils.loss_utils import ssim
    import lpips
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

    data = SimpleNamespace(source_path=str(Path(a.dataset).resolve()), model_path=str(Path(a.model).resolve()), images=a.images,
                           resolution=1, eval=True, white_background=False, data_device='cpu', sh_degree=3)
    mesh = TriangleModel(3); mesh.scaling = a.supersample
    scene = Scene(data, mesh, None, None, load_iteration=a.iteration, shuffle=False)
    for v in vars(mesh).values():
        if isinstance(v, torch.Tensor): v.requires_grad_(False)
    if a.split == 'test':
        cams = sorted(scene.getTestCameras(), key=lambda c: (c.image_name, c.uid))
    else:
        cams = sorted(scene.getTrainCameras(), key=lambda c: (c.image_name, c.uid))[::8]
    if not cams: p.error('empty camera split')
    log(f'{len(cams)} {a.split} views, {len(mesh.vertices)} vertices, opacity_floor={mesh.opacity_floor}')
    bg = torch.zeros(3, device='cuda'); pipe = SimpleNamespace(debug=False, convert_SHs_python=False, depth_ratio=1.)
    perceptual = lpips.LPIPS(net='vgg').cuda().eval()
    for q in perceptual.parameters(): q.requires_grad_(False)

    override = None
    if a.asset:
        if not a.codec_dir: p.error('--asset needs --codec-dir')
        sys.path.insert(0, str(Path(a.codec_dir).resolve()))
        from appearance import expand_wire
        from codec_benchmark import unpack
        wire = unpack(a.asset)
        if len(wire['triangles_points']) != len(mesh.vertices): p.error('asset vertex count differs from checkpoint')
        if float(np.abs(np.asarray(wire['features_dc']) - mesh._features_dc.cpu().numpy()).max()) != 0.0: p.error('asset DC differs: not original order')
        rest = torch.from_numpy(expand_wire(wire)).cuda(); dc = mesh._features_dc
        def override(cam):
            d = mesh.vertices - cam.camera_center[None]; d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
            return (eval_sh(3, torch.cat((dc, rest), 1).transpose(1, 2), d) + .5).clamp_min(0).contiguous()
        # gate python-SH path against the official forward on the independent SH
        with torch.no_grad():
            d = mesh.vertices - cams[0].camera_center[None]; d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
            col = (eval_sh(3, torch.cat((dc, mesh._features_rest), 1).transpose(1, 2), d) + .5).clamp_min(0).contiguous()
            x = render(cams[0], mesh, pipe, bg, override_color=col)['render'].clamp(0, 1)
            y = render(cams[0], mesh, pipe, bg)['render'].clamp(0, 1)
            gate = float((x - y).abs().max())
        log('python-SH gate max error', gate)
        if gate > 1e-4: raise RuntimeError('python SH path mismatch')

    def draw(cam):
        kw = dict(override_color=override(cam)) if override else {}
        return render(cam, mesh, pipe, bg, **kw)['render'].clamp(0, 1)

    w0 = mesh.vertex_weight.detach().clone()
    variants = dict(original=w0, floor=torch.full_like(w0, -30.), one=torch.full_like(w0, 30.), median=torch.full_like(w0, float(w0.median())))
    report = dict(split=a.split, views=len(cams), supersample=a.supersample, opacity_floor=float(mesh.opacity_floor),
                  colours='shared asset ' + a.asset if a.asset else 'independent checkpoint SH',
                  weight_stats=dict(min=float(w0.min()), median=float(w0.median()), max=float(w0.max()),
                                    opacity_min=float(mesh.opacity_activation(w0).min()), opacity_max=float(mesh.opacity_activation(w0).max())),
                  variants={})
    reference = {}
    with torch.no_grad():
        for name, w in variants.items():
            mesh.vertex_weight = w.clone()
            rows = []
            for i, cam in enumerate(cams):
                im = draw(cam); gt = cam.original_image[:3].cuda().clamp(0, 1)
                row = dict(camera=cam.image_name, psnr=float(-10 * torch.log10((im - gt).square().mean().clamp_min(1e-12))),
                           ssim=float(ssim(im, gt)), lpips=float(perceptual(im[None] * 2 - 1, gt[None] * 2 - 1).mean()))
                if name == 'original': reference[i] = im.cpu()
                else:
                    diff = (im.cpu() - reference[i]).abs(); row['max_abs_diff'] = float(diff.max()); row['mean_abs_diff'] = float(diff.mean())
                    row['changed_pixels'] = int((diff.max(0).values > 1 / 255).sum())
                if a.png and i == 0:
                    from PIL import Image
                    Path(a.png).mkdir(parents=True, exist_ok=True)
                    Image.fromarray((im.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)).save(Path(a.png) / f'{name}.png')
                rows.append(row)
            keys = [k for k in rows[0] if k != 'camera']
            report['variants'][name] = dict(**{k: float(np.mean([r[k] for r in rows])) for k in keys},
                                            worst_view_psnr=float(min(r['psnr'] for r in rows)), per_view=rows)
            s = report['variants'][name]
            log(f"{name:9s} PSNR {s['psnr']:.4f}  SSIM {s['ssim']:.5f}  LPIPS {s['lpips']:.5f}" +
                (f"  max|diff| {s['max_abs_diff']:.2e}  changed px/view {s['changed_pixels']:.0f}" if name != 'original' else ''))
    mesh.vertex_weight = w0
    base = report['variants']['original']['psnr']
    report['verdict'] = {k: dict(delta_psnr=v['psnr'] - base) for k, v in report['variants'].items() if k != 'original'}
    Path(a.out).write_text(json.dumps(report, indent=2)); log('wrote', a.out)


if __name__ == '__main__':
    main()
