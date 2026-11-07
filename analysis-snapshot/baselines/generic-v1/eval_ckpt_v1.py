"""eval_ckpt_v1.py (2026-10-24): render a (decoded) checkpoint on the test views with the upstream renderer, 4x supersample.
Cameras/splits come from the ORIGINAL run dir (same as every Stage-2 number); the mesh tensors are swapped in from --ckpt
(triangles_points, _triangle_indices, features_dc, features_rest, vertex_weight; sigma / sh degree / opacity floor stay
the original's).  Prints one row and writes <out>.json (mean PSNR / SSIM / LPIPS-vgg, worst view, per-view rows).
  python eval_ckpt_v1.py --lab /scratch/zr2245/meshsplatting-original-lossless-v1 --model <run>/original \
      --dataset /scratch/zr2245/datasets/m360/room --images images_2 --ckpt <dir>/point_cloud_state_dict.pt --out <dir>/eval.json
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
    p.add_argument('--lab', required=True); p.add_argument('--model', required=True); p.add_argument('--dataset', required=True)
    p.add_argument('--images', default='images_2'); p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True)
    p.add_argument('--supersample', type=int, default=4); p.add_argument('--label', default='')
    a = p.parse_args()
    if not torch.cuda.is_available(): sys.exit('FAILED: CUDA required (run on the GPU node)')
    upstream = Path(a.lab).resolve() / 'upstream'
    sys.path.insert(0, str(upstream / 'submodules/effrdel/src')); sys.path.insert(0, str(upstream))
    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render
    from utils.loss_utils import ssim
    import lpips
    data = SimpleNamespace(source_path=str(Path(a.dataset).resolve()), model_path=str(Path(a.model).resolve()), images=a.images,
                           resolution=1, eval=True, white_background=False, data_device='cpu', sh_degree=3)
    mesh = TriangleModel(3); scene = Scene(data, mesh, None, None, load_iteration=30000, shuffle=False)
    test = sorted(scene.getTestCameras(), key=lambda c: (c.image_name, c.uid))
    w = torch.load(a.ckpt, map_location='cpu', weights_only=False); dev = 'cuda'
    mesh.vertices = w['triangles_points'].to(dev).float().contiguous()
    mesh._triangle_indices = w['_triangle_indices'].to(dev).to(torch.int32).contiguous()
    mesh.vertex_weight = w['vertex_weight'].to(dev).float().reshape(-1, 1).contiguous()
    mesh._features_dc = w['features_dc'].to(dev).float().reshape(len(mesh.vertices), 1, 3).contiguous()
    mesh._features_rest = w['features_rest'].to(dev).float().reshape(len(mesh.vertices), 15, 3).contiguous()
    nf = len(mesh._triangle_indices)
    for name in ('image_size', 'importance_score', 'pixel_count'):
        setattr(mesh, name, torch.zeros(nf, device=dev))
    mesh.scaling = a.supersample; del w
    log(f'{a.label or a.ckpt}: {len(mesh.vertices)} vertices, {nf} faces, {len(test)} test views')
    bg = torch.zeros(3, device=dev); pipe = SimpleNamespace(debug=False, convert_SHs_python=False, depth_ratio=1.)
    perceptual = lpips.LPIPS(net='vgg').cuda().eval()
    rows = []
    with torch.no_grad():
        for c in test:
            im = render(c, mesh, pipe, bg)['render'].clamp(0, 1); gt = c.original_image[:3].cuda().clamp(0, 1)
            rows.append(dict(camera=c.image_name, psnr=float(-10 * torch.log10((im - gt).square().mean().clamp_min(1e-12))), ssim=float(ssim(im, gt)),
                             lpips=float(perceptual(im[None] * 2 - 1, gt[None] * 2 - 1).mean())))
    res = dict(label=a.label, ckpt=a.ckpt, vertices=len(mesh.vertices), faces=nf, views=len(rows),
               psnr=float(np.mean([r['psnr'] for r in rows])), ssim=float(np.mean([r['ssim'] for r in rows])), lpips=float(np.mean([r['lpips'] for r in rows])),
               worst_view_psnr=float(min(r['psnr'] for r in rows)), per_view=rows)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f'EVAL {a.label} | V {res["vertices"]} F {nf} | test {res["psnr"]:.3f} / {res["ssim"]:.4f} / {res["lpips"]:.4f} | worst {res["worst_view_psnr"]:.2f}', flush=True)


if __name__ == '__main__':
    main()
