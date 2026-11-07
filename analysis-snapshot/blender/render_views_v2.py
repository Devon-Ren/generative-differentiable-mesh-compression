"""render_views_v2.py (VERSION render_views_v2 2026-11-04): render chosen TEST views of a MeshSplatting model and write render / GT / error images.

v2: --all_views (every test camera), --no_png (metrics only), SSIM per view, mean PSNR/SSIM line "RENDER_MEAN ..." for supersampling comparisons.

Loads the model through upstream's own Scene(load_iteration=N) (black background, like the delivery chain), renders the
requested test views at a given supersampling, and writes per view: <out>/view-<name>-render.png, -gt.png, -err.png
(|render-gt| summed over RGB, x4, clipped) plus a montage.png (one row per view: render | GT | error) and psnr.txt.

Usage (GPU node):
  python render_views_v1.py --lab $LAB --dataset $DATA --model $G-best/original --iteration 30000 --supersample 2 --views 0 25 50 75 100 125 150 175 --out $G-best/renders
"""
import argparse, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

VERSION = 'render_views_v2 2026-11-04'


def to_u8(x):
    return (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lab', required=True); ap.add_argument('--dataset', required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--iteration', type=int, default=30000); ap.add_argument('--supersample', type=int, default=2)
    ap.add_argument('--views', nargs='+', type=int, default=[0, 25, 50, 75, 100, 125, 150, 175], help='indices into the sorted test cameras')
    ap.add_argument('--err_gain', type=float, default=4.0); ap.add_argument('--out', required=True)
    ap.add_argument('--all_views', action='store_true'); ap.add_argument('--no_png', action='store_true', help='metrics only, no images')
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit('FAILED: CUDA required')
    lab = Path(a.lab).resolve(); up = lab / 'upstream'
    sys.path.insert(0, str(up / 'submodules/effrdel/src')); sys.path.insert(0, str(up))
    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render
    from PIL import Image
    from utils.loss_utils import ssim
    data = SimpleNamespace(source_path=str(Path(a.dataset).resolve()), model_path=str(Path(a.model).resolve()), images='train', resolution=1, eval=True,
                           white_background=False, data_device='cpu', sh_degree=3)
    mesh = TriangleModel(3); scene = Scene(data, mesh, None, None, load_iteration=a.iteration, shuffle=False)
    cams = sorted(scene.getTestCameras(), key=lambda c: (c.image_name, c.uid))
    print(f'[render] {VERSION} | model {a.model} iter {a.iteration} | vertices {int(mesh.vertices.shape[0])} faces {int(mesh._triangle_indices.shape[0])} | test views {len(cams)} | ss {a.supersample}', flush=True)
    mesh.scaling = a.supersample
    bg = torch.zeros(3, device='cuda'); pipe = SimpleNamespace(debug=False, convert_SHs_python=False, depth_ratio=1.)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, lines, ps, ss_ = [], [], [], []
    views = list(range(len(cams))) if a.all_views else a.views
    with torch.no_grad():
        for i in views:
            if i >= len(cams):
                print(f'[render] skip view index {i} (only {len(cams)} test views)'); continue
            c = cams[i]
            im = render(c, mesh, pipe, bg)['render'].clamp(0, 1); gt = c.original_image[:3].cuda().clamp(0, 1)
            psnr = float(-10 * torch.log10((im - gt).square().mean().clamp_min(1e-12))); s_ = float(ssim(im, gt))
            ps.append(psnr); ss_.append(s_); lines.append(f'{c.image_name} psnr {psnr:.2f} ssim {s_:.4f}')
            if not a.no_png:
                err = ((im - gt).abs().sum(0, keepdim=True) * a.err_gain / 3).clamp(0, 1).repeat(3, 1, 1)
                r, g, e = to_u8(im), to_u8(gt), to_u8(err)
                for tag, x in (('render', r), ('gt', g), ('err', e)):
                    Image.fromarray(x).save(out / f'view-{c.image_name}-{tag}.png')
                rows.append(np.concatenate([r, g, e], axis=1))
            if not a.all_views or i % 25 == 0:
                print(f'[render] {c.image_name}: PSNR {psnr:.2f} SSIM {s_:.4f}', flush=True)
    if not ps:
        sys.exit('FAILED: no views rendered')
    if rows:
        Image.fromarray(np.concatenate(rows, axis=0)).save(out / 'montage.png')
    (out / f'psnr-ss{a.supersample}.txt').write_text('\n'.join(lines) + '\n')
    print(f'RENDER_MEAN model {a.model} iter {a.iteration} ss {a.supersample} views {len(ps)} psnr {np.mean(ps):.3f} ssim {np.mean(ss_):.4f} worst {min(ps):.2f}', flush=True)


if __name__ == '__main__':
    main()
