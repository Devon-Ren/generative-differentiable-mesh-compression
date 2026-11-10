"""render_ckpt_v1.py (VERSION render_ckpt_v1 2026-11-10): render chosen TEST views of several checkpoints side by side (render_views_v1 + the checkpoint
loading of eval_ckpt_v1).  Cameras come from upstream's Scene(load_iteration=30000) of --model (any model dir of the scene); each --ckpt NAME=PATH is a
state dict with triangles_points, _triangle_indices, vertex_weight, features_dc, features_rest (an FP32 original, or a decoded asset from
decode_asset_v1.py).  Black background, supersample as the metrics (4).  Writes per view and checkpoint: view-<cam>-<name>.png, view-<cam>-<name>-err.png,
view-<cam>-gt.png; montage.png (one row per view: GT | ckpt1 | ckpt2 ...; the error row beneath) and psnr.txt.
  python render_ckpt_v1.py --lab $LAB --dataset $DATA --images images_2 --model /scratch/zr2245/room-indoor/original --views 2 10 \
     --ckpt FP32=/.../point_cloud_state_dict.pt --ckpt 20MB=/.../decoded.pt ... --out /scratch/zr2245/renders-b23/room
"""
import argparse, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

VERSION = 'render_ckpt_v1 2026-11-10'


def to_u8(x): return (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lab', required=True); ap.add_argument('--dataset', required=True); ap.add_argument('--images', default='images_2'); ap.add_argument('--model', required=True)
    ap.add_argument('--ckpt', action='append', required=True, help='NAME=PATH, repeatable, in display order'); ap.add_argument('--views', nargs='+', type=int, default=[2, 10])
    ap.add_argument('--supersample', type=int, default=4); ap.add_argument('--err_gain', type=float, default=4.0); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if not torch.cuda.is_available(): sys.exit('FAILED: CUDA required')
    up = Path(a.lab).resolve() / 'upstream'; sys.path.insert(0, str(up / 'submodules/effrdel/src')); sys.path.insert(0, str(up))
    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render
    from PIL import Image, ImageDraw
    data = SimpleNamespace(source_path=str(Path(a.dataset).resolve()), model_path=str(Path(a.model).resolve()), images=a.images, resolution=1, eval=True, white_background=False, data_device='cpu', sh_degree=3)
    mesh = TriangleModel(3); scene = Scene(data, mesh, None, None, load_iteration=30000, shuffle=False)
    cams = sorted(scene.getTestCameras(), key=lambda c: (c.image_name, c.uid)); dev = 'cuda'
    names = [c.split('=', 1)[0] for c in a.ckpt]; paths = [c.split('=', 1)[1] for c in a.ckpt]
    print(f'[render] {VERSION} | {len(cams)} test views, rendering {a.views} | ss {a.supersample} | columns GT + {names}', flush=True)
    bg = torch.zeros(3, device=dev); pipe = SimpleNamespace(debug=False, convert_SHs_python=False, depth_ratio=1.)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    views = [i for i in a.views if i < len(cams)]
    gts = {}; renders = {n: {} for n in names}; errs = {n: {} for n in names}; lines = []
    for n, p in zip(names, paths):
        w = torch.load(p, map_location='cpu', weights_only=False); nv = len(w['triangles_points'])
        mesh.vertices = w['triangles_points'].to(dev).float().reshape(nv, 3).contiguous()
        mesh._triangle_indices = w['_triangle_indices'].to(dev).to(torch.int32).reshape(-1, 3).contiguous()
        mesh.vertex_weight = w['vertex_weight'].to(dev).float().reshape(-1, 1).contiguous()
        mesh._features_dc = w['features_dc'].to(dev).float().reshape(nv, 1, 3).contiguous()
        mesh._features_rest = w['features_rest'].to(dev).float().reshape(nv, 15, 3).contiguous()
        nf = len(mesh._triangle_indices)
        for k in ('image_size', 'importance_score', 'pixel_count'): setattr(mesh, k, torch.zeros(nf, device=dev))
        mesh.scaling = a.supersample; del w
        print(f'[render] {n}: {nv} vertices {nf} faces', flush=True)
        with torch.no_grad():
            for i in views:
                c = cams[i]; im = render(c, mesh, pipe, bg)['render'].clamp(0, 1); gt = c.original_image[:3].cuda().clamp(0, 1)
                psnr = float(-10 * torch.log10((im - gt).square().mean().clamp_min(1e-12)))
                err = ((im - gt).abs().sum(0, keepdim=True) * a.err_gain / 3).clamp(0, 1).repeat(3, 1, 1)
                renders[n][i] = to_u8(im); errs[n][i] = to_u8(err); gts[i] = to_u8(gt)
                Image.fromarray(renders[n][i]).save(out / f'view-{c.image_name}-{n}.png'); Image.fromarray(errs[n][i]).save(out / f'view-{c.image_name}-{n}-err.png')
                lines.append(f'{c.image_name} {n} psnr {psnr:.2f}'); print(f'[render] {c.image_name} {n}: PSNR {psnr:.2f}', flush=True)
        torch.cuda.empty_cache()
    rows = []
    for i in views:
        c = cams[i]; Image.fromarray(gts[i]).save(out / f'view-{c.image_name}-gt.png')
        top = [gts[i]] + [renders[n][i] for n in names]; bot = [np.zeros_like(gts[i])] + [errs[n][i] for n in names]
        rows.append(np.concatenate(top, axis=1)); rows.append(np.concatenate(bot, axis=1))
    mont = Image.fromarray(np.concatenate(rows, axis=0)); d = ImageDraw.Draw(mont); W = gts[views[0]].shape[1]
    for j, n in enumerate(['GT'] + names): d.text((j * W + 8, 8), n, fill=(255, 255, 0))
    mont.save(out / 'montage.png'); (out / 'psnr.txt').write_text('\n'.join(lines) + '\n')
    print(f'[render] wrote {out}/montage.png ({len(views)} views x [GT | {" | ".join(names)}], error rows beneath) and per-view PNGs', flush=True)
    print('RENDER OK')


if __name__ == '__main__':
    main()
