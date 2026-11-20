"""render_ids_v1.py (2026-11-20): where the triangles are.  For chosen TEST views (by image name) and several meshes, renders the image and reads the
triangle-id map of the renderer (render_pkg['rend_ids'], the triangle that covers each pixel; the same map the training uses for the per-triangle
error).  From the id map, at the supersampled resolution:
  edges : a pixel is on an edge when its right or lower neighbour belongs to another triangle  -> wireframe, saved as coverage per image pixel (8 bit)
  area  : screen area of the triangle that covers the pixel, in image pixels                   -> saved as 255 * clip((log10(area) + 1) / 4, 0, 1)
Loading and rendering as render_ckpt_v1.py (black background, supersampling 4).  Writes <out>/<view>-<name>-{render.jpg,edges.png,area.png}, <view>-gt.jpg
and stats.txt (visible triangles, median and quartiles of the triangle area over the pixels).
  python render_ids_v1.py --lab $LAB --dataset $DATA --images images_4 --model <original model dir> --names DSC08076 --ckpt A=<state dict> --ckpt B=... --out <dir>"""
import argparse, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

def maps(ids, nf, ss):
    """ids [Hs, Ws] long at the supersampled resolution -> (edge coverage [H, W] in 0..1, area in image pixels [H, W], valid [H, W])"""
    valid = (ids >= 0) & (ids < nf); idc = torch.where(valid, ids, torch.full_like(ids, nf))
    e = torch.zeros_like(idc, dtype=torch.bool); e[:, :-1] |= idc[:, 1:] != idc[:, :-1]; e[:-1, :] |= idc[1:, :] != idc[:-1, :]
    cov = F.avg_pool2d(e[None, None].float(), ss)[0, 0]
    cnt = torch.bincount(idc.reshape(-1), minlength=nf + 1).float() / (ss * ss); small = idc[ss // 2::ss, ss // 2::ss]
    H, W = cov.shape; small = small[:H, :W]; area = cnt[small]; return cov, area, small < nf

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--lab'); ap.add_argument('--dataset'); ap.add_argument('--images', default='images_2'); ap.add_argument('--model')
    ap.add_argument('--ckpt', action='append'); ap.add_argument('--names', nargs='+'); ap.add_argument('--supersample', type=int, default=4); ap.add_argument('--out'); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        ids = torch.zeros(8, 8, dtype=torch.long); ids[:, 4:] = 1; ids[4:, :4] = 2; cov, area, valid = maps(ids, 3, 2)
        assert cov.shape == (4, 4) and float(area[0, 0]) == 4.0 and float(area[0, 3]) == 8.0 and float(cov[0, 0]) == 0.0 and float(cov[0, 1]) == 0.5 and bool(valid.all()), (cov, area)
        print('selftest ok'); return
    if not torch.cuda.is_available(): sys.exit('FAILED: CUDA required')
    up = Path(a.lab).resolve() / 'upstream'; sys.path.insert(0, str(up / 'submodules/effrdel/src')); sys.path.insert(0, str(up))
    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render
    from PIL import Image
    data = SimpleNamespace(source_path=str(Path(a.dataset).resolve()), model_path=str(Path(a.model).resolve()), images=a.images, resolution=1, eval=True, white_background=False, data_device='cpu', sh_degree=3)
    mesh = TriangleModel(3); scene = Scene(data, mesh, None, None, load_iteration=30000, shuffle=False); dev = 'cuda'
    cams = {c.image_name: c for c in scene.getTestCameras()}; miss = [n for n in a.names if n not in cams]
    if miss: sys.exit('FAILED: not a test view: %s (test views: %s ...)' % (miss, sorted(cams)[:8]))
    bg = torch.zeros(3, device=dev); pipe = SimpleNamespace(debug=False, convert_SHs_python=False, depth_ratio=1.); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); lines = []
    u8 = lambda x: (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    for spec in a.ckpt:
        n, p = spec.split('=', 1); w = torch.load(p, map_location='cpu', weights_only=False); nv = len(w['triangles_points'])
        mesh.vertices = w['triangles_points'].to(dev).float().reshape(nv, 3).contiguous(); mesh._triangle_indices = w['_triangle_indices'].to(dev).to(torch.int32).reshape(-1, 3).contiguous()
        mesh.vertex_weight = w['vertex_weight'].to(dev).float().reshape(-1, 1).contiguous(); mesh._features_dc = w['features_dc'].to(dev).float().reshape(nv, 1, 3).contiguous()
        mesh._features_rest = w['features_rest'].to(dev).float().reshape(nv, 15, 3).contiguous(); nf = len(mesh._triangle_indices)
        for k in ('image_size', 'importance_score', 'pixel_count'): setattr(mesh, k, torch.zeros(nf, device=dev))
        mesh.scaling = a.supersample; del w
        with torch.no_grad():
            for name in a.names:
                c = cams[name]; pk = render(c, mesh, pipe, bg)
                if 'rend_ids' not in pk: sys.exit('FAILED: the renderer returned no rend_ids (keys: %s)' % sorted(pk.keys()))
                im = pk['render'].clamp(0, 1); gt = c.original_image[:3].cuda().clamp(0, 1); ids = pk['rend_ids']; ids = ids.reshape(ids.shape[-2], ids.shape[-1]).round().long()
                ss = max(1, round(ids.shape[-1] / im.shape[-1])); cov, area, valid = maps(ids, nf, ss); H, W = min(cov.shape[0], im.shape[1]), min(cov.shape[1], im.shape[2])
                psnr = float(-10 * torch.log10((im - gt).square().mean().clamp_min(1e-12))); av = area[:H, :W][valid[:H, :W]]
                q = torch.quantile(av[torch.randperm(av.numel(), device=av.device)[:2000000]], torch.tensor([0.25, 0.5, 0.75], device=av.device)).tolist() if av.numel() else [float('nan')] * 3
                vis = int(torch.unique(ids[(ids >= 0) & (ids < nf)]).numel())
                Image.fromarray(u8(im)).save(out / f'{name}-{n}-render.jpg', quality=93); Image.fromarray(u8(gt)).save(out / f'{name}-gt.jpg', quality=93)
                Image.fromarray((cov[:H, :W].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)).save(out / f'{name}-{n}-edges.png')
                Image.fromarray((((torch.log10(area[:H, :W].clamp_min(1e-3)) + 1) / 4).clamp(0, 1) * valid[:H, :W]).cpu().numpy().__mul__(255).round().astype(np.uint8)).save(out / f'{name}-{n}-area.png')
                line = f'WIRE {name} {n} | V {nv} F {nf} | id map {tuple(ids.shape)} ss {ss} | PSNR {psnr:.2f} | visible triangles {vis} | triangle area over the pixels (image px): q25 {q[0]:.2f} median {q[1]:.2f} q75 {q[2]:.2f} | covered pixels {float(valid.float().mean()):.3f}'
                lines.append(line); print(line, flush=True)
        torch.cuda.empty_cache()
    with open(out / 'stats.txt', 'a') as f: f.write('\n'.join(lines) + '\n')
    print('WIRE OK')

if __name__ == '__main__':
    main()
