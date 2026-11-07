"""Per-vertex screen-space footprint of the ORIGINAL mesh over the training views.  GPU, read-only.

For every training camera (validation split train[::8] excluded, same split as Stage-2 / coarse_finetune.py)
the official upstream renderer is run once and, for every vertex v that belongs to a rendered face:
    pixel_scale[v] = max over such views of  focal_px / depth_v          [pixels per world unit, native resolution]
i.e. how many pixels a unit world displacement of v moves on screen in the view where v is largest.
This is the quantity the pixel-space collapse criterion (native_qem_vis.cpp) multiplies the cluster
position radius by.  Also written: views_seen[v], max_face_pixels[v] (largest native-res pixel count of an
incident face in any view), residual_mean[v] (|render-GT| at the projected vertex, averaged over views
that render it), plus the same per face.  Unseen vertices get pixel_scale 0 / views_seen 0.

Render sensitivity (default on, --no-sensitivity to skip; one backward pass per view):
    sens_pos[v] = sum over views of || d(sum of rendered pixels) / d vertex_v ||      (c3dgs-style: how much the image
    sens_dc[v]  = sum over views of || d(sum of rendered pixels) / d DC_v ||           changes when the vertex / its colour moves)
    l1_pos[v]   = sum over views of || d L1(render, GT) / d vertex_v ||                (residual-driven; ~0 at convergence)
coarse_vis.py can turn these into per-vertex geometric / appearance weights (--geo-weight / --app-weight).

Run on a GPU node in the original-lossless-v1 .venv, from this directory:
  python vertex_footprint.py --lab /scratch/zr2245/meshsplatting-original-lossless-v1 \
      --model /scratch/zr2245/room-fixed-sharing-0N4OSU/run/original \
      --dataset /scratch/zr2245/datasets/m360/room --images images_2 \
      --out /scratch/zr2245/room-vis-collapse/footprint.npz
"""
import argparse, json, math, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def log(*a): print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


PACKAGE_VERSION = (Path(__file__).resolve().parent / 'VERSION').read_text().split()[0] if (Path(__file__).resolve().parent / 'VERSION').exists() else 'unknown'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lab', required=True); p.add_argument('--model', required=True)
    p.add_argument('--dataset', required=True); p.add_argument('--images', default='images_2')
    p.add_argument('--iteration', type=int, default=30000)
    p.add_argument('--supersample', type=int, default=4, help='renderer supersampling (4 = what the sharing / Stage-2 runs used)')
    p.add_argument('--views', choices=['train', 'train+validation', 'all'], default='train', help='train = training cameras minus the validation split (default)')
    p.add_argument('--no-sensitivity', action='store_true', help='skip the backward passes (pixel_scale / views only)')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if not torch.cuda.is_available(): p.error('CUDA required')
    upstream = Path(a.lab).resolve() / 'upstream'
    sys.path.insert(0, str(upstream / 'submodules/effrdel/src')); sys.path.insert(0, str(upstream))
    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render
    data = SimpleNamespace(source_path=str(Path(a.dataset).resolve()), model_path=str(Path(a.model).resolve()), images=a.images,
                           resolution=1, eval=True, white_background=False, data_device='cpu', sh_degree=3)
    mesh = TriangleModel(3); mesh.scaling = a.supersample
    scene = Scene(data, mesh, None, None, load_iteration=a.iteration, shuffle=False)
    for v in vars(mesh).values():
        if isinstance(v, torch.Tensor): v.requires_grad_(False)
    sens = not a.no_sensitivity
    if sens: mesh.vertices.requires_grad_(True); mesh._features_dc.requires_grad_(True)
    train_all = sorted(scene.getTrainCameras(), key=lambda c: (c.image_name, c.uid))
    validation = train_all[::8]; train = [c for i, c in enumerate(train_all) if i % 8]
    test = sorted(scene.getTestCameras(), key=lambda c: (c.image_name, c.uid))
    cams = {'train': train, 'train+validation': train_all, 'all': train_all + test}[a.views]
    V = len(mesh.vertices); F = len(mesh._triangle_indices); faces = mesh._triangle_indices.long()
    log(f'{V} vertices, {F} faces, {len(cams)} views ({a.views}), supersample {a.supersample}')
    bg = torch.zeros(3, device='cuda'); pipe = SimpleNamespace(debug=False, convert_SHs_python=False, depth_ratio=1.)
    dev = 'cuda'
    pixel_scale = torch.zeros(V, device=dev); views_seen = torch.zeros(V, device=dev, dtype=torch.int32)
    residual_sum = torch.zeros(V, device=dev); vmax_px = torch.zeros(V, device=dev)
    f_views = torch.zeros(F, device=dev, dtype=torch.int32); f_max_px = torch.zeros(F, device=dev)
    sens_pos = torch.zeros(V, device=dev); sens_dc = torch.zeros(V, device=dev); l1_pos = torch.zeros(V, device=dev)
    ss2 = float(a.supersample ** 2); per_view = []
    for i, cam in enumerate(cams):
        if sens:
            out = render(cam, mesh, pipe, bg)
            gt0 = cam.original_image[:3].to(dev).clamp(0, 1)
            # pass 1: image energy -> |dI/dv|, |dI/dDC| ; pass 2: L1 loss -> |dL/dv|
            g = torch.autograd.grad(out['render'].sum(), [mesh.vertices, mesh._features_dc], retain_graph=True, allow_unused=True)
            if g[0] is not None: sens_pos += g[0].norm(dim=1)
            if g[1] is not None: sens_dc += g[1].reshape(V, -1).norm(dim=1)
            g = torch.autograd.grad((out['render'] - gt0).abs().mean(), [mesh.vertices], allow_unused=True)
            if g[0] is not None: l1_pos += g[0].norm(dim=1)
            out = {k: (t.detach() if isinstance(t, torch.Tensor) else t) for k, t in out.items()}
        with torch.no_grad():
            if not sens: out = render(cam, mesh, pipe, bg)
            was = out['triangle_was_rendered'][:F].float() / ss2            # native-res pixel equivalents per face
            rendered_f = was > 0
            vr = torch.zeros(V, device=dev, dtype=torch.bool); vr[faces[rendered_f].reshape(-1)] = True
            hom = torch.cat((mesh.vertices, torch.ones(V, 1, device=dev)), 1) @ cam.world_view_transform
            z = hom[:, 2]
            fx = cam.image_width / (2 * math.tan(cam.FoVx * .5)); fy = cam.image_height / (2 * math.tan(cam.FoVy * .5))
            k = torch.where((z > 1e-6) & vr, max(fx, fy) / z.clamp_min(1e-6), torch.zeros_like(z))
            pixel_scale = torch.maximum(pixel_scale, k); views_seen += vr.int()
            im = out['render'].clamp(0, 1); gt = cam.original_image[:3].to(dev).clamp(0, 1)
            res = (im - gt).abs().mean(0)
            xy = out['image_2D']; ok = vr & torch.isfinite(xy).all(1)
            xi = xy[:, 0].round().long().clamp(0, cam.image_width - 1); yi = xy[:, 1].round().long().clamp(0, cam.image_height - 1)
            residual_sum[ok] += res[yi[ok], xi[ok]]
            f_views += rendered_f.int(); f_max_px = torch.maximum(f_max_px, was)
            vmax_px.scatter_reduce_(0, faces.reshape(-1), was[:, None].expand(-1, 3).reshape(-1), reduce='amax')
            per_view.append(dict(camera=cam.image_name, rendered_faces=int(rendered_f.sum()), rendered_vertices=int(vr.sum()),
                                 psnr=float(-10 * torch.log10((im - gt).square().mean().clamp_min(1e-12))), focal_px=float(max(fx, fy))))
            if i % 20 == 0 or i == len(cams) - 1: log(f'view {i + 1}/{len(cams)} {cam.image_name}: {int(rendered_f.sum())} faces rendered')
    seen = views_seen > 0
    residual_mean = torch.where(seen, residual_sum / views_seen.clamp_min(1).float(), torch.zeros_like(residual_sum))
    ps = pixel_scale.cpu().numpy(); q = lambda t, qs: [float(x) for x in np.quantile(t, qs)]
    summary = dict(package_version=PACKAGE_VERSION, vertices=V, faces=F, views=len(cams), view_split=a.views, supersample=a.supersample,
                   seen_vertex_fraction=float(seen.float().mean()), seen_face_fraction=float((f_views > 0).float().mean()),
                   pixel_scale_quantiles_seen=dict(zip(['q05', 'q25', 'q50', 'q75', 'q95', 'max'], q(ps[ps > 0], [.05, .25, .5, .75, .95, 1.]))),
                   views_seen_quantiles=dict(zip(['q05', 'q50', 'q95'], q(views_seen.cpu().numpy(), [.05, .5, .95]))),
                   max_face_pixels_quantiles=dict(zip(['q50', 'q90', 'q99'], q(f_max_px.cpu().numpy(), [.5, .9, .99]))),
                   per_view=per_view)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, pixel_scale=ps.astype(np.float32), views_seen=views_seen.cpu().numpy().astype(np.int32),
                        max_face_pixels=vmax_px.cpu().numpy().astype(np.float32), residual_mean=residual_mean.cpu().numpy().astype(np.float32),
                        face_views_seen=f_views.cpu().numpy().astype(np.int32), face_max_pixels=f_max_px.cpu().numpy().astype(np.float32),
                        sens_pos=sens_pos.cpu().numpy().astype(np.float32), sens_dc=sens_dc.cpu().numpy().astype(np.float32),
                        l1_pos=l1_pos.cpu().numpy().astype(np.float32), view_names=np.array([c.image_name for c in cams]))
    if sens:
        sp = sens_pos.cpu().numpy(); sd = sens_dc.cpu().numpy()
        summary['sensitivity'] = dict(sens_pos_quantiles_seen=dict(zip(['q05', 'q25', 'q50', 'q75', 'q95', 'max'], q(sp[ps > 0], [.05, .25, .5, .75, .95, 1.]))),
                                      sens_dc_quantiles_seen=dict(zip(['q05', 'q25', 'q50', 'q75', 'q95', 'max'], q(sd[ps > 0], [.05, .25, .5, .75, .95, 1.]))),
                                      zero_sens_pos_fraction=float((sp == 0).mean()))
    out.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    log('seen vertices', f"{summary['seen_vertex_fraction']:.4f}", 'pixel_scale quantiles (seen)', summary['pixel_scale_quantiles_seen'])
    log('wrote', out, 'and', out.with_suffix('.json'))


if __name__ == '__main__':
    main()
