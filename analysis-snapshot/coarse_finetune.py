"""How much of the collapse quality loss can plain retraining recover?  GPU, read-only on inputs.

Loads a collapsed mesh (coarse-initial.pt: independent SH, fixed topology) and fine-tunes
vertices + DC + SH with the ORIGINAL upstream renderer and the original photometric loss
(0.8 L1 + 0.2 D-SSIM), Default is appearance only (--geo-lr 0): DC + SH, geometry frozen.  Geometry can be enabled
with --geo-lr in units of each vertex's minimum incident altitude (Adam moves a parameter by
~lr per step regardless of gradient, so world-unit steps of 1e-4 destroy sliver triangles:
a first attempt with lr 1.5e-4 world units flipped 8% of faces in 1500 steps and lost 3.4 dB).  Validation every --eval-every steps on the same split as the Stage-2
runs (train[::8]); best-by-validation checkpoint; final test metrics with LPIPS.

Run in the original-lossless-v1 .venv on a GPU node (Room: ~5k steps, 4x supersample, ~30 min):
  python coarse_finetune.py --lab /scratch/zr2245/meshsplatting-original-lossless-v1 \
      --model /scratch/zr2245/room-fixed-sharing-0N4OSU/run/original \
      --dataset /scratch/zr2245/datasets/m360/room --images images_2 \
      --coarse /scratch/zr2245/room-shared-collapse-QGQgyr/run/coarse-initial.pt \
      --steps 3000 --out /scratch/zr2245/room-coarse-finetune-appearance        # geometry frozen
  ... --geo-lr 0.005 --out /scratch/zr2245/room-coarse-finetune-geo             # + geometry, 0.5% altitude/step
Reference numbers (validation / test): original 30.715 / 28.263; coarse_initial 28.972 / -;
coarse after 800 constrained steps 29.289 / 27.390.
"""
import argparse, json, math, sys, time, csv
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def save_json(path, obj):
    path = Path(path); tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=2)); tmp.replace(path)


def health(reference, current, faces):
    """Flipped / zero-area faces relative to the reference geometry (same as stage2/algorithm.py)."""
    a = reference.detach().cpu().numpy(); b = current.detach().cpu().numpy(); f = faces.detach().cpu().numpy().astype(np.int64)
    out = dict(nonfinite_vertices=int((~np.isfinite(b).all(1)).sum()), new_zero_area_faces=0, flipped_faces=0, faces=len(f))
    for s in range(0, len(f), 200000):
        i = f[s:s + 200000]; x = a[i].astype(np.float64); y = b[i].astype(np.float64)
        an = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]); bn = np.cross(y[:, 1] - y[:, 0], y[:, 2] - y[:, 0])
        out['new_zero_area_faces'] += int((((an * an).sum(1) > 0) & ((bn * bn).sum(1) <= 1e-30)).sum())
        out['flipped_faces'] += int(((an * bn).sum(1) < 0).sum())
    out['flipped_fraction'] = out['flipped_faces'] / max(1, len(f)); return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lab', required=True); p.add_argument('--model', required=True, help='original model dir (for cameras/splits only)')
    p.add_argument('--dataset', required=True); p.add_argument('--images', default='images_2')
    p.add_argument('--coarse', required=True, help='coarse-initial.pt (or any .pt with independent SH)')
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=5000); p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--geo-lr', type=float, default=0.0, help='0 = appearance only (DC+SH). With --geo-unit altitude the step is a '
                   'fraction of each vertex\'s minimum incident triangle altitude (0.005 = 0.5%% per step); with world it is world units')
    p.add_argument('--geo-unit', choices=['altitude', 'world'], default='altitude')
    p.add_argument('--geo-lr-final-ratio', type=float, default=0.1, help='exponential decay of the geometry lr over the run')
    p.add_argument('--max-displacement', type=float, default=0.25, help='altitude mode: clamp total |x| to this many altitudes (0 = off)')
    p.add_argument('--stop-drop', type=float, default=0.5, help='stop when validation PSNR falls this far below step 0')
    p.add_argument('--dc-lr', type=float, default=0.002, help='original: 0.004 indoor, 0.0016 outdoor')
    p.add_argument('--rest-lr-ratio', type=float, default=1 / 20)
    p.add_argument('--flip-guard', type=float, default=1e-3, help='0 disables the orientation guard')
    p.add_argument('--supersample', type=int, default=4); p.add_argument('--eval-supersample', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--png', action='store_true', help='save test renders of the best model')
    a = p.parse_args()
    if not torch.cuda.is_available(): p.error('CUDA required')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False); save_json(out / 'config.json', vars(a))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    upstream = Path(a.lab).resolve() / 'upstream'
    sys.path.insert(0, str(upstream / 'submodules/effrdel/src')); sys.path.insert(0, str(upstream))
    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render
    from utils.loss_utils import ssim
    import lpips

    # Cameras and split exactly as the sharing / Stage-2 runs (Scene loads the ORIGINAL model; we then swap in the coarse mesh).
    data = SimpleNamespace(source_path=str(Path(a.dataset).resolve()), model_path=str(Path(a.model).resolve()), images=a.images,
                           resolution=1, eval=True, white_background=False, data_device='cpu', sh_degree=3)
    mesh = TriangleModel(3); scene = Scene(data, mesh, None, None, load_iteration=30000, shuffle=False)
    train_all = sorted(scene.getTrainCameras(), key=lambda c: (c.image_name, c.uid))
    validation = train_all[::8]; train = [c for i, c in enumerate(train_all) if i % 8]
    test = sorted(scene.getTestCameras(), key=lambda c: (c.image_name, c.uid))
    save_json(out / 'splits.json', dict(train=[c.image_name for c in train], validation=[c.image_name for c in validation], test=[c.image_name for c in test]))

    w = torch.load(a.coarse, map_location='cpu', weights_only=False)
    dev = 'cuda'
    mesh.vertices = w['triangles_points'].to(dev).float().detach().clone().requires_grad_(True)
    mesh._triangle_indices = w['_triangle_indices'].to(dev).to(torch.int32)
    mesh.vertex_weight = w['vertex_weight'].to(dev).float().detach().clone().reshape(-1, 1)
    mesh._features_dc = w['features_dc'].to(dev).float().detach().clone().requires_grad_(True)
    mesh._features_rest = w['features_rest'].to(dev).float().detach().clone().requires_grad_(True)
    mesh._sigma = w['sigma']; mesh.active_sh_degree = int(w['active_sh_degree'])
    mesh.opacity_floor = float(w.get('opacity_floor', 0.999))
    nf = len(mesh._triangle_indices)
    mesh.image_size = torch.zeros(nf, device=dev); mesh.importance_score = torch.zeros(nf, device=dev); mesh.pixel_count = torch.zeros(nf, device=dev)
    del w
    n, f = len(mesh.vertices), nf
    log(f'coarse mesh: {n} vertices, {f} faces; train {len(train)} / val {len(validation)} / test {len(test)}')
    bg = torch.zeros(3, device=dev); pipe = SimpleNamespace(debug=False, convert_SHs_python=False, depth_ratio=1.)
    perceptual = lpips.LPIPS(net='vgg').cuda().eval()
    for q in perceptual.parameters(): q.requires_grad_(False)

    @torch.no_grad()
    def evaluate(cams, with_lpips=False, png_dir=None):
        mesh.scaling = a.eval_supersample; rows = []
        for i, c in enumerate(cams):
            im = render(c, mesh, pipe, bg)['render'].clamp(0, 1); gt = c.original_image[:3].cuda().clamp(0, 1)
            row = dict(camera=c.image_name, psnr=float(-10 * torch.log10((im - gt).square().mean().clamp_min(1e-12))), ssim=float(ssim(im, gt)))
            row['loss'] = .8 * float((im - gt).abs().mean()) + .2 * (1 - row['ssim'])
            if with_lpips: row['lpips'] = float(perceptual(im[None] * 2 - 1, gt[None] * 2 - 1).mean())
            if png_dir:
                from PIL import Image
                png_dir.mkdir(parents=True, exist_ok=True)
                for tag, x in (('render', im), ('gt', gt)):
                    Image.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)).save(png_dir / f'{i:04d}-{tag}.png')
            rows.append(row)
        mesh.scaling = a.supersample
        keys = [k for k in rows[0] if k != 'camera']
        return dict(**{k: float(np.mean([r[k] for r in rows])) for k in keys}, worst_view_psnr=float(min(r['psnr'] for r in rows)), views=len(rows), per_view=rows)

    def state():
        return dict(triangles_points=mesh.vertices.detach().cpu().clone(), features_dc=mesh._features_dc.detach().cpu().clone(),
                    features_rest=mesh._features_rest.detach().cpu().clone(), _triangle_indices=mesh._triangle_indices.cpu().clone(),
                    vertex_weight=mesh.vertex_weight.detach().cpu().clone(), sigma=mesh._sigma, active_sh_degree=mesh.active_sh_degree,
                    opacity_floor=mesh.opacity_floor)

    reference = mesh.vertices.detach().clone()
    # Geometry: vertices = reference + h * x.  Adam moves x by ~lr per step whatever the gradient, so in
    # 'altitude' mode h = each vertex's minimum incident triangle altitude and lr is a fraction of that
    # (same coordinates as stage2/geometry_trust.py, but as a soft parametrisation, no hard projection).
    if a.geo_lr > 0 and a.geo_unit == 'altitude':
        v = reference.cpu().numpy().astype(np.float64); fc = mesh._triangle_indices.cpu().numpy()
        h = np.full(n, np.inf)
        for s_ in range(0, f, 200000):
            ff = fc[s_:s_ + 200000]; A, B, C = v[ff[:, 0]], v[ff[:, 1]], v[ff[:, 2]]
            longest = np.maximum.reduce([np.linalg.norm(B - A, axis=1), np.linalg.norm(C - A, axis=1), np.linalg.norm(C - B, axis=1)])
            alt = np.linalg.norm(np.cross(B - A, C - A), axis=1) / np.maximum(longest, 1e-300); alt[~np.isfinite(alt)] = 0
            for k in range(3): np.minimum.at(h, ff[:, k], alt)
        h[~np.isfinite(h)] = 0
        log('altitude quantiles (world units):', np.quantile(h, [0, .1, .5, .9, 1]).round(6).tolist())
        hscale = torch.as_tensor(h, device=dev, dtype=torch.float32)[:, None]
    else:
        hscale = torch.ones(n, 1, device=dev)
    x = torch.zeros(n, 3, device=dev, requires_grad=True)
    def apply_geometry():
        mesh.vertices = reference + hscale * x
    apply_geometry()
    groups = [{'params': [mesh._features_dc], 'lr': a.dc_lr, 'name': 'dc'},
              {'params': [mesh._features_rest], 'lr': a.dc_lr * a.rest_lr_ratio, 'name': 'rest'}]
    if a.geo_lr > 0: groups.insert(0, {'params': [x], 'lr': a.geo_lr, 'name': 'vertices'})
    opt = torch.optim.Adam(groups, lr=0., eps=1e-15)
    log('geometry:', 'frozen (appearance only)' if a.geo_lr == 0 else f'{a.geo_unit} units, lr {a.geo_lr} -> {a.geo_lr * a.geo_lr_final_ratio}')
    base_lr = [g['lr'] for g in opt.param_groups]
    rng = np.random.default_rng(a.seed)
    sampled = torch.as_tensor(rng.integers(0, f, min(65536, f)), device=dev)
    sf = mesh._triangle_indices[sampled].long(); t0 = reference[sf]
    n0 = torch.cross(t0[:, 1] - t0[:, 0], t0[:, 2] - t0[:, 0], dim=1); den = n0.square().sum(1).clamp_min(1e-24)

    history = []
    initial = evaluate(validation); log('step 0 validation PSNR', round(initial['psnr'], 4))
    history.append(dict(step=0, **{k: initial[k] for k in ('psnr', 'ssim', 'loss', 'worst_view_psnr')}, flipped_faces=0))
    best = dict(step=0, loss=initial['loss'], psnr=initial['psnr']); torch.save(state(), out / 'best.pt')
    order = rng.permutation(len(train)); cursor = 0; mesh.scaling = a.supersample; ema = None
    for step in range(1, a.steps + 1):
        decay = a.geo_lr_final_ratio ** (step / a.steps)                 # exponential, like the original position schedule
        for g, lr in zip(opt.param_groups, base_lr): g['lr'] = lr * (decay if g['name'] == 'vertices' else .25 + .75 * .5 * (1 + math.cos(math.pi * step / a.steps)))
        if cursor == len(order): order = rng.permutation(len(train)); cursor = 0
        c = train[int(order[cursor])]; cursor += 1
        opt.zero_grad(set_to_none=True); apply_geometry()
        im = render(c, mesh, pipe, bg)['render']; gt = c.original_image[:3].cuda()
        l1 = (im - gt).abs().mean(); ss = ssim(im, gt); loss = .8 * l1 + .2 * (1 - ss)
        if a.flip_guard > 0:
            t = mesh.vertices[sf]; nn_ = torch.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=1)
            loss = loss + a.flip_guard * (.1 - (nn_ * n0).sum(1) / den).clamp_min(0).square().mean()
        if not torch.isfinite(loss): raise FloatingPointError(f'nonfinite loss at step {step}')
        loss.backward(); opt.step()
        if a.geo_lr > 0 and a.geo_unit == 'altitude' and a.max_displacement > 0:
            with torch.no_grad(): x.mul_(torch.clamp(a.max_displacement / x.norm(dim=1, keepdim=True).clamp_min(1e-30), max=1))
        ema = float(loss) if ema is None else .98 * ema + .02 * float(loss)
        if step % 100 == 0: log(f'step {step} loss(ema) {ema:.5f} geo_lr {opt.param_groups[0]["lr"]:.2e}')
        if step % a.eval_every == 0 or step == a.steps or step in (100, 250):
            apply_geometry(); val = evaluate(validation); geo = health(reference, mesh.vertices, mesh._triangle_indices)
            history.append(dict(step=step, **{k: val[k] for k in ('psnr', 'ssim', 'loss', 'worst_view_psnr')}, **{k: geo[k] for k in ('flipped_faces', 'new_zero_area_faces')}))
            save_json(out / 'history.json', history)
            log(f'validation step {step}: PSNR {val["psnr"]:.4f} (step0 {initial["psnr"]:.4f}, original 30.7145) SSIM {val["ssim"]:.4f} flips {geo["flipped_faces"]} zero-area {geo["new_zero_area_faces"]}')
            if val['loss'] < best['loss'] and geo['nonfinite_vertices'] == 0:
                best = dict(step=step, loss=val['loss'], psnr=val['psnr']); torch.save(state(), out / 'best.pt'); log('  new best')
            if val['psnr'] < initial['psnr'] - a.stop_drop:
                log('STOP: validation fell more than', a.stop_drop, 'dB below step 0; keeping best'); break
    log('loading best', best); b = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    mesh.vertices = b['triangles_points'].to(dev); mesh._features_dc = b['features_dc'].to(dev); mesh._features_rest = b['features_rest'].to(dev)
    apply_geometry = lambda: None
    test_best = evaluate(test, True, out / 'png-test' if a.png else None)
    summary = dict(coarse=a.coarse, vertices=n, faces=f, steps=a.steps, best=best, validation_initial={k: initial[k] for k in ('psnr', 'ssim')},
                   validation_best=dict(psnr=best['psnr']), test_best={k: test_best[k] for k in ('psnr', 'ssim', 'lpips', 'worst_view_psnr')},
                   geometry_best=health(reference, mesh.vertices, mesh._triangle_indices), history=history,
                   reference=dict(original_val=30.7145, original_test=28.2634, stage2_800steps_val=29.2892, stage2_800steps_test=27.3904))
    save_json(out / 'summary.json', summary)
    with (out / 'history.csv').open('w', newline='') as s:
        wr = csv.DictWriter(s, fieldnames=list(history[-1].keys())); wr.writeheader(); [wr.writerow({k: r.get(k, '') for k in history[-1]}) for r in history]
    log('TEST best:', {k: round(v, 4) for k, v in summary['test_best'].items()}); log('wrote', out / 'summary.json')


if __name__ == '__main__':
    main()
