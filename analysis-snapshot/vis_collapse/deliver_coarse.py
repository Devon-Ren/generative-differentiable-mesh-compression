"""Take a fine-tuned coarse mesh (best.pt from coarse_finetune.py) through Wenhao's Stage-2 delivery:
shared SH conversion (sharing_after.convert) -> shared appearance fit -> export_asset (q18 XYZ / q8 DC,
strip-fetch reorder, edge-arithmetic codec) -> decoded validation/test metrics.  Exactly the tail of
coarse_once.py, with the coarse mesh supplied from outside instead of built by the hierarchy.

Only IMPORTS the stage2 package (v12 default: shared-collapse-stage2-v2); nothing there is modified.  Defaults for seed
asset / source / dataset / lab / learning rates are read from the historical run's config.json
(--historical-config), so the delivery is comparable with room-shared-collapse-QGQgyr.

Run through deliver_coarse.sh (it sets the same PYTHONPATH / CUDA env as run_room.sh).  Outputs in --output:
progress.json, shared-final.pt, delivery/asset.mslb + codec-report.json, summary.json, results.csv, png-delivery/.

v6 additions (RD axes + rate-aware fit):
  --k K --r R          build the SH dictionaries by k-means on THIS mesh (K primary, R residual codewords) instead of
                       taking the seed asset's banks; axis A of the RD study.  --r 1 = residual effectively off.
  --xyz qN --dc qN     delivery precision (default q18 / q8 = historical).  Modes outside the stage2 lists
                       (q12 XYZ, q7/q6/q5 DC) are enabled by extending delivery.XYZ_MODES/DC_MODES at runtime;
                       the stage2 decoder (dequantize) already accepts any 1-24 bits.
  --qat                rate-aware fit: geometry is quantised to --xyz BEFORE conversion/fitting (so DC and the banks
                       are fitted on top of the exact delivered geometry) and the per-vertex DC is quantised to --dc
                       inside the forward pass with a straight-through estimator.  Export then reproduces the same
                       grids, so the "shared_fitted" and "decoded_delivery" stages coincide up to codec round-trip.

v12 (stage2-v2 delivery format):
  --weight const|keep  const (default) transmits one opacity logit instead of per-vertex vertex_weight (the trained
                       logits are all ~equal; see stage2-v2 README); keep = v1 format.
  --index-codec ctx|zstd  ctx (default) codes app_mapping / app_residual_mapping with the neighbour-context arithmetic
                       coder (codec/index_ctx.cpp); zstd = v1 format.  Both are passed to export_asset only when the
                       stage2 package supports them (inspect.signature); with stage2-v1 the defaults fall back to
                       keep / zstd and the effective values are written to config.json / summary.json.
  --seed-asset is only needed with --k 0 (seed banks); --historical-config none disables the Room defaults.
"""
import argparse, csv, gc, inspect, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

PACKAGE_VERSION = (Path(__file__).resolve().parent / 'VERSION').read_text().split()[0] if (Path(__file__).resolve().parent / 'VERSION').exists() else 'unknown'


def log(*a): print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--coarse', required=True, help='best.pt from coarse_finetune.py (independent SH)')
    p.add_argument('--output', required=True)
    p.add_argument('--stage2', default='/scratch/zr2245/shared-collapse-stage2-v2')
    p.add_argument('--historical-config', default='/scratch/zr2245/room-shared-collapse-QGQgyr/run/config.json')
    for k in ('source', 'seed-asset', 'dataset', 'lab'): p.add_argument('--' + k, help='override the historical config')
    p.add_argument('--images'); p.add_argument('--shared-steps', type=int, help='default: historical (400)')
    p.add_argument('--dictionary-rounds', type=int); p.add_argument('--png', action='store_true', help='save decoded test renders')
    p.add_argument('--k', type=int, default=0, help='primary dictionary size built by k-means on this mesh (0 = seed asset banks)')
    p.add_argument('--r', type=int, default=0, help='residual dictionary size (with --k); 1 = residual off')
    p.add_argument('--kmeans-samples', type=int, default=65536); p.add_argument('--kmeans-rounds', type=int, default=8)
    p.add_argument('--xyz', default='q14', help='XYZ delivery precision: f32,q20,q18,q16,q15,q14,q13,q12 (v8 default q14: precision sweep 2026-10-13 showed q18->q14 costs 0.06 dB for 3 MB)')
    p.add_argument('--dc', default='q8', help='DC delivery precision: f32,f16,q12,q10,q8,q7,q6,q5,q4 (q6 costs 0.14 dB / -0.02 SSIM for 1.5 MB: keep q8)')
    p.add_argument('--qat', action='store_true', help='quantise geometry before fitting and use STE DC quantisation in the fit')
    p.add_argument('--weight', choices=['const', 'keep'], default='const', help='v12: const = one opacity logit for all vertices (stage2-v2), keep = per-vertex vertex_weight (v1 format)')
    p.add_argument('--index-codec', choices=['ctx', 'zstd'], default='ctx', help='v12: ctx = neighbour-context arithmetic coder for the SH indices (stage2-v2), zstd = v1 format')
    a = p.parse_args()
    if (a.k > 0) != (a.r > 0): p.error('--k and --r go together')
    hist = {} if a.historical_config in ('none', '') or not Path(a.historical_config).exists() else json.loads(Path(a.historical_config).read_text())
    if not hist: log('historical config not used (' + a.historical_config + '); source / dataset / lab come from the command line')
    cfg = dict(hist)
    for k in ('source', 'seed_asset', 'dataset', 'lab', 'images', 'shared_steps', 'dictionary_rounds'):
        v = getattr(a, k)
        if v is not None: cfg[k] = v
    defaults = dict(images='images_2', shared_steps=400, dictionary_rounds=2, geo_lr=.005, dc_lr=.002, bank_lr=.0001, residual_lr=.0003,
                    anchor_weight=.005, seed=17, eval_every=200, min_steps=400, patience=3, local_steps=0, max_flip_fraction=.0001,
                    train_factor=4, eval_factor=4)
    for k, v in defaults.items(): cfg.setdefault(k, v)
    for k in ('source', 'dataset', 'lab') + (() if a.k > 0 else ('seed_asset',)):
        if k not in cfg: p.error(f'--{k.replace("_", "-")} required (historical config missing)')
    cfg.setdefault('seed_asset', None)
    args = argparse.Namespace(**cfg)
    if not torch.cuda.is_available(): p.error('CUDA required')
    stage2 = Path(a.stage2).resolve()
    for d in (stage2 / 'renderer', stage2, stage2 / 'codec'):
        sys.path.insert(0, str(d))
    try:
        from stage2_triangle_rasterization import _C
    except ImportError as e:
        p.error(f'stage2 renderer extension not importable ({e}); run through deliver_coarse.sh or build it with "cd {stage2}/renderer && python setup.py build_ext --inplace"')
    if _C.shared_stage2_backward_revision() != 2: raise RuntimeError('Renderer revision 2 required')
    from runtime import load_scene, Evaluator, export_asset
    from algorithm import save_json, health
    from train_collapse_first import assess
    from optimize import fit
    from sharing_after import convert
    from codec_benchmark import unpack
    import delivery
    delivery.XYZ_MODES = tuple(dict.fromkeys(delivery.XYZ_MODES + ('q15', 'q13', 'q12')))
    delivery.DC_MODES = tuple(dict.fromkeys(delivery.DC_MODES + ('q7', 'q6', 'q5', 'q4')))
    if a.xyz not in delivery.XYZ_MODES or a.dc not in delivery.DC_MODES: p.error(f'unknown precision {a.xyz}/{a.dc}')
    # v12: stage2-v2 export_asset(wire, folder, xyz, dc, weight, index_codec); stage2-v1 has only the first four
    export_params = inspect.signature(export_asset).parameters
    export_kw = {}
    for name, value, v1_value in (('weight', a.weight, 'keep'), ('index_codec', a.index_codec, 'zstd')):
        if name in export_params: export_kw[name] = value
        elif value != v1_value: p.error(f'--{name.replace("_", "-")} {value} needs stage2-v2 (export_asset in {stage2} has no {name} argument)')
    effective = dict(weight=export_kw.get('weight', 'keep'), index_codec=export_kw.get('index_codec', 'zstd'))
    stage2_version = (stage2 / 'STAGE2_VERSION').read_text().split()[0] if (stage2 / 'STAGE2_VERSION').exists() else 'v1'
    if effective != dict(weight=a.weight, index_codec=a.index_codec): log('WARNING stage2 package is v1: delivering with', effective)
    log('delivery format:', stage2_version, effective)

    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'config.json', dict(vars(args), coarse=a.coarse, package_version=PACKAGE_VERSION, stage2=str(stage2), stage2_version=stage2_version, k=a.k, r=a.r, xyz=a.xyz, dc=a.dc, qat=a.qat, **effective))
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.backends.cuda.matmul.allow_tf32 = False
    log('vis_collapse package', PACKAGE_VERSION, '| stage2', stage2, '| seed asset', 'k-means on this mesh' if a.k > 0 else args.seed_asset)
    if a.k > 0:
        seed = None  # built below from the coarse mesh itself
    else:
        seed = unpack(args.seed_asset)
        if seed.get('app_schema') != 2 or np.asarray(seed.get('app_gain', [])).size: raise ValueError('Seed must contain residual SH dictionaries without gain')
    train, val, test, ssim = load_scene(args, False); evaluate = Evaluator(ssim, args.eval_factor)
    save_json(out / 'splits.json', dict(train=[c.image_name for c in train], validation=[c.image_name for c in val], test=[c.image_name for c in test]))

    w = torch.load(a.coarse, map_location='cpu', weights_only=False)
    keys = ('triangles_points', 'features_dc', 'features_rest', '_triangle_indices', 'vertex_weight', 'sigma', 'active_sh_degree')
    coarse = {k: (w[k].detach().cpu().clone() if isinstance(w[k], torch.Tensor) else w[k]) for k in keys}
    coarse['_triangle_indices'] = coarse['_triangle_indices'].to(torch.int32).contiguous()
    coarse['opacity_floor'] = float(w.get('opacity_floor', .999)); del w
    rows = []
    if a.qat and a.xyz != 'f32':
        q, meta = delivery.quantize(coarse['triangles_points'].numpy(), a.xyz)
        before = coarse['triangles_points']; coarse['triangles_points'] = torch.from_numpy(delivery.dequantize(q, meta)).float().contiguous()
        log('QAT: geometry pre-quantised to', a.xyz, '| max move', float((coarse['triangles_points'] - before).norm(dim=1).max()), '|', health(before, coarse['triangles_points'], coarse['_triangle_indices']))
        del before, q
    if a.k > 0:
        from appearance import initialize, assign
        src = coarse['features_rest'].cuda().float(); samples = max(a.kmeans_samples, 4 * max(a.k, a.r))
        log('k-means seed on this mesh: K', a.k, 'R', a.r, 'samples', samples, 'rounds', a.kmeans_rounds)
        bank = initialize(src, a.k, False, samples, a.kmeans_rounds, args.seed).bank.detach()
        z, _ = assign(src, bank); resid = src - bank[z]
        rbank = initialize(resid, a.r, False, samples, a.kmeans_rounds, args.seed + 1).bank.detach() if a.r > 1 else resid.mean(0, keepdim=True)
        seed = dict(app_bank=bank.cpu().numpy(), app_residual_bank=rbank.cpu().numpy()); del src, bank, z, resid, rbank; torch.cuda.empty_cache()

    def measure(name, wire):
        row = dict(stage=name, vertices=len(wire['triangles_points']), faces=len(wire['_triangle_indices']), validation=assess(wire, evaluate, val))
        rows.append(row); save_json(out / 'progress.json', rows); log('MEASURE', name, row['faces'], round(row['validation']['psnr'], 4)); return row

    measure('coarse_input', coarse)
    input_test = assess(coarse, evaluate, test); log('coarse_input test PSNR', round(input_test['psnr'], 4))
    shared = convert(coarse, seed, args.dictionary_rounds); del seed; gc.collect(); torch.cuda.empty_cache()
    measure('shared_initial', shared)
    if args.shared_steps:
        if a.qat and a.dc not in ('f32', 'f16'):
            shared = qat_fit(shared, args.shared_steps, train, val, evaluate, ssim, args, out / 'shared-fit', int(a.dc[1:]))
        else:
            shared = fit(shared, None, args.shared_steps, train, val, evaluate, ssim, args, out / 'shared-fit', appearance_only=True)
    torch.save(shared, out / 'shared-final.pt'); measure('shared_fitted', shared)
    delivery_dir = out / 'delivery'
    if delivery_dir.exists():
        import shutil; shutil.rmtree(delivery_dir)
    decoded, codec = export_asset(shared, delivery_dir, a.xyz, a.dc, **export_kw); measure('decoded_delivery', decoded)
    delivery_test = assess(decoded, evaluate, test, png_dir=out / 'png-delivery' if a.png else None)
    pb = codec['payload_bytes']
    summary = dict(package_version=PACKAGE_VERSION, coarse=a.coarse, k=a.k, r=a.r, xyz=a.xyz, dc=a.dc, qat=a.qat, stage2_version=stage2_version, **effective,
                   stages=rows, codec=codec, input_test=input_test, delivery_test=delivery_test,
                   file_MB=codec['file_bytes'] / 1e6, vertices=len(coarse['triangles_points']), faces=len(coarse['_triangle_indices']),
                   mapping_MB=sum(pb.get(k, 0) for k in ('app_mapping', 'app_residual_mapping', 'ctx_mapping', 'ctx_residual')) / 1e6,
                   banks_MB=sum(pb.get(k, 0) for k in ('app_bank', 'app_residual_bank')) / 1e6,
                   vertex_weight_MB=pb.get('vertex_weight', 0) / 1e6,
                   geometry=health(coarse['triangles_points'], shared['triangles_points'], coarse['_triangle_indices']))
    if hist: summary['historical'] = dict(delivery_MB=26.884, delivery_test_psnr=27.3467, note='room-shared-collapse-QGQgyr, same face count')
    save_json(out / 'summary.json', summary)
    with (out / 'results.csv').open('w', newline='') as stream:
        wr = csv.DictWriter(stream, fieldnames=['stage', 'split', 'vertices', 'faces', 'psnr', 'ssim', 'lpips', 'file_MB']); wr.writeheader()
        for row in rows:
            wr.writerow(dict(stage=row['stage'], split='validation', vertices=row['vertices'], faces=row['faces'],
                             **{k: row['validation'].get(k, '') for k in ('psnr', 'ssim', 'lpips')}, file_MB=codec['file_bytes'] / 1e6 if row['stage'] == 'decoded_delivery' else ''))
        for name, r in (('coarse_input', input_test), ('decoded_delivery', delivery_test)):
            wr.writerow(dict(stage=name, split='test', vertices=summary['vertices'], faces=summary['faces'], **{k: r[k] for k in ('psnr', 'ssim', 'lpips')},
                             file_MB=codec['file_bytes'] / 1e6 if name == 'decoded_delivery' else ''))
    log(f"DELIVERY {codec['file_bytes'] / 1e6:.3f} MB  test PSNR {delivery_test['psnr']:.4f} SSIM {delivery_test['ssim']:.4f} LPIPS {delivery_test['lpips']:.4f}"
        f"  | {stage2_version} weight={effective['weight']} indices={effective['index_codec']} mapping {summary['mapping_MB']:.3f} MB banks {summary['banks_MB']:.3f} MB vertex_weight {summary['vertex_weight_MB']:.3f} MB")
    log('wrote', out / 'summary.json')


def qat_fit(wire, steps, cameras, validation, evaluate, ssim, args, folder, dc_bits):
    """Appearance-only shared fit (same loss/schedule/selection as stage2 optimize.fit with appearance_only=True)
    with the per-vertex DC quantised to dc_bits inside the forward pass through a straight-through estimator.
    The grid (per-channel min/span over all vertices) is recomputed from the current DC every step, which is
    exactly what delivery.quantize does at export time, so the exported DC lands on the grid the fit saw."""
    from model import SharedMesh, optimizer, render
    from algorithm import save_json

    class QATMesh(SharedMesh):
        """v8: fixed grid from the initial DC range (v7 recomputed min/max every step; result was worse than post-hoc
        quantisation).  DC is clamped to the grid range and the two extreme vertices per channel are pinned so that
        delivery.quantize at export reproduces exactly this grid."""
        def __init__(self, wire):
            super().__init__(wire)
            flat = self.dc.detach().reshape(-1, 3); low = flat.min(0).values; high = flat.max(0).values
            self.register_buffer('q_low', low.clone()); self.register_buffer('q_high', high.clone())
            self.register_buffer('q_scale', ((high - low) / ((1 << dc_bits) - 1)).clamp_min(1e-12))
            self.register_buffer('pin_lo', flat.argmin(0)); self.register_buffer('pin_hi', flat.argmax(0))
        def colors(self, center):
            dc = torch.maximum(torch.minimum(self.dc, self.q_high), self.q_low)
            q = torch.round((dc - self.q_low) / self.q_scale) * self.q_scale + self.q_low
            dc = dc + (q - dc).detach()
            v = self.vertices; direction = v - center[None]; direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            rest = self.bank[self.z] + self.residual[self.u]
            from sh_utils import eval_sh
            sh = torch.cat((dc, rest), dim=1).transpose(1, 2)
            return (eval_sh(self.degree, sh, direction) + .5).clamp_min(0).contiguous()

    folder.mkdir(parents=True, exist_ok=True)
    model = QATMesh(wire); opt = optimizer(model, args.geo_lr, args.dc_lr, args.bank_lr, args.residual_lr)
    model.x.requires_grad_(False); lrs = [g['lr'] for g in opt.param_groups]
    generator = np.random.default_rng(args.seed)
    appearance0 = {k: p.detach().clone() for k, p in model.named_parameters() if k in ('bank', 'residual')}
    initial = evaluate(model, validation, include_lpips=False); best_score = initial['loss']; best = model.wire(); best_step = 0; stale = 0
    save_json(folder / 'initial-validation.json', dict(initial, note=f'QAT forward: DC q{dc_bits} STE'))
    order = generator.permutation(len(cameras)); cursor = 0; stop_reason = 'step budget'
    for step in range(1, steps + 1):
        for group, lr in zip(opt.param_groups, lrs): group['lr'] = lr * (.25 + .75 * .5 * (1 + math.cos(math.pi * (step - 1) / max(1, steps))))
        if cursor == len(order): order = generator.permutation(len(cameras)); cursor = 0
        c = cameras[int(order[cursor])]; cursor += 1; opt.zero_grad(set_to_none=True)
        image = render(c, model, args.train_factor)['render'].clamp(0, 1); gt = c.original_image[:3].to(image.device)
        image_loss = .8 * (image - gt).abs().mean() + .2 * (1 - ssim(image, gt))
        anchor = sum((p - appearance0[k]).square().mean() for k, p in model.named_parameters() if k in appearance0)
        loss = image_loss + args.anchor_weight * anchor
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
        loss.backward()
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if param.grad is None or not torch.isfinite(param.grad).all(): raise FloatingPointError('Missing/nonfinite gradient: ' + name)
            torch.nn.utils.clip_grad_norm_([param], 1.)
        opt.step()
        if step == 1 or step % 100 == 0:
            with (folder / 'training.jsonl').open('a') as stream: stream.write(json.dumps(dict(step=step, phase=f'qat-dc-q{dc_bits}', image_loss=float(image_loss))) + '\n')
            print('QAT:', step, round(float(image_loss), 5), flush=True)
        if step % args.eval_every == 0 or step == steps:
            val = evaluate(model, validation, include_lpips=False)
            if val['loss'] < best_score: best_score = val['loss']; best = model.wire(); best_step = step; stale = 0
            else: stale += 1
            save_json(folder / f'validation-{step:05d}.json', dict(metrics=val, best_step=best_step))
            print('Validation', step, 'PSNR', val['psnr'], 'initial', initial['psnr'], 'best step', best_step, flush=True)
            if not math.isfinite(val['psnr']) or val['psnr'] < initial['psnr'] - .5: stop_reason = 'quality guard'; break
            if step >= getattr(args, 'min_steps', 400) and stale >= getattr(args, 'patience', 3): break
    save_json(folder / 'adaptation.json', dict(steps=step, requested_steps=steps, best_step=best_step, best_validation_loss=best_score, stop_reason=stop_reason, qat_dc_bits=dc_bits))
    # pin the grid: clamp into the training range and put the extreme vertices exactly on low/high per channel
    dc = best['features_dc'].reshape(-1, 3); low = model.q_low.cpu(); high = model.q_high.cpu()
    dc = torch.maximum(torch.minimum(dc, high), low)
    for c in range(3): dc[int(model.pin_lo[c]), c] = low[c]; dc[int(model.pin_hi[c]), c] = high[c]
    best['features_dc'] = dc.reshape(best['features_dc'].shape).contiguous()
    return best


if __name__ == '__main__':
    main()
