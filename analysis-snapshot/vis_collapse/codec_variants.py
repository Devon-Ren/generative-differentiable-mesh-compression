"""v12: measure the two stage2-v2 format changes on an EXISTING fitted shared mesh, without training.

    bash stage2_run.sh codec_variants.py --deliver-dir $W/axisB/v0.8/deliver --xyz q14 --dc q7

Re-exports <deliver-dir>/shared-final.pt for each (weight, index_codec) variant:
    keep/zstd  = stage2-v1 format (per-vertex vertex_weight, zstd-coded SH indices)
    const/zstd = constant opacity logit only
    keep/ctx   = neighbour-context index coder only
    const/ctx  = stage2-v2 default
and records file MB, payload bytes per field, decoded validation and test metrics.  Needs stage2-v2 (export_asset with
weight / index_codec arguments; --stage2 default /scratch/zr2245/shared-collapse-stage2-v2).  Output:
<deliver-dir>/codec-variants/<weight>_<codec>/asset.mslb + codec-report.json, and codec-variants/sweep.json + sweep.txt.
The decoded quality of the const variants equals keep up to the (tiny) opacity error reported by stage2-v2
(vertex_weight_const.max_abs_opacity_error in codec-report.json); the ctx variants are lossless, so their PSNR must
match zstd exactly.  Restartable: measured variants are skipped.
"""
import argparse, inspect, json, shutil, sys, time
from pathlib import Path
import torch

PACKAGE_VERSION = (Path(__file__).resolve().parent / 'VERSION').read_text().split()[0] if (Path(__file__).resolve().parent / 'VERSION').exists() else 'unknown'
MAPPING = ('app_mapping', 'app_residual_mapping', 'ctx_mapping', 'ctx_residual')


def log(*a): print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--deliver-dir', required=True, help='output dir of deliver_coarse.py (shared-final.pt + config.json)')
    p.add_argument('--shared', help='default: <deliver-dir>/shared-final.pt')
    p.add_argument('--xyz', default='q14'); p.add_argument('--dc', default='q7')
    p.add_argument('--variants', default='keep/zstd,const/zstd,keep/ctx,const/ctx', help='comma-separated weight/index_codec pairs')
    p.add_argument('--stage2', default='/scratch/zr2245/shared-collapse-stage2-v2'); p.add_argument('--skip-test', action='store_true')
    a = p.parse_args()
    d = Path(a.deliver_dir); cfg = json.loads((d / 'config.json').read_text()); args = argparse.Namespace(**cfg)
    stage2 = Path(a.stage2).resolve()
    for sub in (stage2 / 'renderer', stage2, stage2 / 'codec'): sys.path.insert(0, str(sub))
    from runtime import load_scene, Evaluator, export_asset
    from algorithm import save_json
    from train_collapse_first import assess
    import delivery
    delivery.XYZ_MODES = tuple(dict.fromkeys(delivery.XYZ_MODES + ('q15', 'q13', 'q12')))
    delivery.DC_MODES = tuple(dict.fromkeys(delivery.DC_MODES + ('q7', 'q6', 'q5', 'q4')))
    if a.xyz not in delivery.XYZ_MODES or a.dc not in delivery.DC_MODES: p.error(f'unknown precision {a.xyz}/{a.dc}')
    params = inspect.signature(export_asset).parameters
    if 'weight' not in params or 'index_codec' not in params:
        p.error(f'export_asset in {stage2} has no weight/index_codec arguments: this needs stage2-v2 (stage2-v2-patch-v1/apply_stage2_v2.sh)')
    stage2_version = (stage2 / 'STAGE2_VERSION').read_text().split()[0] if (stage2 / 'STAGE2_VERSION').exists() else 'unknown'
    log('vis_collapse', PACKAGE_VERSION, '| stage2', stage2, stage2_version, '| precision', a.xyz, a.dc)
    shared = torch.load(a.shared or d / 'shared-final.pt', map_location='cpu', weights_only=False)
    train, val, test, ssim = load_scene(args, False); evaluate = Evaluator(ssim, args.eval_factor); del train
    out = d / 'codec-variants'; out.mkdir(exist_ok=True)
    prev = out / 'sweep.json'; rows = json.loads(prev.read_text())['rows'] if prev.exists() else []
    done = {(r['weight'], r['index_codec'], r['xyz'], r['dc']) for r in rows}
    for variant in a.variants.split(','):
        weight, codec_name = variant.split('/')
        if weight not in ('keep', 'const') or codec_name not in ('zstd', 'ctx'): p.error('bad variant ' + variant)
        if (weight, codec_name, a.xyz, a.dc) in done: log('already measured', variant); continue
        folder = out / f'{weight}_{codec_name}_{a.xyz}_{a.dc}'
        if folder.exists(): shutil.rmtree(folder)
        t0 = time.time(); decoded, codec = export_asset(shared, folder, a.xyz, a.dc, weight=weight, index_codec=codec_name); seconds = time.time() - t0
        v = assess(decoded, evaluate, val); t = None if a.skip_test else assess(decoded, evaluate, test)
        pb = codec['payload_bytes']
        row = dict(weight=weight, index_codec=codec_name, xyz=a.xyz, dc=a.dc, file_MB=codec['file_bytes'] / 1e6,
                   mapping_MB=sum(pb.get(k, 0) for k in MAPPING) / 1e6, vertex_weight_MB=pb.get('vertex_weight', 0) / 1e6,
                   payload_MB={k: round(b / 1e6, 4) for k, b in pb.items() if b > 20000}, export_seconds=round(seconds, 1),
                   vertex_weight_const=codec.get('vertex_weight_const'),
                   validation={k: v[k] for k in ('psnr', 'ssim', 'lpips')}, test=None if t is None else {k: t[k] for k in ('psnr', 'ssim', 'lpips')})
        rows.append(row)
        save_json(out / 'sweep.json', dict(package_version=PACKAGE_VERSION, stage2=str(stage2), stage2_version=stage2_version,
                                           shared=str(a.shared or d / 'shared-final.pt'), vertices=len(shared['triangles_points']), rows=rows))
        shutil.rmtree(folder / 'native', ignore_errors=True)
        log(f"{variant}: {row['file_MB']:.3f} MB (mapping {row['mapping_MB']:.3f}, vertex_weight {row['vertex_weight_MB']:.3f}) | val {v['psnr']:.4f} | test {'-' if t is None else round(t['psnr'], 4)}")
    lines = [f'{"weight/idx":12s} {"MB":>8s} {"mapMB":>7s} {"vwMB":>6s} {"val":>8s} {"test":>8s} {"ssim":>7s} {"lpips":>7s}  opacity_err']
    for r in rows:
        t = r['test'] or {}; c = r.get('vertex_weight_const') or {}
        lines.append(f"{r['weight'] + '/' + r['index_codec']:12s} {r['file_MB']:8.3f} {r['mapping_MB']:7.3f} {r['vertex_weight_MB']:6.3f} {r['validation']['psnr']:8.4f} "
                     f"{t.get('psnr', float('nan')):8.4f} {t.get('ssim', float('nan')):7.4f} {t.get('lpips', float('nan')):7.4f}  {c.get('max_abs_opacity_error', '-')}")
    lines.append('ctx rows must have exactly the PSNR of their zstd twin (lossless); const rows differ from keep only through the opacity error.')
    (out / 'sweep.txt').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    main()
