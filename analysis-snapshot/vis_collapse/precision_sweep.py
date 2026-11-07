"""Re-export an already fitted shared mesh (shared-final.pt from deliver_coarse.py) at several XYZ/DC precisions,
WITHOUT any training, and measure decoded validation/test quality per precision.  Answers "how many bits does the
geometry/DC really need?" before spending GPU time on quantisation-aware fitting.

    bash deliver_coarse.sh  (sets the stage2 env)  -- or run with the same PYTHONPATH --
    python precision_sweep.py --deliver-dir $W/deliver-geo-only --modes q18/q8,q16/q8,q14/q8,q12/q8,q14/q6,q14/q5

--deliver-dir must contain shared-final.pt and config.json (written by deliver_coarse.py).  Output:
<deliver-dir>/precision/<xyz>_<dc>/asset.mslb + codec-report.json and <deliver-dir>/precision/sweep.json + sweep.txt.
Only imports the stage2 package; the extra modes (q12 XYZ, q7..q4 DC) are enabled by extending the mode lists at runtime.
"""
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np
import torch

PACKAGE_VERSION = (Path(__file__).resolve().parent / 'VERSION').read_text().split()[0] if (Path(__file__).resolve().parent / 'VERSION').exists() else 'unknown'


def log(*a): print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--deliver-dir', required=True); p.add_argument('--shared', help='default: <deliver-dir>/shared-final.pt')
    p.add_argument('--modes', default='q18/q8,q16/q8,q14/q8,q12/q8,q16/q6,q14/q6,q14/q5')
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
    shared = torch.load(a.shared or d / 'shared-final.pt', map_location='cpu', weights_only=False)
    train, val, test, ssim = load_scene(args, False); evaluate = Evaluator(ssim, args.eval_factor); del train
    out = d / 'precision'; out.mkdir(exist_ok=True)
    prev = out / 'sweep.json'; rows = json.loads(prev.read_text())['rows'] if prev.exists() else []
    done = {(r['xyz'], r['dc']) for r in rows}
    base = assess(shared, evaluate, val); log('shared-final (FP32) validation', round(base['psnr'], 4))
    for mode in a.modes.split(','):
        xyz, dc = mode.split('/')
        if xyz not in delivery.XYZ_MODES or dc not in delivery.DC_MODES: log('skip unknown mode', mode); continue
        if (xyz, dc) in done: log('already measured', mode); continue
        folder = out / f'{xyz}_{dc}'
        if folder.exists(): shutil.rmtree(folder)
        decoded, codec = export_asset(shared, folder, xyz, dc)
        v = assess(decoded, evaluate, val); t = None if a.skip_test else assess(decoded, evaluate, test)
        row = dict(xyz=xyz, dc=dc, file_MB=codec['file_bytes'] / 1e6, payload_MB={k: round(b / 1e6, 4) for k, b in codec['payload_bytes'].items() if b > 50000},
                   quantization_geometry=codec['quantization_geometry'], validation={k: v[k] for k in ('psnr', 'ssim', 'lpips')},
                   test=None if t is None else {k: t[k] for k in ('psnr', 'ssim', 'lpips')}, worst_test_view=None if t is None else min(r['psnr'] for r in t['per_view']))
        rows.append(row); save_json(out / 'sweep.json', dict(package_version=PACKAGE_VERSION, shared=str(a.shared or d / 'shared-final.pt'), fp32_validation=base, rows=rows))
        shutil.rmtree(folder / 'native', ignore_errors=True)
        log(f"{xyz}/{dc}: {row['file_MB']:.3f} MB | val {v['psnr']:.3f} | test {'-' if t is None else round(t['psnr'], 3)} {'' if t is None else round(t['ssim'], 4)} {'' if t is None else round(t['lpips'], 4)} | flips {codec['quantization_geometry']['flipped_faces']}")
    lines = [f'{"xyz/dc":10s} {"MB":>8s} {"val":>8s} {"test":>8s} {"ssim":>7s} {"lpips":>7s} {"flips":>6s}']
    for r in rows:
        t = r['test'] or {}
        lines.append(f"{r['xyz']}/{r['dc']:<5s} {r['file_MB']:8.3f} {r['validation']['psnr']:8.3f} {t.get('psnr', float('nan')):8.3f} {t.get('ssim', float('nan')):7.4f} {t.get('lpips', float('nan')):7.4f} {r['quantization_geometry']['flipped_faces']:6d}")
    (out / 'sweep.txt').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    main()
