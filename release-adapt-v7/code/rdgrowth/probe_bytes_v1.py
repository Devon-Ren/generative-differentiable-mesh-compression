"""probe_bytes_v1 (2026-11-06): the delivery encoder as a byte meter for the rate controller (train_rd_adapt_v2).

Runs INSIDE the stage2 env through vis_collapse/stage2_run.sh (which prepends --stage2 <pkg>):
    bash $PKG/stage2_run.sh $AN/rdgrowth/probe_bytes_v1.py --coarse probe.pt --out probe.json --k 4096 --r 256 --xyz q16 --dc q7
Takes a mesh state (same keys as coarse_finetune best.pt / the upstream point_cloud_state_dict.pt) and runs exactly the delivery
chain of deliver_coarse.py without the 400-step bank refit and without any rendering: k-means dictionaries (K primary, R residual)
on THIS mesh, sharing_after.convert (dictionary_rounds), export_asset (XYZ / DC grids, constant opacity logit, neighbour-context
index coder, edge-arithmetic connectivity; includes the codec's own decode round-trip check).  Writes <out>: file_bytes,
payload_bytes per stream, vertices, faces, seconds.  The bank refit changes bank values, not the index statistics, so the bytes
measured here are the delivered bytes up to the DC/bank drift of the refit (checked on 12 adapt-v1 runs: delivered / probe at the
final mesh is reported by the sbatch).  Nothing in the stage2 package is modified; the asset folder is deleted after measuring."""
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np
import torch

PROBE_VERSION = 'probe-v1'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage2', required=True); p.add_argument('--coarse', required=True); p.add_argument('--out', required=True)
    p.add_argument('--k', type=int, default=4096); p.add_argument('--r', type=int, default=256)
    p.add_argument('--xyz', default='q14'); p.add_argument('--dc', default='q7')
    p.add_argument('--dictionary-rounds', type=int, default=2); p.add_argument('--kmeans-samples', type=int, default=65536)
    p.add_argument('--kmeans-rounds', type=int, default=8); p.add_argument('--seed', type=int, default=17)
    a = p.parse_args()
    t0 = time.time()
    stage2 = Path(a.stage2).resolve()
    for d in (stage2 / 'renderer', stage2, stage2 / 'codec'): sys.path.insert(0, str(d))
    from runtime import export_asset
    from sharing_after import convert
    from appearance import initialize, assign
    import delivery
    delivery.XYZ_MODES = tuple(dict.fromkeys(delivery.XYZ_MODES + ('q15', 'q13', 'q12')))
    delivery.DC_MODES = tuple(dict.fromkeys(delivery.DC_MODES + ('q7', 'q6', 'q5', 'q4')))
    if a.xyz not in delivery.XYZ_MODES or a.dc not in delivery.DC_MODES: p.error(f'unknown precision {a.xyz}/{a.dc}')
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.backends.cuda.matmul.allow_tf32 = False
    w = torch.load(a.coarse, map_location='cpu', weights_only=False)
    keys = ('triangles_points', 'features_dc', 'features_rest', '_triangle_indices', 'vertex_weight', 'sigma', 'active_sh_degree')
    coarse = {k: (w[k].detach().cpu().clone() if isinstance(w[k], torch.Tensor) else w[k]) for k in keys}
    coarse['_triangle_indices'] = coarse['_triangle_indices'].to(torch.int32).contiguous()
    coarse['opacity_floor'] = float(w.get('opacity_floor', .999)); del w
    V, F = len(coarse['triangles_points']), len(coarse['_triangle_indices'])
    src = coarse['features_rest'].cuda().float(); samples = max(a.kmeans_samples, 4 * max(a.k, a.r))
    bank = initialize(src, a.k, False, samples, a.kmeans_rounds, a.seed).bank.detach()
    z, _ = assign(src, bank); resid = src - bank[z]
    rbank = initialize(resid, a.r, False, samples, a.kmeans_rounds, a.seed + 1).bank.detach() if a.r > 1 else resid.mean(0, keepdim=True)
    seed = dict(app_bank=bank.cpu().numpy(), app_residual_bank=rbank.cpu().numpy()); del src, bank, z, resid, rbank; torch.cuda.empty_cache()
    t1 = time.time()
    shared = convert(coarse, seed, a.dictionary_rounds); del seed; torch.cuda.empty_cache()
    t2 = time.time()
    folder = Path(a.out).with_suffix('.asset')
    shutil.rmtree(folder, ignore_errors=True)
    decoded, codec = export_asset(shared, folder, a.xyz, a.dc, weight='const', index_codec='ctx')
    t3 = time.time()
    pb = {k: int(v) for k, v in codec['payload_bytes'].items()}
    out = dict(probe_version=PROBE_VERSION, file_bytes=int(codec['file_bytes']), payload_bytes=pb, vertices=V, faces=F, k=a.k, r=a.r, xyz=a.xyz, dc=a.dc,
               dictionary_rounds=a.dictionary_rounds, seconds=dict(kmeans=round(t1 - t0, 1), convert=round(t2 - t1, 1), export=round(t3 - t2, 1), total=round(t3 - t0, 1)))
    Path(a.out).write_text(json.dumps(out, indent=1))
    shutil.rmtree(folder, ignore_errors=True)
    print(f"[probe] V {V} F {F} -> {out['file_bytes'] / 1e6:.3f} MB in {out['seconds']['total']} s | " + ' '.join(f"{k}:{v / 1e6:.3f}" for k, v in sorted(pb.items(), key=lambda kv: -kv[1]) if v > 20000), flush=True)


if __name__ == '__main__':
    main()
