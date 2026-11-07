"""Rate-aware step 2: neighbour-context RD reassignment of the shared-SH indices.

The delivered asset spends ~22 bits/vertex on app_mapping + app_residual_mapping (5.1 MB of 25.5 MB on Room), coded
only with zstd.  Measured neighbour agreement of the k-means assignment is ~30 %.  This script asks: if a vertex may
take a NEIGHBOUR's codeword whenever the extra SH distortion is small (cost = D + lambda * bits), how fast does the
index rate fall, and what does it cost in rendered PSNR -- with and without a short appearance re-fit?

    bash stage2_run.sh index_rd.py --deliver-dir $W/deliver-geo-only --lambdas 0,0.1,0.3,1,3 [--refit-steps 200]

Per lambda it reports: neighbour hit rate of z and u, estimated context-coder bits/vertex (hit: 1+log2(#distinct
neighbour codes); miss: 1+log2(K)), proxy SH distortion, validation PSNR after reassignment (and after re-fit),
REAL asset bytes from export_asset (zstd also benefits from repeated indices) and decoded test metrics.
Output: <deliver-dir>/index_rd/<lambda>/..., index_rd/sweep.json + sweep.txt.  Only imports stage2.
"""
import argparse, json, math, shutil, sys, time
from pathlib import Path
import numpy as np
import torch

PACKAGE_VERSION = (Path(__file__).resolve().parent / 'VERSION').read_text().split()[0] if (Path(__file__).resolve().parent / 'VERSION').exists() else 'unknown'


def log(*a): print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def adjacency(faces, n):
    f = np.asarray(faces, np.int64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]], f[:, [1, 0]], f[:, [2, 1]], f[:, [0, 2]]])
    e = e[e[:, 0] != e[:, 1]]
    key = np.unique(e[:, 0] * n + e[:, 1]); src = key // n; dst = key % n
    counts = np.bincount(src, minlength=n); ptr = np.zeros(n + 1, np.int64); ptr[1:] = np.cumsum(counts)
    return torch.from_numpy(ptr), torch.from_numpy(dst), int(counts.max())


def neighbour_codes(code, ptr, dst, maxdeg, device):
    """[N, maxdeg] matrix of neighbour codes, -1 padded."""
    n = len(code); deg = (ptr[1:] - ptr[:-1]).to(device)
    out = torch.full((n, maxdeg), -1, dtype=torch.int64, device=device)
    src = torch.repeat_interleave(torch.arange(n, device=device), deg)
    slot = torch.arange(len(dst), device=device) - ptr[:-1].to(device)[src]
    out[src, slot] = code[dst.to(device)]
    return out


@torch.no_grad()
def reassign(target, bank, z, ptr, dst, maxdeg, lam, unit, K, sweeps=3):
    """Jacobi sweeps: each vertex may switch to a neighbour's code if D + lam*unit*bits is lower.  target [N,D] is
    what the (bank[z] part of the) vertex should reproduce; D_j = ||bank[j]-target||^2."""
    device = target.device; z = z.clone()
    bits_hit = 1 + math.log2(max(1, maxdeg)); bits_miss = 1 + math.log2(K)
    for _ in range(sweeps):
        nb = neighbour_codes(z, ptr, dst, maxdeg, device)
        cand = torch.cat([z[:, None], nb], 1)  # [N, 1+maxdeg]
        valid = cand >= 0; safe = cand.clamp_min(0)
        d = (bank[safe] - target[:, None]).square().sum(-1)  # [N, 1+maxdeg]
        # rate: keeping own code counts as a miss unless a neighbour already has it
        own_hit = (nb == z[:, None]).any(1)
        rate = torch.full_like(d, bits_hit); rate[:, 0] = torch.where(own_hit, bits_hit, bits_miss)
        cost = d + lam * unit * rate; cost[~valid] = float('inf')
        z = cand.gather(1, cost.argmin(1, keepdim=True))[:, 0]
    return z


def context_bits(z, ptr, dst, maxdeg, K, device):
    """Sequential estimate in index order: context = neighbours with a smaller index."""
    nb = neighbour_codes(z, ptr, dst, maxdeg, device)
    idx = torch.arange(len(z), device=device)
    nbi = neighbour_codes(idx, ptr, dst, maxdeg, device)
    earlier = (nbi >= 0) & (nbi < idx[:, None])
    hit = ((nb == z[:, None]) & earlier).any(1)
    # distinct earlier neighbour codes (upper bound by count of earlier neighbours)
    m = earlier.sum(1).clamp_min(1).float()
    bits = torch.where(hit, 1 + torch.log2(m), torch.full_like(m, 1 + math.log2(K)))
    return float(hit.float().mean()), float(bits.mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--deliver-dir', required=True); p.add_argument('--shared', help='default: <deliver-dir>/shared-final.pt')
    p.add_argument('--lambdas', default='0,0.1,0.3,1,3'); p.add_argument('--sweeps', type=int, default=3)
    p.add_argument('--refit-steps', type=int, default=0, help='appearance-only shared fit after reassignment (banks + DC adapt to the new indices)')
    p.add_argument('--xyz', default='q18'); p.add_argument('--dc', default='q8')
    p.add_argument('--stage2', default='/scratch/zr2245/shared-collapse-stage2-v2'); p.add_argument('--skip-test', action='store_true')
    a = p.parse_args()
    d = Path(a.deliver_dir); cfg = json.loads((d / 'config.json').read_text()); args = argparse.Namespace(**cfg)
    stage2 = Path(a.stage2).resolve()
    for sub in (stage2 / 'renderer', stage2, stage2 / 'codec'): sys.path.insert(0, str(sub))
    from runtime import load_scene, Evaluator, export_asset
    from algorithm import save_json
    from train_collapse_first import assess
    from optimize import fit
    from model import SharedMesh
    shared = torch.load(a.shared or d / 'shared-final.pt', map_location='cpu', weights_only=False)
    train, val, test, ssim = load_scene(args, False); evaluate = Evaluator(ssim, args.eval_factor)
    out = d / 'index_rd'; out.mkdir(exist_ok=True)
    dev = 'cuda'
    b = torch.as_tensor(shared['app_bank'], device=dev).float().reshape(len(shared['app_bank']), -1)
    r = torch.as_tensor(shared['app_residual_bank'], device=dev).float().reshape(len(shared['app_residual_bank']), -1)
    z0 = torch.as_tensor(np.asarray(shared['app_mapping']).astype(np.int64), device=dev)
    u0 = torch.as_tensor(np.asarray(shared['app_residual_mapping']).astype(np.int64), device=dev)
    n = len(z0); K, R = len(b), len(r)
    ptr, dst, maxdeg = adjacency(shared['_triangle_indices'].numpy(), n); maxdeg = min(maxdeg, 32)
    if (ptr[1:] - ptr[:-1]).max() > maxdeg:  # cap degree for memory: keep the first 32 neighbours
        keep = []; p0 = ptr.numpy(); dd = dst.numpy()
        for v in range(n): keep.append(dd[p0[v]:min(p0[v + 1], p0[v] + maxdeg)])
        dst = torch.from_numpy(np.concatenate(keep)); counts = np.minimum(np.diff(p0), maxdeg); ptr = torch.from_numpy(np.concatenate([[0], np.cumsum(counts)]))
    full = b[z0] + r[u0]  # current SH rest per vertex (= what the renderer uses); reassignment targets this
    unit = float((full - b[z0]).square().sum(1).median())  # median residual energy: lambda=1 trades one such unit per bit
    base_hit_z, base_bits_z = context_bits(z0, ptr, dst, maxdeg, K, dev); base_hit_u, base_bits_u = context_bits(u0, ptr, dst, maxdeg, R, dev)
    base_val = assess(shared, evaluate, val)
    log(f'N {n} K {K} R {R} maxdeg {maxdeg} | baseline hit z {base_hit_z:.3f} u {base_hit_u:.3f} | est bits z {base_bits_z:.2f} u {base_bits_u:.2f} | val {base_val["psnr"]:.3f} | unit {unit:.4g}')
    rows = [dict(lam=None, hit_z=base_hit_z, hit_u=base_hit_u, bits_z=base_bits_z, bits_u=base_bits_u, proxy_mse=0., validation=base_val['psnr'], note='baseline (k-means assignment)')]
    for lam in [float(x) for x in a.lambdas.split(',')]:
        z = reassign(full - r[u0], b, z0, ptr, dst, maxdeg, lam, unit, K, a.sweeps) if lam > 0 else z0.clone()
        u = reassign(full - b[z], r, u0, ptr, dst, maxdeg, lam, unit, R, a.sweeps) if lam > 0 else u0.clone()
        proxy = float((b[z] + r[u] - full).square().sum(1).mean())
        hz, bz = context_bits(z, ptr, dst, maxdeg, K, dev); hu, bu = context_bits(u, ptr, dst, maxdeg, R, dev)
        w = dict(shared); w['app_mapping'] = z.cpu().numpy().astype(np.uint32); w['app_residual_mapping'] = u.cpu().numpy().astype(np.uint32)
        w['app_rows'] = np.arange(n, dtype=np.uint32); w = SharedMesh(w).wire()
        v = assess(w, evaluate, val); row = dict(lam=lam, hit_z=hz, hit_u=hu, bits_z=bz, bits_u=bu, proxy_mse=proxy, validation=v['psnr'], changed_z=float((z != z0).float().mean()), changed_u=float((u != u0).float().mean()))
        log(f'lambda {lam}: hit z {hz:.3f} u {hu:.3f} | est bits z {bz:.2f} u {bu:.2f} | changed z {row["changed_z"]:.3f} u {row["changed_u"]:.3f} | val {v["psnr"]:.3f}')
        folder = out / f'lam{lam:g}'
        if folder.exists(): shutil.rmtree(folder)
        folder.mkdir()
        if a.refit_steps:
            w = fit(w, None, a.refit_steps, train, val, evaluate, ssim, args, folder / 'refit', appearance_only=True); v2 = assess(w, evaluate, val); row['validation_refit'] = v2['psnr']
            log(f'lambda {lam}: after {a.refit_steps}-step refit val {v2["psnr"]:.3f}')
        decoded, codec = export_asset(w, folder / 'delivery', a.xyz, a.dc); shutil.rmtree(folder / 'delivery' / 'native', ignore_errors=True)
        pb = codec['payload_bytes']; mb = sum(pb.get(k, 0) for k in ('app_mapping', 'app_residual_mapping', 'ctx_mapping', 'ctx_residual'))  # v12: ctx_* with stage2-v2
        row.update(file_MB=codec['file_bytes'] / 1e6, mapping_MB=mb / 1e6, real_bits_per_vertex_mapping=8 * mb / n)
        t = None if a.skip_test else assess(decoded, evaluate, test)
        if t is not None: row['test'] = {k: t[k] for k in ('psnr', 'ssim', 'lpips')}
        rows.append(row); save_json(out / 'sweep.json', dict(package_version=PACKAGE_VERSION, unit=unit, K=K, R=R, rows=rows))
        log(f'lambda {lam}: REAL {row["file_MB"]:.3f} MB, mapping {row["mapping_MB"]:.3f} MB ({row["real_bits_per_vertex_mapping"]:.2f} bits/vertex) | test {"-" if t is None else round(t["psnr"], 3)}')
    lines = [f'{"lambda":>7s} {"hit_z":>6s} {"hit_u":>6s} {"estbits":>8s} {"val":>7s} {"val_refit":>9s} {"MB":>8s} {"mapMB":>7s} {"test":>7s}']
    for q in rows:
        name = 'base' if q['lam'] is None else f"{q['lam']:g}"
        tpsnr = (q.get('test') or {}).get('psnr', float('nan'))
        lines.append(f"{name:>7s} {q['hit_z']:6.3f} {q['hit_u']:6.3f} {q['bits_z'] + q['bits_u']:8.2f} {q['validation']:7.3f} {q.get('validation_refit', float('nan')):9.3f} {q.get('file_MB', float('nan')):8.3f} {q.get('mapping_MB', float('nan')):7.3f} {tpsnr:7.3f}")
    (out / 'sweep.txt').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    main()
