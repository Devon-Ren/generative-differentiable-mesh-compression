"""decode_asset_v1.py (VERSION decode_asset_v1 2026-11-10): decode a delivered asset (deliver/delivery/asset.mslb) with the stage-2 decoder and write a plain
MeshSplatting-style checkpoint (triangles_points, _triangle_indices, vertex_weight, features_dc, features_rest) that render_ckpt_v1.py / eval_ckpt_v1.py can render.
Run inside the stage-2 env:  bash $PKG/stage2_run.sh decode_asset_v1.py --deliver-dir <...>/axisB/v1.0/deliver --out <dir>/decoded.pt
(stage2_run.sh prepends --stage2 <PKG>).  Uses the delivery's own compiled native coder (delivery/native) as the cache.  If the decode fails, falls
back to shared-final.pt (the fitted shared mesh before quantisation and entropy coding) and writes <out>-prequant.pt instead, with a WARNING.
"""
import argparse, sys, tempfile, traceback
from pathlib import Path
import numpy as np
import torch

VERSION = 'decode_asset_v1 2026-11-10'


def T(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu()
    return torch.as_tensor(np.asarray(x))


def to_ckpt(w, n_hint=None):
    xyz = T(w['triangles_points']).float().reshape(-1, 3); n = xyz.shape[0]
    faces = T(w['_triangle_indices']).to(torch.int32).reshape(-1, 3)
    dc = T(w['features_dc']).float().reshape(n, 1, 3)
    vw = T(w['vertex_weight']).float().reshape(-1)
    if vw.numel() == 1: vw = vw.expand(n).clone()
    assert vw.numel() == n, f'vertex_weight has {vw.numel()} entries for {n} vertices'
    return dict(triangles_points=xyz.contiguous(), _triangle_indices=faces.contiguous(), vertex_weight=vw.reshape(n, 1).contiguous(), features_dc=dc.contiguous()), n


def rest_of(w, n):
    from appearance import expand_wire
    r = T(expand_wire(w)).float()
    if r.ndim == 2 and r.shape[1] == 45: r = r.reshape(-1, 15, 3)
    if r.ndim == 3 and r.shape[1] == 3 and r.shape[2] == 15: r = r.permute(0, 2, 1)
    assert r.shape == (n, 15, 3), f'expand_wire returned {tuple(r.shape)}, expected ({n}, 15, 3)'
    return r.contiguous()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage2', required=True); ap.add_argument('--deliver-dir', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    pkg = Path(a.stage2).resolve()
    for d in (pkg / 'renderer', pkg, pkg / 'codec'): sys.path.insert(0, str(d))
    D = Path(a.deliver_dir).resolve(); out = Path(a.out)
    assets = sorted((D / 'delivery').glob('*.mslb')) if (D / 'delivery').exists() else []
    print(f'[decode] {VERSION} | deliver dir {D} | assets {[p.name for p in assets]}', flush=True)
    try:
        if not assets: raise FileNotFoundError('no *.mslb under ' + str(D / 'delivery'))
        from topology import read_asset
        cache = D / 'delivery' / 'native' if (D / 'delivery' / 'native').exists() else Path(tempfile.mkdtemp()) / 'native'
        tmp = Path(tempfile.mkdtemp())
        dec = read_asset(str(assets[0]), cache, tmp)
        print('[decode] decoded keys:', {k: (tuple(np.asarray(v).shape) if hasattr(v, '__len__') and not isinstance(v, (str, bytes)) else type(v).__name__) for k, v in dec.items()}, flush=True)
        ck, n = to_ckpt(dec); ck['features_rest'] = rest_of(dec, n); tag = 'decoded'
    except Exception as e:
        print('[decode] WARNING: decoding the asset failed (' + repr(e) + '); falling back to shared-final.pt (pre-quantisation fitted mesh)', flush=True); traceback.print_exc()
        w = torch.load(D / 'shared-final.pt', map_location='cpu', weights_only=False)
        ck, n = to_ckpt(w); ck['features_rest'] = rest_of(w, n); tag = 'prequant'; out = out.with_name(out.stem + '-prequant.pt')
    for k, v in ck.items(): assert torch.isfinite(v.float()).all(), f'{k} has non-finite values'
    torch.save(ck, out)
    print(f'[decode] {tag}: wrote {out} | vertices {n} faces {ck["_triangle_indices"].shape[0]} | ' + ' '.join(f'{k} {tuple(v.shape)} {v.dtype}' for k, v in ck.items()), flush=True)
    print('DECODE OK ' + tag)


if __name__ == '__main__':
    main()
