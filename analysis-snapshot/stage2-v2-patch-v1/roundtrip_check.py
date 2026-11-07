import sys, json, shutil, time
from pathlib import Path
import numpy as np, torch
pkg = Path(sys.argv[1]); out = Path(sys.argv[2]); shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
for d in (pkg / 'renderer', pkg, pkg / 'codec'): sys.path.insert(0, str(d))
from runtime import export_asset
from fixtures import wire_grid
from appearance import expand_wire
from model import SharedMesh
def big(W=200, K=256, R=64, corr=.5, seed=0):
    rng = np.random.default_rng(seed); n = W * W; idx = np.arange(n).reshape(W, W)
    f = np.concatenate([np.stack([idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:]], -1).reshape(-1, 3), np.stack([idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 3)])
    x, y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, W)); v = np.stack((x, y, 3 + .1 * np.sin(3 * x) * y), -1).reshape(-1, 3).astype(np.float32)
    def codes(Kc):
        c = rng.integers(0, Kc, n).reshape(W, W)
        for i in range(W):
            for j in range(W):
                if rng.random() < corr and (i or j): c[i, j] = c[i - 1, j] if (i and (not j or rng.random() < .5)) else c[i, j - 1]
        return c.reshape(-1)
    w = dict(triangles_points=torch.from_numpy(v), features_dc=torch.from_numpy(rng.normal(0, .05, (n, 1, 3)).astype(np.float32)), _triangle_indices=torch.tensor(f, dtype=torch.int32),
             vertex_weight=torch.from_numpy(rng.normal(8, .5, (n, 1)).astype(np.float32)), sigma=torch.tensor(-.2), active_sh_degree=3, app_schema=2,
             app_bank=rng.normal(0, .025, (K, 15, 3)).astype(np.float32), app_mapping=codes(K).astype(np.uint16), app_gain=np.empty((0, 3), np.float32), app_rows=np.arange(n, dtype=np.uint32),
             app_correction=np.empty((0, 15, 3), np.float32), app_residual_bank=rng.normal(0, .01, (R, 15, 3)).astype(np.float32), app_residual_mapping=codes(R).astype(np.uint16))
    return w
for name, w in (('grid6', wire_grid()), ('big200', big())):
    for kw in ({}, dict(weight='keep', index_codec='zstd')) if 'index_codec' in export_asset.__code__.co_varnames else ({},):
        tag = name + ('-' + '-'.join(f'{k}={v}' for k, v in kw.items()) if kw else '-default')
        t = time.time(); dec, rep = export_asset(w, out / tag, 'q14', 'q8', **kw); dt = time.time() - t
        m = SharedMesh(dec, 'cpu'); col = m.colors(torch.zeros(3))
        same_idx = np.array_equal(np.sort(np.asarray(dec['app_mapping'])), np.sort(np.asarray(w['app_mapping'])))
        rest_ok = np.allclose(expand_wire(dec), expand_wire(w)[np.argsort(np.argsort(np.arange(len(w['app_mapping']))))] if False else True)
        pb = rep['payload_bytes']; keys = {k: v for k, v in pb.items() if k.startswith(('app_mapping', 'app_residual_mapping', 'ctx_', 'vertex_weight'))}
        print(f'{tag:28s} file {rep["file_bytes"]:8d} B  {dt:5.1f}s  audit {rep["semantic_audit"]["same_all_attributes"]}  colors finite {bool(torch.isfinite(col).all())}  '
              f'vw {tuple(dec["vertex_weight"].shape)} opacity err {rep.get("vertex_weight_const", {}).get("max_abs_opacity_error", 0):.2e}  fields {keys}')
