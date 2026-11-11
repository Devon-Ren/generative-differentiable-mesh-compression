"""dump_asset_fields_v1.py (2026-11-11): step 1 of the Draco + VQ baseline.  Runs in the stage-2 env (bash $PKG/stage2_run.sh dump_asset_fields_v1.py ...).
Reads our delivered asset (*.mslb) with the codec's own reader and writes an .npz with: xyz (float32, de-quantised), faces (int32), dc_q (the DC
colour as stored, integer), every per-vertex INTEGER field of the decoded asset (the appearance dictionary indices among them), and the dictionary
tables' shapes.  Nothing is re-encoded here; step 2 (draco_vq_v1.py, DracoPy env) does the generic encodes.
  bash $PKG/stage2_run.sh dump_asset_fields_v1.py --deliver-dir <...>/axisB/v1.0/deliver --out <npz>
"""
import argparse, sys, tempfile
from pathlib import Path
import numpy as np
VERSION = 'dump_asset_fields_v1 2026-11-11'
ap = argparse.ArgumentParser(); ap.add_argument('--stage2', required=True); ap.add_argument('--deliver-dir', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); pkg = Path(a.stage2).resolve()
for d in (pkg / 'renderer', pkg, pkg / 'codec'): sys.path.insert(0, str(d))
D = Path(a.deliver_dir).resolve(); assets = sorted((D / 'delivery').glob('*.mslb')); assert assets, 'no *.mslb under ' + str(D / 'delivery')
from topology import read_asset
cache = D / 'delivery' / 'native' if (D / 'delivery' / 'native').exists() else Path(tempfile.mkdtemp()) / 'native'
dec = read_asset(str(assets[0]), cache, Path(tempfile.mkdtemp()))
raw = {k: np.asarray(v) for k, v in dec.items() if hasattr(v, '__len__') and not isinstance(v, (str, bytes, dict))}
n = int(np.asarray(dec['triangles_points']).reshape(-1, 3).shape[0])
print(f'[dump] {VERSION} | {assets[0].name} {assets[0].stat().st_size} bytes | vertices {n} | fields: ' + ', '.join(f'{k} {v.shape} {v.dtype}' for k, v in raw.items()), flush=True)
ints = {k: v for k, v in raw.items() if v.dtype.kind in 'iu' and v.shape[:1] == (n,) and k not in ('_triangle_indices',)}
dc_q = raw['features_dc'] if raw['features_dc'].dtype.kind in 'iu' else None
try:
    from delivery import restore
    dec_r = restore(dict(dec)); xyz = np.asarray(dec_r['triangles_points'], np.float32).reshape(-1, 3)
except Exception as e:
    print('[dump] note: restore() failed (' + repr(e) + '); positions used as stored'); xyz = np.asarray(dec['triangles_points'], np.float32).reshape(-1, 3)
out = dict(xyz=xyz, faces=np.asarray(dec['_triangle_indices'], np.int32).reshape(-1, 3), asset_bytes=np.int64(assets[0].stat().st_size), n=np.int64(n))
if dc_q is not None: out['dc_q'] = dc_q.reshape(n, -1)
else: out['dc_f'] = np.asarray(dec['features_dc'], np.float32).reshape(n, -1)
for k, v in ints.items(): out['int_' + k] = v
tables = {k: v.shape for k, v in raw.items() if v.ndim == 2 and v.shape[0] in (4096, 256, 1024, 16384) and v.dtype.kind == 'f'}
print('[dump] integer per-vertex fields:', {k: (v.shape, v.dtype.name, int(v.min()), int(v.max())) for k, v in ints.items()}, '| float tables:', tables, '| dc', ('int ' + str(dc_q.dtype) + ' max ' + str(int(dc_q.max()))) if dc_q is not None else 'float', flush=True)
np.savez_compressed(a.out, **out); print('DUMP OK', a.out)
