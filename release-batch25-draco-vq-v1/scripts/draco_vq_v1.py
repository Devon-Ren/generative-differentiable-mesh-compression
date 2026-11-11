"""draco_vq_v1.py (2026-11-11): step 2 of the Draco + VQ baseline (DracoPy env, CPU).  From the .npz of dump_asset_fields_v1.py:
  draco_geo      Draco (Edgebreaker, level 10) positions on the qp grid + faces                                   -> connectivity + geometry reference
  draco_vq       Draco + ONE uint8 generic attribute carrying exactly our appearance payload per vertex:
                 dictionary index (16 bit, 2 bytes), residual index (8 bit), DC colour (3 x 7-bit as stored) = 6 bytes per vertex
  draco_geo+zstd Draco geometry + the same index / DC bytes entropy-coded separately with zstd -19 (no prediction)
  ours           the delivered file
The decoded quality of draco_vq is ours up to Draco's position re-quantisation on its own bounding box (same bit depth), so only bytes are compared.
  python draco_vq_v1.py --npz <npz> --qp 14 --scene room --target 10 --k_field <name> --r_field <name> [--rows <jsonl>]
Field names come from the dump's '[dump] integer per-vertex fields' line; with --auto the two index fields are picked as the integer fields whose
max is < 65536 (dictionary) and < 256 (residual), largest range first."""
import argparse, json, time, io
import numpy as np
import DracoPy
VERSION = 'draco_vq_v1 2026-11-11'
ap = argparse.ArgumentParser(); ap.add_argument('--npz', required=True); ap.add_argument('--qp', type=int, required=True); ap.add_argument('--scene', required=True); ap.add_argument('--target', required=True)
ap.add_argument('--k_field', default=None); ap.add_argument('--r_field', default=None); ap.add_argument('--auto', action='store_true'); ap.add_argument('--rows', default=None)
a = ap.parse_args(); z = np.load(a.npz); n = int(z['n']); xyz = z['xyz'].astype(np.float32); faces = z['faces'].astype(np.uint32)
ints = {k[4:]: z[k] for k in z.files if k.startswith('int_')}
if a.auto or not (a.k_field and a.r_field):
    cand = sorted(((int(v.max()), k) for k, v in ints.items() if v.ndim == 1 or v.shape[1] == 1), reverse=True)
    kf = next((k for m, k in cand if 256 <= m < 65536), None); rf = next((k for m, k in cand if m < 256 and k != kf), None)
    a.k_field = a.k_field or kf; a.r_field = a.r_field or rf
assert a.k_field in ints and a.r_field in ints, f'index fields not found; available: { {k: (v.shape, int(v.max())) for k, v in ints.items()} }'
ki = ints[a.k_field].reshape(n).astype(np.uint16); ri = ints[a.r_field].reshape(n).astype(np.uint8)
dc = z['dc_q'].reshape(n, -1).astype(np.uint8) if 'dc_q' in z.files else np.clip(np.rint((0.28209479177387814 * z['dc_f'].reshape(n, -1) + 0.5) * 127), 0, 127).astype(np.uint8)
attr = np.concatenate([(ki >> 8).astype(np.uint8)[:, None], (ki & 255).astype(np.uint8)[:, None], ri[:, None], dc[:, :3]], axis=1)  # (n, 6) uint8
def enc(**kw):
    t = time.time(); buf = DracoPy.encode(xyz, faces, quantization_bits=a.qp, compression_level=10, preserve_order=False, **kw); return buf, round(time.time() - t, 1)
geo, tg = enc(); vq, tv = enc(generic_attributes={'vq': attr})
try:
    import zstandard as zstd; zb = len(zstd.ZstdCompressor(level=19).compress(attr.tobytes(order='F'))); zhow = 'zstd19'  # channel-major so each stream is contiguous
except ImportError:
    import lzma; zb = len(lzma.compress(attr.tobytes(order='F'), preset=9)); zhow = 'lzma9 (zstandard not installed)'
ours = int(z['asset_bytes'])
row = dict(version=VERSION, scene=a.scene, target=a.target, vertices=n, faces=int(faces.shape[0]), qp=a.qp, k_field=a.k_field, r_field=a.r_field, k_max=int(ki.max()), r_max=int(ri.max()),
           ours_bytes=ours, draco_geo_bytes=len(geo), draco_vq_bytes=len(vq), draco_geo_plus_zstd_idx_bytes=len(geo) + zb, zstd_idx_bytes=zb, idx_coder=zhow,
           draco_vq_over_ours=round(len(vq) / ours, 3), draco_geo_zstd_over_ours=round((len(geo) + zb) / ours, 3), draco_geo_over_ours=round(len(geo) / ours, 3), enc_s=(tg, tv))
print('ROW ' + json.dumps(row))
if a.rows:
    with open(a.rows, 'a') as f: f.write(json.dumps(row) + '\n')
