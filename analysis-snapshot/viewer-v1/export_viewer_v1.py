"""export_viewer_v1.py (2026-10-25): decode a delivered asset.mslb (or read an original checkpoint .pt) and write the
compact viewer file <out>.msv for viewer.html (standard mesh + per-vertex SH via codebooks, standard GPU pipeline).

usage (inside the stage2 env, e.g. via vis_collapse/stage2_run.sh):
  python export_viewer_v1.py --stage2 /scratch/zr2245/shared-collapse-stage2-v2 --asset .../delivery/asset.mslb \
      --cameras /scratch/zr2245/room-vis-collapse/original/cameras.json --out room-lam0.0062.msv [--label room]
  python export_viewer_v1.py --stage2 ... --ckpt .../point_cloud_state_dict.pt --cameras ... --out room-original.msv

.msv layout: b'MSV1' | uint32 header_len | JSON header | blobs in header order (little endian):
  xyz float32 [V,3] | faces uint32 [F,3] | dc float32 [V,3] | mapping uint32 [V] | residual uint32 [V] (0xFFFFFFFF = none)
  | bank float32 [K,45] | residual_bank float32 [R,45]
Also prints decode wall time (read + rebuild, native decoders precompiled into --native-cache) for the paper's decode-time column."""
import argparse, json, sys, time, struct, tempfile
from pathlib import Path
import numpy as np

def arr(t):
    return t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t)

def from_asset(path, stage2, native_cache):
    for sub in ('renderer', '', 'codec'):
        sys.path.insert(0, str(Path(stage2) / sub))
    import topology as tc, local_codec as lc, index_context as ic
    from delivery import restore
    cache = Path(native_cache); cache.mkdir(parents=True, exist_ok=True)
    lc.native(cache); tc.edge_native(cache); ic.native(cache)     # compile the three native decoders once, outside the timer
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wire = tc.read_asset(path, cache, tmp)                    # container -> faces -> XYZ -> SH indices (all streams)
    t_read = time.perf_counter() - t0
    state = restore(dict(wire))                                   # dequantise XYZ / DC, constant opacity
    t_all = time.perf_counter() - t0
    xyz = arr(state['triangles_points']).astype(np.float32).reshape(-1, 3)
    faces = arr(state['_triangle_indices']).astype(np.uint32).reshape(-1, 3)
    dc = arr(state['features_dc']).astype(np.float32).reshape(len(xyz), 3)
    if wire.get('app_schema') != 2 or np.asarray(wire.get('app_gain', ())).size or np.asarray(wire.get('app_correction', ())).size:
        raise SystemExit('FAILED: expected app_schema 2 without gains/corrections; got schema %r' % wire.get('app_schema'))
    bank = np.asarray(wire['app_bank'], np.float32).reshape(-1, 45)
    mapping = np.asarray(wire['app_mapping']).astype(np.uint32)
    rbank = np.asarray(wire['app_residual_bank'], np.float32).reshape(-1, 45)
    rows = np.asarray(wire['app_rows']).astype(np.int64); rmap = np.asarray(wire['app_residual_mapping']).astype(np.uint32)
    residual = np.full(len(xyz), 0xFFFFFFFF, np.uint32); residual[rows] = rmap
    meta = dict(source='asset', asset=str(path), file_bytes=Path(path).stat().st_size, decode_read_s=round(t_read, 3), decode_total_s=round(t_all, 3),
                opacity_const=float(wire.get('vertex_weight_const', 0.0)), opacity_floor=float(wire.get('opacity_floor', 0.999)))
    return xyz, faces, dc, mapping, residual, bank, rbank, meta

def from_ckpt(path):
    import torch
    t0 = time.perf_counter(); s = torch.load(path, map_location='cpu', weights_only=False); t_all = time.perf_counter() - t0
    g = lambda *n: next(s[k] for k in n if k in s)
    xyz = arr(g('triangles_points')).astype(np.float32).reshape(-1, 3)
    faces = arr(g('_triangle_indices')).astype(np.uint32).reshape(-1, 3)
    dc = arr(g('features_dc', '_features_dc')).astype(np.float32).reshape(len(xyz), 3)
    rest = arr(g('features_rest', '_features_rest')).astype(np.float32).reshape(len(xyz), 45)
    meta = dict(source='checkpoint', asset=str(path), file_bytes=Path(path).stat().st_size, decode_total_s=round(t_all, 3))
    return xyz, faces, dc, np.arange(len(xyz), dtype=np.uint32), np.full(len(xyz), 0xFFFFFFFF, np.uint32), rest, np.zeros((0, 45), np.float32), meta

def load_cameras(path, limit):
    if not path: return []
    cams = json.loads(Path(path).read_text()); out = []
    for c in cams[:limit]:
        out.append(dict(id=c.get('id'), name=c.get('img_name'), width=int(c['width']), height=int(c['height']), fx=float(c['fx']), fy=float(c['fy']),
                        position=[float(v) for v in c['position']], rotation=[[float(v) for v in r] for r in c['rotation']]))
    return out

def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage2', default='/scratch/zr2245/shared-collapse-stage2-v2')
    p.add_argument('--asset'); p.add_argument('--ckpt'); p.add_argument('--cameras'); p.add_argument('--out', required=True)
    p.add_argument('--label', default=''); p.add_argument('--max-cameras', type=int, default=400)
    p.add_argument('--native-cache', default='', help='dir for compiled decoders (default <out dir>/native-cache)')
    a = p.parse_args()
    if bool(a.asset) == bool(a.ckpt): raise SystemExit('FAILED: give exactly one of --asset / --ckpt')
    xyz, faces, dc, mapping, residual, bank, rbank, meta = from_asset(a.asset, a.stage2, a.native_cache or Path(a.out).resolve().parent / 'native-cache') if a.asset else from_ckpt(a.ckpt)
    if faces.max() >= len(xyz): raise SystemExit('FAILED: face index out of range')
    if mapping.max() >= len(bank): raise SystemExit('FAILED: mapping out of range')
    if len(rbank) and residual[residual != 0xFFFFFFFF].size and residual[residual != 0xFFFFFFFF].max() >= len(rbank): raise SystemExit('FAILED: residual mapping out of range')
    blobs = [('xyz', xyz), ('faces', faces), ('dc', dc), ('mapping', mapping), ('residual', residual), ('bank', bank), ('residual_bank', rbank)]
    lo, hi = xyz.min(0).tolist(), xyz.max(0).tolist()
    header = dict(version=1, label=a.label or Path(a.out).stem, V=int(len(xyz)), F=int(len(faces)), K=int(len(bank)), R=int(len(rbank)),
                  bbox=[lo, hi], cameras=load_cameras(a.cameras, a.max_cameras), meta=meta,
                  blobs=[dict(name=n, dtype=str(b.dtype), shape=list(b.shape), bytes=int(b.nbytes)) for n, b in blobs])
    hb = json.dumps(header, separators=(',', ':')).encode(); hb += b' ' * (-len(hb) % 4)   # keep blobs 4-byte aligned for typed-array views
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('wb') as f:
        f.write(b'MSV1'); f.write(struct.pack('<I', len(hb))); f.write(hb)
        for _, b in blobs: f.write(np.ascontiguousarray(b).astype(b.dtype.newbyteorder('<') if b.dtype.byteorder == '>' else b.dtype).tobytes())
    print(f"VIEWER {header['label']} | V {header['V']} F {header['F']} K {header['K']} R {header['R']} | cameras {len(header['cameras'])} | "
          f"source {meta['source']} {meta['file_bytes']/1e6:.3f} MB | decode {meta.get('decode_read_s', '-')} s read, {meta['decode_total_s']} s total | "
          f"msv {out.stat().st_size/1e6:.1f} MB -> {out}")

if __name__ == '__main__': main()
