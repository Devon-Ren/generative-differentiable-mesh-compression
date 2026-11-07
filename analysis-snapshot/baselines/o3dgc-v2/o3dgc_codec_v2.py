"""o3dgc_codec_v2.py (2026-11-01, VERSION o3dgc-v2 = v1 rows + geometry-only row + MPEG-geometry texture row): MPEG-4 Part 16 SC3DMC / TFAN (Open3DGC) rows on the SAME MeshSplatting asset,
in the generic-v1 row format (rows.json next to the generic rows; decoded checkpoints for eval_ckpt_v1.py).  CPU only.
Rows:
  o3dgc_tfan_sh8int   positions q<qp> (parallelogram) + SH pre-quantised per channel to uint8 (same grid as quant_ply_zstd19) carried as
                      six 8-dim INTEGER attributes (lossless, differential prediction) -> decoded quality == quant_ply_zstd19 row.
  o3dgc_tfan_sh8f     positions q<qp> + SH as six 8-dim FLOAT attributes quantised by Open3DGC itself at 8 bits (its range rule) -> own checkpoint.
  o3dgc_tfan_geo      positions q<qp> + faces only (no attributes): the MPEG geometry budget, for the "MPEG geometry + our SH streams" ablation.
  o3dgc_texture_<R>   texture route with MPEG geometry: Open3DGC(positions[vmapping] q<qp>, atlas faces, UV as a 2-dim float attribute at 12 bit)
                      + the JPEG bytes of the existing texture_<R> row (generic_codec_v1 --texture); needs atlas.npz + texture_<R> in rows.json.
                      Quality = draco_rgb8 / texture_<R> row by construction (vertex colours only).
usage: python o3dgc_codec_v2.py --ckpt <state_dict.pt> --qp 14 --out <generic-v1/room dir> --driver <o3dgc_driver> --scene room"""
import argparse, json, struct, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'generic-v1'))
from generic_codec_v1 import load_ckpt, sh_quant, sh_dequant, save_ckpt, log  # noqa: E402
VERSION = 'o3dgc-v2 2026-11-01'
def write_bundle(path, xyz, faces, sh, mode):
    with open(path, 'wb') as f:
        f.write(b'O3DGCB01' + struct.pack('<IIII', len(xyz), len(faces), mode, sh.shape[1]))
        f.write(np.ascontiguousarray(xyz, np.float32).tobytes()); f.write(np.ascontiguousarray(faces, np.int32).tobytes())
        f.write(np.ascontiguousarray(sh, np.float32 if mode == 0 else np.uint8).tobytes())
def read_bundle(path):
    b = open(path, 'rb').read(); assert b[:8] == b'O3DGCB01', 'bad decoded bundle'
    V, F, mode, D = struct.unpack('<IIII', b[8:24]); o = 24
    xyz = np.frombuffer(b, np.float32, V * 3, o).reshape(V, 3); o += V * 12
    faces = np.frombuffer(b, np.int32, F * 3, o).reshape(F, 3); o += F * 12
    sh = (np.frombuffer(b, np.float32, V * D, o) if mode == 0 else np.frombuffer(b, np.uint8, V * D, o)).reshape(V, D)
    return xyz.copy(), faces.astype(np.int64), sh.copy(), mode
def main():
    p = argparse.ArgumentParser(); p.add_argument('--ckpt', required=True); p.add_argument('--qp', type=int, required=True); p.add_argument('--out', required=True)
    p.add_argument('--driver', required=True); p.add_argument('--scene', default=''); p.add_argument('--qattr', type=int, default=8); a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rows_path = out / 'rows.json'; rows = json.load(open(rows_path)) if rows_path.exists() else {}
    def put(name, **kw):
        rows[name] = dict(name=name, scene=a.scene, qp=a.qp, **kw); json.dump(rows, open(rows_path, 'w'), indent=1); log(f'ROW {name}: ' + ', '.join(f'{k}={v}' for k, v in kw.items() if k != 'ckpt'))
    log(VERSION, '| loading', a.ckpt); m = load_ckpt(a.ckpt); xyz, faces, dc, rest, w = m['xyz'], m['faces'], m['dc'], m['rest'], m['weight']
    V, F = len(xyz), len(faces); sh = np.concatenate([dc, rest], 1); wlogit = float(np.median(w)) if w is not None else 7.0; MB = lambda b: round(b / 1e6, 3)
    log(f'{V} vertices, {F} faces')
    for name, mode in (('o3dgc_tfan_sh8int', 1), ('o3dgc_tfan_sh8f', 0)):
        if name in rows: log('skip', name, '(exists)'); continue
        d = out / name; d.mkdir(exist_ok=True); bundle = d / 'in.bundle'; stream = d / 'mesh.o3dgc'; dec = d / 'out.decoded'
        if mode == 1: sq, slo, shi = sh_quant(sh, 8); write_bundle(bundle, xyz, faces, sq, 1)
        else: write_bundle(bundle, xyz, faces, sh, 0)
        t = time.time(); r = subprocess.run([a.driver, str(bundle), str(stream), str(dec), str(a.qp), str(a.qattr)], capture_output=True, text=True); enc = time.time() - t
        log(r.stdout.strip()); 
        if r.returncode != 0: log('FAILED driver:', r.stderr.strip()[-800:]); json.dump(rows, open(rows_path, 'w'), indent=1); sys.exit(1)
        xyz2, faces2, sh2, _ = read_bundle(dec); n = stream.stat().st_size + (96 * 4 if mode == 1 else 0)
        shr = sh_dequant(sh2, slo, shi, 8) if mode == 1 else sh2.astype(np.float32)
        ck = d / 'point_cloud_state_dict.pt'; save_ckpt(ck, xyz2, faces2, shr[:, :3], shr[:, 3:], wlogit)
        note = (f'MPEG-4 SC3DMC TFAN (Open3DGC): positions q{a.qp} parallelogram + 48-ch uint8 SH as 6x8-dim int attributes (lossless; same grid as quant_ply_zstd19)' if mode == 1
                else f'MPEG-4 SC3DMC TFAN (Open3DGC): positions q{a.qp} + 48-ch SH as 6x8-dim float attributes, Open3DGC {a.qattr}-bit quantisation')
        put(name, bytes=n, MB=MB(n), decoded_vertices=int(len(xyz2)), vertex_split=round(len(xyz2) / V, 3), encode_decode_s=round(enc, 1), note=note, ckpt=str(ck))
        bundle.unlink(); dec.unlink()
    # --- geometry only (mode 1, D = 0): MPEG budget for connectivity + positions
    if 'o3dgc_tfan_geo' in rows: log('skip o3dgc_tfan_geo (exists)')
    else:
        d = out / 'o3dgc_tfan_geo'; d.mkdir(exist_ok=True); bundle = d / 'in.bundle'; stream = d / 'mesh.o3dgc'; dec = d / 'out.decoded'
        write_bundle(bundle, xyz, faces, np.zeros((V, 0), np.uint8), 1)
        t = time.time(); r = subprocess.run([a.driver, str(bundle), str(stream), str(dec), str(a.qp), str(a.qattr)], capture_output=True, text=True); enc = time.time() - t
        log(r.stdout.strip())
        if r.returncode != 0: log('FAILED driver (geo):', r.stderr.strip()[-800:]); sys.exit(1)
        xyz2, faces2, _, _ = read_bundle(dec); n = stream.stat().st_size
        put('o3dgc_tfan_geo', bytes=n, MB=MB(n), decoded_vertices=int(len(xyz2)), decoded_faces=int(len(faces2)), vertex_split=round(len(xyz2) / V, 3), encode_decode_s=round(enc, 1),
            note=f'MPEG-4 SC3DMC TFAN (Open3DGC): positions q{a.qp} + faces only, no attributes (geometry budget)')
        bundle.unlink(); dec.unlink()
    # --- texture route with MPEG geometry: atlas mesh + 12-bit UV through Open3DGC, colour = the existing baked JPEG
    atlas = out / 'atlas.npz'; tex_rows = [k for k in rows if k.startswith('texture_') and 'jpeg_bytes' in rows[k]]
    if not atlas.exists() or not tex_rows: log('no atlas.npz / texture_<R> rows in', out, '-> skipping o3dgc_texture rows (run generic_codec_v1 --texture first)')
    else:
        z = np.load(atlas); vmapping, indices, uvs = z['vmapping'], z['indices'], z['uvs'].astype(np.float32)
        name_geo = 'o3dgc_texture_geo_uv'
        if name_geo in rows: log('skip', name_geo, '(exists)')
        else:
            d = out / name_geo; d.mkdir(exist_ok=True); bundle = d / 'in.bundle'; stream = d / 'mesh.o3dgc'; dec = d / 'out.decoded'
            write_bundle(bundle, xyz[vmapping], indices.astype(np.int32), uvs, 0)
            t = time.time(); r = subprocess.run([a.driver, str(bundle), str(stream), str(dec), str(a.qp), '12'], capture_output=True, text=True); enc = time.time() - t
            log(r.stdout.strip())
            if r.returncode != 0: log('FAILED driver (texture geo+uv):', r.stderr.strip()[-800:]); sys.exit(1)
            xyz2, faces2, uv2, _ = read_bundle(dec); n = stream.stat().st_size
            put(name_geo, bytes=n, MB=MB(n), decoded_vertices=int(len(xyz2)), atlas_vertices=int(len(vmapping)), encode_decode_s=round(enc, 1),
                note=f'Open3DGC positions q{a.qp} + atlas faces + 12-bit UV (2-dim float attribute) on the xatlas mesh')
            bundle.unlink(); dec.unlink()
        geo = rows[name_geo]['bytes']
        for k in tex_rows:
            name = 'o3dgc_' + k
            if name in rows: log('skip', name, '(exists)'); continue
            jb = rows[k]['jpeg_bytes']; put(name, bytes=geo + jb, MB=MB(geo + jb), geo_bytes=geo, jpeg_bytes=jb,
                note=f'texture route with MPEG geometry: {name_geo} + JPEG of {k}; quality = draco_rgb8 row (vertex colours only, no SH)')
    log('done ->', rows_path)
if __name__ == '__main__': main()
