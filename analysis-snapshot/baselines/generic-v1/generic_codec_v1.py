"""generic_codec_v1.py (2026-10-24): generic / off-the-shelf codecs applied to the SAME MeshSplatting asset.  CPU only.

Reads an original checkpoint (point_cloud_state_dict.pt: triangles_points, _triangle_indices, features_dc, features_rest,
vertex_weight) and produces, per row, the byte count of an honest container and (where lossy) a decoded checkpoint that
eval_ckpt_v1.py renders on the test views.  Rows:

  fp32_minimal      V x (3 xyz + 48 SH + 1 weight) x 4 B + F x 12 B                       computed, no file
  ply_zstd19        the same bytes through zstd -19 (write bytes, entropy-code them)      measured
  quant_ply_zstd19  xyz uint16 on the qp grid + SH uint8 (per-channel min/max) + int32 faces, zstd -19   measured + ckpt
  draco_geo         Draco (Edgebreaker, level 10) positions qp + faces only: connectivity + geometry reference   measured
  draco_full_sh8    Draco + 48-channel uint8 SH generic attribute (dc + rest, per-channel min/max)      measured + ckpt
  draco_full_sh16   same with uint16 SH                                                                  measured + ckpt
  draco_rgb8        Draco + uint8 RGB vertex colour (SH0 only -> the authors' RGB-only PLY in Draco form) measured + ckpt (rest = 0)
  texture_<R>       --texture: xatlas UV atlas, Draco(pos qp + faces + uv 12 bit) + baked vertex-colour texture R x R,
                    PNG and JPEG q90 bytes (the MPEG / texture-mapped route; carries no view dependence, quality = draco_rgb8)

Draco's quantization_bits is the per-axis bit depth over the bounding cube, i.e. the same meaning as our XYZ q<b>.
Decoded checkpoints keep Draco's vertex order / splits (Draco duplicates non-manifold vertices); vertex_weight is set to
the original's median logit (constant, as in our delivery format).  Usage:
  python generic_codec_v1.py --ckpt .../point_cloud_state_dict.pt --qp 14 --out /scratch/zr2245/analysis/generic/room [--texture 4096 8192] [--skip-zstd]
Writes <out>/rows.json (+ rows.md) and <out>/<row>/point_cloud_state_dict.pt for lossy rows.
"""
import argparse, io, json, math, sys, time, zlib
from pathlib import Path
import numpy as np

VERSION = 'generic-v1 2026-10-24'


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def to_np(t):
    return t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t)


def load_ckpt(path):
    import torch
    s = torch.load(path, map_location='cpu', weights_only=False)
    g = lambda *names: next(s[n] for n in names if n in s)
    xyz = to_np(g('triangles_points')).astype(np.float32)
    faces = to_np(g('_triangle_indices')).astype(np.int64)
    dc = to_np(g('features_dc', '_features_dc')).reshape(len(xyz), -1).astype(np.float32)     # (V, 3)
    rest = to_np(g('features_rest', '_features_rest')).reshape(len(xyz), -1).astype(np.float32)  # (V, 45)
    w = to_np(g('vertex_weight')).reshape(-1).astype(np.float32) if 'vertex_weight' in s else None
    return dict(xyz=xyz, faces=faces, dc=dc, rest=rest, weight=w, keys=sorted(s.keys()))


def sh_quant(sh, bits):
    lo = sh.min(0); hi = sh.max(0); scale = np.where(hi > lo, (2 ** bits - 1) / np.maximum(hi - lo, 1e-12), 0.0)
    q = np.rint((sh - lo) * scale).astype(np.uint8 if bits == 8 else np.uint16)
    return q, lo.astype(np.float32), hi.astype(np.float32)


def sh_dequant(q, lo, hi, bits):
    return (lo + q.astype(np.float32) * (hi - lo) / (2 ** bits - 1)).astype(np.float32)


def xyz_quant(xyz, qp):
    lo = xyz.min(0); extent = float((xyz.max(0) - lo).max()); step = extent / (2 ** qp - 1)
    q = np.rint((xyz - lo) / step).astype(np.uint16 if qp <= 16 else np.uint32)
    return q, lo.astype(np.float32), np.float32(step)


def rgb8_from_dc(dc):
    return np.clip(np.rint((0.28209479177387814 * dc + 0.5) * 255), 0, 255).astype(np.uint8)


def dc_from_rgb8(rgb):
    return ((rgb.astype(np.float32) / 255 - 0.5) / 0.28209479177387814).astype(np.float32)


def save_ckpt(path, xyz, faces, dc, rest, weight_logit):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    V = len(xyz)
    torch.save(dict(triangles_points=torch.from_numpy(np.ascontiguousarray(xyz, np.float32)),
                    _triangle_indices=torch.from_numpy(np.ascontiguousarray(faces, np.int32)),
                    features_dc=torch.from_numpy(np.ascontiguousarray(dc, np.float32)).reshape(V, 1, 3),
                    features_rest=torch.from_numpy(np.ascontiguousarray(rest, np.float32)).reshape(V, 15, 3),
                    vertex_weight=torch.full((V, 1), float(weight_logit), dtype=torch.float32), generic_codec=VERSION), path)


def zstd_bytes(blob, level=19):
    try:
        import zstandard as zstd
        return len(zstd.ZstdCompressor(level=level, threads=-1).compress(blob)), f'zstd{level}'
    except ImportError:
        import lzma
        return len(lzma.compress(blob, preset=9)), 'lzma9 (zstandard not installed)'


def draco_roundtrip(xyz, faces, qp, **attrs):
    import DracoPy
    t = time.time()
    buf = DracoPy.encode(xyz.astype(np.float32), faces.astype(np.uint32), quantization_bits=qp, compression_level=10, preserve_order=False, **attrs)
    enc = time.time() - t; d = DracoPy.decode(buf)
    return buf, d, enc


def rasterize_gouraud(uv_tri, col_tri, R):
    """uv_tri (T,3,2) in [0,1], col_tri (T,3,3) uint8 -> (R,R,3) uint8 texture, Gouraud-interpolated.  Triangles are
    processed in bounding-box size classes so the inner work is vectorised (no per-triangle Python loop)."""
    img = np.zeros((R, R, 3), np.uint8); px = uv_tri * (R - 1)
    lo = np.floor(px.min(1)).astype(np.int64); hi = np.ceil(px.max(1)).astype(np.int64)
    size = (hi - lo).max(1) + 1; order = np.argsort(size); classes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 100000]
    start = 0; done = 0
    for s in classes:
        end = np.searchsorted(size[order], s, side='right')
        ids = order[start:end]; start = end
        if len(ids) == 0: continue
        s_eff = min(s, int(size[ids].max())) if s < 100000 else int(size[ids].max())
        off = np.stack(np.meshgrid(np.arange(s_eff), np.arange(s_eff), indexing='ij'), -1).reshape(-1, 2)   # (s*s, 2) as (y, x)
        chunk = max(1, int(4e7 // (len(off) * 3)))
        for c0 in range(0, len(ids), chunk):
            t = ids[c0:c0 + chunk]; p = px[t]; o = lo[t]
            X = o[:, None, 0] + off[None, :, 1]; Y = o[:, None, 1] + off[None, :, 0]              # (n, s*s)
            x0, y0 = p[:, 0, 0][:, None], p[:, 0, 1][:, None]; x1, y1 = p[:, 1, 0][:, None], p[:, 1, 1][:, None]; x2, y2 = p[:, 2, 0][:, None], p[:, 2, 1][:, None]
            det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2); det = np.where(np.abs(det) < 1e-12, 1e-12, det)
            cx = X + 0.5; cy = Y + 0.5
            l0 = ((y1 - y2) * (cx - x2) + (x2 - x1) * (cy - y2)) / det; l1 = ((y2 - y0) * (cx - x2) + (x0 - x2) * (cy - y2)) / det; l2 = 1 - l0 - l1
            eps = -0.002; inside = (l0 >= eps) & (l1 >= eps) & (l2 >= eps) & (X < R) & (Y < R) & (X >= 0) & (Y >= 0)
            # also stamp the three vertex texels so 1-texel triangles are not lost
            col = col_tri[t].astype(np.float32)
            rgb = (l0[..., None] * col[:, None, 0] + l1[..., None] * col[:, None, 1] + l2[..., None] * col[:, None, 2])
            yi, xi = Y[inside], X[inside]; img[yi, xi] = np.clip(np.rint(rgb[inside]), 0, 255).astype(np.uint8)
            vx = np.clip(np.rint(p[..., 0]).astype(np.int64), 0, R - 1); vy = np.clip(np.rint(p[..., 1]).astype(np.int64), 0, R - 1)
            img[vy.reshape(-1), vx.reshape(-1)] = col.reshape(-1, 3).astype(np.uint8)
            done += len(t)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True); p.add_argument('--qp', type=int, required=True, help='XYZ bits (Room 14, garden 15, ...)')
    p.add_argument('--out', required=True); p.add_argument('--scene', default='')
    p.add_argument('--texture', type=int, nargs='*', default=[], help='atlas resolutions for the texture route, e.g. 4096 8192 (xatlas; slow on 6M faces)')
    p.add_argument('--skip-zstd', action='store_true'); p.add_argument('--jpeg-quality', type=int, default=90)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows_path = out / 'rows.json'; rows = json.load(open(rows_path)) if rows_path.exists() else {}
    def put(name, **kw):
        rows[name] = dict(name=name, scene=a.scene, qp=a.qp, **kw); json.dump(rows, open(rows_path, 'w'), indent=1); log(f'ROW {name}: ' + ', '.join(f'{k}={v}' for k, v in kw.items() if k != 'ckpt'))

    log(VERSION, '| loading', a.ckpt)
    m = load_ckpt(a.ckpt); xyz, faces, dc, rest, w = m['xyz'], m['faces'], m['dc'], m['rest'], m['weight']
    V, F = len(xyz), len(faces); sh = np.concatenate([dc, rest], 1)                     # (V, 48)
    wlogit = float(np.median(w)) if w is not None else 7.0
    log(f'{V} vertices, {F} faces, keys {m["keys"]}, weight logit median {wlogit:.3f}')
    MB = lambda b: round(b / 1e6, 3)

    put('fp32_minimal', bytes=V * 52 * 4 + F * 12, MB=MB(V * 52 * 4 + F * 12), note='xyz + 48 SH + weight float32, int32 faces')

    if not a.skip_zstd and 'ply_zstd19' not in rows:
        blob = xyz.tobytes() + sh.tobytes() + (w.tobytes() if w is not None else b'') + faces.astype(np.int32).tobytes()
        t = time.time(); n, how = zstd_bytes(blob); put('ply_zstd19', bytes=n, MB=MB(n), seconds=round(time.time() - t, 1), note=how)
        del blob

    # quantised PLY + zstd (same precision as ours: xyz qp, SH 8 bit)
    if 'quant_ply_zstd19' not in rows:
        xq, xlo, xstep = xyz_quant(xyz, a.qp); sq, slo, shi = sh_quant(sh, 8)
        blob = xq.tobytes() + sq.tobytes() + faces.astype(np.int32).tobytes()
        n, how = (len(blob), 'raw') if a.skip_zstd else zstd_bytes(blob)
        xr = (xlo + xq.astype(np.float32) * xstep); shr = sh_dequant(sq, slo, shi, 8)
        ck = out / 'quant_ply_zstd19' / 'point_cloud_state_dict.pt'; save_ckpt(ck, xr, faces, shr[:, :3], shr[:, 3:], wlogit)
        put('quant_ply_zstd19', bytes=n, MB=MB(n), note=f'xyz uint16 q{a.qp} + SH uint8 + int32 faces, {how}', ckpt=str(ck)); del blob, xq, sq

    try:
        import DracoPy  # noqa
    except ImportError:
        log('FAILED: pip install DracoPy'); sys.exit(1)

    if 'draco_geo' not in rows:
        buf, d, enc = draco_roundtrip(xyz, faces, a.qp)
        put('draco_geo', bytes=len(buf), MB=MB(len(buf)), decoded_vertices=int(len(d.points)), vertex_split=round(len(d.points) / V, 3), encode_s=round(enc, 1), note='positions + faces only')

    for bits, name in ((8, 'draco_full_sh8'), (16, 'draco_full_sh16')):
        if name in rows: continue
        sq, slo, shi = sh_quant(sh, bits)
        buf, d, enc = draco_roundtrip(xyz, faces, a.qp, generic_attributes={'sh': sq})
        att = d.get_attribute_by_name('sh')['data']; shr = sh_dequant(att, slo, shi, bits)
        ck = out / name / 'point_cloud_state_dict.pt'; save_ckpt(ck, np.asarray(d.points, np.float32), np.asarray(d.faces, np.int64), shr[:, :3], shr[:, 3:], wlogit)
        n = len(buf) + 96 * 4
        put(name, bytes=n, MB=MB(n), decoded_vertices=int(len(d.points)), vertex_split=round(len(d.points) / V, 3), encode_s=round(enc, 1), note=f'positions q{a.qp} + faces + 48-ch uint{bits} SH generic attribute', ckpt=str(ck))
        del buf, d, att, shr

    if 'draco_rgb8' not in rows:
        rgb = rgb8_from_dc(dc)
        buf, d, enc = draco_roundtrip(xyz, faces, a.qp, colors=rgb)
        col = np.asarray(d.colors)[:, :3].astype(np.uint8); dcr = dc_from_rgb8(col)
        ck = out / 'draco_rgb8' / 'point_cloud_state_dict.pt'; save_ckpt(ck, np.asarray(d.points, np.float32), np.asarray(d.faces, np.int64), dcr, np.zeros((len(dcr), 45), np.float32), wlogit)
        put('draco_rgb8', bytes=len(buf), MB=MB(len(buf)), decoded_vertices=int(len(d.points)), vertex_split=round(len(d.points) / V, 3), encode_s=round(enc, 1), note='positions + faces + uint8 RGB (SH0 only, no view dependence)', ckpt=str(ck))
        del buf, d

    if a.texture:
        try:
            import xatlas
            from PIL import Image
        except ImportError as e:
            log(f'FAILED texture route: {e!r} (pip install xatlas pillow)'); return
        if 'texture_atlas' not in rows:
            t = time.time(); log('xatlas.parametrize on', F, 'faces (can take a long time) ...')
            vmapping, indices, uvs = xatlas.parametrize(xyz.astype(np.float32), faces.astype(np.uint32))
            np.savez(out / 'atlas.npz', vmapping=vmapping, indices=indices, uvs=uvs)
            put('texture_atlas', seconds=round(time.time() - t, 1), atlas_vertices=int(len(vmapping)), atlas_faces=int(len(indices)), note='xatlas UV parametrisation (vertices duplicated at chart seams)')
        z = np.load(out / 'atlas.npz'); vmapping, indices, uvs = z['vmapping'], z['indices'], z['uvs'].astype(np.float32)
        if 'texture_draco_geo_uv' not in rows:
            import DracoPy
            buf = DracoPy.encode(xyz[vmapping].astype(np.float32), indices.astype(np.uint32), quantization_bits=a.qp, compression_level=10, preserve_order=False, tex_coord=np.ascontiguousarray(uvs, np.float64), tex_coord_quantization_bits=12)
            put('texture_draco_geo_uv', bytes=len(buf), MB=MB(len(buf)), note='Draco positions + faces + 12-bit UV on the atlas mesh')
        rgb = rgb8_from_dc(dc)[vmapping]; uv_tri = uvs[indices]; col_tri = rgb[indices]
        for R in a.texture:
            name = f'texture_{R}'
            if name in rows: continue
            t = time.time(); img = rasterize_gouraud(uv_tri, col_tri, R); im = Image.fromarray(img)
            png = io.BytesIO(); im.save(png, format='PNG', optimize=True); jpg = io.BytesIO(); im.save(jpg, format='JPEG', quality=a.jpeg_quality, subsampling=0)
            im.save(out / f'{name}.jpg', quality=a.jpeg_quality, subsampling=0)
            geo = rows['texture_draco_geo_uv']['bytes']; cov = float((img.sum(2) > 0).mean())
            put(name, png_bytes=png.tell(), jpeg_bytes=jpg.tell(), total_png_MB=MB(geo + png.tell()), total_jpeg_MB=MB(geo + jpg.tell()), texel_coverage=round(cov, 3), seconds=round(time.time() - t, 1),
                note=f'baked vertex colour, {R}x{R}, JPEG q{a.jpeg_quality}; quality = draco_rgb8 row (vertex colours only, no SH)')

    lines = ['| row | MB | decoded V | note |', '|---|---:|---:|---|'] + [f'| {r["name"]} | {r.get("MB", r.get("total_jpeg_MB", ""))} | {r.get("decoded_vertices", "")} | {r.get("note", "")} |' for r in rows.values()]
    (out / 'rows.md').write_text('\n'.join(lines) + '\n'); log('wrote', rows_path, 'and rows.md')


if __name__ == '__main__':
    main()
