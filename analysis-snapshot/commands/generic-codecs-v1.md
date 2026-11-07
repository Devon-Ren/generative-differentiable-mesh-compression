# generic-codecs-v1（2026-10-24）：同一 MeshSplatting asset 上的通用 codec baseline（Draco / zstd / texture 路线）+ stage2 codec 源码打包

全部在你的 GPU node 上跑（CPU 部分：zstd、Draco、xatlas 烤 texture；GPU 部分：每个解码后的 checkpoint 在 test views 上渲染，4× supersample，和所有 Stage-2 数字同一 protocol）。Room ≈ 20–40 min（xatlas 6.4M 面是大头，时间未知），garden 同。输出一行一个 row 到 `/scratch/zr2245/analysis/generic-codecs.txt`。

## 0. 打包 stage2-v2 的 codec 源码（只打源码，不打 .so / 数据）
```bash
cd /scratch/zr2245 && Z=stage2-v2-src-$(date +%m%d).zip && rm -f $Z && zip -r $Z shared-collapse-stage2-v2 -i '*.py' '*.cpp' '*.cc' '*.h' '*.hpp' '*.cu' '*.sh' '*.md' '*.txt' '*.json' '*/STAGE2_VERSION' -x '*/__pycache__/*' '*/.git/*' && ls -la $Z && unzip -l $Z | tail -3
```
把 `/scratch/zr2245/$Z` 发我（VS Code 右键 Download，或者 scp）。

## 1. 安装脚本（写三个文件到 /scratch/zr2245/analysis/generic-v1/）
```bash
G=/scratch/zr2245/analysis/generic-v1; mkdir -p $G
cat > $G/generic_codec_v1.py <<'PYEOF1'
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
PYEOF1
cat > $G/eval_ckpt_v1.py <<'PYEOF2'
"""eval_ckpt_v1.py (2026-10-24): render a (decoded) checkpoint on the test views with the upstream renderer, 4x supersample.
Cameras/splits come from the ORIGINAL run dir (same as every Stage-2 number); the mesh tensors are swapped in from --ckpt
(triangles_points, _triangle_indices, features_dc, features_rest, vertex_weight; sigma / sh degree / opacity floor stay
the original's).  Prints one row and writes <out>.json (mean PSNR / SSIM / LPIPS-vgg, worst view, per-view rows).
  python eval_ckpt_v1.py --lab /scratch/zr2245/meshsplatting-original-lossless-v1 --model <run>/original \
      --dataset /scratch/zr2245/datasets/m360/room --images images_2 --ckpt <dir>/point_cloud_state_dict.pt --out <dir>/eval.json
"""
import argparse, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lab', required=True); p.add_argument('--model', required=True); p.add_argument('--dataset', required=True)
    p.add_argument('--images', default='images_2'); p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True)
    p.add_argument('--supersample', type=int, default=4); p.add_argument('--label', default='')
    a = p.parse_args()
    if not torch.cuda.is_available(): sys.exit('FAILED: CUDA required (run on the GPU node)')
    upstream = Path(a.lab).resolve() / 'upstream'
    sys.path.insert(0, str(upstream / 'submodules/effrdel/src')); sys.path.insert(0, str(upstream))
    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render
    from utils.loss_utils import ssim
    import lpips
    data = SimpleNamespace(source_path=str(Path(a.dataset).resolve()), model_path=str(Path(a.model).resolve()), images=a.images,
                           resolution=1, eval=True, white_background=False, data_device='cpu', sh_degree=3)
    mesh = TriangleModel(3); scene = Scene(data, mesh, None, None, load_iteration=30000, shuffle=False)
    test = sorted(scene.getTestCameras(), key=lambda c: (c.image_name, c.uid))
    w = torch.load(a.ckpt, map_location='cpu', weights_only=False); dev = 'cuda'
    mesh.vertices = w['triangles_points'].to(dev).float().contiguous()
    mesh._triangle_indices = w['_triangle_indices'].to(dev).to(torch.int32).contiguous()
    mesh.vertex_weight = w['vertex_weight'].to(dev).float().reshape(-1, 1).contiguous()
    mesh._features_dc = w['features_dc'].to(dev).float().reshape(len(mesh.vertices), 1, 3).contiguous()
    mesh._features_rest = w['features_rest'].to(dev).float().reshape(len(mesh.vertices), 15, 3).contiguous()
    nf = len(mesh._triangle_indices)
    for name in ('image_size', 'importance_score', 'pixel_count'):
        setattr(mesh, name, torch.zeros(nf, device=dev))
    mesh.scaling = a.supersample; del w
    log(f'{a.label or a.ckpt}: {len(mesh.vertices)} vertices, {nf} faces, {len(test)} test views')
    bg = torch.zeros(3, device=dev); pipe = SimpleNamespace(debug=False, convert_SHs_python=False, depth_ratio=1.)
    perceptual = lpips.LPIPS(net='vgg').cuda().eval()
    rows = []
    with torch.no_grad():
        for c in test:
            im = render(c, mesh, pipe, bg)['render'].clamp(0, 1); gt = c.original_image[:3].cuda().clamp(0, 1)
            rows.append(dict(camera=c.image_name, psnr=float(-10 * torch.log10((im - gt).square().mean().clamp_min(1e-12))), ssim=float(ssim(im, gt)),
                             lpips=float(perceptual(im[None] * 2 - 1, gt[None] * 2 - 1).mean())))
    res = dict(label=a.label, ckpt=a.ckpt, vertices=len(mesh.vertices), faces=nf, views=len(rows),
               psnr=float(np.mean([r['psnr'] for r in rows])), ssim=float(np.mean([r['ssim'] for r in rows])), lpips=float(np.mean([r['lpips'] for r in rows])),
               worst_view_psnr=float(min(r['psnr'] for r in rows)), per_view=rows)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f'EVAL {a.label} | V {res["vertices"]} F {nf} | test {res["psnr"]:.3f} / {res["ssim"]:.4f} / {res["lpips"]:.4f} | worst {res["worst_view_psnr"]:.2f}', flush=True)


if __name__ == '__main__':
    main()
PYEOF2
cat > $G/run_generic_v1.sh <<'SHEOF'
#!/bin/bash
# run_generic_v1.sh (2026-10-24): generic-codec baselines on one scene's ORIGINAL (level-1.0) mesh. Run on the GPU node:
#   bash /scratch/zr2245/analysis/generic-v1/run_generic_v1.sh room    (then garden)
# CPU part: zstd / Draco / xatlas texture rows -> $OUT/rows.json ; GPU part: render each decoded checkpoint on the test views.
# Rows are appended to /scratch/zr2245/analysis/generic-codecs.txt. Restartable: existing rows / eval.json are skipped.
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
AN=/scratch/zr2245/analysis; G=$AN/generic-v1; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; TAB=$AN/generic-codecs.txt
SCENE=${1:?usage: run_generic_v1.sh room|garden}; TEX=${TEX:-"4096 8192"}
case $SCENE in
  room)   RUN=/scratch/zr2245/room-fixed-sharing-0N4OSU/run/original; DATA=/scratch/zr2245/datasets/m360/room;   IMAGES=images_2; QP=14;;
  garden) RUN=/scratch/zr2245/garden-vis-collapse/original;            DATA=/scratch/zr2245/datasets/m360/garden; IMAGES=images_4; QP=15;;
  *)      RUN=${RUN:?set RUN=<dir with point_cloud/iteration_30000>}; DATA=${DATA:?}; IMAGES=${IMAGES:-images_4}; QP=${QP:?};;
esac
CK=$RUN/point_cloud/iteration_30000/point_cloud_state_dict.pt; OUT=$G/$SCENE; mkdir -p $OUT
for f in $CK $DATA/$IMAGES/. $LAB/upstream/train.py $G/generic_codec_v1.py $G/eval_ckpt_v1.py; do [ -e $f ] || { echo "FAILED: missing $f"; exit 1; }; done
python -c "import DracoPy, xatlas, zstandard, PIL" 2>/dev/null || { echo "=== pip install DracoPy xatlas zstandard pillow ==="; pip install DracoPy xatlas zstandard pillow > $OUT/pip.log 2>&1 || { echo "FAILED: pip (see $OUT/pip.log)"; tail -5 $OUT/pip.log; exit 1; }; }
python -c "import DracoPy, xatlas, zstandard" || { echo "FAILED: imports after pip"; exit 1; }
echo "=== $SCENE: CPU rows (zstd, Draco, texture $TEX) -> $OUT  $(date) ==="
python $G/generic_codec_v1.py --ckpt $CK --qp $QP --out $OUT --scene $SCENE --texture $TEX > $OUT/codec.log 2>&1 || { echo "WARNING: generic_codec_v1.py exited non-zero (rows written so far are kept)"; tail -15 $OUT/codec.log; }
grep '^\[.*ROW' $OUT/codec.log
[ -f $OUT/rows.json ] || { echo "FAILED: no rows.json"; tail -20 $OUT/codec.log; exit 1; }
echo "=== $SCENE: GPU eval of decoded checkpoints  $(date) ==="
ev () { # label ckpt
  [ -f $OUT/$1.eval.json ] && { echo "skip eval $1 (exists)"; return; }
  python $G/eval_ckpt_v1.py --lab $LAB --model $RUN --dataset $DATA --images $IMAGES --ckpt $2 --out $OUT/$1.eval.json --label "$SCENE $1" > $OUT/$1.eval.log 2>&1 \
    && grep '^EVAL' $OUT/$1.eval.log || { echo "FAILED: eval $1"; tail -8 $OUT/$1.eval.log; }
}
ev original $CK
for row in quant_ply_zstd19 draco_full_sh8 draco_full_sh16 draco_rgb8; do [ -f $OUT/$row/point_cloud_state_dict.pt ] && ev $row $OUT/$row/point_cloud_state_dict.pt || echo "no checkpoint for $row"; done
python - <<ASM
import json, os
out='$OUT'; rows=json.load(open(out+'/rows.json')); ev={}
for f in os.listdir(out):
    if f.endswith('.eval.json'): ev[f[:-10]]=json.load(open(out+'/'+f))
o=ev.get('original'); lines=[]
def q(r): return f"{r['psnr']:.3f} / {r['ssim']:.4f} / {r['lpips']:.4f}" if r else 'n/a'
lines.append(f"$SCENE original | V {o['vertices']} F {o['faces']} | test {q(o)}" if o else "$SCENE original | eval missing")
for name,r in rows.items():
    mb=r.get('MB', r.get('total_jpeg_MB')); e=ev.get(name) or (ev.get('draco_rgb8') if name.startswith('texture_') and 'total_jpeg_MB' in r else None)
    extra=f" | PNG total {r['total_png_MB']} MB" if 'total_png_MB' in r else ''
    dv=f" | decoded V {r['decoded_vertices']}" if 'decoded_vertices' in r else ''
    lines.append(f"$SCENE {name} | {mb} MB{extra}{dv} | test {q(e)}" + (f" ({e['psnr']-o['psnr']:+.3f} dB)" if e and o else '') + f" | {r.get('note','')}")
open('$TAB','a').write('\n'.join(lines)+'\n'); print('\n'.join(lines))
ASM
echo "generic $SCENE done $(date)  -> $TAB"
SHEOF
chmod +x $G/run_generic_v1.sh; python3 -m py_compile $G/generic_codec_v1.py $G/eval_ckpt_v1.py && echo "installed generic-v1 OK" || echo "FAILED: compile"
```

## 2. 跑 Room，再跑 garden（GPU node 上，前台；一个场景结束再起下一个）
```bash
G=/scratch/zr2245/analysis/generic-v1
bash $G/run_generic_v1.sh room 2>&1 | tee $G/room.log
```
```bash
G=/scratch/zr2245/analysis/generic-v1
bash $G/run_generic_v1.sh garden 2>&1 | tee $G/garden.log
```
第一次会自动 `pip install DracoPy xatlas zstandard pillow` 到 original-lossless-v1 的 .venv（纯 wheel，无 CUDA）。脚本不会 `set -e`；每步失败打印 `FAILED: ...` 和 log 尾巴，texture 阶段失败不影响前面的 row。可重跑：已有 row / eval 跳过。xatlas 太慢想跳过：`TEX="" bash $G/run_generic_v1.sh room`。

## 3. 跑完给我
```bash
cat /scratch/zr2245/analysis/generic-codecs.txt
```
每行：`<scene> <row> | MB | decoded V | test PSNR / SSIM / LPIPS (ΔdB vs original)`。`original` 行应复现 Room 28.263 / garden 24.732（protocol 自检）。

## rows 的含义
| row | 是什么 | 质量 |
|---|---|---|
| fp32_minimal | V×52 float32 + 12 B/face（742 MB 基准）| = original |
| ply_zstd19 | 同样的字节过 zstd -19（"写成字节再熵编码"）| = original |
| quant_ply_zstd19 | xyz uint16 @ q14/q15 + SH uint8（逐通道 min/max）+ int32 faces，zstd -19 | 渲染 |
| draco_geo | Draco level 10（Edgebreaker）只编 positions q + faces：拓扑+几何参照（会看到顶点拆分 4.5×）| — |
| draco_full_sh8 / sh16 | Draco + 48 通道 uint8 / uint16 SH generic attribute（glTF Draco 扩展做法）| 渲染（Draco 解码顺序 + 拆分后的顶点）|
| draco_rgb8 | Draco + uint8 RGB 顶点色（只有 SH0 = 作者 RGB-only PLY 的 Draco 版）| 渲染，rest = 0 |
| texture_atlas / texture_draco_geo_uv / texture_4096 / _8192 | xatlas UV 展开 → Draco(positions + faces + 12-bit UV) + 烤顶点色的 texture（PNG 无损 / JPEG q90）= MPEG / texture-mapped 路线 | = draco_rgb8（texture 只有顶点色，无 view dependence；JPEG 损失未计入 = 该路线的 upper bound）|

V-DMC（MPEG 的 video-based dynamic mesh coding）test model 需要 base mesh + UV atlas + 视频编码器，这个 texture 路线就是它的静态等价物；真跑 V-DMC 要先把它的参考软件编起来，我建议等这两行结果出来再决定值不值。
