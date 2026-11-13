"""generic_rd_v1.py (2026-11-13): rate points of the generic codecs on ONE mesh at several SH precisions.  CPU only.
For each precision b in --bits: positions on the q<qp> grid, the 48 SH channels quantised per channel to b bits (min/max per channel), then
  zstd   : the quantised arrays + int32 faces through zstd -19
  draco  : Draco (Edgebreaker, level 10), positions q<qp> + faces + the SH as a 48-channel integer attribute
  tfan   : MPEG-4 SC3DMC / TFAN (Open3DGC driver), positions q<qp> + faces + the SH as integer attributes        (skipped without --driver)
All three are lossless after the quantisation, so they share ONE decoded mesh per precision: <out>/sh<b>/point_cloud_state_dict.pt
(rendered by eval_ckpt_v1.py).  Sizes go to <out>/rows.json.  Restartable.
  python generic_rd_v1.py --ckpt <mesh .pt> --qp 15 --out <dir> --bits 8 6 4 [--driver <o3dgc_driver>] --generic <dir with generic_codec_v1.py>"""
import argparse, json, struct, subprocess, sys, time
from pathlib import Path
import numpy as np
p = argparse.ArgumentParser(); p.add_argument('--ckpt', required=True); p.add_argument('--qp', type=int, required=True); p.add_argument('--out', required=True)
p.add_argument('--bits', type=int, nargs='+', default=[8, 6, 4]); p.add_argument('--driver', default=''); p.add_argument('--generic', required=True); p.add_argument('--label', default='')
a = p.parse_args(); sys.path.insert(0, a.generic)
from generic_codec_v1 import load_ckpt, xyz_quant, save_ckpt, zstd_bytes, log  # noqa: E402
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rp = out / 'rows.json'; rows = json.load(open(rp)) if rp.exists() else {}
m = load_ckpt(a.ckpt); xyz, faces, w = m['xyz'], m['faces'], m['weight']; sh = np.concatenate([m['dc'], m['rest']], 1); V, F = len(xyz), len(faces)
wlogit = float(np.median(w)) if w is not None else 7.0; MB = lambda b: round(b / 1e6, 3)
log(f'generic-rd-v1 | {a.label} | {V} vertices, {F} faces | qp {a.qp} | bits {a.bits} | driver {a.driver or "none"}')
xq, xlo, xstep = xyz_quant(xyz, a.qp); xr = (xlo + xq.astype(np.float32) * xstep)
lo = sh.min(0); hi = sh.max(0)
for b in a.bits:
    key = f'sh{b}'; r = rows.get(key, {}); L = 2 ** b - 1
    sq = np.rint((sh - lo) * np.where(hi > lo, L / np.maximum(hi - lo, 1e-12), 0.0)).astype(np.uint8)
    if 'zstd_MB' not in r:
        t = time.time(); n, how = zstd_bytes(xq.tobytes() + sq.tobytes() + faces.astype(np.int32).tobytes()); r.update(zstd_MB=MB(n + 96 * 4), zstd_s=round(time.time() - t, 1))
    if 'draco_MB' not in r:
        try:
            import DracoPy
            t = time.time(); buf = DracoPy.encode(xyz.astype(np.float32), faces.astype(np.uint32), quantization_bits=a.qp, compression_level=10, preserve_order=False, generic_attributes={'sh': sq})
            r.update(draco_MB=MB(len(buf) + 96 * 4), draco_s=round(time.time() - t, 1)); del buf
        except Exception as e: log(f'WARNING: Draco failed at {key}: {e!r}')
    if a.driver and 'tfan_MB' not in r:
        d = out / key; d.mkdir(exist_ok=True); bundle, stream, dec = d / 'in.bundle', d / 'mesh.o3dgc', d / 'out.decoded'
        with open(bundle, 'wb') as f:
            f.write(b'O3DGCB01' + struct.pack('<IIII', V, F, 1, sq.shape[1])); f.write(np.ascontiguousarray(xyz, np.float32).tobytes()); f.write(np.ascontiguousarray(faces, np.int32).tobytes()); f.write(np.ascontiguousarray(sq, np.uint8).tobytes())
        t = time.time(); pr = subprocess.run([a.driver, str(bundle), str(stream), str(dec), str(a.qp), '8'], capture_output=True, text=True)
        if pr.returncode == 0 and stream.exists(): r.update(tfan_MB=MB(stream.stat().st_size + 96 * 4), tfan_s=round(time.time() - t, 1))
        else: log(f'WARNING: TFAN driver failed at {key}: {pr.stderr.strip()[-300:]}')
        for x in (bundle, dec, stream): x.unlink(missing_ok=True)
    ck = out / key / 'point_cloud_state_dict.pt'
    if not ck.exists():
        shr = (lo + sq.astype(np.float32) * (hi - lo) / L).astype(np.float32); save_ckpt(ck, xr, faces, shr[:, :3], shr[:, 3:], wlogit)
    r.update(bits=b, vertices=V, faces=F, qp=a.qp, ckpt=str(ck)); rows[key] = r; json.dump(rows, open(rp, 'w'), indent=1)
    log(f'ROW {a.label} {key}: zstd {r.get("zstd_MB")} MB, draco {r.get("draco_MB")} MB, tfan {r.get("tfan_MB")} MB')
log('done ->', rp)
