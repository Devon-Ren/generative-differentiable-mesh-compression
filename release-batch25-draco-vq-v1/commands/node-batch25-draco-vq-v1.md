# node-batch25-draco-vq-v1 (2026-11-11): Draco carrying our own appearance payload, on every delivered asset (CPU, one job, about 1 h)

Purpose: reviewer W6. Our generic-codec table compares Draco/MPEG with 8-bit SH (near-lossless) against our file (lossy). The fair generic baseline
is Draco given exactly what our file stores per vertex: the two appearance-dictionary indices (16 + 8 bit) and the 7-bit DC colour, as one uint8
generic attribute, on the same positions (same grid bit depth) and faces. Quality is then ours up to Draco's own position re-quantisation, so the
comparison is bytes only, which is what the table needs. Also reported: Draco geometry alone, and Draco geometry + the index bytes through zstd
without prediction. 63 assets: 13 COLMAP scenes x 5/10/20 MB, 8 NeRF-Synthetic objects x 1.5/2.5/4 MB.

One paste on a login node. It writes the two scripts, the sbatch, checks md5s and submits one cpu_short job.

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/draco-vq
cat > $AN/draco-vq/dump_asset_fields_v1.py <<'PYDUMP'
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
PYDUMP
cat > $AN/draco-vq/draco_vq_v1.py <<'PYDRACO'
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
PYDRACO
cat > $AN/draco-vq-v1.sbatch <<'SBDRACO'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=dracovq
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/zr2245/analysis/dracovq-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/dracovq-%x-%j.err
# draco-vq-v1.sbatch (2026-11-11, VERSION dracovq-v1): the generic baseline the reviewer asked for: Draco carrying EXACTLY our appearance payload
# (dictionary index 16 bit + residual index 8 bit + 3 x 7-bit DC, as a uint8 generic attribute) on the same mesh as our file, for every delivered
# BudgetMesh asset (Mip-360 / T&T 5/10/20 MB, Deep Blending 5/10/20 MB, NeRF-Synthetic 1.5/2.5/4 MB).  CPU only.  Two steps per asset: the codec's
# own reader dumps the fields (stage-2 env), DracoPy encodes (main env).  Rows -> $AN/draco-vq-v1.jsonl (restartable: assets with a row are skipped).
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; S1=$AN/draco-vq/dump_asset_fields_v1.py; S2=$AN/draco-vq/draco_vq_v1.py; ROWS=$AN/draco-vq-v1.jsonl; NPZ=$AN/draco-vq/npz; mkdir -p $NPZ
for f in $S1 $S2 $PKG/stage2_run.sh; do [ -f $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
python -c "import DracoPy, numpy; print('DracoPy', getattr(DracoPy, '__version__', '?'))" || { echo "FAILED: DracoPy not importable in this env"; exit 1; }
qp_of () { case $1 in room|lego|chair|drums|ficus|hotdog|materials|mic|ship) echo 14;; garden|bonsai|counter|flowers|drjohnson|playroom) echo 15;; kitchen|bicycle|stump|treehill) echo 16;; truck|train) echo 18;; esac; }
n=0; nf=0
one () { S=$1; T=$2; DIR=$3; [ -d $DIR/delivery ] || { echo "SKIP $S $T: no delivery at $DIR"; return; }
  grep -q "\"scene\": \"$S\", \"target\": \"$T\"" $ROWS 2>/dev/null && { echo "exists: $S $T"; return; }
  echo "=== $S $T ($(date)) ==="
  bash $PKG/stage2_run.sh $S1 --deliver-dir $DIR --out $NPZ/$S-$T.npz > $NPZ/$S-$T.dump.log 2>&1 || { echo "FAILED: dump $S $T"; tail -6 $NPZ/$S-$T.dump.log; nf=$((nf+1)); return; }
  grep -h "integer per-vertex fields" $NPZ/$S-$T.dump.log | cut -c1-300
  python $S2 --npz $NPZ/$S-$T.npz --qp $(qp_of $S) --scene $S --target $T --auto --rows $ROWS > $NPZ/$S-$T.draco.log 2>&1 || { echo "FAILED: draco $S $T"; tail -6 $NPZ/$S-$T.draco.log; nf=$((nf+1)); return; }
  grep -h "^ROW" $NPZ/$S-$T.draco.log | cut -c1-400; n=$((n+1)); }
for S in room kitchen counter bonsai garden bicycle stump flowers treehill truck train drjohnson playroom; do for T in 5 10 20; do one $S $T /scratch/zr2245/$S-adapt9x-t${T}mb/axisB/v1.0/deliver; done; done
for S in lego chair drums ficus hotdog materials mic ship; do for T in 1.5 2.5 4; do one $S $T /scratch/zr2245/$S-adapt9xb-t${T}mb-best/axisB/v1.0/deliver; done; done
echo "===== $ROWS ($(wc -l < $ROWS) rows) ====="; cat $ROWS
echo "draco-vq-v1 exit 0 ($n new rows, $nf failures)  $(date)"
SBDRACO
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/draco-vq/dump_asset_fields_v1.py 34 92bcedcc85bbe0113f941b3b149ba651
chk $AN/draco-vq/draco_vq_v1.py 40 b89866aa48872abb550e49d71ba47097
chk $AN/draco-vq-v1.sbatch 36 3bec86765ab4eefe22fd30e65bd712c6
bash -n $AN/draco-vq-v1.sbatch && python3 -m py_compile $AN/draco-vq/dump_asset_fields_v1.py $AN/draco-vq/draco_vq_v1.py || { echo "CHECK FAILED: syntax"; exit 1; }
[ -f $AN/vis_collapse-v12/stage2_run.sh ] || { echo "CHECK FAILED: no stage2_run.sh"; exit 1; }
echo "CHECK OK"; sbatch $AN/draco-vq-v1.sbatch
```

When it ends (`squeue --me | grep dracovq` empty): send me `$AN/draco-vq-v1.jsonl` and the tail of the .out:

```bash
AN=/scratch/zr2245/analysis; f=$(ls -t $AN/dracovq-*.out | head -1); grep -c "^ROW" $f; grep "FAILED\|SKIP\|exit 0" $f | head -20; grep -m1 "integer per-vertex fields" $f | cut -c1-400; wc -l $AN/draco-vq-v1.jsonl
```

If the first asset prints `FAILED: draco ... index fields not found`, the dump log line "integer per-vertex fields" tells me the field names and I will give you the two `--k_field/--r_field` names to put into the sbatch (one-line change); the dumps themselves are kept.
