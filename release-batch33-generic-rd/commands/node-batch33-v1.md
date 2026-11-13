# node-batch33-v1 (2026-11-13): rate-distortion points of the generic codecs (11 GPU jobs, about 10 GPU-h)

## Why

In the RD figure each generic codec (quantised PLY + zstd, Draco, MPEG TFAN) has one point: the full FP32 mesh with 8-bit SH. A codec deserves a curve. Two things move a generic codec along the rate axis, and this batch measures both:

1. **Precision**: the SH coefficients at 8, 6 and 4 bits.
2. **Vertex count**: the FP32 model simplified after training and fine-tuned, at the three vertex counts already used in the paper. This is the conventional pipeline with nothing of ours in it: MeshSplatting, simplify, generic codec.

Per scene: 4 meshes x 3 precisions = 12 operating points, each with the size under the three codecs and the PSNR / SSIM / LPIPS of the decoded mesh. The three codecs are lossless after quantisation, so they share one render per operating point. Indoor scenes use the `--indoor` FP32 models, the same models the simplified meshes come from.

Tested here: the size script on a synthetic mesh (zstd and Draco paths). Not tested here: the TFAN path (it needs the Open3DGC driver on the cluster; a failure there leaves that column empty and does not stop the job) and the render step (GPU).

## One paste (login node)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/generic_rd_v1.py <<'GENRDPY'
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
GENRDPY
cat > $AN/generic-rd-v1.sbatch <<'GENRDSB'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=genrd
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/zr2245/analysis/genrd-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/genrd-%x-%j.err
# generic-rd-v1.sbatch (2026-11-13, VERSION genrd-v1): rate-distortion points of the generic codecs (quantised PLY + zstd, Draco, MPEG TFAN) on ONE scene.
# Meshes: the FP32 model, and the same model simplified after training and fine-tuned at three vertex counts (the POST-HOC-COLLAPSE controls of the
# cap jobs, $ORIG/axisB/g0ctrl-<B>/finetune/best.pt).  For each mesh: SH precision 8 / 6 / 4 bits -> sizes of the three codecs (CPU) and one render of
# the decoded mesh (GPU; the codecs are lossless after quantisation, so they share it), plus one render of the mesh before quantisation.
# Rows -> $AN/generic-rd.txt:  SCENE | mesh | V F | shB | zstd MB | draco MB | tfan MB | test PSNR SSIM LPIPS | unquantised PSNR
#   sbatch --export=ALL,SCENE=garden --job-name=garden-genrd-b33 .../generic-rd-v1.sbatch
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; BITS=${BITS:-"8 6 4"}
AN=/scratch/zr2245/analysis; G=$AN/generic-v1; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; D=/scratch/zr2245/datasets; TAB=$AN/generic-rd.txt; DRV=$AN/o3dgc-v1/o3dgc_driver
case $SCENE in
  room)     ORIG=/scratch/zr2245/$SCENE-indoor; DATA=$D/m360/$SCENE; IMAGES=images_2; QP=14; BUD="2198000 1020000 477000";;
  kitchen)  ORIG=/scratch/zr2245/$SCENE-indoor; DATA=$D/m360/$SCENE; IMAGES=images_2; QP=16; BUD="1895000 928000 409000";;
  counter)  ORIG=/scratch/zr2245/$SCENE-indoor; DATA=$D/m360/$SCENE; IMAGES=images_2; QP=15; BUD="2040000 978000 449000";;
  bonsai)   ORIG=/scratch/zr2245/$SCENE-indoor; DATA=$D/m360/$SCENE; IMAGES=images_2; QP=15; BUD="2090000 1005000 473000";;
  garden)   ORIG=/scratch/zr2245/$SCENE-vis-collapse; DATA=$D/m360/$SCENE; IMAGES=images_4; QP=15; BUD="2015000 936000 405000";;
  bicycle)  ORIG=/scratch/zr2245/$SCENE-vis-collapse; DATA=$D/m360/$SCENE; IMAGES=images_4; QP=16; BUD="2029000 986000 440000";;
  stump)    ORIG=/scratch/zr2245/$SCENE-vis-collapse; DATA=$D/m360/$SCENE; IMAGES=images_4; QP=16; BUD="2015000 983000 440000";;
  flowers)  ORIG=/scratch/zr2245/$SCENE-vis-collapse; DATA=$D/m360/$SCENE; IMAGES=images_4; QP=15; BUD="2068000 992000 447000";;
  treehill) ORIG=/scratch/zr2245/$SCENE-vis-collapse; DATA=$D/m360/$SCENE; IMAGES=images_4; QP=16; BUD="2054000 1002000 449000";;
  truck)    ORIG=/scratch/zr2245/$SCENE-vis-collapse; DATA=$D/tandt/$SCENE; IMAGES=images; QP=18; BUD="1853000 929000 421000";;
  train)    ORIG=/scratch/zr2245/$SCENE-vis-collapse; DATA=$D/tandt/$SCENE; IMAGES=images; QP=18; BUD="2040000 975000 436000";;
  *) echo "FAILED: no settings for scene $SCENE"; exit 1;;
esac
RUN=$ORIG/original; CK=$RUN/point_cloud/iteration_30000/point_cloud_state_dict.pt; OUT=$AN/generic-rd/$SCENE; mkdir -p $OUT
for f in $CK $DATA/$IMAGES/. $G/generic_codec_v1.py $G/eval_ckpt_v1.py $AN/generic_rd_v1.py; do [ -e $f ] || { echo "FAILED: missing $f"; exit 1; }; done
python -c "import DracoPy, zstandard" 2>/dev/null || pip install DracoPy zstandard > $OUT/pip.log 2>&1 || { echo "FAILED: pip install DracoPy zstandard"; tail -5 $OUT/pip.log; exit 1; }
DOPT=""; [ -x $DRV ] && DOPT="--driver $DRV" || echo "note: no Open3DGC driver at $DRV, the TFAN column stays empty"
echo "generic-rd-v1 | scene $SCENE | original $ORIG | qp $QP | bits $BITS | simplified at $BUD | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ev () { # $1 label  $2 ckpt  $3 out json
  [ -f $3 ] && return 0
  python $G/eval_ckpt_v1.py --lab $LAB --model $RUN --dataset $DATA --images $IMAGES --ckpt $2 --out $3 --label "$SCENE $1" > ${3%.json}.log 2>&1 && grep '^EVAL' ${3%.json}.log || { echo "FAILED: eval $1"; tail -6 ${3%.json}.log; return 1; }; }
nf=0
for M in full $BUD; do
  if [ $M = full ]; then MK=$CK; else MK=$ORIG/axisB/g0ctrl-$M/finetune/best.pt; fi
  [ -f $MK ] || { echo "note: $SCENE mesh $M missing ($MK), skipped"; continue; }
  O=$OUT/$M; mkdir -p $O
  echo "=== $SCENE mesh $M: sizes ($(date)) ==="
  python $AN/generic_rd_v1.py --ckpt $MK --qp $QP --out $O --bits $BITS $DOPT --generic $G --label "$SCENE $M" > $O/codec.log 2>&1 || { echo "FAILED: generic_rd_v1 $SCENE $M"; tail -8 $O/codec.log; nf=$((nf+1)); continue; }
  grep -h "ROW\|WARNING" $O/codec.log
  echo "=== $SCENE mesh $M: renders ($(date)) ==="
  ev "$M unquantised" $MK $O/ref.eval.json || nf=$((nf+1))
  for B in $BITS; do ev "$M sh$B" $O/sh$B/point_cloud_state_dict.pt $O/sh$B.eval.json || nf=$((nf+1)); done
  python - "$SCENE" "$M" "$O" <<'PY' | tee -a $TAB
import json, os, sys
S, M, O = sys.argv[1:4]; rows = json.load(open(O + '/rows.json')); ref = json.load(open(O + '/ref.eval.json')) if os.path.exists(O + '/ref.eval.json') else None
for k, r in sorted(rows.items(), key=lambda kv: -kv[1]['bits']):
    e = json.load(open(f'{O}/{k}.eval.json')) if os.path.exists(f'{O}/{k}.eval.json') else None
    q = (f"{e['psnr']:.3f} {e['ssim']:.4f} {e['lpips']:.4f}" if e else 'n/a')
    print(f"{S} GENERIC-RD | mesh {M} | V {r['vertices']} F {r['faces']} | sh{r['bits']} | zstd {r.get('zstd_MB', 'n/a')} MB | draco {r.get('draco_MB', 'n/a')} MB | tfan {r.get('tfan_MB', 'n/a')} MB | test {q} | unquantised {ref['psnr']:.3f}" if ref else f"{S} GENERIC-RD | mesh {M} | sh{r['bits']} | no reference render")
PY
  rm -f $O/sh*/point_cloud_state_dict.pt   # the decoded meshes are large and no longer needed
done
echo "generic-rd-v1 $SCENE exit 0 ($nf failures)  $(date)"
GENRDSB
cat > $AN/submit-batch33-v1.sh <<'SUB33'
#!/usr/bin/env bash
# submit-batch33-v1.sh (2026-11-13): rate-distortion points of the generic codecs on the 11 Mip-NeRF 360 / T&T scenes (11 GPU jobs, about 10 GPU-h;
# most of each job is CPU encoding).  Per scene: 4 meshes (FP32 model; simplified after training + fine-tuned at three vertex counts) x SH precision 8 / 6 / 4 bits.
AN=/scratch/zr2245/analysis
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/generic_rd_v1.py 46 17c568d6da5cfaf473cfe7067bf38b60
chk $AN/generic-rd-v1.sbatch 68 a9ad66a9e7bd9a740b82f5b7bff42681
python3 -m py_compile $AN/generic_rd_v1.py && bash -n $AN/generic-rd-v1.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
for f in $AN/generic-v1/generic_codec_v1.py $AN/generic-v1/eval_ckpt_v1.py; do [ -f $f ] || { echo "CHECK FAILED: missing $f"; exit 1; }; done
[ -x $AN/o3dgc-v1/o3dgc_driver ] || echo "note: no Open3DGC driver, the TFAN column will be empty"
miss=0; for S in room kitchen counter bonsai; do for B in $(case $S in room) echo 2198000 1020000 477000;; kitchen) echo 1895000 928000 409000;; counter) echo 2040000 978000 449000;; bonsai) echo 2090000 1005000 473000;; esac); do [ -f /scratch/zr2245/$S-indoor/axisB/g0ctrl-$B/finetune/best.pt ] || { echo "note: simplified mesh missing: $S $B"; miss=$((miss+1)); }; done; done
for S in garden truck; do [ -f /scratch/zr2245/$S-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no original for $S"; exit 1; }; done
echo "CHECK OK ($miss simplified meshes missing among the indoor scenes)"; n=0
for S in garden room truck kitchen counter bonsai bicycle stump flowers treehill train; do
  j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=8 --export=ALL,SCENE=$S --job-name=$S-genrd-b33 $AN/generic-rd-v1.sbatch) && { n=$((n+1)); echo "submitted $j  $S-genrd-b33"; } || echo "FAILED: sbatch $S"; done
echo "$n GPU jobs submitted"
SUB33
bash $AN/submit-batch33-v1.sh
```

Expected: `CHECK OK (0 simplified meshes missing among the indoor scenes)` (a non-zero count only means fewer points for that scene), 11 `submitted` lines, `11 GPU jobs submitted`.

After the first job has run for about 15 minutes:

```bash
tail -5 /scratch/zr2245/analysis/genrd-garden-genrd-b33-*.out; cat /scratch/zr2245/analysis/generic-rd.txt 2>/dev/null | head
```

## Collect

```bash
AN=/scratch/zr2245/analysis; O=$AN/batch33-collect-$(date +%Y-%m-%d-%H%M).txt
{ echo "===== generic-rd rows"; cat $AN/generic-rd.txt; echo; echo "===== notes and failures"; grep -h "note:\|FAILED\|WARNING\|exit 0" $AN/genrd-*-b33-*.out | cut -c1-200; echo; echo "===== sacct"; sacct -X --starttime=now-2days --format=JobID,JobName%28,State,Elapsed,ExitCode | grep b33; } > $O 2>&1; wc -l $O; echo $O
```
