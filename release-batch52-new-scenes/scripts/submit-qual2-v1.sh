#!/usr/bin/env bash
# submit-qual2-v1.sh (2026-11-20): renders of two scenes that the paper does not show, for the qualitative figure and the triangle-density figure of
# the Appendix.  drjohnson (Deep Blending) at about 15 MB and chair (NeRF-Synthetic) at about 2.5 MB; bicycle and truck are already rendered.
#   FP32 : MeshSplatting, released settings
#   TFAN : MeshSplatting-cap, SH quantised to 6 bits        drjohnson cap 600,000 (TFAN 15.42 MB)     chair cap 66,755 (TFAN 2.56 MB)
#   SIMP : FP32 collapsed after training, BM-Codec          drjohnson g0ctrl-1370000 (15.42 MB)       chair g0ctrl-137117 (2.55 MB)
#   OURS : ByteMesh                                         drjohnson target 15 MB (14.87 MB)         chair target 2.5 MB (2.51 MB)
#   CAP  : MeshSplatting-cap of about the size of OURS (density maps only)   drjohnson cap 1,370,000 (14.53 MB)   chair cap 178,000 (2.63 MB)
# Job 1 (CPU, minutes) writes the quantised meshes of the TFAN column.  Jobs 2 and 3 (GPU, render-b23.sbatch, start after job 1) render 8 test views
# of FP32 / TFAN / SIMP / OURS per scene.  Job 4 (GPU) writes the triangle-area maps of CAP and OURS for the same views (render_ids_v3.py).
# Nothing is deleted or overwritten: every run writes to a new directory /scratch/zr2245/qual2-b52-<time>.
# Usage:  bash submit-qual2-v1.sh          (needs render_ids_v3.py in /scratch/zr2245/analysis)
#         bash submit-qual2-v1.sh pack     when the jobs are done -> /scratch/zr2245/qual2-b52-<time>/qual2-b52.zip (JPG / PNG / txt)
AN=/scratch/zr2245/analysis; ROOT=/scratch/zr2245; D=$ROOT/datasets; ACC=torch_pr_38_general
ENV="source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate"
if [ "$1" = pack ]; then
  O=$(cat $AN/b52-outdir.txt 2>/dev/null); [ -d "$O" ] || { echo "FAILED: no output directory recorded in $AN/b52-outdir.txt"; exit 0; }
  for S in drjohnson chair; do [ -f $O/$S/psnr.txt ] || echo "not finished: $S renders ($(ls -t $AN/rend23-$S-qual2-*.out 2>/dev/null | head -1 | xargs -r tail -2 | tr '\n' ' '))"; [ -f $O/ids/$S/stats.txt ] || echo "not finished: $S area maps ($(ls -t $AN/b52-ids-*.out 2>/dev/null | head -1 | xargs -r tail -2 | tr '\n' ' '))"; done
  eval "$ENV"
  python - "$O" <<'PY'
import sys, glob, os, zipfile
from PIL import Image
O = sys.argv[1]; P = os.path.join(O, 'pack'); n = 0
for S in ('drjohnson', 'chair'):
    os.makedirs(os.path.join(P, S), exist_ok=True)
    for f in sorted(glob.glob('%s/%s/view-*.png' % (O, S))):
        if f.endswith('-err.png'): continue
        Image.open(f).convert('RGB').save(os.path.join(P, S, os.path.basename(f)[5:-4] + '.jpg'), quality=96, subsampling=0); n += 1
    for f in glob.glob('%s/%s/psnr.txt' % (O, S)): open(os.path.join(P, S, 'psnr.txt'), 'w').write(open(f).read())
    for f in sorted(glob.glob('%s/ids/%s/*' % (O, S))):
        if f.endswith(('-area.png', '-valid.png', 'stats.txt')): open(os.path.join(P, S, 'ids-' + os.path.basename(f)), 'wb').write(open(f, 'rb').read()); n += 1
fs = sorted(glob.glob(P + '/*/*')); z = os.path.join(O, 'qual2-b52.zip')
with zipfile.ZipFile(z, 'w', zipfile.ZIP_STORED) as zf:
    for f in fs: zf.write(f, os.path.relpath(f, P))
print('%d files written, %d packed, %.1f MB -> %s' % (n, len(fs), os.path.getsize(z) / 1e6, z))
PY
  exit 0
fi
O=$ROOT/qual2-b52-$(date +%m%d-%H%M%S); miss=0
first () { for p in "$@"; do [ -e "$p" ] && { echo "$p"; return; }; done; echo "$1"; }
cfg () { case $1 in
  drjohnson) ORIG=$ROOT/drjohnson-indoor; DATA=$D/db/drjohnson; IMAGES=images; QP=15; VIEWS="0 4 8 12 16 20 24 28"
             CAPT=$ROOT/drjohnson-budget-600000; SIMP=$ORIG/axisB/g0ctrl-1370000/deliver; HOLD=$(first $ROOT/drjohnson-abl24-hold-t15mb $ROOT/drjohnson-abl24b-hold-t15mb); CAPD=$ROOT/drjohnson-budget-1370000;;
  chair)     ORIG=$ROOT/chair-blender4-best; DATA=$D/nerf_synthetic/chair; IMAGES=train; QP=14; VIEWS="0 25 50 75 100 125 150 175"
             CAPT=$ROOT/chair-budgetb-66755-best; SIMP=$ORIG/axisB/g0ctrl-137117/deliver; HOLD=$(first $ROOT/chair-abl24b-hold-t2.5mb-best $ROOT/chair-abl24-hold-t2.5mb-best); CAPD=$ROOT/chair-budgetb-178000-best;; esac
  FP=$ORIG/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; MK=$CAPT/axisB/v1.0/finetune/best.pt; OU=$HOLD/axisB/v1.0/deliver; OUF=$HOLD/axisB/v1.0/finetune/best.pt; CAF=$CAPD/axisB/v1.0/finetune/best.pt; }
for f in $AN/render_ids_v3.py $AN/render-b23.sbatch $AN/generic-v1/generic_codec_v1.py; do [ -f $f ] || { echo "MISSING: $f"; miss=1; }; done
for S in drjohnson chair; do cfg $S
  for p in $FP $MK $OUF $CAF; do [ -f $p ] || { echo "MISSING ($S): $p"; miss=1; }; done
  for p in $SIMP $OU $DATA; do [ -d $p ] || { echo "MISSING ($S): $p"; miss=1; }; done
done
[ $miss = 0 ] || { echo "nothing submitted.  Directories that exist:"; ls -d $ROOT/drjohnson-* $ROOT/chair-* 2>/dev/null | tr '\n' ' '; echo; ls -d $ROOT/drjohnson-indoor/axisB/g0ctrl-* $ROOT/chair-blender4-best/axisB/g0ctrl-* 2>/dev/null | tr '\n' ' '; echo; exit 0; }
mkdir -p $O/tfan $O/ids; echo $O > $AN/b52-outdir.txt
# ---- job 1: quantised meshes for the TFAN column (positions as in the paper, SH coefficients to 6 bits, as generic_rd_v1.py)
{ echo '#!/usr/bin/env bash'; echo "$ENV"; for S in drjohnson chair; do cfg $S; cat <<SH
python - "$AN/generic-v1" "$MK" $QP "$O/tfan/$S-sh6.pt" 6 <<'PY' || exit 1
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from generic_codec_v1 import load_ckpt, xyz_quant, save_ckpt
m = load_ckpt(sys.argv[2]); xyz, faces, w = m['xyz'], m['faces'], m['weight']; sh = np.concatenate([m['dc'], m['rest']], 1)
xq, xlo, xstep = xyz_quant(xyz, int(sys.argv[3])); xr = xlo + xq.astype(np.float32) * xstep
lo, hi = sh.min(0), sh.max(0); L = 2 ** int(sys.argv[5]) - 1
sq = np.rint((sh - lo) * np.where(hi > lo, L / np.maximum(hi - lo, 1e-12), 0.0)).astype(np.uint8)
shr = (lo + sq.astype(np.float32) * (hi - lo) / L).astype(np.float32)
save_ckpt(Path(sys.argv[4]), xr, faces, shr[:, :3], shr[:, 3:], float(np.median(w)) if w is not None else 7.0)
print('written', sys.argv[4], len(xyz), 'vertices', len(faces), 'faces')
PY
SH
done; echo 'echo "b52-tfan exit 0  $(date)"'; } > $O/b52-tfan.sh
# ---- job 4: triangle-area maps of CAP and OURS (after the same fine-tuning), same views
{ echo '#!/usr/bin/env bash'; echo "$ENV"; echo "python $AN/render_ids_v3.py --selftest || { echo 'FAILED: selftest'; exit 1; }"; for S in drjohnson chair; do cfg $S
  echo "python $AN/render_ids_v3.py --lab $ROOT/meshsplatting-original-lossless-v1 --dataset $DATA --images $IMAGES --model $ORIG/original --views $VIEWS --ckpt CAP=$CAF --ckpt OURS=$OUF --out $O/ids/$S > $O/ids/$S.log 2>&1 && grep -h '^WIRE' $O/ids/$S.log || { echo 'FAILED: $S'; grep -n -m 6 -i 'error\|Traceback\|FAILED' $O/ids/$S.log; tail -6 $O/ids/$S.log; }"
done; echo 'echo "b52-ids exit 0  $(date)"'; } > $O/b52-ids.sh
bash -n $O/b52-tfan.sh && bash -n $O/b52-ids.sh || { echo "FAILED: syntax of the job scripts"; exit 0; }
j1=$(sbatch --parsable --account=$ACC --cpus-per-task=4 --mem=32G --time=00:20:00 --job-name=b52-tfan --output=$AN/b52-tfan-%j.out $O/b52-tfan.sh) || { echo "FAILED: sbatch b52-tfan"; exit 0; }
echo "submitted $j1  b52-tfan (CPU)"
for S in drjohnson chair; do cfg $S
  j=$(sbatch --parsable --dependency=afterok:$j1 --account=$ACC --cpus-per-task=6 --job-name=$S-qual2 --export=ALL,SCENE=$S,DATA=$DATA,IMAGES=$IMAGES,MODEL=$ORIG/original,VIEWS="$VIEWS",OUT=$O/$S,ASSETS="FP32=$FP|TFAN=$O/tfan/$S-sh6.pt|SIMP=$SIMP|OURS=$OU" $AN/render-b23.sbatch) && echo "submitted $j  $S-qual2 (GPU, starts after $j1)" || echo "FAILED: sbatch $S-qual2"
done
j=$(sbatch --parsable --account=$ACC --gres=gpu:1 --constraint=l40s --cpus-per-task=6 --mem=48G --time=00:45:00 --job-name=b52-ids --output=$AN/b52-ids-%j.out $O/b52-ids.sh) && echo "submitted $j  b52-ids (GPU)" || echo "FAILED: sbatch b52-ids"
echo "output: $O"; squeue -u $USER -h -o "%i %j %T" | grep -E "b52|qual2"
