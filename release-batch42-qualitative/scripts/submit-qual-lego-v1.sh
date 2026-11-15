#!/usr/bin/env bash
# submit-qual-lego-v1.sh (2026-11-15): renders of one object (lego) for the qualitative figure, at about 2.5 MB.
#   FP32 : MeshSplatting, released settings          TFAN : MeshSplatting with a vertex cap of 78,124, SH quantised to 6 bits (TFAN 2.77 MB, batch38)
#   SIMP : FP32 simplified after training, our codec (2.73 MB)      OURS : hold run, target 2.5 MB (2.51 MB)
# Job 1 (CPU, 1 minute) writes the quantised mesh for the TFAN column; job 2 (GPU, render-b23.sbatch) starts when job 1 is done.
# Then: bash submit-qual-lego-v1.sh pack     -> /scratch/zr2245/renders-qual/lego.zip (all views as JPEG, a few MB)
AN=/scratch/zr2245/analysis; ROOT=/scratch/zr2245; OUTR=$ROOT/renders-qual; S=lego
if [ "$1" = pack ]; then
  [ -f $OUTR/$S/psnr.txt ] || { echo "not finished: $(ls -t $AN/rend23-$S-qual-*.out 2>/dev/null | head -1 | xargs -r tail -3)"; exit 0; }
  source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
  rm -rf $OUTR/lego-pack; mkdir -p $OUTR/lego-pack; cp $OUTR/$S/psnr.txt $OUTR/lego-pack/
  python - "$OUTR/$S" "$OUTR/lego-pack" <<'PY'
import sys, glob, os
from PIL import Image
n = 0
for f in sorted(glob.glob(sys.argv[1] + '/view-*.png')):
    if f.endswith('-err.png'): continue
    Image.open(f).convert('RGB').save(os.path.join(sys.argv[2], 'lego-' + os.path.basename(f)[5:-4] + '.jpg'), quality=96, subsampling=0); n += 1
print(n, 'images')
PY
  cd $OUTR && rm -f lego.zip && zip -q -r lego.zip lego-pack && ls -la $OUTR/lego.zip; exit 0
fi
FP=$ROOT/lego-blender4-best/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; MK=$ROOT/lego-budgetb-78124-best/axisB/v1.0/finetune/best.pt
TF=$AN/generic-rd/lego/cap78124/sh6/point_cloud_state_dict.pt; SI=$ROOT/lego-blender4-best/axisB/g0ctrl-142000/deliver; OU=$ROOT/lego-abl24b-hold-t2.5mb-best/axisB/v1.0/deliver; miss=0
for p in $FP $MK $AN/render-b23.sbatch $AN/generic-v1/generic_codec_v1.py; do [ -f $p ] || { echo "MISSING: $p"; miss=1; }; done
for p in $SI $OU; do [ -d $p ] || { echo "MISSING: $p"; miss=1; }; done
[ $miss = 0 ] || exit 0
cat > $AN/regen-lego-sh6.sh <<SH
#!/usr/bin/env bash
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
[ -f $TF ] && { echo "exists: $TF"; exit 0; }
python - "$AN/generic-v1" "$MK" 14 "$TF" 6 <<'PY'
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
print('written', sys.argv[4], len(xyz), 'vertices')
PY
SH
j1=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=4 --mem=16G --time=00:15:00 --job-name=regen-lego --output=$AN/regen-lego-%j.out $AN/regen-lego-sh6.sh) || { echo "FAILED: sbatch regen"; exit 0; }
echo "submitted $j1  regen-lego (CPU)"
j2=$(sbatch --parsable --dependency=afterok:$j1 --account=torch_pr_38_general --cpus-per-task=6 --job-name=$S-qual --export=ALL,SCENE=$S,DATA=$ROOT/datasets/nerf_synthetic/lego,IMAGES=train,MODEL=$ROOT/lego-blender4-best/original,VIEWS="0 25 50 75 100 125 150 175",OUT=$OUTR/$S,ASSETS="FP32=$FP|TFAN=$TF|SIMP=$SI|OURS=$OU" $AN/render-b23.sbatch) && echo "submitted $j2  $S-qual (GPU, starts after $j1)" || echo "FAILED: sbatch render"
squeue -u $USER -h -o "%i %j %T" | grep -E "regen-lego|lego-qual"
