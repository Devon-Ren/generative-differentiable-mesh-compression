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
