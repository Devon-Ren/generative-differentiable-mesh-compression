# node-batch20-v1 (2026-11-09): indoor G0 refresh — CPU smoke of the `--indoor` / `--max_points` interaction, then 4 jobs

**The pitfall, re-read from the upstream source.** `update_indoor()` in `arguments/__init__.py` overwrites add_percentage, the densify window, feature_lr, start_pruning, prune_size, iteration_mesh and a few loss weights; it does **not** touch `max_points`. The 11/04 note lumped max_points in with add_percentage (that experiment needed add_percentage, which the preset does overwrite). So plain `train.py --indoor --max_points B` should be a valid indoor budget run. Since a wrong assumption here costs a night, §1 first runs a CPU check that reproduces train.py's own parse → extract → update_indoor flow with `--indoor --max_points 123456` and prints the value after the preset; the submit script refuses to submit unless it prints PASS. Seconds, no GPU.

**Then** `budget-sweep-v3.sbatch` (88 lines, md5 3c4ba68fe54bc67ddd7852f39d5365f5, VERSION g0r-v3) = v2 + `--indoor` passed through + ORIG defaulting to the 11/01 `--indoor` originals (`$SCENE-indoor`) + an in-job grep that fails if this upstream's `update_indoor` ever sets max_points. Four jobs: room, kitchen, counter, bonsai (images_2; XYZ q14 / q16 / q15 / q15), budgets = V of the scene's own `--indoor` λ 1.85e-3 / 6.2e-3 / 1.85e-2 assets ÷ 0.9: room 1215000 486000 220000; kitchen 1349000 622000 272000; counter 1162000 526000 245000; bonsai 1328000 557000 290000. Rows append to `g0-refresh-$SCENE.txt` with `| --indoor` at the end. About 3.5–4 h per job, ~14 GPU-h, done in about 4 h on 4 GPUs.

## 1. CPU check + write the sbatch + submit (one paste; the submit stops if the check fails)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/indoor_budget_check_v1.py <<'PYCHK'
import sys, os
LAB = '/scratch/zr2245/meshsplatting-original-lossless-v1'; up = os.path.join(LAB, 'upstream'); sys.path.insert(0, up); os.chdir(up)
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams, update_indoor
p = ArgumentParser(); lp = ModelParams(p); op = OptimizationParams(p); pp = PipelineParams(p); p.add_argument('--indoor', action='store_true', default=False)
args = p.parse_args(['-s', '/tmp/x', '-m', '/tmp/y', '--indoor', '--max_points', '123456'])
ops = op.extract(args); before = (ops.max_points, ops.add_percentage); ops = update_indoor(ops); after = (ops.max_points, ops.add_percentage)
print(f'after extract: max_points {before[0]} add_percentage {before[1]} | after update_indoor: max_points {after[0]} add_percentage {after[1]}')
ok = (after[0] == 123456)
print('INDOOR BUDGET CHECK', 'PASS: --max_points survives --indoor (add_percentage is the one the preset overwrites)' if ok else 'FAIL: update_indoor changed max_points')
sys.exit(0 if ok else 1)
PYCHK
cat > $AN/budget-sweep-v3.sbatch <<'SBEOF_G0R3'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=g0r
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/zr2245/analysis/g0r-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/g0r-%x-%j.err
# budget-sweep-v3.sbatch (2026-11-09, VERSION g0r-v3): = budget-sweep-v2 with INDOOR=1 allowed: upstream's update_indoor() does NOT touch max_points
# (it overwrites add_percentage, densify window, lrs, prune_size, iteration_mesh ...; verified by the CPU check in node-batch20 that reproduces train.py's
# parse -> extract -> update_indoor flow), so `train.py --indoor --max_points B` is the indoor budget run.  --indoor is passed to train.py when INDOOR=1;
# ORIG defaults to /scratch/zr2245/$SCENE-indoor for INDOOR=1 (the 11/01 --indoor originals), $SCENE-vis-collapse otherwise.  v2 header follows.
# budget-sweep-v2.sbatch (2026-11-08, VERSION g0r-v2): the G0 comparison on one scene, three vertex budgets, current protocol.
#   = budget-sweep-v1.sbatch (train from scratch with upstream --max_points B, then the level-1.0 protocol) + the CONTROL of
#   g0-budget-v1.sbatch (collapse the scene's dense FP32 original to exactly the vertex count the budget run ended with, same
#   3000-step finetune, same K4096 R256 delivery) + DATA / ORIG overrides (T&T scenes) + explicit XYZ.  One row per path and budget
#   in $AN/g0-refresh-$SCENE.txt: 'TRAIN-AT-BUDGET' and 'POST-HOC-COLLAPSE'; the lambda-grown assets of the same scene (rd-prune3 tables)
#   are the third path, compared at equal bytes.  Budgets are chosen as V(lambda asset) / 0.9 so the three paths meet at ~equal V.
#   INDOOR=1 is REFUSED here: upstream's update_indoor() overwrites --max_points after parsing (known pitfall), so an indoor budget run
#   needs the train_ckpt_v4 --override path, which this file does not use.  Restartable: every stage is skipped when its output exists.
#   sbatch --export=ALL,SCENE=truck,IMAGES=images,XYZ=q18,DATA=/scratch/zr2245/datasets/tandt/truck,BUDGETS="1934000 853000 354000" --job-name=truck-g0r .../budget-sweep-v2.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_4}; BUDGETS=${BUDGETS:?set BUDGETS="2000000 900000 400000" via --export}; INDOOR=${INDOOR:-0}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}; ODEF=/scratch/zr2245/$SCENE-vis-collapse; [ "$INDOOR" = "1" ] && ODEF=/scratch/zr2245/$SCENE-indoor; ORIG=${ORIG:-$ODEF}; TAB=$AN/g0-refresh-$SCENE.txt
CORIG=$ORIG/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; MORIG=$ORIG/original; FOOT=$ORIG/footprint-sens.npz
CRIT=${CRIT:-"--geo-weight sens_pos --app-weight none --weight-power 0.5 --weight-clip 0.1,10 --weight-divide-pixel-scale"}
echo "g0-refresh | scene $SCENE | IMAGES $IMAGES | budgets $BUDGETS | indoor flag '$FLAG' | original $ORIG | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $PKG/scripts/run_garden.sh $PKG/coarse_vis.py $PKG/vertex_footprint.py $PKG/deliver_coarse.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $DATA/sparse/0/cameras.bin $CORIG $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
ls $PKG/native/_native_qem_vis*.so >/dev/null 2>&1 || { echo "FAILED: native module missing in $PKG/native"; exit 1; }
(cd $LAB/upstream && python train.py -h 2>&1 | grep -q -- "--max_points") || { echo "FAILED: train.py does not accept --max_points"; (cd $LAB/upstream && python train.py -h 2>&1 | tail -5); exit 1; }
[ "$INDOOR" = "0" ] || grep -A 20 "def update_indoor" $LAB/upstream/arguments/__init__.py | grep -q "max_points" && { echo "FAILED: this upstream's update_indoor() sets max_points; the CLI budget would be overwritten"; exit 1; }
echo "indoor preset check: update_indoor() leaves max_points alone ($FLAG)"
echo "preflight ok: --max_points accepted; original $(python -c "import torch; sd=torch.load('$CORIG',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))") vertices"
if [ ! -f $FOOT ]; then
  echo "=== footprint of the original (missing) -> $FOOT ($(date)) ==="
  (cd $PKG && python vertex_footprint.py --lab $LAB --model $MORIG --dataset $DATA --images $IMAGES --out $FOOT > $ORIG/footprint.log 2>&1) || { echo "FAILED: FOOTPRINT of the original"; tail -8 $ORIG/footprint.log; exit 1; }
fi
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
row () { # $1 tag  $2 report.json  $3 finetune/summary.json  $4 deliver/summary.json
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$2')); f=json.load(open('$3')); d=json.load(open('$4')); t=d['delivery_test']; h=f['history']
print('$1 | vertices', b['vertices'], 'faces', b['faces'], '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3),
      'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | $IMAGES | $FLAG')
PY
}
for B in $BUDGETS; do
  G=/scratch/zr2245/$SCENE-budget-$B; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== A: $SCENE budget $B: train with --max_points $B ($(date)) ==="
  if [ ! -f $CB ]; then
    (cd $LAB/upstream && python train.py -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --max_points $B $FLAG > $G/train.log 2>&1) \
      || { echo "FAILED: TRAIN $SCENE budget $B"; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE budget $B"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  VB=$(nverts $CB); echo "budget $B -> trained vertices $VB ($(date))"; grep -h "\[ITER 30000\] Evaluating test" $G/train.log | tail -1
  echo "=== A: $SCENE budget $B: level-1.0 protocol ($(date)) ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 || { echo "FAILED: LEVEL-1.0 $SCENE budget $B"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE budget $B"; tail -8 $G/level.log; exit 1; }
  row "$SCENE budget $B TRAIN-AT-BUDGET" $NA/report.json $NA/finetune/summary.json $NA/deliver/summary.json
  echo "=== B: $SCENE control: collapse the dense original to $VB vertices ($(date)) ==="
  NC=$ORIG/axisB/g0ctrl-$B; mkdir -p $ORIG/axisB
  if [ ! -f $NC/report.json ]; then rm -rf $NC
    (cd $PKG && python coarse_vis.py --source $CORIG --footprint $FOOT --base uniform --target-vertices $VB --output $NC $CRIT > $NC.build.log 2>&1) || { echo "FAILED: CONTROL BUILD $SCENE $B"; tail -8 $NC.build.log; exit 1; }
  fi
  if [ ! -f $NC/finetune/summary.json ]; then rm -rf $NC/finetune
    (cd $AN && python coarse_finetune.py --lab $LAB --model $MORIG --dataset $DATA --images $IMAGES --coarse $NC/coarse-initial.pt --steps 3000 --out $NC/finetune > $NC.finetune.log 2>&1) || { echo "FAILED: CONTROL FINETUNE $SCENE $B"; tail -8 $NC.finetune.log; exit 1; }
  fi
  if [ ! -f $NC/deliver/summary.json ]; then rm -rf $NC/deliver
    bash $PKG/deliver_coarse.sh --coarse $NC/finetune/best.pt --output $NC/deliver --historical-config none --source $CORIG --dataset $DATA --lab $LAB --images $IMAGES --k $K --r $R --xyz $XYZ --dc $DC --dictionary-rounds 4 > $NC.deliver.log 2>&1 || { echo "FAILED: CONTROL DELIVER $SCENE $B"; tail -8 $NC.deliver.log; exit 1; }
  fi
  row "$SCENE budget $B POST-HOC-COLLAPSE" $NC/report.json $NC/finetune/summary.json $NC/deliver/summary.json
done
echo "===== $TAB ====="; cat $TAB
echo "g0-refresh $SCENE exit 0  $(date)"
SBEOF_G0R3
cat > $AN/submit-batch20-v1.sh <<'SUB20'
AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2 (paste truncated?)"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3 (paste altered?)"; exit 1; }; }
chk $AN/budget-sweep-v3.sbatch 88 3c4ba68fe54bc67ddd7852f39d5365f5
grep -q "VERSION g0r-v3" $AN/budget-sweep-v3.sbatch || { echo "CHECK FAILED: sbatch version"; exit 1; }; bash -n $AN/budget-sweep-v3.sbatch || { echo "CHECK FAILED: sbatch syntax"; exit 1; }
echo "=== smoke (CPU, seconds): does --max_points survive --indoor in THIS upstream? ==="
python $AN/indoor_budget_check_v1.py || { echo "CHECK FAILED: the indoor preset overwrites max_points on this upstream; do NOT submit (needs the train_ckpt_v4 --override path)"; exit 1; }
grep -n "max_points" /scratch/zr2245/meshsplatting-original-lossless-v1/upstream/arguments/__init__.py | head -3
ok=""; for S in room kitchen counter bonsai; do O=/scratch/zr2245/$S-indoor
  if [ -f $O/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ]; then ok="$ok $S"; [ -f $O/footprint-sens.npz ] || echo "note: $S indoor original has no footprint-sens.npz (the job computes it)"; else echo "SKIP $S: no --indoor FP32 original at $O"; fi; done
echo "indoor G0 scenes with an original:$ok"
for S in room kitchen counter bonsai; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
busy=$(squeue -u $USER -h -o "%j" | grep -cE "g0r|adapt9|rd3i"); [ "$busy" = 0 ] || echo "note: $busy related jobs still in the queue"
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"; scontrol show reservation $RES | grep -o "EndTime=[^ ]*"
echo "CHECK OK"; k=0
g0i () { S=$1; X=$2; BUD=$3; case " $ok " in *" $S "*) ;; *) return;; esac
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=05:00:00 --export=ALL,SCENE=$S,IMAGES=images_2,XYZ=$X,INDOOR=1,BUDGETS="$BUD" --job-name=$S-g0ri $AN/budget-sweep-v3.sbatch) && { echo "submitted $S-g0ri $j  budgets $BUD"; k=$((k+1)); } || echo "FAILED: sbatch $S"; }
g0i room q14 "1215000 486000 220000"; g0i kitchen q16 "1349000 622000 272000"; g0i counter q15 "1162000 526000 245000"; g0i bonsai q15 "1328000 557000 290000"
echo "submitted $k of up to 4 indoor G0 jobs (~3.5-4 h each, 4 GPUs -> done in ~4 h; rows append to g0-refresh-<scene>.txt with '| --indoor')"
SUB20
bash $AN/submit-batch20-v1.sh
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R"
```

If the check prints `INDOOR BUDGET CHECK FAIL`, paste me its two lines and nothing is submitted; the fallback is the train_ckpt_v4 `--override max_points=B` wrapper (ready to write, half an hour).

## 2. After ~4 h: collect

```bash
AN=/scratch/zr2245/analysis; OUT=$AN/batch20-results-$(date +%m%d-%H%M).txt
{
echo "===== sacct"; sacct -X -u $USER -S now-8hours -o JobID,JobName%24,State,Elapsed,ExitCode | grep -E "g0ri|JobID"
for S in room kitchen counter bonsai; do echo "===== g0-refresh-$S.txt"; cat $AN/g0-refresh-$S.txt 2>/dev/null; for d in /scratch/zr2245/$S-budget-*; do [ -f $d/train.log ] && echo "$d: $(grep -h '\[ITER 30000\] Evaluating test' $d/train.log | tail -1)"; done; done
echo "===== FAILED lines"; grep -l "FAILED" $AN/g0r-*-g0ri-*.out 2>/dev/null | while read f; do echo "--- $f"; grep -h "FAILED" $f | head -3; tail -8 $f | cut -c1-300; done
} > $OUT 2>&1
echo "wrote $OUT ($(grep -c '' $OUT) lines)"
```
