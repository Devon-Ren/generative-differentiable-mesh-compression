# node-batch19-v1 (2026-11-08 night): G0 refresh on 7 scenes + Deep Blending full hull + 2.5 MB controller runs — ~35 GPU-h, one paste

**1. G0 refresh (the paper's premise, on the current protocol).** For garden, bicycle, stump, flowers, treehill, truck, train: train from scratch with upstream `--max_points B` at three budgets B = V of the scene's own λ 1.85e-3 / 6.2e-3 / 1.85e-2 asset divided by 0.9 (so the three paths meet at ~equal vertex count), level-1.0 delivery; and for each, the control: collapse the scene's dense FP32 original to exactly the vertex count the budget run ended with (coarse_vis, same 3000-step finetune, same K4096 R256 codec). `budget-sweep-v2.sbatch` (82 lines, md5 247b78f2bad723ecd7cae26e236076d9, VERSION g0r-v2) = budget-sweep-v1 + the g0-budget-v1 control + DATA / ORIG overrides. Rows `TRAIN-AT-BUDGET` / `POST-HOC-COLLAPSE` in `g0-refresh-$SCENE.txt`; the λ-grown assets are the third path (rd-prune3 tables), compared at equal bytes. Indoor scenes are NOT in this batch: upstream's `update_indoor()` overwrites `--max_points` after parsing (known pitfall), so they need the train_ckpt_v4 override path, to be smoke-tested in daylight. The job refuses INDOOR=1.
Budgets: truck 1934000 853000 354000; train 2546000 1270000 553000; flowers 2639000 1227000 522000; treehill 3053000 1245000 482000; garden 2346000 919000 414000; bicycle 2886000 1201000 474000; stump 2292000 888000 344000.

**2. Deep Blending, the two remaining λ points** (6.2e-4 and 1.85e-2) for drjohnson and playroom → a full 4-point hull on the third dataset. Same `rd-prune-v3c.sbatch` as tonight's first batch.

**3. Controller at 2.5 MB** on the 11 Mip-360 / T&T scenes, v9d as is (SMOKE off, self-check off). Rows append to `adapt-v9d.txt`.

Order in the queue: the 7 G0 jobs (longest) first, then the 2 DB jobs, then the 11 short ones; ~35 GPU-h, about 8.5–9 h on 4 GPUs. Every stage fails loudly and is restartable.

## 1. Write the sbatch + submit everything (one paste)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/budget-sweep-v2.sbatch <<'SBEOF_G0R'
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
[ "$INDOOR" = "0" ] || { echo "FAILED: INDOOR=1 is not supported by this sbatch (update_indoor overrides --max_points); indoor scenes need the train_ckpt_v4 --override path"; exit 1; }
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}; ORIG=${ORIG:-/scratch/zr2245/$SCENE-vis-collapse}; TAB=$AN/g0-refresh-$SCENE.txt
CORIG=$ORIG/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; MORIG=$ORIG/original; FOOT=$ORIG/footprint-sens.npz
CRIT=${CRIT:-"--geo-weight sens_pos --app-weight none --weight-power 0.5 --weight-clip 0.1,10 --weight-divide-pixel-scale"}
echo "g0-refresh | scene $SCENE | IMAGES $IMAGES | budgets $BUDGETS | original $ORIG | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $PKG/scripts/run_garden.sh $PKG/coarse_vis.py $PKG/vertex_footprint.py $PKG/deliver_coarse.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $DATA/sparse/0/cameras.bin $CORIG $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
ls $PKG/native/_native_qem_vis*.so >/dev/null 2>&1 || { echo "FAILED: native module missing in $PKG/native"; exit 1; }
(cd $LAB/upstream && python train.py -h 2>&1 | grep -q -- "--max_points") || { echo "FAILED: train.py does not accept --max_points"; (cd $LAB/upstream && python train.py -h 2>&1 | tail -5); exit 1; }
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
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | $IMAGES')
PY
}
for B in $BUDGETS; do
  G=/scratch/zr2245/$SCENE-budget-$B; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== A: $SCENE budget $B: train with --max_points $B ($(date)) ==="
  if [ ! -f $CB ]; then
    (cd $LAB/upstream && python train.py -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --max_points $B > $G/train.log 2>&1) \
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
SBEOF_G0R
cat > $AN/submit-batch19-v1.sh <<'SUB19'
AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2 (paste truncated?)"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3 (paste altered?)"; exit 1; }; }
chk $AN/budget-sweep-v2.sbatch 82 247b78f2bad723ecd7cae26e236076d9
for f in $AN/adapt-v9d.sbatch $AN/rd-prune-v3c.sbatch $AN/rdgrowth/train_rd_adapt_v9d.py $AN/rdgrowth/rd_prune_v2.py $AN/rdgrowth/train_rd_v3.py $AN/coarse_finetune.py $AN/vis_collapse-v12/coarse_vis.py $AN/vis_collapse-v12/scripts/run_garden.sh $AN/vis_collapse-v12/deliver_coarse.sh; do [ -f $f ] || { echo "CHECK FAILED: missing $f"; exit 1; }; done
grep -q "VERSION v9d-r1" $AN/adapt-v9d.sbatch && grep -q "VERSION v3c" $AN/rd-prune-v3c.sbatch && grep -q "VERSION g0r-v2" $AN/budget-sweep-v2.sbatch || { echo "CHECK FAILED: sbatch versions"; exit 1; }
bash -n $AN/budget-sweep-v2.sbatch && bash -n $AN/adapt-v9d.sbatch && bash -n $AN/rd-prune-v3c.sbatch || { echo "CHECK FAILED: sbatch syntax"; exit 1; }
ls $AN/vis_collapse-v12/native/_native_qem_vis*.so >/dev/null 2>&1 || { echo "CHECK FAILED: native qem module missing (coarse_vis needs it)"; exit 1; }
for S in garden bicycle stump flowers treehill; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in truck train; do [ -f $D/tandt/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in room kitchen counter bonsai; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in drjohnson playroom; do [ -f $D/db/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for db/$S"; exit 1; }; done
ok=""; for S in truck train flowers treehill garden bicycle stump; do O=/scratch/zr2245/$S-vis-collapse
  if [ -f $O/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ]; then ok="$ok $S"; [ -f $O/footprint-sens.npz ] || echo "note: $S original has no footprint-sens.npz yet (the job computes it, ~5 min)"; else echo "SKIP $S: no FP32 original at $O (G0 job not submitted for it)"; fi; done
echo "G0 scenes with an original:$ok"
busy=$(squeue -u $USER -h -o "%j" | grep -cE "adapt9|g0r|rd3i|origi|gate1"); [ "$busy" = 0 ] || echo "note: $busy related jobs still in the queue"
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"; scontrol show reservation $RES | grep -o "EndTime=[^ ]*"
echo "CHECK OK"; k=0
sb () { j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 "$@") && { echo "submitted $j  ${@: -1}  (${*:3:1})"; k=$((k+1)); } || echo "FAILED: sbatch $*"; }
# 1) G0 refresh: 7 outdoor / T&T scenes x 3 budgets (= V of the lambda 1.85e-3 / 6.2e-3 / 1.85e-2 assets / 0.9), train-at-budget + post-hoc-collapse control each; ~3.5 h per job
g0 () { S=$1; X=$2; BUD=$3; EXTRA=$4; case " $ok " in *" $S "*) ;; *) return;; esac
  IM=images_4; [ "$S" = truck ] || [ "$S" = train ] && IM=images
  sb --time=05:00:00 --export=ALL,SCENE=$S,IMAGES=$IM,XYZ=$X,INDOOR=0,BUDGETS="$BUD"$EXTRA --job-name=$S-g0r $AN/budget-sweep-v2.sbatch; }
g0 truck q18 "1934000 853000 354000" ",DATA=$D/tandt/truck"
g0 train q18 "2546000 1270000 553000" ",DATA=$D/tandt/train"
g0 flowers q15 "2639000 1227000 522000" ""
g0 treehill q16 "3053000 1245000 482000" ""
g0 garden q15 "2346000 919000 414000" ""
g0 bicycle q16 "2886000 1201000 474000" ""
g0 stump q16 "2292000 888000 344000" ""
# 2) Deep Blending: the two remaining fixed-lambda points (6.2e-4, 1.85e-2) -> full 4-point hull on the third dataset; ~2 h per job
for S in drjohnson playroom; do sb --time=03:30:00 --export=ALL,SCENE=$S,IMAGES=images,XYZ=q15,INDOOR=1,DATA=$D/db/$S,LAMS="0.00062 0.0185" --job-name=$S-rd3i2 $AN/rd-prune-v3c.sbatch; done
# 3) adaptive v9d at 2.5 MB on the 11 Mip-360 / T&T scenes (the low end of the controller's range); ~35 min per job
m () { sb --time=01:30:00 --export=ALL,SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS=2.5,SMOKE=0,SELFCHECK=0$5 --job-name=$1-2.5-adapt9d $AN/adapt-v9d.sbatch; }
m room 1 images_2 q14; m kitchen 1 images_2 q16; m counter 1 images_2 q15; m bonsai 1 images_2 q15; m garden 0 images_4 q15; m bicycle 0 images_4 q16; m stump 0 images_4 q16; m flowers 0 images_4 q15; m treehill 0 images_4 q16
m truck 0 images q18 ",DATA=$D/tandt/truck"; m train 0 images q18 ",DATA=$D/tandt/train"
echo "submitted $k of up to 20 jobs (7 G0 ~3.5 h each first, then 2 DB ~2 h, then 11 x 2.5 MB ~35 min; ~35 GPU-h, ~8.5-9 h on 4 GPUs)"
SUB19
bash $AN/submit-batch19-v1.sh
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R" | head -30
```

## 2. Optional, after 10 min: preflight lines of the first four G0 jobs

```bash
AN=/scratch/zr2245/analysis
for f in $AN/g0r-*-g0r-*.out; do [ -f $f ] || continue; echo "=== $f"; grep -h "preflight\|footprint\|=== A\|FAILED\|MISSING" $f | tail -4 | cut -c1-240; done
```

## 3. Morning: everything to one file, upload it

```bash
AN=/scratch/zr2245/analysis; OUT=$AN/batch19-results-$(date +%m%d-%H%M).txt
{
echo "===== sacct"; sacct -X -u $USER -S now-14hours -o JobID,JobName%24,State,Elapsed,ExitCode | grep -E "g0r|rd3i2|2.5-adapt9d|JobID"
echo "===== squeue"; squeue -u $USER -o "%.10i %.22j %.3t %.8M %R"
for S in truck train flowers treehill garden bicycle stump; do echo "===== g0-refresh-$S.txt"; cat $AN/g0-refresh-$S.txt 2>/dev/null; echo "--- test lines of the budget trainings"; for d in /scratch/zr2245/$S-budget-*; do [ -f $d/train.log ] && echo "$d: $(grep -h '\[ITER 30000\] Evaluating test' $d/train.log | tail -1)"; done; done
for S in drjohnson playroom; do echo "===== rd-prune3i-$S.txt"; cat $AN/rd-prune3i-$S.txt 2>/dev/null; done
echo "===== adapt-v9d.txt (2.5 MB rows)"; grep -E "\| target 2.5 MB" $AN/adapt-v9d.txt
echo "===== FAILED lines"; grep -l "FAILED" $AN/g0r-*.out $AN/rd3i-*-rd3i2-*.out $AN/adapt9d-*-2.5-adapt9d-*.out 2>/dev/null | while read f; do echo "--- $f"; grep -h "FAILED" $f | head -3; tail -8 $f | cut -c1-300; done
} > $OUT 2>&1
echo "wrote $OUT ($(grep -c '' $OUT) lines, $(du -h $OUT | cut -f1))"
```
