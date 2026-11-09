# node-batch21-v1 (2026-11-09 night): the completeness batch, all at once, last GPU window

Paste §1 in one go on the login/GPU terminal (no `set -e`, no bare `exit` outside the heredoc scripts). It writes four files, verifies them by line count and md5, runs the CPU check that `--indoor` leaves `max_points` alone, lists which Blender originals / hull points / DB lambda runs exist, refuses to double-submit, then submits 59 jobs with their dependencies. Reservation ends 11/12 09:35; ~100 GPU-h -> ~26 h wall on 4 GPUs. Every job name contains `b21`; outputs are `$AN/<prefix>-<jobname>-<jobid>.out`.

What it fills (per claim, per dataset; see submit script header for the job list):

| claim | Mip-360 / T&T | NeRF-Synthetic | Deep Blending |
|---|---|---|---|
| fixed-λ 4-point hull | done | **+ λ 1.85e-3, 1.85e-2 (C, C2)** | done |
| controller hit | done 5/10/20, **+ 7.5/15 (E2)** | done 2.5/4, **+ 1.5/3/6 (E3)** | done (v9d), **+ v9b 5/10/20 (E1)** |
| train-at-budget vs post-hoc (G0) | done | **all 4 λ counts (B)** | **3 counts (A)** |
| no-codec equal-V (derived) | done | from B | from A |
| λ × dictionary fan + DC q8 | done | **(D)** | **(D)** |
| generic codecs Draco / MPEG | room, garden, **+ 9 (F)** | – (black-bg protocol; not run) | **(F)** |

## §1 write files, check, submit (one paste)

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN
cat > $AN/budget-sweep-blender-v1.sbatch <<'SBEOF_G0RB'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=g0rb
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/zr2245/analysis/g0rb-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/g0rb-%x-%j.err
# budget-sweep-blender-v1.sbatch (2026-11-09, VERSION g0rb-v1): the G0 comparison (train-at-budget vs post-hoc collapse) on ONE NeRF-Synthetic scene,
# Blender protocol: --indoor, black background, checkpoints every 1k from 12k (train_ckpt_v2 wrapping upstream train.py --max_points B), the best
# OFFICIAL-val checkpoint chosen by val_select_v3 (importance recomputed at the final floor / 4x), then the level-1.0 protocol on $G-best.
# CONTROL: the scene's val-selected dense original ($ORIG, default $SCENE-blender4-best) collapsed by coarse_vis to the vertex count the budget
# run ended with, same 3000-step finetune, same K4096 R256 delivery.  Budgets come from LAMS_AUTO: for each lambda L the delivered vertex count of
# /scratch/zr2245/$SCENE-rd3b4-lam$L-mu1.0-best3 (the val_select_v3 re-delivery, the hull's points) divided by 0.9, so the three paths meet at ~equal V;
# a lambda whose -best3 delivery is missing is SKIPPED with a note (never a failure).  Explicit BUDGETS="..." are appended.  Rows -> $AN/g0-refresh-blender-$SCENE.txt
# ('TRAIN-AT-BUDGET-BLENDER' / 'POST-HOC-COLLAPSE-BLENDER').  A failing CONTROL stage prints FAILED and moves on (the budget row is kept).  Restartable.
#   sbatch --export=ALL,SCENE=lego,XYZ=q14,LAMS_AUTO="0.00062 0.00185 0.0062 0.0185" --job-name=lego-b21-g0rb .../budget-sweep-blender-v1.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=train; INDOOR=${INDOOR:-1}; LAMS_AUTO=${LAMS_AUTO:-"0.00062 0.00185 0.0062 0.0185"}; BUDGETS=${BUDGETS:-}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
SAVES="12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; VS=$AN/blender/val_select_v3.py; CK=$AN/blender/train_ckpt_v2.py
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; ORIG=${ORIG:-/scratch/zr2245/$SCENE-blender4-best}; TAB=$AN/g0-refresh-blender-$SCENE.txt
CORIG=$ORIG/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; MORIG=$ORIG/original; FOOT=$ORIG/footprint-sens.npz
CRIT=${CRIT:-"--geo-weight sens_pos --app-weight none --weight-power 0.5 --weight-clip 0.1,10 --weight-divide-pixel-scale"}
echo "g0-refresh-blender | scene $SCENE (black bg, val-selected) | lams_auto '$LAMS_AUTO' budgets '$BUDGETS' | indoor flag '$FLAG' | original $ORIG | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $PKG/scripts/run_garden.sh $PKG/coarse_vis.py $PKG/vertex_footprint.py $PKG/deliver_coarse.sh $AN/coarse_finetune.py $DATA/transforms_train.json $DATA/transforms_val.json $DATA/transforms_test.json $CORIG $VS $CK $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
ls $PKG/native/_native_qem_vis*.so >/dev/null 2>&1 || { echo "FAILED: native module missing in $PKG/native"; exit 1; }
grep -q "VERSION = 'val_select_v3" $VS || { echo "FAILED: $VS is not val_select_v3"; exit 1; }
grep -q "VERSION = 'train_ckpt_v2" $CK || { echo "FAILED: $CK is not train_ckpt_v2"; exit 1; }
(cd $LAB/upstream && python train.py -h 2>&1 | grep -q -- "--max_points") || { echo "FAILED: train.py does not accept --max_points"; exit 1; }
grep -A 20 "def update_indoor" $LAB/upstream/arguments/__init__.py | grep -q "max_points" && { echo "FAILED: this upstream's update_indoor() sets max_points; the CLI budget would be overwritten"; exit 1; }
echo "preflight ok: --max_points accepted, update_indoor() leaves it alone; original $(python -c "import torch; sd=torch.load('$CORIG',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))") vertices"
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
for L in $LAMS_AUTO; do
  RJ=/scratch/zr2245/$SCENE-rd3b4-lam$L-mu1.0-best3/axisB/v1.0/report.json
  if [ -f $RJ ]; then V=$(python -c "import json; print(json.load(open('$RJ'))['vertices'])"); B=$(( (V * 10 + 8) / 9 )); BUDGETS="$BUDGETS $B"; echo "lambda $L: delivered V $V -> budget $B"
  else echo "note: lambda $L has no -best3 delivery at $RJ; skipped"; fi
done
BUDGETS=$(echo $BUDGETS | tr ' ' '\n' | sort -nr | uniq | tr '\n' ' '); [ -n "$(echo $BUDGETS)" ] || { echo "FAILED: no budgets (no lambda delivery found and BUDGETS empty)"; exit 1; }
echo "budgets for $SCENE: $BUDGETS"
if [ ! -f $FOOT ]; then
  echo "=== footprint of the original (missing) -> $FOOT ($(date)) ==="
  (cd $PKG && python vertex_footprint.py --lab $LAB --model $MORIG --dataset $DATA --images $IMAGES --out $FOOT > $ORIG/footprint.log 2>&1) || { echo "FAILED: FOOTPRINT of the original (controls will be skipped)"; tail -8 $ORIG/footprint.log; }
fi
row () { # $1 tag  $2 report.json  $3 finetune/summary.json  $4 deliver/summary.json  $5 extra
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$2')); f=json.load(open('$3')); d=json.load(open('$4')); t=d['delivery_test']; h=f['history']
print('$1 | vertices', b['vertices'], 'faces', b['faces'], '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3),
      'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | $IMAGES | blackbg $FLAG | $5')
PY
}
nfail=0
for B in $BUDGETS; do
  G=/scratch/zr2245/$SCENE-budgetb-$B; GB=$G-best; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== A: $SCENE budget $B: train.py --max_points $B $FLAG, checkpoints every 1k from 12k ($(date)) ==="
  if [ ! -f $CB ]; then
    python $CK --saves $SAVES --script $LAB/upstream/train.py -- -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --max_points $B --test_iterations 7000 30000 $FLAG > $G/train.log 2>&1 \
      || { echo "FAILED: TRAIN $SCENE budget $B"; grep -n -m 6 -i 'error\|Traceback' $G/train.log; tail -10 $G/train.log; nfail=$((nfail+1)); continue; }
    [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE budget $B"; tail -8 $G/train.log; nfail=$((nfail+1)); continue; }
  else echo "exists: $CB"; fi
  grep -h "^\[ckpt\] all\|FAILED" $G/train.log | tail -2; grep -h "\[ITER 30000\] Evaluating test" $G/train.log | tail -1
  echo "=== A: $SCENE budget $B: val_select_v3 ($(date)) ==="
  if [ ! -f $GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ]; then
    rm -rf $GB
    python $VS --lab $LAB --dataset $DATA --model $G/original --best $GB/original --delete_unselected > $G/val_select.log 2>&1 || { echo "FAILED: VAL_SELECT $SCENE budget $B"; grep -n -m 6 -i 'error\|Traceback\|FAILED' $G/val_select.log; tail -10 $G/val_select.log; nfail=$((nfail+1)); continue; }
  fi
  grep -h "BEST\|VAL_SELECT_RESULT" $G/val_select.log | tail -2
  [ -f $GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "FAILED: val_select output missing for $SCENE budget $B"; nfail=$((nfail+1)); continue; }
  VB=$(nverts $GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt); echo "budget $B -> val-selected pruned vertices $VB"
  echo "=== A: $SCENE budget $B: level-1.0 protocol on $GB ($(date)) ==="
  if [ ! -f $GB/axisB/v1.0/deliver/summary.json ]; then
    G=$GB DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $GB/level.log 2>&1 || { echo "FAILED: LEVEL-1.0 $SCENE budget $B"; tail -8 $GB/level.log; nfail=$((nfail+1)); continue; }
  fi
  NA=$GB/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE budget $B"; tail -8 $GB/level.log; nfail=$((nfail+1)); continue; }
  row "$SCENE budget $B TRAIN-AT-BUDGET-BLENDER" $NA/report.json $NA/finetune/summary.json $NA/deliver/summary.json "max_points $B"
  [ -f $FOOT ] || { echo "skip control for budget $B: no footprint"; continue; }
  echo "=== B: $SCENE control: collapse the dense original to $VB vertices ($(date)) ==="
  NC=$ORIG/axisB/g0ctrl-$B; mkdir -p $ORIG/axisB
  if [ ! -f $NC/report.json ]; then rm -rf $NC
    (cd $PKG && python coarse_vis.py --source $CORIG --footprint $FOOT --base uniform --target-vertices $VB --output $NC $CRIT > $NC.build.log 2>&1) || { echo "FAILED: CONTROL BUILD $SCENE $B"; tail -8 $NC.build.log; nfail=$((nfail+1)); continue; }
  fi
  if [ ! -f $NC/finetune/summary.json ]; then rm -rf $NC/finetune
    (cd $AN && python coarse_finetune.py --lab $LAB --model $MORIG --dataset $DATA --images $IMAGES --coarse $NC/coarse-initial.pt --steps 3000 --out $NC/finetune > $NC.finetune.log 2>&1) || { echo "FAILED: CONTROL FINETUNE $SCENE $B"; tail -8 $NC.finetune.log; nfail=$((nfail+1)); continue; }
  fi
  if [ ! -f $NC/deliver/summary.json ]; then rm -rf $NC/deliver
    bash $PKG/deliver_coarse.sh --coarse $NC/finetune/best.pt --output $NC/deliver --historical-config none --source $CORIG --dataset $DATA --lab $LAB --images $IMAGES --k $K --r $R --xyz $XYZ --dc $DC --dictionary-rounds 4 > $NC.deliver.log 2>&1 || { echo "FAILED: CONTROL DELIVER $SCENE $B"; tail -8 $NC.deliver.log; nfail=$((nfail+1)); continue; }
  fi
  row "$SCENE budget $B POST-HOC-COLLAPSE-BLENDER" $NC/report.json $NC/finetune/summary.json $NC/deliver/summary.json "target-vertices $VB"
done
echo "===== $TAB ====="; cat $TAB 2>/dev/null
echo "g0-refresh-blender $SCENE exit 0 ($nfail stage failures)  $(date)"
SBEOF_G0RB

cat > $AN/lambda-k-fan-v3.sbatch <<'SBEOF_KFAN3'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=kfan3
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/zr2245/analysis/kfan3-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/kfan3-%x-%j.err
# lambda-k-fan-v3.sbatch (2026-11-09, VERSION v3): = lambda-k-fan-v2 (the joint lambda x dictionary-size fan + DC q8, no training) generalised to the
# other datasets: XYZ via --export (overrides xyz_of), SUFFIX appended to the run dir (Blender hull points live in $SCENE-rd3b4-lam$L-mu1.0-best3),
# DATA / IMAGES via --export (Deep Blending: datasets/db/$SCENE, images; NeRF-Synthetic: datasets/nerf_synthetic/$SCENE, train, transforms_*.json),
# the dataset check accepting either a COLMAP image folder or Blender transforms.  Same rows, same table $AN/lambda-k-fan-v1.txt.  A lambda whose
# delivery is missing is skipped with FAILED: MISSING (counted, not fatal).  Restartable.
#   sbatch --export=ALL,SCENE=drjohnson,TAG=rd3i,XYZ=q15,IMAGES=images,DATA=/scratch/zr2245/datasets/db/drjohnson --job-name=drjohnson-b21-kfan .../lambda-k-fan-v3.sbatch
#   sbatch --export=ALL,SCENE=lego,TAG=rd3b4,SUFFIX=-best3,XYZ=q14,IMAGES=train,DATA=/scratch/zr2245/datasets/nerf_synthetic/lego --job-name=lego-b21-kfan .../lambda-k-fan-v3.sbatch
set -o pipefail
export TORCH_HOME=/scratch/zr2245/cache/torch
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; TAG=${TAG:?set TAG=rd3i (indoor / Deep Blending) or rd3 (outdoor) or rd3b4 (Blender) via --export}; IMAGES=${IMAGES:-images_2}; SUFFIX=${SUFFIX:-}
LAMS=${LAMS:-"0.00062 0.00185 0.0062 0.0185"}; KRS=${KRS:-"4096 1|1024 256|4096 1024|16384 1024"}
AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; TAB=$AN/lambda-k-fan-v1.txt
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}
xyz_of () { case $1 in room) echo q14;; garden|bonsai|counter|flowers|drjohnson|playroom) echo q15;; kitchen|bicycle|stump|treehill) echo q16;; truck|train) echo q18;; chair|drums|ficus|hotdog|lego|materials|mic|ship) echo q14;; *) echo "FAILED: no XYZ recorded for $1" >&2; return 1;; esac; }
XYZ=${XYZ:-$(xyz_of $SCENE)} || exit 1
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
echo "lambda-k-fan-v3 | scene $SCENE tag $TAG suffix '$SUFFIX' | DATA $DATA/$IMAGES | lams $LAMS | K x R $KRS | XYZ $XYZ | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $PKG/codec_variants.py $PKG/deliver_coarse.sh $PKG/stage2_run.sh /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
[ -e $DATA/$IMAGES/. ] || [ -f $DATA/transforms_train.json ] || { echo "FAILED: neither $DATA/$IMAGES nor $DATA/transforms_train.json exists"; exit 1; }
nf=0
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-$TAG-lam$L-mu1.0$SUFFIX; d=$G/axisB/v1.0/deliver; MESH=$G/axisB/v1.0/finetune/best.pt; C=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
  miss=0; for f in $d/shared-final.pt $d/config.json $d/summary.json $MESH $C; do [ -e $f ] || { echo "FAILED: MISSING $f"; miss=1; }; done
  [ $miss = 0 ] || { nf=$((nf+1)); continue; }
  python -c "import json; c=json.load(open('$d/config.json')); k=c.get('k',4096); r=c.get('r',256); assert (k,r)==(4096,256), ('baseline delivery is not K4096 R256', k, r); assert c.get('xyz')=='$XYZ', ('delivered xyz', c.get('xyz'), 'differs from', '$XYZ')" || { echo "FAILED: $SCENE lam $L baseline delivery is not K4096 R256 / $XYZ"; nf=$((nf+1)); continue; }
  echo "=== $SCENE lam $L: baseline + (a) DC q8 on $d (XYZ $XYZ) ($(date +%H:%M)) ==="
  for V in "const/ctx q7" "const/ctx q8"; do set -- $V; W=$1; DC=$2
    if python -c "import json,sys; s=json.load(open('$d/codec-variants/sweep.json')); sys.exit(0 if any((r['weight']+'/'+r['index_codec'],r['xyz'],r['dc'])==('$W','$XYZ','$DC') and r.get('test') for r in s['rows']) else 1)" 2>/dev/null; then echo "exists: $W $XYZ/$DC"; continue; fi
    bash $PKG/stage2_run.sh codec_variants.py --deliver-dir $d --xyz $XYZ --dc $DC --variants $W > $d/codec-variants-fan-${W/\//-}-$DC.log 2>&1 || { echo "FAILED: codec_variants $SCENE lam $L $W $XYZ/$DC"; tail -5 $d/codec-variants-fan-${W/\//-}-$DC.log; nf=$((nf+1)); }
  done
  echo "=== $SCENE lam $L: (d) dictionary re-fits on $MESH ($(date +%H:%M)) ==="
  OUT=$G/kfan-v1; mkdir -p $OUT
  echo "$KRS" | tr '|' '\n' | while read K R; do name=$OUT/k${K}_r${R}
    if [ -f $name/summary.json ]; then echo "exists: k${K}_r${R}"; continue; fi
    rm -rf $name; echo "--- K $K R $R $(date +%H:%M:%S)"
    bash $PKG/deliver_coarse.sh --coarse $MESH --output $name --source $C --dataset $DATA --lab $LAB --images $IMAGES --xyz $XYZ --dc q7 \
      --historical-config none --k $K --r $R --dictionary-rounds 4 > $name.log 2>&1 || { echo "FAILED: deliver_coarse $SCENE lam $L k${K}_r${R}"; tail -8 $name.log; }
    [ -f $name/summary.json ] || echo "FAILED: no summary for $SCENE lam $L k${K}_r${R}"
  done
  python - "$SCENE" "$L" "$d" "$XYZ" "$OUT" "$KRS" "$TAG$SUFFIX" <<'PY' | tee -a $TAB
import json, sys, pathlib
S, L, d, xyz, out, krs, tag = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]), sys.argv[4], pathlib.Path(sys.argv[5]), sys.argv[6], sys.argv[7]
s = json.load(open(d / 'codec-variants' / 'sweep.json')); rows = {}
for r in s['rows']:
    if r['xyz'] == xyz and r.get('test'): rows[(r['weight'] + '/' + r['index_codec'], r['dc'])] = r
base = rows.get(('const/ctx', 'q7'))
if base is None: print(f"{S} | lam {L} | baseline MISSING | {tag}"); sys.exit(0)
bp = base['test']['psnr']; bm = base['file_MB']
def line(name, mb, t): print(f"{S} | lam {L} | {name:22s} | {mb:7.3f} MB | test {t['psnr']:.3f} {t['ssim']:.4f} {t['lpips']:.4f} | dPSNR {t['psnr'] - bp:+.3f} | dMB {mb - bm:+.3f} | {tag}")
line('K4096 R256 q7 (base)', bm, base['test'])
r = rows.get(('const/ctx', 'q8')); line('K4096 R256 q8', r['file_MB'], r['test']) if r else print(f"{S} | lam {L} | K4096 R256 q8: MISSING | {tag}")
for kr in krs.split('|'):
    K, R = kr.split(); p = out / f'k{K}_r{R}' / 'summary.json'
    if p.exists(): j = json.load(open(p)); line(f'K{K} R{R} q7', j['file_MB'], j['delivery_test'])
    else: print(f"{S} | lam {L} | K{K} R{R} q7: MISSING | {tag}")
PY
done
echo "===== $TAB ($SCENE rows) ====="; grep "^$SCENE " $TAB
echo "lambda-k-fan-v3 $SCENE exit 0 ($nf missing / failed)  $(date)"
SBEOF_KFAN3

cat > $AN/generic-codecs-b21.sbatch <<'SBEOF_GENC'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=genc
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/zr2245/analysis/genc-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/genc-%x-%j.err
# generic-codecs-b21.sbatch (2026-11-09, VERSION genc-b21): the generic-codec baselines (Draco / zstd / texture route: generic-v1/run_generic_v1.sh,
# and MPEG-4 SC3DMC-TFAN: o3dgc-v1/run_o3dgc_v3.sh) on the ORIGINAL mesh of every scene in SCENES, one after another, in one job.  So far only
# room and garden have rows (generic-codecs.txt, o3dgc rows).  Both runners are restartable and non-fatal per row; a scene whose runner fails
# prints FAILED and the job moves to the next scene.  Deep Blending scenes use the --indoor originals ($SCENE-indoor) through the runners'
# generic branch (RUN / DATA / IMAGES / QP exported).  TEX="4096" (one texture resolution: the xatlas bake is the slow part).
#   sbatch --export=ALL,SCENES="kitchen counter bonsai drjohnson playroom" --job-name=b21-genc-a .../generic-codecs-b21.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENES=${SCENES:?set SCENES="kitchen counter ..." via --export}; export TEX=${TEX:-"4096"}
AN=/scratch/zr2245/analysis; G=$AN/generic-v1; O=$AN/o3dgc-v1; D=/scratch/zr2245/datasets
for f in $G/run_generic_v1.sh $G/generic_codec_v1.py $G/eval_ckpt_v1.py $O/run_o3dgc_v3.sh $O/o3dgc_codec_v2.py; do [ -f $f ] || { echo "FAILED: MISSING $f (generic-codecs-v1 / o3dgc-v3 blocks were never installed?)"; exit 1; }; done
echo "generic-codecs-b21 | scenes $SCENES | TEX '$TEX' | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nf=0
for S in $SCENES; do
  unset RUN DATA IMAGES QP
  case $S in
    drjohnson|playroom) export RUN=/scratch/zr2245/$S-indoor/original DATA=$D/db/$S IMAGES=images QP=15;;
    room|kitchen|counter|bonsai|garden|bicycle|stump|flowers|treehill|truck|train) ;;   # known to both runners' case tables
    *) echo "FAILED: no settings for scene $S"; nf=$((nf+1)); continue;;
  esac
  if [ -n "$RUN" ] && [ ! -f $RUN/point_cloud/iteration_30000/point_cloud_state_dict.pt ]; then echo "FAILED: $S original missing at $RUN"; nf=$((nf+1)); continue; fi
  echo "=== $S: generic-v1 (zstd / Draco / texture) $(date) ==="
  bash $G/run_generic_v1.sh $S > $G/$S-b21.log 2>&1 || { echo "FAILED: run_generic_v1 $S"; tail -12 $G/$S-b21.log; nf=$((nf+1)); }
  grep -h "^$S \|FAILED\|WARNING" $G/$S-b21.log | tail -14
  echo "=== $S: o3dgc-v3 (MPEG SC3DMC / TFAN) $(date) ==="
  EVAL=1 bash $O/run_o3dgc_v3.sh $S > $O/$S-b21.log 2>&1 || { echo "FAILED: run_o3dgc_v3 $S"; tail -12 $O/$S-b21.log; nf=$((nf+1)); }
  grep -ih "ROW\|FAILED\|done" $O/$S-b21.log | tail -8
done
echo "===== generic-codecs.txt (these scenes) ====="; for S in $SCENES; do grep -h "^$S " $AN/generic-codecs.txt 2>/dev/null; done
echo "generic-codecs-b21 exit 0 ($nf failures)  $(date)"
SBEOF_GENC

cat > $AN/submit-batch21-v1.sh <<'SHEOF_B21'
#!/usr/bin/env bash
# submit-batch21-v1.sh (2026-11-09, r2: rd-prune-blender-v5 = the Blender lambda sbatch that exists on HPC; v4 was superseded the same day): the completeness batch, everything at once, last GPU window (reservation ends 11/12 09:35).
# Fills every hole in the main tables so that each claim covers all four datasets, plus redundancy on the controller.
#   A  G0 (train-at-budget vs post-hoc collapse) on Deep Blending          2 jobs   budget-sweep-v3.sbatch (INDOOR=1, DB originals)
#   C  NeRF-Synthetic fixed-lambda 1.85e-3 and 1.85e-2 (4-point hull)       8 jobs   rd-prune-blender-v5.sbatch
#   C2 their val_select_v3 re-delivery (-best3, the hull's protocol)         8 jobs   redeliver-v3-blender.sbatch      afterok C
#   B  G0 on NeRF-Synthetic at all four lambda vertex counts                 8 jobs   budget-sweep-blender-v1.sbatch   afterok C2
#   D  lambda x dictionary fan + DC q8 on Deep Blending and NeRF-Synthetic  10 jobs   lambda-k-fan-v3.sbatch           (Blender: afterok C2)
#   E1 controller v9b (the headline version) on Deep Blending 5/10/20 MB     2 jobs   adapt-v9b.sbatch
#   E2 controller v9b on Mip-360 / T&T at 7.5 and 15 MB (5 targets total)   11 jobs   adapt-v9b.sbatch
#   E3 controller v9b on NeRF-Synthetic at 1.5 / 3 / 6 MB (5 targets total)  8 jobs   adapt-v9b-blender.sbatch
#   F  generic codecs (Draco / zstd / texture / MPEG) on the 9 remaining Mip/T&T scenes + Deep Blending   2 jobs   generic-codecs-b21.sbatch
# ~100 GPU-h; 4 GPUs -> ~26 h wall.  Every job name carries 'b21'.  Nothing here overwrites an existing run directory (new names throughout).
AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets; ACC=torch_pr_38_general
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2 (paste truncated?)"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3 (paste altered?)"; exit 1; }; }
chk $AN/budget-sweep-blender-v1.sbatch 105 d720b8ccba85622f90f25ee0b52b77ee
chk $AN/lambda-k-fan-v3.sbatch 75 5d0dd7ff4012fa9a858c3a0ed1aff8d6
chk $AN/generic-codecs-b21.sbatch 45 28807d9a2ed0a77d4a29fd4e75f910a3
for f in $AN/budget-sweep-v3.sbatch $AN/rd-prune-blender-v5.sbatch $AN/redeliver-v3-blender.sbatch $AN/adapt-v9b.sbatch $AN/adapt-v9b-blender.sbatch $AN/budget-sweep-blender-v1.sbatch $AN/lambda-k-fan-v3.sbatch $AN/generic-codecs-b21.sbatch; do [ -f $f ] || { echo "CHECK FAILED: missing $f"; exit 1; }; bash -n $f || { echo "CHECK FAILED: syntax $f"; exit 1; }; done
grep -q "VERSION g0r-v3" $AN/budget-sweep-v3.sbatch && grep -q "VERSION v5" $AN/rd-prune-blender-v5.sbatch && grep -q "VERSION rd3" $AN/redeliver-v3-blender.sbatch && grep -q "VERSION g0rb-v1" $AN/budget-sweep-blender-v1.sbatch && grep -q "VERSION v3" $AN/lambda-k-fan-v3.sbatch && grep -q "VERSION genc-b21" $AN/generic-codecs-b21.sbatch || { echo "CHECK FAILED: sbatch versions"; exit 1; }
grep -q "train_rd_adapt_v9b.py" $AN/adapt-v9b.sbatch && grep -q "train_rd_adapt_v9b.py" $AN/adapt-v9b-blender.sbatch || { echo "CHECK FAILED: adapt-v9b sbatch files do not reference train_rd_adapt_v9b.py"; exit 1; }
for f in $AN/rdgrowth/train_rd_adapt_v9b.py $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/probe_bytes_v1.py $AN/blender/val_select_v3.py $AN/blender/val_select_v2.py $AN/blender/train_ckpt_v2.py $AN/vis_collapse-v12/coarse_vis.py $AN/coarse_finetune.py $AN/generic-v1/run_generic_v1.sh $AN/o3dgc-v1/run_o3dgc_v3.sh; do [ -f $f ] || { echo "CHECK FAILED: missing $f"; exit 1; }; done
python -m py_compile $AN/rdgrowth/train_rd_adapt_v9b.py $AN/blender/val_select_v3.py $AN/blender/train_ckpt_v2.py || { echo "CHECK FAILED: py_compile"; exit 1; }
python $AN/indoor_budget_check_v1.py || { echo "CHECK FAILED: the indoor preset overwrites max_points on this upstream (budget runs with --indoor would be wrong)"; exit 1; }
BL="chair drums ficus hotdog lego materials mic ship"; MIP="room kitchen counter bonsai garden bicycle stump flowers treehill truck train"; DB="drjohnson playroom"
for S in $BL; do [ -f $D/nerf_synthetic/$S/transforms_val.json ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in $DB; do [ -f $D/db/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; [ -f /scratch/zr2245/$S-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no --indoor FP32 original for $S at /scratch/zr2245/$S-indoor"; exit 1; }; done
for S in room kitchen counter bonsai; do [ -f /scratch/zr2245/$S-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || echo "note: $S has no -indoor original (only matters for the generic-codec job's o3dgc branch, which uses -vis-collapse anyway)"; done
echo "=== Blender originals for the G0 control (val-selected -blender4-best preferred) ==="; declare -A BORIG
for S in $BL; do O=""; for c in /scratch/zr2245/$S-blender4-best /scratch/zr2245/$S-blender4 /scratch/zr2245/$S-blender2; do [ -f $c/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] && { O=$c; break; }; done
  [ -n "$O" ] && { BORIG[$S]=$O; echo "$S: $O"; } || echo "WARNING: $S has no Blender original; its G0 job will fail at preflight (train-at-budget rows still need it for the control)"; done
echo "=== existing Blender hull points (-best3) and DB lambda runs for the fan ==="
for S in $BL; do e=""; for L in 0.00062 0.0062; do [ -f /scratch/zr2245/$S-rd3b4-lam$L-mu1.0-best3/axisB/v1.0/deliver/summary.json ] && e="$e $L"; done; echo "$S best3:$e"; done
for S in $DB; do e=""; for L in 0.00062 0.00185 0.0062 0.0185; do [ -f /scratch/zr2245/$S-rd3i-lam$L-mu1.0/axisB/v1.0/deliver/summary.json ] && e="$e $L"; done; echo "$S rd3i:$e"; done
for S in $BL; do for L in 0.00185 0.0185; do [ -e /scratch/zr2245/$S-rd3b4-lam$L-mu1.0 ] && echo "note: /scratch/zr2245/$S-rd3b4-lam$L-mu1.0 exists (restart: finished stages skipped)"; done; done
busy=$(squeue -u $USER -h -o "%j" | grep -c "b21"); [ "$busy" = 0 ] || { echo "CHECK FAILED: $busy b21 jobs already in the queue (double submission?)"; squeue -u $USER -o "%.10i %.24j %.2t %.10M"; exit 1; }
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"; scontrol show reservation $RES | grep -o "EndTime=[^ ]*"
echo "CHECK OK"; k=0; declare -A JC JC2
sb () { j=$(sbatch --parsable --reservation=$RES --account=$ACC --cpus-per-task=6 "$@") && { echo "submitted $j  $(echo "$@" | grep -o 'job-name=[^ ]*' | cut -d= -f2)"; k=$((k+1)); LASTJ=$j; } || { echo "FAILED: sbatch $*"; LASTJ=""; }; }
# --- A: G0 Deep Blending (budgets = lambda asset V / 0.9 from deep-blending-fixed-lambda-2026-11-09: 1232667/478718/200920 and 1080549/396809/171479)
sb --time=06:00:00 --export=ALL,SCENE=drjohnson,IMAGES=images,XYZ=q15,INDOOR=1,DATA=$D/db/drjohnson,BUDGETS="1370000 532000 224000" --job-name=drjohnson-b21-g0r $AN/budget-sweep-v3.sbatch
sb --time=06:00:00 --export=ALL,SCENE=playroom,IMAGES=images,XYZ=q15,INDOOR=1,DATA=$D/db/playroom,BUDGETS="1201000 441000 191000" --job-name=playroom-b21-g0r $AN/budget-sweep-v3.sbatch
# --- C: NeRF-Synthetic fixed-lambda 1.85e-3 / 1.85e-2 (same sbatch, prices, schedule, codec as the 11/04 hull points), then C2 re-delivery
for S in $BL; do
  sb --time=03:30:00 --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,LAMS="0.00185 0.0185",BPV=10.0,BPF=2.0,MU=1.0,MAXF=0.15 --job-name=$S-b21-b4-185 $AN/rd-prune-blender-v5.sbatch; JC[$S]=$LASTJ
  if [ -n "${JC[$S]}" ]; then sb --time=01:30:00 --dependency=afterok:${JC[$S]} --export=ALL,RUNS="$S-rd3b4-lam0.00185-mu1.0 $S-rd3b4-lam0.0185-mu1.0",XYZ=q14 --job-name=$S-b21-redeliv3 $AN/redeliver-v3-blender.sbatch; JC2[$S]=$LASTJ; fi
done
# --- D (Deep Blending now; Blender after C2): lambda x dictionary fan + DC q8
for S in $DB; do sb --time=04:00:00 --export=ALL,SCENE=$S,TAG=rd3i,XYZ=q15,IMAGES=images,DATA=$D/db/$S --job-name=$S-b21-kfan $AN/lambda-k-fan-v3.sbatch; done
# --- E1: controller v9b on Deep Blending (the headline controller version; batch18 ran v9d there)
for S in $DB; do sb --time=06:00:00 --export=ALL,SCENE=$S,INDOOR=1,IMAGES=images,XYZ=q15,TARGETS="5 10 20",DATA=$D/db/$S,SMOKE=1,SELFCHECK=0 --job-name=$S-b21-adapt9x $AN/adapt-v9b.sbatch; done
# --- B + D(Blender): after each scene's C2 (the budgets and the fan read the -best3 deliveries; existing 6.2e-4 / 6.2e-3 points are used as they are)
for S in $BL; do dep=""; [ -n "${JC2[$S]}" ] && dep="--dependency=afterany:${JC2[$S]}"
  sb --time=04:00:00 $dep --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,ORIG=${BORIG[$S]:-/scratch/zr2245/$S-blender4-best},LAMS_AUTO="0.00062 0.00185 0.0062 0.0185" --job-name=$S-b21-g0rb $AN/budget-sweep-blender-v1.sbatch
  sb --time=03:00:00 $dep --export=ALL,SCENE=$S,TAG=rd3b4,SUFFIX=-best3,XYZ=q14,IMAGES=train,DATA=$D/nerf_synthetic/$S --job-name=$S-b21-kfan $AN/lambda-k-fan-v3.sbatch
done
# --- E2: controller v9b on Mip-360 / T&T at 7.5 and 15 MB (adds two targets to the 5/10/20 table)
m () { sb --time=03:30:00 --export=ALL,SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS="7.5 15"$5,SMOKE=0,SELFCHECK=0 --job-name=$1-b21-adapt9x $AN/adapt-v9b.sbatch; }
m room 1 images_2 q14; m kitchen 1 images_2 q16; m counter 1 images_2 q15; m bonsai 1 images_2 q15
m garden 0 images_4 q15; m bicycle 0 images_4 q16; m stump 0 images_4 q16; m flowers 0 images_4 q15; m treehill 0 images_4 q16
m truck 0 images q18 ",DATA=$D/tandt/truck"; m train 0 images q18 ",DATA=$D/tandt/train"
# --- E3: controller v9b on NeRF-Synthetic at 1.5 / 3 / 6 MB (adds three targets to the 2.5/4 table; ficus/mic at 6 MB report the capacity limit)
for S in $BL; do sb --time=02:30:00 --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,TARGETS="1.5 3 6",SMOKE=0,SELFCHECK=0 --job-name=$S-b21-adapt9xb $AN/adapt-v9b-blender.sbatch; done
# --- F: generic codecs on the remaining scenes (room / garden already have rows)
sb --time=10:00:00 --export=ALL,SCENES="kitchen counter bonsai drjohnson playroom truck" --job-name=b21-genc-a $AN/generic-codecs-b21.sbatch
sb --time=10:00:00 --export=ALL,SCENES="bicycle stump flowers treehill train" --job-name=b21-genc-b $AN/generic-codecs-b21.sbatch
echo "submitted $k of 59 jobs (A 2, C 8, C2 8, D 10, E1 2, B 8, E2 11, E3 8, F 2); dependent jobs wait on their scene's C / C2"
squeue -u $USER -o "%.10i %.26j %.2t %.10M %.12E" | head -70
SHEOF_B21

bash $AN/submit-batch21-v1.sh 2>&1 | tee $AN/submit-batch21-v1.log
```

If it prints `CHECK FAILED: ...` nothing is submitted: paste me the line. `WARNING:` lines (a Blender original missing) do not stop the submission; the affected G0 job fails at its own preflight and the rest runs.

## §1b fix after the first paste (11/09 night): the Blender lambda sbatch on HPC is rd-prune-blender-v5 (output names rd3b4), not v4

The three new sbatch files are already on HPC and passed their md5 checks; only the submit script changes (v4 -> v5, version grep). Paste this instead of §1:

```bash
AN=/scratch/zr2245/analysis
cat > $AN/submit-batch21-v1.sh <<'SHEOF_B21R2'
#!/usr/bin/env bash
# submit-batch21-v1.sh (2026-11-09, r2: rd-prune-blender-v5 = the Blender lambda sbatch that exists on HPC; v4 was superseded the same day): the completeness batch, everything at once, last GPU window (reservation ends 11/12 09:35).
# Fills every hole in the main tables so that each claim covers all four datasets, plus redundancy on the controller.
#   A  G0 (train-at-budget vs post-hoc collapse) on Deep Blending          2 jobs   budget-sweep-v3.sbatch (INDOOR=1, DB originals)
#   C  NeRF-Synthetic fixed-lambda 1.85e-3 and 1.85e-2 (4-point hull)       8 jobs   rd-prune-blender-v5.sbatch
#   C2 their val_select_v3 re-delivery (-best3, the hull's protocol)         8 jobs   redeliver-v3-blender.sbatch      afterok C
#   B  G0 on NeRF-Synthetic at all four lambda vertex counts                 8 jobs   budget-sweep-blender-v1.sbatch   afterok C2
#   D  lambda x dictionary fan + DC q8 on Deep Blending and NeRF-Synthetic  10 jobs   lambda-k-fan-v3.sbatch           (Blender: afterok C2)
#   E1 controller v9b (the headline version) on Deep Blending 5/10/20 MB     2 jobs   adapt-v9b.sbatch
#   E2 controller v9b on Mip-360 / T&T at 7.5 and 15 MB (5 targets total)   11 jobs   adapt-v9b.sbatch
#   E3 controller v9b on NeRF-Synthetic at 1.5 / 3 / 6 MB (5 targets total)  8 jobs   adapt-v9b-blender.sbatch
#   F  generic codecs (Draco / zstd / texture / MPEG) on the 9 remaining Mip/T&T scenes + Deep Blending   2 jobs   generic-codecs-b21.sbatch
# ~100 GPU-h; 4 GPUs -> ~26 h wall.  Every job name carries 'b21'.  Nothing here overwrites an existing run directory (new names throughout).
AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets; ACC=torch_pr_38_general
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2 (paste truncated?)"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3 (paste altered?)"; exit 1; }; }
chk $AN/budget-sweep-blender-v1.sbatch 105 d720b8ccba85622f90f25ee0b52b77ee
chk $AN/lambda-k-fan-v3.sbatch 75 5d0dd7ff4012fa9a858c3a0ed1aff8d6
chk $AN/generic-codecs-b21.sbatch 45 28807d9a2ed0a77d4a29fd4e75f910a3
for f in $AN/budget-sweep-v3.sbatch $AN/rd-prune-blender-v5.sbatch $AN/redeliver-v3-blender.sbatch $AN/adapt-v9b.sbatch $AN/adapt-v9b-blender.sbatch $AN/budget-sweep-blender-v1.sbatch $AN/lambda-k-fan-v3.sbatch $AN/generic-codecs-b21.sbatch; do [ -f $f ] || { echo "CHECK FAILED: missing $f"; exit 1; }; bash -n $f || { echo "CHECK FAILED: syntax $f"; exit 1; }; done
grep -q "VERSION g0r-v3" $AN/budget-sweep-v3.sbatch && grep -q "VERSION v5" $AN/rd-prune-blender-v5.sbatch && grep -q "VERSION rd3" $AN/redeliver-v3-blender.sbatch && grep -q "VERSION g0rb-v1" $AN/budget-sweep-blender-v1.sbatch && grep -q "VERSION v3" $AN/lambda-k-fan-v3.sbatch && grep -q "VERSION genc-b21" $AN/generic-codecs-b21.sbatch || { echo "CHECK FAILED: sbatch versions"; exit 1; }
grep -q "train_rd_adapt_v9b.py" $AN/adapt-v9b.sbatch && grep -q "train_rd_adapt_v9b.py" $AN/adapt-v9b-blender.sbatch || { echo "CHECK FAILED: adapt-v9b sbatch files do not reference train_rd_adapt_v9b.py"; exit 1; }
for f in $AN/rdgrowth/train_rd_adapt_v9b.py $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/probe_bytes_v1.py $AN/blender/val_select_v3.py $AN/blender/val_select_v2.py $AN/blender/train_ckpt_v2.py $AN/vis_collapse-v12/coarse_vis.py $AN/coarse_finetune.py $AN/generic-v1/run_generic_v1.sh $AN/o3dgc-v1/run_o3dgc_v3.sh; do [ -f $f ] || { echo "CHECK FAILED: missing $f"; exit 1; }; done
python -m py_compile $AN/rdgrowth/train_rd_adapt_v9b.py $AN/blender/val_select_v3.py $AN/blender/train_ckpt_v2.py || { echo "CHECK FAILED: py_compile"; exit 1; }
python $AN/indoor_budget_check_v1.py || { echo "CHECK FAILED: the indoor preset overwrites max_points on this upstream (budget runs with --indoor would be wrong)"; exit 1; }
BL="chair drums ficus hotdog lego materials mic ship"; MIP="room kitchen counter bonsai garden bicycle stump flowers treehill truck train"; DB="drjohnson playroom"
for S in $BL; do [ -f $D/nerf_synthetic/$S/transforms_val.json ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in $DB; do [ -f $D/db/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; [ -f /scratch/zr2245/$S-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no --indoor FP32 original for $S at /scratch/zr2245/$S-indoor"; exit 1; }; done
for S in room kitchen counter bonsai; do [ -f /scratch/zr2245/$S-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || echo "note: $S has no -indoor original (only matters for the generic-codec job's o3dgc branch, which uses -vis-collapse anyway)"; done
echo "=== Blender originals for the G0 control (val-selected -blender4-best preferred) ==="; declare -A BORIG
for S in $BL; do O=""; for c in /scratch/zr2245/$S-blender4-best /scratch/zr2245/$S-blender4 /scratch/zr2245/$S-blender2; do [ -f $c/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] && { O=$c; break; }; done
  [ -n "$O" ] && { BORIG[$S]=$O; echo "$S: $O"; } || echo "WARNING: $S has no Blender original; its G0 job will fail at preflight (train-at-budget rows still need it for the control)"; done
echo "=== existing Blender hull points (-best3) and DB lambda runs for the fan ==="
for S in $BL; do e=""; for L in 0.00062 0.0062; do [ -f /scratch/zr2245/$S-rd3b4-lam$L-mu1.0-best3/axisB/v1.0/deliver/summary.json ] && e="$e $L"; done; echo "$S best3:$e"; done
for S in $DB; do e=""; for L in 0.00062 0.00185 0.0062 0.0185; do [ -f /scratch/zr2245/$S-rd3i-lam$L-mu1.0/axisB/v1.0/deliver/summary.json ] && e="$e $L"; done; echo "$S rd3i:$e"; done
for S in $BL; do for L in 0.00185 0.0185; do [ -e /scratch/zr2245/$S-rd3b4-lam$L-mu1.0 ] && echo "note: /scratch/zr2245/$S-rd3b4-lam$L-mu1.0 exists (restart: finished stages skipped)"; done; done
busy=$(squeue -u $USER -h -o "%j" | grep -c "b21"); [ "$busy" = 0 ] || { echo "CHECK FAILED: $busy b21 jobs already in the queue (double submission?)"; squeue -u $USER -o "%.10i %.24j %.2t %.10M"; exit 1; }
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"; scontrol show reservation $RES | grep -o "EndTime=[^ ]*"
echo "CHECK OK"; k=0; declare -A JC JC2
sb () { j=$(sbatch --parsable --reservation=$RES --account=$ACC --cpus-per-task=6 "$@") && { echo "submitted $j  $(echo "$@" | grep -o 'job-name=[^ ]*' | cut -d= -f2)"; k=$((k+1)); LASTJ=$j; } || { echo "FAILED: sbatch $*"; LASTJ=""; }; }
# --- A: G0 Deep Blending (budgets = lambda asset V / 0.9 from deep-blending-fixed-lambda-2026-11-09: 1232667/478718/200920 and 1080549/396809/171479)
sb --time=06:00:00 --export=ALL,SCENE=drjohnson,IMAGES=images,XYZ=q15,INDOOR=1,DATA=$D/db/drjohnson,BUDGETS="1370000 532000 224000" --job-name=drjohnson-b21-g0r $AN/budget-sweep-v3.sbatch
sb --time=06:00:00 --export=ALL,SCENE=playroom,IMAGES=images,XYZ=q15,INDOOR=1,DATA=$D/db/playroom,BUDGETS="1201000 441000 191000" --job-name=playroom-b21-g0r $AN/budget-sweep-v3.sbatch
# --- C: NeRF-Synthetic fixed-lambda 1.85e-3 / 1.85e-2 (same sbatch, prices, schedule, codec as the 11/04 hull points), then C2 re-delivery
for S in $BL; do
  sb --time=03:30:00 --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,LAMS="0.00185 0.0185",BPV=10.0,BPF=2.0,MU=1.0,MAXF=0.15 --job-name=$S-b21-b4-185 $AN/rd-prune-blender-v5.sbatch; JC[$S]=$LASTJ
  if [ -n "${JC[$S]}" ]; then sb --time=01:30:00 --dependency=afterok:${JC[$S]} --export=ALL,RUNS="$S-rd3b4-lam0.00185-mu1.0 $S-rd3b4-lam0.0185-mu1.0",XYZ=q14 --job-name=$S-b21-redeliv3 $AN/redeliver-v3-blender.sbatch; JC2[$S]=$LASTJ; fi
done
# --- D (Deep Blending now; Blender after C2): lambda x dictionary fan + DC q8
for S in $DB; do sb --time=04:00:00 --export=ALL,SCENE=$S,TAG=rd3i,XYZ=q15,IMAGES=images,DATA=$D/db/$S --job-name=$S-b21-kfan $AN/lambda-k-fan-v3.sbatch; done
# --- E1: controller v9b on Deep Blending (the headline controller version; batch18 ran v9d there)
for S in $DB; do sb --time=06:00:00 --export=ALL,SCENE=$S,INDOOR=1,IMAGES=images,XYZ=q15,TARGETS="5 10 20",DATA=$D/db/$S,SMOKE=1,SELFCHECK=0 --job-name=$S-b21-adapt9x $AN/adapt-v9b.sbatch; done
# --- B + D(Blender): after each scene's C2 (the budgets and the fan read the -best3 deliveries; existing 6.2e-4 / 6.2e-3 points are used as they are)
for S in $BL; do dep=""; [ -n "${JC2[$S]}" ] && dep="--dependency=afterany:${JC2[$S]}"
  sb --time=04:00:00 $dep --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,ORIG=${BORIG[$S]:-/scratch/zr2245/$S-blender4-best},LAMS_AUTO="0.00062 0.00185 0.0062 0.0185" --job-name=$S-b21-g0rb $AN/budget-sweep-blender-v1.sbatch
  sb --time=03:00:00 $dep --export=ALL,SCENE=$S,TAG=rd3b4,SUFFIX=-best3,XYZ=q14,IMAGES=train,DATA=$D/nerf_synthetic/$S --job-name=$S-b21-kfan $AN/lambda-k-fan-v3.sbatch
done
# --- E2: controller v9b on Mip-360 / T&T at 7.5 and 15 MB (adds two targets to the 5/10/20 table)
m () { sb --time=03:30:00 --export=ALL,SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS="7.5 15"$5,SMOKE=0,SELFCHECK=0 --job-name=$1-b21-adapt9x $AN/adapt-v9b.sbatch; }
m room 1 images_2 q14; m kitchen 1 images_2 q16; m counter 1 images_2 q15; m bonsai 1 images_2 q15
m garden 0 images_4 q15; m bicycle 0 images_4 q16; m stump 0 images_4 q16; m flowers 0 images_4 q15; m treehill 0 images_4 q16
m truck 0 images q18 ",DATA=$D/tandt/truck"; m train 0 images q18 ",DATA=$D/tandt/train"
# --- E3: controller v9b on NeRF-Synthetic at 1.5 / 3 / 6 MB (adds three targets to the 2.5/4 table; ficus/mic at 6 MB report the capacity limit)
for S in $BL; do sb --time=02:30:00 --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,TARGETS="1.5 3 6",SMOKE=0,SELFCHECK=0 --job-name=$S-b21-adapt9xb $AN/adapt-v9b-blender.sbatch; done
# --- F: generic codecs on the remaining scenes (room / garden already have rows)
sb --time=10:00:00 --export=ALL,SCENES="kitchen counter bonsai drjohnson playroom truck" --job-name=b21-genc-a $AN/generic-codecs-b21.sbatch
sb --time=10:00:00 --export=ALL,SCENES="bicycle stump flowers treehill train" --job-name=b21-genc-b $AN/generic-codecs-b21.sbatch
echo "submitted $k of 59 jobs (A 2, C 8, C2 8, D 10, E1 2, B 8, E2 11, E3 8, F 2); dependent jobs wait on their scene's C / C2"
squeue -u $USER -o "%.10i %.26j %.2t %.10M %.12E" | head -70
SHEOF_B21R2
bash $AN/submit-batch21-v1.sh 2>&1 | tee $AN/submit-batch21-v1.log
```

## §2 progress (any time)

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.26j %.2t %.10M %.12E" | sort -k2
echo "===== done / failed so far"; sacct -X -u $USER -S now-36hours -o JobID,JobName%26,State,Elapsed,ExitCode | grep -E "b21|JobID" | grep -vE "RUNNING|PENDING"
echo "===== FAILED lines"; grep -l "FAILED\|CHECK FAILED" $AN/*b21*.out 2>/dev/null | while read f; do echo "--- $f"; grep -h "FAILED" $f | head -3; done
```

## §3 collect (after the queue is empty; send me the whole output as a file)

```bash
AN=/scratch/zr2245/analysis; O=$AN/batch21-collect-$(date +%m%d-%H%M).txt
{
echo "===== sacct"; sacct -X -u $USER -S now-48hours -o JobID,JobName%26,State,Elapsed,ExitCode | grep -E "b21|JobID"
echo "===== g0-refresh Deep Blending"; cat $AN/g0-refresh-drjohnson.txt $AN/g0-refresh-playroom.txt 2>/dev/null
echo "===== g0-refresh Blender"; cat $AN/g0-refresh-blender-*.txt 2>/dev/null
echo "===== Blender new lambda points (rd-prune3b4, 1.85e-3 / 1.85e-2)"; grep -h "lam 0.00185\|lam 0.0185" $AN/rd-prune3b4-*.txt 2>/dev/null
echo "===== redeliver-v3-blender (new -best3 rows)"; grep -h "lam0.00185\|lam0.0185" $AN/redeliver-v3-blender.txt 2>/dev/null
echo "===== lambda-k-fan (DB + Blender rows)"; grep -hE "^(drjohnson|playroom|chair|drums|ficus|hotdog|lego|materials|mic|ship) " $AN/lambda-k-fan-v1.txt 2>/dev/null
echo "===== adapt-v9b (DB rows + 7.5 / 15 MB rows)"; grep -hE "^(drjohnson|playroom) |target (7.5|15) MB" $AN/adapt-v9b.txt 2>/dev/null
echo "===== adapt-v9b-blender (1.5 / 3 / 6 MB rows)"; grep -hE "target (1.5|3|6) MB" $AN/adapt-v9b-blender.txt 2>/dev/null
echo "===== generic codecs"; cat $AN/generic-codecs.txt 2>/dev/null; echo "--- o3dgc"; for S in kitchen counter bonsai bicycle stump flowers treehill truck train drjohnson playroom; do grep -ih "ROW" $AN/o3dgc-v1/$S-b21.log 2>/dev/null | head -6; done
echo "===== Blender G0 budgets used"; grep -h "lambda .*-> budget\|budgets for" $AN/g0rb-*b21*.out 2>/dev/null
echo "===== controller landing lines"; grep -h "Training is done" /scratch/zr2245/*-adapt9x-t7.5mb/train.log /scratch/zr2245/*-adapt9x-t15mb/train.log /scratch/zr2245/drjohnson-adapt9x-t*mb/train.log /scratch/zr2245/playroom-adapt9x-t*mb/train.log /scratch/zr2245/*-adapt9xb-t1.5mb/train.log /scratch/zr2245/*-adapt9xb-t3mb/train.log /scratch/zr2245/*-adapt9xb-t6mb/train.log 2>/dev/null | cut -c1-260
echo "===== FAILED lines"; grep -l "FAILED" $AN/*b21*.out 2>/dev/null | while read f; do echo "--- $f"; grep -h "FAILED" $f | head -4; tail -4 $f | cut -c1-240; done
} > $O 2>&1; wc -l $O; echo "send me $O"
```

## Notes

- Nothing is overwritten: new run directories throughout (`*-budgetb-*`, `*-rd3b4-lam0.00185-*`, `*-adapt9x-t7.5mb`, `*-adapt9xb-t1.5mb`, `drjohnson-adapt9x-*`, `kfan-v1/` inside existing λ dirs, `g0ctrl-*` inside the originals).
- Order of submission = rough order of execution: core gaps (A, C, D-DB, E1) first; B and the Blender fan wait for their scene's re-delivery; the redundancy runs (E2, E3) and generic codecs fill the GPUs in between.
- If the reservation runs out before the queue empties, the jobs keep running on the general partition (same account) at normal priority; restartable sbatch files mean a re-submit resumes.
