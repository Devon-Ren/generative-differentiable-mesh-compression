#!/usr/bin/env bash
# submit-batch43-v1.sh (2026-11-15), three independent parts:
#   db    : zstd / Draco / MPEG TFAN on Deep Blending (drjohnson, playroom): the FP32 model and the cap-trained meshes at 600,000 / 1,081,000 /
#           3,000,000, SH precision 8 / 6 / 4 bits.  No training.  2 GPU jobs, under 1 hour each.  Rows -> $AN/generic-rd.txt
#   rank  : the selection-rule ablation of batch35 (split by importance / collapse by importance) on three more scenes, bicycle, kitchen and
#           truck, target 10 MB.  6 GPU jobs of about 1 hour.  Rows -> $AN/ablation-b24.txt (abl24-holdis / abl24-holdic)
#   white : NeRF-Synthetic on a WHITE background, evaluation only: the FP32 model and the hold files at 4 / 2.5 / 1.5 MB of the 8 objects, as
#           trained (on black).  1 GPU job, about 45 minutes.  Rows -> $AN/white-eval.txt.  Exploratory: the meshes were not trained for it.
# Usage: bash submit-batch43-v1.sh db | rank | white | all        Collect: bash submit-batch43-v1.sh collect
AN=/scratch/zr2245/analysis; ROOT=/scratch/zr2245; W=${1:?db | rank | white | all | collect}
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; return 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; return 1; }; }
if [ $W = collect ]; then
  echo "== db"; grep -h -E "^(drjohnson|playroom) GENERIC-RD" $AN/generic-rd.txt
  echo "== rank"; grep -h -E "^(bicycle|kitchen|truck) abl24-(hold|holdis|holdic) \| target 10 MB" $AN/ablation-b24.txt | cut -c1-330
  echo "== white"; cat $AN/white-eval.txt 2>/dev/null
  echo "== unfinished"; for f in $(ls -t $AN/genrd-*-b43-*.out $AN/abl24-*-b43-*.out $AN/white-b43-*.out 2>/dev/null); do tail -1 $f | grep -q "exit 0" || echo "$f : $(tail -1 $f | cut -c1-140)"; done; exit 0
fi
if [ $W = db ] || [ $W = all ]; then
  chk $AN/generic_rd_v1.py 46 17c568d6da5cfaf473cfe7067bf38b60 && chk $AN/generic-rd-v4.sbatch 73 a48f1e5f1aed893f3d413030de658301 && bash -n $AN/generic-rd-v4.sbatch && for S in drjohnson playroom; do miss=0
    for p in $ROOT/$S-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt $ROOT/$S-budget-600000 $ROOT/$S-budget-1081000 $ROOT/$S-budget-3000000; do [ -e $p ] || { echo "MISSING ($S): $p"; miss=1; }; done
    [ $miss = 0 ] || continue
    j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=8 --time=01:15:00 --export=ALL,SCENE=$S,CAPS="full 600000 1081000 3000000" --job-name=$S-gendb-b43 $AN/generic-rd-v4.sbatch) && echo "submitted $j  $S-gendb-b43" || echo "FAILED: sbatch $S"; done
fi
if [ $W = rank ] || [ $W = all ]; then
  cfg () { case $1 in bicycle) echo "INDOOR=0,IMAGES=images_4,XYZ=q16";; kitchen) echo "INDOOR=1,IMAGES=images_2,XYZ=q16";; truck) echo "INDOOR=0,IMAGES=images,XYZ=q18,DATA=$ROOT/datasets/tandt/truck";; esac; }
  chk $AN/rdgrowth/train_rd_adapt_v9b_abl8.py 1092 0f6ca00cec9da3f1cf3da0f6d9f59bc5 && chk $AN/ablation-b24-r9.sbatch 220 86899591acad64fe4783baab438a0093 && for V in holdis holdic; do for S in bicycle kitchen truck; do
    [ -e $ROOT/$S-abl24-$V-t10mb ] && { echo "exists, skipped: $S $V"; continue; }
    j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 --time=02:30:00 --export=ALL,SCENE=$S,$(cfg $S),SMOKE=0,SELFCHECK=0,ABL=$V,TARGETS="10" --job-name=$S-$V-b43 $AN/ablation-b24-r9.sbatch) && echo "submitted $j  $S-$V-b43" || echo "FAILED: sbatch $S $V"; done; done
fi
if [ $W = white ] || [ $W = all ]; then
  for f in $AN/generic-v1/eval_ckpt_v1.py $AN/render/decode_asset_v2.py $AN/vis_collapse-v12/stage2_run.sh; do [ -f $f ] || { echo "MISSING: $f"; exit 0; }; done
  cat > $AN/white-eval-b43.sbatch <<'SB'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=white-b43
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/zr2245/analysis/white-b43-%j.out
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
AN=/scratch/zr2245/analysis; ROOT=/scratch/zr2245; D=$ROOT/datasets/nerf_synthetic; LAB=$ROOT/meshsplatting-original-lossless-v1; PKG=$AN/vis_collapse-v12; O=$AN/white-eval; TAB=$AN/white-eval.txt; mkdir -p $O
# the evaluator with a white background: ground truth composited on white, renders on white.  Everything else is eval_ckpt_v1.py.
sed -e 's/white_background=False/white_background=True/' -e 's/bg = torch.zeros(3, device=dev)/bg = torch.ones(3, device=dev)/' $AN/generic-v1/eval_ckpt_v1.py > $O/eval_ckpt_white.py
grep -q "white_background=True" $O/eval_ckpt_white.py && grep -q "torch.ones(3, device=dev)" $O/eval_ckpt_white.py || { echo "FAILED: the evaluator could not be patched"; exit 1; }
for S in lego chair drums ficus hotdog materials mic ship; do M=$ROOT/$S-blender4-best/original
  ev () { python $O/eval_ckpt_white.py --lab $LAB --model $M --dataset $D/$S --images train --ckpt $2 --out $O/$S-$1.json --label "$S $1 white" 2>&1 | grep "^EVAL" | sed "s/^EVAL/WHITE/" | tee -a $TAB || echo "FAILED: eval $S $1"; }
  ev fp32 $M/point_cloud/iteration_30000/point_cloud_state_dict.pt
  for T in 4 2.5 1.5; do DL=$ROOT/$S-abl24b-hold-t${T}mb-best/axisB/v1.0/deliver; [ -d $DL ] || { echo "note: no $DL"; continue; }
    bash $PKG/stage2_run.sh $AN/render/decode_asset_v2.py --deliver-dir $DL --out $O/$S-t$T.pt > $O/$S-t$T.dec.log 2>&1; [ -f $O/$S-t$T.pt ] || { echo "FAILED: decode $S $T"; tail -3 $O/$S-t$T.dec.log; continue; }
    ev hold-${T}MB $O/$S-t$T.pt; rm -f $O/$S-t$T.pt; done
done
echo "white-b43 exit 0"
SB
  j=$(sbatch --parsable $AN/white-eval-b43.sbatch) && echo "submitted $j  white-b43" || echo "FAILED: sbatch white"
fi
squeue -u $USER -h -o "%j %T" | grep -E "b43" | sort | uniq -c
