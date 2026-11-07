# rd-indoor-v1：室内四场景用 upstream `--indoor` 重跑 original + 4 个 λ 点（2026-11-01）

**为什么**：所有现有 run（original 和 λ）都是室外默认超参训的；upstream `update_indoor` 改 14 个超参。室内四场景对论文 PSNR −0.26/−0.57/−0.97/−0.90、LPIPS +0.07…+0.14（plan/figure-review-2026-11-01.md §0.1–0.2）。
**做什么**：room / kitchen / counter / bonsai 各 1 个 original job（upstream `train.py --indoor` + level-1.0 交付）+ 2 个 λ job（`train_rd_v3.py --indoor`，每个 job 跑 2 个 λ，共 4 λ = 6.2e-4 / 1.85e-3 / 6.2e-3 / 1.85e-2）。12 个 job，≈ 22 GPU-h；general 6 个（room、kitchen）、tandon_advanced 6 个（counter、bonsai）。
**不改任何旧文件**：新建 `originals-v2.sbatch`（= v1 + `--indoor`、新输出目录、XYZ 必须显式）、`rd-prune-v3c.sbatch`（= v3b + `--indoor`、新输出目录、去掉 XYZ=auto）。旧的室外参数结果留在原目录，不覆盖。
**输出**：`/scratch/zr2245/<scene>-indoor/`（original）、`/scratch/zr2245/<scene>-rd3i-lam<L>-mu1.0/`（λ）；表 `$AN/originals-indoor.txt`、`$AN/rd-prune3i-<scene>.txt`，行格式与旧表相同（可直接贴给我）。
λ job 不依赖 original job（train_rd 从头训；XYZ 显式传入），12 个一起提交。

## §1 写三个文件并提交（任意节点，整块粘贴）

```bash
AN=/scratch/zr2245/analysis
cat > $AN/originals-v2.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=origi
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/zr2245/analysis/origi-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/origi-%x-%j.err
# originals-v2.sbatch (2026-11-01, VERSION v2): = originals-v1.sbatch + INDOOR=1 passes --indoor to upstream train.py (update_indoor),
# output dir /scratch/zr2245/$SCENE-indoor (the outdoor-parameter original in $SCENE-vis-collapse is kept), XYZ must be explicit (qN), table originals-indoor.txt.
#   sbatch --export=ALL,SCENE=room,IMAGES=images_2,XYZ=q14,INDOOR=1 --job-name=room-origi /scratch/zr2245/analysis/originals-v2.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_2}; INDOOR=${INDOOR:-1}
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}; G=${G:-/scratch/zr2245/$SCENE-indoor}
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12
K=${K:-4096}; R=${R:-256}; DC=${DC:-q7}; XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; TAB=$AN/originals-indoor.txt
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
C=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
echo "scene $SCENE | DATA $DATA | IMAGES $IMAGES | G $G | indoor flag '$FLAG' | XYZ $XYZ | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $DATA/$IMAGES/. $DATA/sparse/0/cameras.bin $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $LAB/upstream/train.py $LAB/upstream/extract_normals.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do
  [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
ls $PKG/native/_native_qem_vis*.so >/dev/null 2>&1 || { echo "FAILED: native module missing in $PKG/native"; exit 1; }
grep -q 'add_argument("--indoor"' $LAB/upstream/train.py || { echo "FAILED: upstream train.py has no --indoor flag"; exit 1; }
mkdir -p $G
# --- stage -1: normals (already extracted for these scenes; re-run only if missing)
NORMALS=$DATA/${IMAGES/images/normals}
n_img=$(ls $DATA/$IMAGES | grep -ci '\.jpe\?g$'); n_nrm=$(ls $NORMALS 2>/dev/null | grep -ci '\.png$')
echo "images: $n_img in $DATA/$IMAGES | normals: $n_nrm in $NORMALS"
if [ "$n_nrm" -lt "$n_img" ]; then
  echo "=== normals -> $NORMALS ($(date)) ==="
  (cd $LAB/upstream && python extract_normals.py -s $DATA -i $IMAGES > $G.normals.log 2>&1) || { echo "FAILED: NORMALS (log $G.normals.log)"; grep -n -m 6 -i 'error\|No module' $G.normals.log; exit 1; }
  n_nrm=$(ls $NORMALS | grep -ci '\.png$'); [ "$n_nrm" -ge "$n_img" ] || { echo "FAILED: normals incomplete: $n_nrm / $n_img"; exit 1; }
fi
# --- stage 0: original WITH --indoor (the only difference from originals-v1)
if [ ! -f $C ]; then
  echo "=== train original $FLAG -> $G/original ($(date)) ==="
  (cd $LAB/upstream && python train.py -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 $FLAG > $G/train.log 2>&1) || { echo "FAILED: TRAIN"; tail -20 $G/train.log; exit 1; }
  [ -f $C ] || { echo "FAILED: TRAIN produced no checkpoint"; tail -20 $G/train.log; exit 1; }
else echo "exists: $C"; fi
TEST30=$(grep -h "\[ITER 30000\] Evaluating test" $G/train.log | tail -1); echo "$TEST30"
# --- level-1.0 protocol (footprint + dense anchor finetune + K4096 R256 delivery) = the 'dense codec' row
echo "=== level-1.0 protocol ($(date)) ==="
G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 || { echo "FAILED: LEVEL-1.0"; tail -12 $G/level.log; exit 1; }
NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery"; tail -12 $G/level.log; exit 1; }
python - <<PY | tee -a $TAB
import json, re
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']
m=re.search(r'PSNR ([0-9.]+) SSIM ([0-9.]+) LPIPS ([0-9.]+)', '''$TEST30''')
print('$SCENE ORIGINAL-INDOOR | vertices', b['vertices'], 'faces', b['faces'], '| original test', m.group(1)[:6] if m else 'n/a', m.group(2)[:6] if m else '', m.group(3)[:6] if m else '',
      '| level-1.0: finetune test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test', round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | $IMAGES | $FLAG')
PY
echo "originals-indoor $SCENE exit 0  $(date)"
SBEOF

cat > $AN/rd-prune-v3c.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=rd3c
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/zr2245/analysis/rd3i-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/rd3i-%x-%j.err
# rd-prune-v3c.sbatch (2026-11-01, VERSION v3c): = rd-prune-v3b.sbatch + INDOOR=1 passes --indoor to train_rd_v3.py (upstream update_indoor:
# feature_lr 0.004, add_percentage 1.27, iteration_mesh 12000, start_pruning 3000, prune_size 1300, depth priors off, ...), output dir
# /scratch/zr2245/$SCENE-rd3i-lam$L-mu$MU (outdoor-parameter runs in $SCENE-rd3-... are kept), table rd-prune3i-$SCENE.txt, smoke stage removed,
# XYZ must be explicit (qN).  Same lambda rule, same prune schedule (12000 14000 16000 18000: after the Delaunay rebuild at 11000), same codec.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_2}; LAMS=${LAMS:?set LAMS="0.00062 0.0185" via --export}; INDOOR=${INDOOR:-1}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}; TAB=$AN/rd-prune3i-$SCENE.txt; TR=$AN/rdgrowth/train_rd_v3.py
echo "scene $SCENE | IMAGES $IMAGES | lams $LAMS | indoor flag '$FLAG' | B/v $BPV B/face $BPF | prune at $PRUNE mu $MU max_frac $MAXF | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
grep -q "'--indoor'" $TR || { echo "FAILED: $TR has no --indoor flag"; exit 1; }
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-rd3i-lam$L-mu$MU; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== $SCENE lam $L mu $MU $FLAG: train+prune ($(date)) ==="
  if [ ! -f $CB ]; then
    python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF $FLAG > $G/train.log 2>&1 \
      || { echo "FAILED: TRAIN $SCENE lam $L"; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE lam $L"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  VB=$(nverts $CB); echo "lam $L mu $MU -> trained vertices $VB ($(date))"; grep -h "^\[RD prune" $G/train.log
  echo "=== $SCENE lam $L mu $MU: level-1.0 protocol ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 \
    || { echo "FAILED: LEVEL-1.0 $SCENE lam $L"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE lam $L"; tail -8 $G/level.log; exit 1; }
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; h=f['history']
ev=[json.loads(l) for l in open('$G/original/rd_log.jsonl')]; dl=[e for e in ev if e['event']=='delaunay']; pr=[e for e in ev if e['event']=='prune']
print('$SCENE lam $L RD-GROWTH+PRUNE-mu$MU-INDOOR | vertices', b['vertices'], 'faces', b['faces'], '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3),
      'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | grow: V@delaunay', dl[0]['vertices'] if dl else None,
      '| prune rounds', [(e['iteration'], e['vertices_before'], e['vertices_after'], 'pairs', e['n_pairs'], 'cap', e['max_pairs']) for e in pr], '| $FLAG')
PY
done
echo "===== $TAB ====="; cat $TAB
echo "rd-prune3i $SCENE exit 0  $(date)"
SBEOF

cat > $AN/submit-indoor-v1.sh <<'SHEOF'
#!/usr/bin/env bash
# submit-indoor-v1.sh (2026-11-01): 4 indoor scenes x (1 original job + 2 lambda jobs of 2 lambdas) = 12 jobs, all with --indoor.
# accounts: room + kitchen -> general (6 jobs); counter + bonsai -> tandon_advanced (6 jobs).  Lambda jobs do not wait for the original.
AN=/scratch/zr2245/analysis
for f in $AN/originals-v2.sbatch $AN/rd-prune-v3c.sbatch $AN/rdgrowth/train_rd_v3.py; do [ -f $f ] || { echo "FAILED: missing $f"; exit 1; }; done
xyz_of () { case $1 in room) echo q14;; counter|bonsai) echo q15;; kitchen) echo q16;; *) echo "";; esac; }
acc_of () { case $1 in room|kitchen) echo torch_pr_38_general;; *) echo torch_pr_38_tandon_advanced;; esac; }
n=0
for S in room kitchen counter bonsai; do
  Q=$(xyz_of $S); [ -n "$Q" ] || { echo "FAILED: no XYZ for $S"; continue; }
  A=$(acc_of $S); IM=images_2
  [ -f /scratch/zr2245/$S-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] && echo "note: $S indoor original exists, job will skip its training"
  j=$(sbatch --parsable --account=$A --time=03:30:00 --export=ALL,SCENE=$S,IMAGES=$IM,XYZ=$Q,INDOOR=1 --job-name=$S-origi $AN/originals-v2.sbatch) \
    && { echo "submitted $S-origi $j  XYZ $Q  account $A"; n=$((n+1)); } || echo "FAILED: sbatch $S original"
  for LAMS in "0.00062 0.0185" "0.00185 0.0062"; do
    for L in $LAMS; do [ -f /scratch/zr2245/$S-rd3i-lam$L-mu1.0/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] && echo "note: $S lam $L indoor checkpoint exists, sbatch will skip its training"; done
    T=$(echo $LAMS | tr ' ' '_' | sed 's/0\.0*//g')
    j=$(sbatch --parsable --account=$A --time=03:30:00 --export=ALL,SCENE=$S,IMAGES=$IM,XYZ=$Q,INDOOR=1,LAMS="$LAMS",BPV=10.0,BPF=2.0,MU=1.0,MAXF=0.15 --job-name=$S-i$T $AN/rd-prune-v3c.sbatch) \
      && { echo "submitted $S-i$T $j  lams $LAMS  XYZ $Q  account $A"; n=$((n+1)); } || echo "FAILED: sbatch $S lams $LAMS"
  done
done
echo "submitted $n jobs (expected 12)"
squeue -u $USER -o "%.10i %.16j %.28a %.2t %.10M %.6D %R"
SHEOF
bash $AN/submit-indoor-v1.sh
```

## §2 看进度 / 结果

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.16j %.28a %.2t %.10M %R"; echo; cat $AN/originals-indoor.txt 2>/dev/null; echo; grep -h "INDOOR" $AN/rd-prune3i-*.txt 2>/dev/null | cut -c1-230; echo; grep -l "FAILED" $AN/origi-*.out $AN/rd3i-*.out 2>/dev/null
```

若某个账户排队明显慢：`scontrol update JobId=<id> Account=torch_pr_38_general`（或 tandon_advanced）。全部完成后把 `originals-indoor.txt` 的 4 行和 `rd-prune3i-*.txt` 的 16 行贴给我，我换掉 11 场景表里的室内行、重画 hull / 主图 / 三指标图。

## §3 顺手补 5 个旧 original 的 SSIM / LPIPS（0 GPU，任意节点）

旧的 bonsai / counter / kitchen / bicycle / stump original 没记 SSIM/LPIPS，但 train.log 的 `[ITER 30000] Evaluating test` 行里有（room 的 original 没有 train.log，跳过）：

```bash
for S in bonsai counter kitchen bicycle stump garden flowers treehill truck train; do echo -n "$S original | "; grep -h "\[ITER 30000\] Evaluating test" /scratch/zr2245/$S-vis-collapse/train.log 2>/dev/null | tail -1 | grep . || echo "no train.log"; done
```

把输出贴给我即可（室外 5 个场景这一行用来和 batch A 的记录交叉核对）。
