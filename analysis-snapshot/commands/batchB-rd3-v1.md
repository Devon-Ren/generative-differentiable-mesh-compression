# batchB-rd3-v1（2026-10-20）：10 场景 × 两档 λ 的 v3（生长 + phase-B 剪枝）headline，10 个 job × ≈2 h，general

Room 的两行已有（`rd-prune3-room.txt`），这里跑其余 10 个场景：Mip-360 的 garden / bonsai / counter / kitchen / bicycle / stump（original 已有，现在就能提交）+ flowers / treehill / truck / train（**批次 A 跑完后再提交**，XYZ 精度用规则从 original checkpoint 自动算）。每个 job 两档 λ = 6.2e-3 / 1.85e-3（v2 mpx 单位，即七场景表里的 6e-3 / 2e-3），每档 train+prune ≈45–50 min + level-1.0 交付 ≈10 min；smoke 关掉（Room 已验证）。输出和 Room 同格式：`/scratch/zr2245/analysis/rd-prune3-<scene>.txt`。

`rd-prune-v3b.sbatch` = `rd-prune-v3.sbatch`（rd-prune-v3.md 已安装的 `train_rd_v3.py` / `rd_prune.py` 不变）+ 三处改动：`DATA` 可覆盖（T&T）、`SMOKE=0` 跳过 smoke、`XYZ=auto` 按精度规则算。已有 6 场景的 XYZ 用当时实测选定的值：garden q15、bonsai q15、counter q15、kitchen q16、bicycle q16、stump q16。

## 1. 写 sbatch + 提交 6 个已有-original 的场景（现在就能提交）

```bash
AN=/scratch/zr2245/analysis
for f in $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/rd_prune.py $AN/rdgrowth/rd_growth_v2.py; do [ -f $f ] || echo "MISSING $f (paste rd-prune-v3.md first)"; done
cat > $AN/rd-prune-v3b.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=rd3b
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/zr2245/analysis/rd3-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/rd3-%x-%j.err
# rd-prune-v3b.sbatch (2026-10-20, batch B): = rd-prune-v3.sbatch + DATA override (T&T), SMOKE=0 to skip the smoke stage, XYZ=auto (precision rule on the scene's ORIGINAL checkpoint).
# rd-prune-v3.sbatch (2026-10-19): for SCENE and each L in LAMS: train with rate-controlled growth AND phase-B lambda-prune
# (rdgrowth/train_rd_v3.py: --lam L --dd_mode error --dd_norm mpx --bytes_per_vertex BPV --prune_iters PRUNE --prune_mu MU
# --prune_max_frac MAXF --bytes_per_face BPF), then the level-1.0 protocol (run_garden.sh FRACS=1.0, K4096 R256 XYZ/DC stage2-v2).
# -> /scratch/zr2245/$SCENE-rd3-lam$L-mu$MU/, one row per lambda in /scratch/zr2245/analysis/rd-prune3-$SCENE.txt.  Restartable.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_2}; LAMS=${LAMS:?set LAMS="0.0062 0.00185" via --export}
XYZ=${XYZ:-q14}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}; TAB=$AN/rd-prune3-$SCENE.txt; TR=$AN/rdgrowth/train_rd_v3.py
SMOKE=${SMOKE:-1}; ORIG=${ORIG:-/scratch/zr2245/$SCENE-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt}
if [ "$XYZ" = "auto" ]; then
  [ -f $ORIG ] || { echo "XYZ=auto needs the original checkpoint: $ORIG (run batch A first or pass XYZ=qN)"; exit 1; }
  XYZ=$(python - <<PYRULE
import torch, numpy as np, math
w=torch.load('$ORIG',map_location='cpu',weights_only=False); x=w['triangles_points'].numpy().astype(np.float64); f=np.asarray(w['_triangle_indices']).astype(np.int64)
extent=float((x.max(0)-x.min(0)).max()); rng=np.random.default_rng(0); sel=rng.choice(len(f),min(len(f),1000000),replace=False)
e=np.concatenate([f[sel][:,[0,1]],f[sel][:,[1,2]],f[sel][:,[2,0]]]); med=float(np.median(np.linalg.norm(x[e[:,0]]-x[e[:,1]],axis=1)))
print(f'q{int(min(18,max(13,math.ceil(math.log2(extent/(0.1*med))))))}')
PYRULE
) || { echo "PRECISION RULE FAILED"; exit 1; }
  echo "XYZ=auto -> $XYZ (rule on $ORIG)"
fi
echo "scene $SCENE | IMAGES $IMAGES | lams $LAMS | B/v $BPV B/face $BPF | prune at $PRUNE mu $MU max_frac $MAXF | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "MISSING: $f"; exit 1; }; done
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
# --- smoke: 1300 iterations, one forced prune round at 1250 (mu 1e6 -> every edge is a candidate, the per-round cap binds),
#     then 50 more iterations of rendering/training on the collapsed mesh and the final save ---
if [ "$SMOKE" = "1" ]; then
SM=$AN/rd3-smoke-$SCENE-$SLURM_JOB_ID; rm -rf $SM; mkdir -p $SM
echo "=== smoke ($(date)) ==="
python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $SM/original --iterations 1300 --test_iterations 1300 --lam 0.0062 --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
  --prune_iters 1250 --allow_any_prune_iter --prune_mu 1e6 --prune_max_frac 0.1 > $SM/smoke.log 2>&1 \
  || { echo "SMOKE FAILED"; tail -40 $SM/smoke.log; exit 1; }
python - <<PY || { echo "SMOKE CHECK FAILED"; grep -h "RD \|Traceback\|Error" $SM/smoke.log | tail -20; exit 1; }
import json
ev=[json.loads(l) for l in open('$SM/original/rd_log.jsonl')]; p=[e for e in ev if e['event']=='prune']; fin=[e for e in ev if e['event']=='final']
assert p, 'no prune event in smoke'; e=p[0]
print('smoke prune:', {k:e[k] for k in ('iteration','views','vertices_before','vertices_after','pairs','vertices_isolated_removed','faces_before','faces_after','faces_degenerate','faces_duplicate','n_cand','n_pairs','max_pairs','n_flip_rejected','passes','dR_mean_sel')})
assert e['n_pairs'] > 0 and e['vertices_after'] < e['vertices_before'], 'collapse did nothing'
assert e['faces_after'] == e['faces_before'] - e['faces_degenerate'] - e['faces_duplicate'], 'face bookkeeping'
assert fin, 'training did not finish after the prune (render on collapsed mesh failed?)'
print('smoke final:', fin[0])
PY
grep -h "^\[RD" $SM/smoke.log | tail -3
else echo "smoke skipped (SMOKE=0)"; fi
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-rd3-lam$L-mu$MU; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== $SCENE lam $L mu $MU: train+prune ($(date)) ==="
  if [ ! -f $CB ]; then
    python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF > $G/train.log 2>&1 \
      || { echo "TRAIN FAILED $SCENE lam $L"; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "no checkpoint for $SCENE lam $L"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  VB=$(nverts $CB); echo "lam $L mu $MU -> trained vertices $VB ($(date))"; grep -h "^\[RD prune" $G/train.log
  echo "=== $SCENE lam $L mu $MU: level-1.0 protocol ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 \
    || { echo "LEVEL-1.0 FAILED $SCENE lam $L"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "no delivery for $SCENE lam $L"; tail -8 $G/level.log; exit 1; }
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; h=f['history']
ev=[json.loads(l) for l in open('$G/original/rd_log.jsonl')]; dn=[e for e in ev if e['event']=='densify']; dl=[e for e in ev if e['event']=='delaunay']; pr=[e for e in ev if e['event']=='prune']
print('$SCENE lam $L RD-GROWTH+PRUNE-mu$MU | vertices', b['vertices'], 'faces', b['faces'], '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3),
      'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | grow: V@delaunay', dl[0]['vertices'] if dl else None,
      '| prune rounds', [(e['iteration'], e['vertices_before'], e['vertices_after'], 'cand%.3f' % e['cand_frac'], 'pairs', e['n_pairs'], 'cap', e['max_pairs'], 'flip', e['n_flip_rejected'],
                          'q50/thr %.2f' % (e['score_q50'] / e['thr'] if e['thr'] else float('nan')), 'sel', e['faces_removed_share_sel'], 'all', e['faces_removed_share_all']) for e in pr])
PY
done
echo "===== $TAB ====="; cat $TAB
echo "rd-prune3 $SCENE exit 0  $(date)"
SBEOF
chmod +x $AN/rd-prune-v3b.sbatch
LAMS="0.0062 0.00185"
sub () { j=$(sbatch --parsable --export=ALL,SCENE=$1,IMAGES=$2,XYZ=$3,LAMS="$LAMS",BPV=10.0,BPF=2.0,MU=1.0,MAXF=0.15,SMOKE=0${4:+,DATA=$4} --job-name=$1-rd3 $AN/rd-prune-v3b.sbatch) && echo "$1-rd3 $j" || echo "SUBMIT FAILED $1"; }
sub garden  images_4 q15
sub bonsai  images_2 q15
sub counter images_2 q15
sub kitchen images_2 q16
sub bicycle images_4 q16
sub stump   images_4 q16
squeue -u $USER -o "%.10i %.14j %.24a %.2t %.10M %.6D %R"
```

## 2. 批次 A 出来后，提交 4 个新场景（XYZ=auto）

```bash
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets; LAMS="0.0062 0.00185"
sub () { j=$(sbatch --parsable --export=ALL,SCENE=$1,IMAGES=$2,XYZ=auto,LAMS="$LAMS",BPV=10.0,BPF=2.0,MU=1.0,MAXF=0.15,SMOKE=0${3:+,DATA=$3} --job-name=$1-rd3 $AN/rd-prune-v3b.sbatch) && echo "$1-rd3 $j" || echo "SUBMIT FAILED $1"; }
for S in flowers treehill truck train; do [ -f /scratch/zr2245/$S-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] && echo "original ok: $S" || echo "NOT YET: $S original"; done
sub flowers  images_4
sub treehill images_4
sub truck    images   $D/tandt/truck
sub train    images   $D/tandt/train
squeue -u $USER -o "%.10i %.14j %.24a %.2t %.10M %.6D %R"
```

跑完给我 `cat /scratch/zr2245/analysis/rd-prune3-*.txt`。

**算力**：10 job × ≈2 h ≈ 20 GPU-h；加批次 A 的 10 h，共约 30 GPU-h。按 FairShare 分几天提交都行，每个 job 独立。

## §2-v2 (2026-10-24, use this instead of §2): explicit XYZ from xyz_rule_v1.py

`XYZ=auto` in rd-prune-v3b.sbatch fails on original checkpoints (`.numpy()` on requires_grad tensors, see
[batchA-resume-v1.md](batchA-resume-v1.md)). Needs `$AN/xyz_rule_v1.py`, which the batch A resume paste writes.
Submit after `originals.txt` has the four new rows.

```bash
AN=/scratch/zr2245/analysis
cat > $AN/submit-rd3-new4-v1.sh <<'SHEOF'
#!/usr/bin/env bash
# submit-rd3-new4-v1.sh (2026-10-24): batch B for flowers/treehill/truck/train with explicit XYZ from xyz_rule_v1.py.
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets; LAMS="0.0062 0.00185"
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
for f in $AN/xyz_rule_v1.py $AN/rd-prune-v3b.sbatch; do [ -f $f ] || { echo "FAILED: missing $f"; exit 1; }; done
for S in flowers treehill truck train; do
  C=/scratch/zr2245/$S-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
  [ -f $C ] || { echo "NOT YET: $S original"; continue; }
  Q=$(python $AN/xyz_rule_v1.py $C) || { echo "FAILED: xyz rule on $S"; continue; }
  [ "$Q" = "q17" ] && Q=q18
  case $S in truck|train) IM=images; DX=",DATA=$D/tandt/$S" ;; *) IM=images_4; DX="" ;; esac
  j=$(sbatch --parsable --export=ALL,SCENE=$S,IMAGES=$IM,XYZ=$Q,LAMS="$LAMS",BPV=10.0,BPF=2.0,MU=1.0,MAXF=0.15,SMOKE=0$DX --job-name=$S-rd3 $AN/rd-prune-v3b.sbatch) \
    && echo "submitted $S-rd3 $j  XYZ $Q" || echo "FAILED: sbatch $S"
done
squeue -u $USER -o "%.10i %.14j %.24a %.2t %.10M %.6D %R"
SHEOF
bash $AN/submit-rd3-new4-v1.sh
```
