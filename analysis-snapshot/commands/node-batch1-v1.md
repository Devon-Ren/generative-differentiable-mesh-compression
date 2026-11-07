# node-batch1-v1：gl030 预留上的第一批（2026-11-05 16:00）— 8 场景 λ×K fan + B_a 价格敏感性

节点：reservation `yw523-l40s`（gl030，4 × L40S，11/05 09:35 → 11/12 09:35，只给 zr2245）。所有 sbatch 加 `--reservation=yw523-l40s`，account 照旧 general。节点上别人的抢占 job 会在你的 job 排队后 ≤30 min 被挤掉。

| 块 | 内容 | job 数 | GPU-h | 产物 |
|---|---|---|---|---|
| §1 fan v2 | counter / bonsai（rd3i, images_2）、bicycle / stump / flowers / treehill（rd3, images_4）、truck / train（rd3, images, tandt 路径）× 4 λ × {DC q8, K4096R1, K1024R256, K4096R1024, K16384R1024} | 8 | ≈12 | `$AN/lambda-k-fan-v1.txt` 追加 48 行 → Table 6 变 11 场景 |
| §2 ba-sens v1 | kitchen（--indoor）/ garden × BPV {8, 13}（B_a 4 / 9，默认 10 / 6 已有）× λ {6.2e-3, 6.2e-4}，训练 + level-1.0 交付 | 4 | ≈8 | `$AN/ba-sens-v1.txt` 8 行 → Yao "not end-to-end" 的敏感性数据 |

fan v2 = v1 + DATA 可覆盖（T&T）+ 交付资产的 XYZ 必须等于 xyz_of（否则在任何 GPU 工作前 FAIL）。ba-sens v1 = rd-prune-v3c + BPV 由 --export 给、输出目录带 `-bpv$BPV`（不碰默认目录）、表单独。两块互不依赖，可以先后粘贴。

## §1 fan v2：写文件 + 自检 + 提交 8 个 job（整块粘贴，任意节点）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN
cat > $AN/lambda-k-fan-v2.sbatch <<'SBEOF_LKF2'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=lkfan2
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/zr2245/analysis/lkfan-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/lkfan-%x-%j.err
# lambda-k-fan-v2.sbatch (2026-11-05, VERSION v2): = v1 + DATA override via --export (T&T: DATA=/scratch/zr2245/datasets/tandt/$SCENE, IMAGES=images)
#   + the delivered asset's XYZ (config.json) must equal xyz_of SCENE, else FAIL before any GPU work.  Same table file as v1 ($AN/lambda-k-fan-v1.txt).
# lambda-k-fan-v1.sbatch (2026-11-05, VERSION v1): Yao Q1/Q2 -- the joint lambda x dictionary-size RD fan, WITHOUT training.
# For one SCENE and every lambda asset $G = /scratch/zr2245/$SCENE-$TAG-lam$L-mu1.0 (level-1.0 protocol already run: axisB/v1.0/{finetune/best.pt, deliver/}):
#   (a) DC q8 instead of q7 on the delivered K4096 R256 asset      codec_variants const/ctx --dc q8   (seconds + test render)
#   (d) dictionary K x R in KRS instead of K4096 R256                deliver_coarse.sh on the same finetuned mesh (k-means 4 rounds + 400-step fit, 3-15 min each)
# Rows -> $AN/lambda-k-fan-v1.txt: scene | lam | lever | MB | test PSNR SSIM LPIPS | dPSNR / dMB vs that lambda's baseline (const/ctx q7 K4096R256).
# Restartable: existing sweep rows / summary.json are skipped.  48G: the dictionary fit holds a 2-3 M-vertex model.
set -o pipefail
export TORCH_HOME=/scratch/zr2245/cache/torch
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; TAG=${TAG:?set TAG=rd3i (indoor rerun) or rd3 (outdoor) via --export}; IMAGES=${IMAGES:-images_2}
LAMS=${LAMS:-"0.00062 0.00185 0.0062 0.0185"}; KRS=${KRS:-"4096 1|1024 256|4096 1024|16384 1024"}
AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; TAB=$AN/lambda-k-fan-v1.txt
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}
xyz_of () { case $1 in room) echo q14;; garden|bonsai|counter|flowers) echo q15;; kitchen|bicycle|stump|treehill) echo q16;; truck|train) echo q18;; *) echo "FAILED: no XYZ recorded for $1" >&2; return 1;; esac; }
XYZ=$(xyz_of $SCENE) || exit 1
echo "lambda-k-fan-v2 | scene $SCENE tag $TAG | DATA $DATA/$IMAGES | lams $LAMS | K x R $KRS | XYZ $XYZ | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $PKG/codec_variants.py $PKG/deliver_coarse.sh $PKG/stage2_run.sh /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION $DATA/$IMAGES/.; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
nf=0
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-$TAG-lam$L-mu1.0; d=$G/axisB/v1.0/deliver; MESH=$G/axisB/v1.0/finetune/best.pt; C=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
  miss=0; for f in $d/shared-final.pt $d/config.json $d/summary.json $MESH $C; do [ -e $f ] || { echo "FAILED: MISSING $f"; miss=1; }; done
  [ $miss = 0 ] || { nf=$((nf+1)); continue; }
  python -c "import json; c=json.load(open('$d/config.json')); k=c.get('k',4096); r=c.get('r',256); assert (k,r)==(4096,256), ('baseline delivery is not K4096 R256', k, r); assert c.get('xyz')=='$XYZ', ('delivered xyz', c.get('xyz'), 'differs from xyz_of', '$XYZ')" || { echo "FAILED: $SCENE lam $L baseline is not K4096 R256 at XYZ $XYZ (see assert above)"; nf=$((nf+1)); continue; }
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
  python - "$SCENE" "$L" "$d" "$XYZ" "$OUT" "$KRS" <<'PY' | tee -a $TAB
import json, sys, pathlib
S, L, d, xyz, out, krs = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]), sys.argv[4], pathlib.Path(sys.argv[5]), sys.argv[6]
s = json.load(open(d / 'codec-variants' / 'sweep.json')); rows = {}
for r in s['rows']:
    if r['xyz'] == xyz and r.get('test'): rows[(r['weight'] + '/' + r['index_codec'], r['dc'])] = r
base = rows.get(('const/ctx', 'q7'))
if base is None: print(f"{S} | lam {L} | baseline MISSING"); sys.exit(0)
bp = base['test']['psnr']; bm = base['file_MB']
def line(name, mb, t): print(f"{S} | lam {L} | {name:22s} | {mb:7.3f} MB | test {t['psnr']:.3f} {t['ssim']:.4f} {t['lpips']:.4f} | dPSNR {t['psnr'] - bp:+.3f} | dMB {mb - bm:+.3f}")
line('K4096 R256 q7 (base)', bm, base['test'])
r = rows.get(('const/ctx', 'q8')); line('K4096 R256 q8', r['file_MB'], r['test']) if r else print(f"{S} | lam {L} | K4096 R256 q8: MISSING")
for kr in krs.split('|'):
    K, R = kr.split(); p = out / f'k{K}_r{R}' / 'summary.json'
    if p.exists(): j = json.load(open(p)); line(f'K{K} R{R} q7', j['file_MB'], j['delivery_test'])
    else: print(f"{S} | lam {L} | K{K} R{R} q7: MISSING")
PY
done
echo "===== $TAB ($SCENE rows) ====="; grep "^$SCENE " $TAB
[ $nf = 0 ] && echo "lambda-k-fan-v2 $SCENE exit 0  $(date)" || { echo "lambda-k-fan-v2 $SCENE FINISHED WITH $nf FAILED items  $(date)"; exit 1; }
SBEOF_LKF2
cat > $AN/submit-fan-v2.sh <<'SUB'
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets; RES=yw523-l40s
n=$(wc -l < $AN/lambda-k-fan-v2.sbatch); [ "$n" = 74 ] || { echo "CHECK FAILED: lambda-k-fan-v2.sbatch has $n lines, expected 74 (paste truncated?)"; exit 1; }
grep -q "<<'PY' | tee -a" $AN/lambda-k-fan-v2.sbatch || { echo "CHECK FAILED: heredoc/tee form"; exit 1; }
grep -q 'DATA=${DATA:-' $AN/lambda-k-fan-v2.sbatch || { echo "CHECK FAILED: no DATA override (v1 pasted instead of v2?)"; exit 1; }
scontrol show res $RES > /dev/null 2>&1 || { echo "CHECK FAILED: reservation $RES not visible"; exit 1; }
miss=0
for S in counter bonsai bicycle stump flowers treehill truck train; do
  case $S in counter|bonsai) TAG=rd3i; IM=images_2; DX="" ;; truck|train) TAG=rd3; IM=images; DX=",DATA=$D/tandt/$S" ;; *) TAG=rd3; IM=images_4; DX="" ;; esac
  for L in 0.00062 0.00185 0.0062 0.0185; do f=/scratch/zr2245/$S-$TAG-lam$L-mu1.0/axisB/v1.0/deliver/summary.json; [ -f $f ] || { echo "MISSING asset: $f"; miss=$((miss+1)); }; done
  case $S in truck|train) DD=$D/tandt/$S/$IM ;; *) DD=$D/m360/$S/$IM ;; esac; [ -d $DD ] || { echo "MISSING dataset: $DD"; miss=$((miss+1)); }
done
[ $miss = 0 ] || { echo "CHECK FAILED: $miss missing items above -- nothing submitted"; exit 1; }
echo "CHECK OK"; k=0
for S in counter bonsai bicycle stump flowers treehill truck train; do
  case $S in counter|bonsai) TAG=rd3i; IM=images_2; DX="" ;; truck|train) TAG=rd3; IM=images; DX=",DATA=$D/tandt/$S" ;; *) TAG=rd3; IM=images_4; DX="" ;; esac
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --export=ALL,SCENE=$S,TAG=$TAG,IMAGES=$IM$DX --job-name=$S-lkfan2 $AN/lambda-k-fan-v2.sbatch) \
    && { echo "submitted $S-lkfan2 $j ($TAG $IM$DX)"; k=$((k+1)); } || echo "FAILED: sbatch $S"
done
echo "submitted $k of 8"
SUB
bash $AN/submit-fan-v2.sh
squeue -u $USER -o "%.10i %.14j %.3t %.10M %.6D %R" | grep -E "lkfan2|basens|JOBID"
```

## §2 ba-sens v1：写文件 + 自检 + 提交 4 个 job（整块粘贴，任意节点）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN
cat > $AN/ba-sens-v1.sbatch <<'SBEOF_BAS1'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=basens
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/zr2245/analysis/basens-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/basens-%x-%j.err
# ba-sens-v1.sbatch (2026-11-05, VERSION v1): price-constant sensitivity = rd-prune-v3c.sbatch with BPV from --export (B_a = BPV - 2 B_f: BPV 8 / 13 = B_a 4 / 9 vs the default 10 / 6),
#   INDOOR explicit (0 outdoor params, 1 --indoor), output dir /scratch/zr2245/$SCENE-rd3<i>-lam$L-mu$MU-bpv$BPV (never the default dirs), table $AN/ba-sens-v1.txt.
#   Reading: a price error c is identical to lambda -> c*lambda, so the bpv8 / bpv13 points should fall ON the scene's lambda hull, not off it.
# rd-prune-v3c.sbatch (2026-11-01, VERSION v3c): = rd-prune-v3b.sbatch + INDOOR=1 passes --indoor to train_rd_v3.py (upstream update_indoor:
# feature_lr 0.004, add_percentage 1.27, iteration_mesh 12000, start_pruning 3000, prune_size 1300, depth priors off, ...), output dir
# /scratch/zr2245/$SCENE-rd3i-lam$L-mu$MU (outdoor-parameter runs in $SCENE-rd3-... are kept), table rd-prune3i-$SCENE.txt, smoke stage removed,
# XYZ must be explicit (qN).  Same lambda rule, same prune schedule (12000 14000 16000 18000: after the Delaunay rebuild at 11000), same codec.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_2}; LAMS=${LAMS:?set LAMS="0.00062 0.0185" via --export}; INDOOR=${INDOOR:?set INDOOR=0 (outdoor params) or 1 (--indoor) via --export}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}; TAB=$AN/ba-sens-v1.txt; SUF=""; [ "$INDOOR" = "1" ] && SUF=i; TR=$AN/rdgrowth/train_rd_v3.py
echo "scene $SCENE | IMAGES $IMAGES | lams $LAMS | indoor flag '$FLAG' | B/v $BPV B/face $BPF | prune at $PRUNE mu $MU max_frac $MAXF | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
grep -q "'--indoor'" $TR || { echo "FAILED: $TR has no --indoor flag"; exit 1; }
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-rd3$SUF-lam$L-mu$MU-bpv$BPV; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
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
print('$SCENE lam $L bpv $BPV bpf $BPF RD-GROWTH+PRUNE-mu$MU indoor=$INDOOR | vertices', b['vertices'], 'faces', b['faces'], '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3),
      'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | grow: V@delaunay', dl[0]['vertices'] if dl else None,
      '| prune rounds', [(e['iteration'], e['vertices_before'], e['vertices_after'], 'pairs', e['n_pairs'], 'cap', e['max_pairs']) for e in pr], '| $FLAG')
PY
done
echo "===== $TAB ====="; cat $TAB
echo "ba-sens-v1 $SCENE bpv $BPV exit 0  $(date)"
SBEOF_BAS1
cat > $AN/submit-ba-sens-v1.sh <<'SUB'
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets; RES=yw523-l40s; LAMS="0.0062 0.00062"
n=$(wc -l < $AN/ba-sens-v1.sbatch); [ "$n" = 61 ] || { echo "CHECK FAILED: ba-sens-v1.sbatch has $n lines, expected 61 (paste truncated?)"; exit 1; }
grep -q -- '-bpv$BPV' $AN/ba-sens-v1.sbatch || { echo "CHECK FAILED: output dir has no -bpv suffix"; exit 1; }
grep -q "'--indoor'" $AN/rdgrowth/train_rd_v3.py || { echo "CHECK FAILED: train_rd_v3.py has no --indoor flag"; exit 1; }
for f in $D/m360/kitchen/images_2 $D/m360/garden/images_4 /scratch/zr2245/kitchen-rd3i-lam0.0062-mu1.0/axisB/v1.0/deliver/summary.json /scratch/zr2245/garden-rd3-lam0.0062-mu1.0/axisB/v1.0/deliver/summary.json; do [ -e $f ] || { echo "CHECK FAILED: MISSING $f"; exit 1; }; done
for d in /scratch/zr2245/kitchen-rd3i-lam0.0062-mu1.0-bpv8 /scratch/zr2245/garden-rd3-lam0.0062-mu1.0-bpv13; do [ -e $d ] && echo "note: $d exists, its training will be skipped"; done
echo "CHECK OK"; k=0
for BPV in 8 13; do
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --export=ALL,SCENE=kitchen,INDOOR=1,IMAGES=images_2,XYZ=q16,LAMS="$LAMS",BPV=$BPV,BPF=2.0,MU=1.0,MAXF=0.15 --job-name=kitchen-bpv$BPV $AN/ba-sens-v1.sbatch) && { echo "submitted kitchen-bpv$BPV $j"; k=$((k+1)); } || echo "FAILED: sbatch kitchen bpv $BPV"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --export=ALL,SCENE=garden,INDOOR=0,IMAGES=images_4,XYZ=q15,LAMS="$LAMS",BPV=$BPV,BPF=2.0,MU=1.0,MAXF=0.15 --job-name=garden-bpv$BPV $AN/ba-sens-v1.sbatch) && { echo "submitted garden-bpv$BPV $j"; k=$((k+1)); } || echo "FAILED: sbatch garden bpv $BPV"
done
echo "submitted $k of 4"
SUB
bash $AN/submit-ba-sens-v1.sh
squeue -u $USER -o "%.10i %.14j %.3t %.10M %.6D %R" | grep -E "lkfan2|basens|bpv|JOBID"
```

## §3 看进度 / 结果（任意节点）

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.14j %.3t %.10M %.6D %R" | grep -E "lkfan2|bpv|JOBID"; echo
for f in $AN/lkfan-*-lkfan2-*.out $AN/basens-*.out; do [ -f $f ] && { echo "== $f"; grep -E "^===|FAILED|exit 0|FINISHED" $f | tail -n 4; }; done 2>/dev/null
echo "===== fan rows so far"; wc -l $AN/lambda-k-fan-v1.txt; grep -E "^(counter|bonsai|bicycle|stump|flowers|treehill|truck|train) " $AN/lambda-k-fan-v1.txt
echo "===== ba-sens"; cat $AN/ba-sens-v1.txt 2>/dev/null
```

跑完把 §3 最后两段贴我：fan 48 行 → Table 6 / Fig 7 扩到 11 场景；ba-sens 8 行 → 和各场景的 λ hull 画在一起（bpv8 / bpv13 点应落在 hull 上，这就是"价格常数只是 λ 的重标定"的证据）。
