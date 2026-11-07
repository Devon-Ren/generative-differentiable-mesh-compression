# indoor 交付损失分解诊断 indoor-diag-v1（2026-11-05，≈0.5-0.7 GPU-h，general）

回答 Yao Q2 的"最高码率点失真是不是 DC 7 bit 造成"：对已有的 room / kitchen `--indoor` 原模型 level-1.0 交付（room 28.480→27.994，kitchen 27.465→26.706），**不训练**，在同一个 finetune 后的 mesh 上把三个固定交付设置分别放开：(a) DC q8；(b) 真实 per-vertex weight（keep）；(c) 两者；(d) 字典 K4096 R1024 / K16384 R1024（k-means 重拟合，每点 4-15 min）。基线行 = const/ctx q7 K4096 R256。输出 `$AN/indoor-diag-v1.txt`，每行带 dPSNR / dMB。

读法：哪一项的 dPSNR 接近 +0.3..+0.5 dB，哪一项就是 indoor 模型的损失来源；若三项各只有 +0.05..+0.1，则损失分散在整个量化+字典链上（和 Room 非 indoor 的结论一致，答案是"不是单一 DC"）。决定 indoor 行的类别默认（DC q8 或 K16384），不逐场景调。

## §1 写文件 + 自检 + 提交（整块粘贴，任意节点）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN
cat > $AN/indoor-diag-v1.sbatch <<'SBEOF_IDG1'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=indoor-diag
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/zr2245/analysis/indoor-diag-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/indoor-diag-%x-%j.err
# indoor-diag-v1.sbatch (2026-11-05, VERSION v1): decompose the level-1.0 delivery loss of the --indoor originals (room -0.49, kitchen -0.76 dB
# vs own FP32) into its three fixed delivery settings, WITHOUT training, on the existing $SCENE-indoor/axisB/v1.0 delivery:
#   (a) DC q8 instead of q7            codec_variants const/ctx --dc q8                 (seconds + test render)
#   (b) true per-vertex weight (keep)  codec_variants keep/ctx  --dc q7                 (seconds + test render)
#   (c) both                           codec_variants keep/ctx  --dc q8
#   (d) dictionary K4096 R1024 / K16384 R1024 instead of K4096 R256   deliver_coarse.sh on the same finetuned mesh (k-means + 400-step fit, ~4-15 min)
# Rows -> $AN/indoor-diag-v1.txt: scene | lever | MB | test PSNR SSIM LPIPS | dPSNR vs the baseline row (const/ctx q7 K4096R256). 48G: dictionary fit holds a 2.5 M-vertex model.
set -o pipefail
export TORCH_HOME=/scratch/zr2245/cache/torch
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENES=${SCENES:-"room kitchen"}; IMAGES=${IMAGES:-images_2}; KRS=${KRS:-"4096 1024|16384 1024"}
AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; TAB=$AN/indoor-diag-v1.txt
echo "indoor-diag-v1 | scenes $SCENES | K x R $KRS | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
xyz_of () { case $1 in room) echo q14;; kitchen) echo q16;; bonsai) echo q14;; counter) echo q14;; *) echo "FAILED: no XYZ recorded for $1 (add it to xyz_of)" >&2; return 1;; esac; }
for S in $SCENES; do
  B=/scratch/zr2245/$S-indoor; d=$B/axisB/v1.0/deliver; MESH=$B/axisB/v1.0/finetune/best.pt; C=$B/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; DATA=/scratch/zr2245/datasets/m360/$S
  XYZ=$(xyz_of $S) || exit 1
  for f in $d/shared-final.pt $d/config.json $d/summary.json $MESH $C $DATA/$IMAGES/. $PKG/codec_variants.py $PKG/deliver_coarse.sh $PKG/stage2_run.sh /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
  python -c "import json; c=json.load(open('$d/config.json')); k=c.get('k',4096); r=c.get('r',256); assert (k,r)==(4096,256), ('baseline delivery is not K4096 R256', k, r)" || { echo "FAILED: $S baseline dictionary is not K4096 R256"; exit 1; }
  echo "=== $S: baseline + (a)(b)(c) codec re-exports on $d (XYZ $XYZ) ($(date +%H:%M)) ==="
  for V in "const/ctx q7" "const/ctx q8" "keep/ctx q7" "keep/ctx q8"; do set -- $V; W=$1; DC=$2
    if python -c "import json,sys; s=json.load(open('$d/codec-variants/sweep.json')); sys.exit(0 if any((r['weight']+'/'+r['index_codec'],r['xyz'],r['dc'])==('$W','$XYZ','$DC') and r.get('test') for r in s['rows']) else 1)" 2>/dev/null; then echo "exists: $W $XYZ/$DC"; continue; fi
    bash $PKG/stage2_run.sh codec_variants.py --deliver-dir $d --xyz $XYZ --dc $DC --variants $W > $d/codec-variants-diag-${W/\//-}-$DC.log 2>&1 || { echo "FAILED: codec_variants $S $W $XYZ/$DC"; tail -5 $d/codec-variants-diag-${W/\//-}-$DC.log; exit 1; }
  done
  echo "=== $S: (d) dictionary re-fits on $MESH ($(date +%H:%M)) ==="
  OUT=$B/diag-v1; mkdir -p $OUT
  echo "$KRS" | tr '|' '\n' | while read K R; do name=$OUT/k${K}_r${R}
    if [ -f $name/summary.json ]; then echo "exists: k${K}_r${R}"; continue; fi
    rm -rf $name; echo "--- K $K R $R $(date +%H:%M:%S)"
    bash $PKG/deliver_coarse.sh --coarse $MESH --output $name --source $C --dataset $DATA --lab $LAB --images $IMAGES --xyz $XYZ --dc q7 \
      --historical-config none --k $K --r $R --dictionary-rounds 4 > $name.log 2>&1 || { echo "FAILED: deliver_coarse $S k${K}_r${R}"; tail -8 $name.log; exit 1; }
    [ -f $name/summary.json ] || { echo "FAILED: no summary for $S k${K}_r${R}"; exit 1; }
  done || exit 1
  python - "$S" "$d" "$XYZ" "$B/diag-v1" "$KRS" <<'PY' | tee -a $TAB
import json, sys, pathlib
S, d, xyz, out, krs = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3], pathlib.Path(sys.argv[4]), sys.argv[5]
s = json.load(open(d / 'codec-variants' / 'sweep.json')); rows = {}
for r in s['rows']:
    if r['xyz'] == xyz and r.get('test'): rows[(r['weight'] + '/' + r['index_codec'], r['dc'])] = r
base = rows[('const/ctx', 'q7')]; bp = base['test']['psnr']
def line(name, mb, t): print(f"{S} | {name:28s} | {mb:7.3f} MB | test {t['psnr']:.3f} {t['ssim']:.4f} {t['lpips']:.4f} | dPSNR {t['psnr'] - bp:+.3f} | dMB {mb - base['file_MB']:+.3f}")
line('baseline const/ctx q7 K4096R256', base['file_MB'], base['test'])
for key, name in ((('const/ctx', 'q8'), '(a) DC q8'), (('keep/ctx', 'q7'), '(b) true weight'), (('keep/ctx', 'q8'), '(c) DC q8 + true weight')):
    r = rows.get(key); line(name, r['file_MB'], r['test']) if r else print(f"{S} | {name}: MISSING")
for kr in krs.split('|'):
    K, R = kr.split(); p = out / f'k{K}_r{R}' / 'summary.json'
    if p.exists(): j = json.load(open(p)); line(f'(d) K{K} R{R} q7', j['file_MB'], j['delivery_test'])
    else: print(f"{S} | (d) K{K} R{R}: MISSING")
PY
done
echo "===== $TAB ====="; cat $TAB
echo "indoor-diag-v1 exit 0  $(date)"
SBEOF_IDG1
n=$(wc -l < $AN/indoor-diag-v1.sbatch); [ "$n" = 67 ] && bash -n $AN/indoor-diag-v1.sbatch && grep -q "VERSION v1" $AN/indoor-diag-v1.sbatch && echo "CHECK OK ($n lines)" || echo "CHECK FAILED: $n lines (expect 67) - do not submit"
for S in room kitchen; do for f in /scratch/zr2245/$S-indoor/axisB/v1.0/deliver/shared-final.pt /scratch/zr2245/$S-indoor/axisB/v1.0/finetune/best.pt; do [ -f $f ] && echo "ok: $f" || echo "MISSING: $f"; done; done
j=$(sbatch --parsable --export=ALL,SCENES="room kitchen",IMAGES=images_2 --job-name=indoor-diag $AN/indoor-diag-v1.sbatch) && echo "submitted indoor-diag $j" || echo "FAILED: sbatch"
squeue -u $USER -o "%.10i %.16j %.28a %.2t %.10M %.6D %R"
```

如果上面有 `MISSING`，先别提交，把那两行贴给我（说明 indoor 原模型目录名和我记的 `$S-indoor` 不一样）。

## §2 看结果

```bash
AN=/scratch/zr2245/analysis
ls -t $AN/indoor-diag-indoor-diag-*.out | head -1 | xargs grep -h "===\|FAILED\|exit 0\|MISSING"
cat $AN/indoor-diag-v1.txt
```
