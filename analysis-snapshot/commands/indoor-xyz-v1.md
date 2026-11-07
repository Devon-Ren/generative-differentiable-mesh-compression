# indoor-xyz-v1：room indoor level-1.0 的 XYZ 精度重导出（2026-11-05 14:30，≈10-15 min GPU，general）

indoor-diag-v1-stages 结果：room 的 quant+coding 损失 −0.35 dB（val），其中 DC q8 只解释 ~0.08，其余 ≈0.27 dB 只能是 XYZ q14 网格（const weight 和 ctx 编码都是无损）。kitchen（q16）quant+coding 只有 −0.12，不需要。10/17 的 7 场景精度扫描已经显示规则 b=⌈log2(extent/(0.1·median edge))⌉ 在 4/7 场景低 1 bit，所以这里对 room indoor 直接重导出 q15 / q16（const/ctx，DC q7），和已有 q14 基线并排。不训练，每点几秒 + 一次 test 渲染。

## §1 写文件 + 自检 + 提交（整块粘贴，任意节点）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN
cat > $AN/indoor-xyz-v1.sbatch <<'SBEOF_IXZ1'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=indoor-xyz
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/zr2245/analysis/indoor-xyz-%j.out
#SBATCH --error=/scratch/zr2245/analysis/indoor-xyz-%j.err
# indoor-xyz-v1.sbatch (2026-11-05, VERSION v1): XYZ precision re-export on the --indoor level-1.0 room delivery.
# indoor-diag-v1-stages showed room quant+coding = -0.35 dB (val) of which DC q8 explains ~0.08 -> the rest is the XYZ q14 grid.
# Re-exports const/ctx q15 / q16 with DC q7 into the same deliver dir (seconds + one test render each, no training) and prints
# rows next to the existing q14 baseline: scene | xyz | MB | test PSNR SSIM LPIPS | dPSNR | dMB. Restartable (existing sweep rows skipped).
set -o pipefail
export TORCH_HOME=/scratch/zr2245/cache/torch
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:-room}; BASE=${BASE:-q14}; XYZS=${XYZS:-"q15 q16"}; DC=q7; W=const/ctx
AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; TAB=$AN/indoor-xyz-v1.txt
d=/scratch/zr2245/$SCENE-indoor/axisB/v1.0/deliver
echo "indoor-xyz-v1 | scene $SCENE base $BASE xyz $XYZS dc $DC | job $SLURM_JOB_ID on $(hostname) $(date)"
for f in $PKG/codec_variants.py $PKG/stage2_run.sh $d/shared-final.pt $d/codec-variants/sweep.json; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
nf=0
for X in $XYZS; do
  if python -c "import json,sys; s=json.load(open('$d/codec-variants/sweep.json')); sys.exit(0 if any((r['weight']+'/'+r['index_codec'],r['xyz'],r['dc'])==('$W','$X','$DC') and r.get('test') for r in s['rows']) else 1)" 2>/dev/null; then echo "exists: $W $X/$DC"; continue; fi
  echo "--- $SCENE $W $X/$DC $(date +%H:%M:%S)"
  bash $PKG/stage2_run.sh codec_variants.py --deliver-dir $d --xyz $X --dc $DC --variants $W > $d/codec-variants-xyz-$X-$DC.log 2>&1 || { echo "FAILED: codec_variants $SCENE $X/$DC"; tail -5 $d/codec-variants-xyz-$X-$DC.log; nf=$((nf+1)); }
done
python - "$SCENE" "$d" "$BASE" "$XYZS" "$DC" "$W" <<'PY' | tee -a $TAB
import json, sys, pathlib
S, d, base, xyzs, dc, w = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4].split(), sys.argv[5], sys.argv[6]
rows = {(r['xyz'], r['dc']): r for r in json.load(open(d / 'codec-variants' / 'sweep.json'))['rows'] if r['weight'] + '/' + r['index_codec'] == w and r.get('test')}
b = rows.get((base, dc))
if b is None: print(f"{S} | baseline {base}/{dc} MISSING"); sys.exit(0)
def line(x, r): t = r['test']; print(f"{S} | xyz {x} dc {dc} | {r['file_MB']:7.3f} MB | test {t['psnr']:.3f} {t['ssim']:.4f} {t['lpips']:.4f} | dPSNR {t['psnr'] - b['test']['psnr']:+.3f} | dMB {r['file_MB'] - b['file_MB']:+.3f}")
line(base + ' (base)', b)
for x in xyzs:
    r = rows.get((x, dc)); line(x, r) if r else print(f"{S} | xyz {x} dc {dc}: MISSING")
PY
[ $nf = 0 ] && echo "indoor-xyz-v1 $SCENE exit 0  $(date)" || { echo "indoor-xyz-v1 $SCENE FINISHED WITH $nf FAILED items  $(date)"; exit 1; }
SBEOF_IXZ1
cat > $AN/check-indoor-xyz-v1.sh <<'CHK'
n=$(wc -l < /scratch/zr2245/analysis/indoor-xyz-v1.sbatch)
[ "$n" = 44 ] || { echo "CHECK FAILED: indoor-xyz-v1.sbatch has $n lines, expected 44 (paste truncated?)"; exit 1; }
grep -q "<<'PY' | tee -a" /scratch/zr2245/analysis/indoor-xyz-v1.sbatch || { echo "CHECK FAILED: heredoc/tee form wrong"; exit 1; }
d=/scratch/zr2245/room-indoor/axisB/v1.0/deliver
for f in $d/shared-final.pt $d/codec-variants/sweep.json /scratch/zr2245/analysis/vis_collapse-v12/codec_variants.py; do [ -e $f ] || { echo "CHECK FAILED: MISSING $f"; exit 1; }; done
echo "CHECK OK"
CHK
bash $AN/check-indoor-xyz-v1.sh && sbatch --parsable --account=torch_pr_38_general --export=ALL,SCENE=room --job-name=room-ixyz $AN/indoor-xyz-v1.sbatch
squeue -u $USER -o "%.10i %.14j %.3t %.10M %.6D %R" | grep -E "ixyz|lkfan|JOBID"
```

## §2 看结果（跑完后，任意节点）

```bash
AN=/scratch/zr2245/analysis
tail -4 $(ls -t $AN/indoor-xyz-*.out | head -1)
cat $AN/indoor-xyz-v1.txt
```

预期 3 行：q14 (base) / q15 / q16，各带 dPSNR、dMB。读法：q14→q15 的 dB/MB 若 ≥0.1（garden 当年 0.147）就说明 indoor room 的规则低了 1 bit，report §2.6 的 indoor 顶端损失写成 "SH sharing（主）+ XYZ 网格（room）"，并把 indoor room 的 XYZ 改为 q15（类别规则加 1 bit 余量，不逐场景调）。
