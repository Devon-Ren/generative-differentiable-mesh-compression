# batchA-originals-v1（2026-10-20）：4 个新场景的 original + level-1.0 交付行（flowers、treehill、truck、train），4 个 job × ≈2.5 h，general

每个 job：Metric3D normals（缺则抽）→ upstream `train.py` 30k（与 7 场景 original 同命令，无 `--indoor`）→ footprint → level-1.0 protocol（K4096 R256，XYZ 按精度规则 b = ceil(log2(extent / 0.1·median edge)) 自动算，DC q7，stage2-v2）→ 一行到 `/scratch/zr2245/analysis/originals.txt`。可重启：已有的阶段跳过。walltime 3.5 h（训练 1–1.5 h + 交付约 40 min + normals），`--mem=64G` 是因为 coarse_vis 需要 ≥48 GB RAM（scene-v1 同样）。

## 1. 数据检查 + T&T 下载（任意节点，cpu-torch 也行；T&T 已有则跳过）

```bash
D=/scratch/zr2245/datasets; mkdir -p $D
for S in m360/flowers m360/treehill tandt/truck tandt/train; do
  if [ -f $D/$S/sparse/0/cameras.bin ]; then echo "ok   $S  images: $(ls -d $D/$S/images* | xargs -n1 basename | tr '\n' ' ')"; else echo "MISS $S"; fi
done
if [ ! -f $D/tandt/truck/sparse/0/cameras.bin ]; then
  cd $D && wget -c https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip && unzip -q -o tandt_db.zip && ls $D/tandt && echo "T&T downloaded"
fi
```

flowers / treehill 如果显示 MISS：这两个场景不在 360_v2.zip（那里只有 7 个），而在 Mip-NeRF 360 官方的 `360_extra_scenes.zip`（含 images / images_2 / images_4 / images_8 和 sparse/0）。下载块（2026-10-24 补，任意节点）：

```bash
D=/scratch/zr2245/datasets/m360; mkdir -p $D; cd $D
wget -c https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip \
  && unzip -q -o 360_extra_scenes.zip && ls $D \
  && for S in flowers treehill; do [ -f $D/$S/sparse/0/cameras.bin ] && echo "ok $S: $(ls -d $D/$S/images* | xargs -n1 basename | tr '\n' ' ')" || echo "STILL MISSING $S"; done
```

## 2. 写 sbatch 并提交 4 个 job（一次粘贴）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN
cat > $AN/originals-v1.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=orig
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/zr2245/analysis/orig-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/orig-%x-%j.err
# originals-v1.sbatch (2026-10-20): original MeshSplatting model + level-1.0 delivery for one scene.
#   sbatch --export=ALL,SCENE=flowers,IMAGES=images_4 --job-name=flowers-orig /scratch/zr2245/analysis/originals-v1.sbatch
#   sbatch --export=ALL,SCENE=truck,IMAGES=images,DATA=/scratch/zr2245/datasets/tandt/truck --job-name=truck-orig ...
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_4}
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}; G=${G:-/scratch/zr2245/$SCENE-vis-collapse}
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12
K=${K:-4096}; R=${R:-256}; DC=${DC:-q7}; XYZ=${XYZ:-auto}; TAB=$AN/originals.txt
C=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
echo "scene $SCENE | DATA $DATA | IMAGES $IMAGES | G $G | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $DATA/$IMAGES/. $DATA/sparse/0/cameras.bin $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $LAB/upstream/train.py $LAB/upstream/extract_normals.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do
  [ -e $f ] || { echo "MISSING: $f"; exit 1; }; done
ls $PKG/native/_native_qem_vis*.so >/dev/null 2>&1 || { echo "native module missing in $PKG/native"; exit 1; }
mkdir -p $G
# --- stage -1: normals (same protocol as the 7 scenes)
NORMALS=$DATA/${IMAGES/images/normals}
n_img=$(ls $DATA/$IMAGES | grep -ci '\.jpe\?g$'); n_nrm=$(ls $NORMALS 2>/dev/null | grep -ci '\.png$')
echo "images: $n_img in $DATA/$IMAGES | normals: $n_nrm in $NORMALS"
if [ "$n_nrm" -lt "$n_img" ]; then
  echo "=== normals -> $NORMALS ($(date)) ==="
  (cd $LAB/upstream && python extract_normals.py -s $DATA -i $IMAGES > $G.normals.log 2>&1) || { echo "NORMALS FAILED (log $G.normals.log)"; grep -n -m 6 -i 'error\|No module' $G.normals.log; exit 1; }
  n_nrm=$(ls $NORMALS | grep -ci '\.png$'); [ "$n_nrm" -ge "$n_img" ] || { echo "normals incomplete: $n_nrm / $n_img"; exit 1; }
fi
# --- stage 0: original (identical to run_garden.sh stage 0; no --indoor, as for the 7 scenes)
if [ ! -f $C ]; then
  echo "=== train original -> $G/original ($(date)) ==="
  (cd $LAB/upstream && python train.py -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 > $G/train.log 2>&1) || { echo "TRAIN FAILED"; tail -20 $G/train.log; exit 1; }
  [ -f $C ] || { echo "TRAIN produced no checkpoint"; tail -20 $G/train.log; exit 1; }
else echo "exists: $C"; fi
TEST30=$(grep -h "\[ITER 30000\] Evaluating test" $G/train.log | tail -1); echo "$TEST30"
# --- XYZ precision by the scene rule (C7): b = ceil(log2(extent / (0.1 * median edge))), clamped to 13..18
if [ "$XYZ" = "auto" ]; then
  XYZ=$(python - <<PY
import torch, numpy as np, math
w=torch.load('$C',map_location='cpu',weights_only=False); x=w['triangles_points'].numpy().astype(np.float64); f=np.asarray(w['_triangle_indices']).astype(np.int64)
extent=float((x.max(0)-x.min(0)).max()); rng=np.random.default_rng(0); sel=rng.choice(len(f),min(len(f),1000000),replace=False)
e=np.concatenate([f[sel][:,[0,1]],f[sel][:,[1,2]],f[sel][:,[2,0]]]); med=float(np.median(np.linalg.norm(x[e[:,0]]-x[e[:,1]],axis=1)))
b=int(min(18,max(13,math.ceil(math.log2(extent/(0.1*med)))))); print(f'q{b}')
import sys; print(f'precision rule: extent {extent:.3f} median edge {med:.5f} -> q{b}', file=sys.stderr)
PY
) || { echo "PRECISION RULE FAILED"; exit 1; }
fi
echo "XYZ precision: $XYZ"
# --- level-1.0 protocol (footprint + dense anchor finetune + K4096 R256 delivery) = the 'dense codec' row
echo "=== level-1.0 protocol ($(date)) ==="
G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 || { echo "LEVEL-1.0 FAILED"; tail -12 $G/level.log; exit 1; }
NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "no delivery"; tail -12 $G/level.log; exit 1; }
python - <<PY | tee -a $TAB
import json, re
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']
m=re.search(r'PSNR ([0-9.]+) SSIM ([0-9.]+) LPIPS ([0-9.]+)', '''$TEST30''')
print('$SCENE ORIGINAL | vertices', b['vertices'], 'faces', b['faces'], '| original test', m.group(1)[:6] if m else 'n/a', m.group(2)[:6] if m else '', m.group(3)[:6] if m else '',
      '| level-1.0: finetune test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test', round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | $IMAGES')
PY
echo "originals $SCENE exit 0  $(date)"
SBEOF
chmod +x $AN/originals-v1.sbatch
D=/scratch/zr2245/datasets
sub () { j=$(sbatch --parsable --export=ALL,$2 --job-name=$1-orig $AN/originals-v1.sbatch) && echo "$1-orig $j" || echo "SUBMIT FAILED $1"; }
[ -f $D/m360/flowers/sparse/0/cameras.bin ]  && sub flowers  "SCENE=flowers,IMAGES=images_4"  || echo "SKIP flowers: dataset missing"
[ -f $D/m360/treehill/sparse/0/cameras.bin ] && sub treehill "SCENE=treehill,IMAGES=images_4" || echo "SKIP treehill: dataset missing"
[ -f $D/tandt/truck/sparse/0/cameras.bin ]   && sub truck    "SCENE=truck,IMAGES=images,DATA=$D/tandt/truck" || echo "SKIP truck: dataset missing"
[ -f $D/tandt/train/sparse/0/cameras.bin ]   && sub train    "SCENE=train,IMAGES=images,DATA=$D/tandt/train" || echo "SKIP train: dataset missing"
squeue -u $USER -o "%.10i %.14j %.24a %.2t %.10M %.6D %R"
```

跑完给我 `cat /scratch/zr2245/analysis/originals.txt`（每场景一行：original 的 test PSNR/SSIM/LPIPS、顶点数、level-1.0 交付的 MB / test）。失败看 `orig-<scene>-orig-<jobid>.out` 最后 40 行。

**算力**：4 job × ≈2.5 h ≈ 10 GPU-h（比估的 4 h 多，因为每个场景顺手把 level-1.0 交付行也做了，省去批次 B 再回头补）。批次 B（11 场景 × 2 λ 的 v3）等这批 original 出来后给。
