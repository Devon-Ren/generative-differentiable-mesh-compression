# nerf-synthetic-v1：NeRF-Synthetic（Blender，8 场景）探针（2026-11-04）

**目的**：决定 P2/P3（plan/object-dataset-decision-2026-11-04.md §5）。三步：§0 CPU 下载 → §1 CPU 检查 upstream 的 Blender loader → §2 GPU lego smoke test（1 个 job，≈0.5-1 GPU-h）→ §3 通过后 8 场景探针（16 个 job，≈10 GPU-h，general）。
**不改任何旧文件**：新建 `originals-blender-v1.sbatch`、`rd-prune-blender-v1.sbatch`、`submit-blender-v1.sh`；表 `originals-blender.txt`、`rd-prune3b-<scene>.txt`；输出目录 `/scratch/zr2245/<scene>-blender/`、`/scratch/zr2245/<scene>-rd3b-lam<L>-mu1.0/`。

**背景色（重要，先说清楚）**：level-1.0 交付链（vertex_footprint / coarse_finetune / deliver_coarse）全部写死黑背景（`white_background=False`、`bg = zeros`）。所以这次探针**全程黑背景**（train.py 不加 `--white_background`，loader 会把 RGBA 合成到黑底，渲染背景也是黑，自洽）。论文上 NeRF-Synthetic 惯例是白背景，数字会差一点（通常 ≤ 0.3 dB，方向不定）；如果探针通过、决定做 P3，我再出交付链的 `-v2`（新文件，加 `--white_background`）重跑白底版本。探针阶段不碰。

## §0 下载数据集（cpu-torch 即可，不花 FairShare）

```bash
DS=/scratch/zr2245/datasets/nerf_synthetic
mkdir -p $DS && cd $DS
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
python -c "import gdown" 2>/dev/null || pip install -q gdown
# 官方 NeRF Google Drive 里的 nerf_synthetic.zip（约 2 GB）
[ -f nerf_synthetic.zip ] || gdown 18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG -O nerf_synthetic.zip || echo "FAILED: gdown（Drive 限流时：本机从 NeRF 项目页的 Drive 下载 nerf_synthetic.zip 再 scp 到 $DS）"
[ -f nerf_synthetic.zip ] && unzip -q -n nerf_synthetic.zip
# 解压后应是 $DS/nerf_synthetic/{chair,drums,ficus,hotdog,lego,materials,mic,ship}；统一成 $DS/<scene>
[ -d $DS/nerf_synthetic ] && for s in chair drums ficus hotdog lego materials mic ship; do [ -d $DS/$s ] || mv $DS/nerf_synthetic/$s $DS/$s; done
for s in chair drums ficus hotdog lego materials mic ship; do
  printf "%-10s train %3s  test %3s  %s\n" $s "$(ls $DS/$s/train 2>/dev/null | grep -c png)" "$(ls $DS/$s/test 2>/dev/null | grep -c png)" "$(ls $DS/$s/transforms_train.json 2>/dev/null)"
done
# 期望：每场景 train 100、test 200（test 目录里还有 depth/normal png，计数会 >200，正常）、transforms_train.json 存在
```

### §0b Drive 限流时：从 nerfbaselines 的 Hugging Face 镜像按场景下载（2026-11-04 补）

```bash
DS=/scratch/zr2245/datasets/nerf_synthetic
mkdir -p $DS && cat > $DS/fetch_hf_v1.sh <<'SHEOF'
#!/usr/bin/env bash
# fetch_hf_v1.sh (2026-11-04): NeRF-Synthetic 8 scenes from https://huggingface.co/datasets/nerfbaselines/nerfbaselines-data/tree/main/blender (8 zips, 1.29 GB total)
DS=/scratch/zr2245/datasets/nerf_synthetic; BASE=https://huggingface.co/datasets/nerfbaselines/nerfbaselines-data/resolve/main/blender
cd $DS || { echo "FAILED: cd $DS"; exit 1; }
for s in chair drums ficus hotdog lego materials mic ship; do
  if [ -f $DS/$s/transforms_train.json ]; then echo "exists: $s"; continue; fi
  echo "=== $s ($(date)) ==="
  wget -q --tries=5 --timeout=60 -c -O $DS/$s.zip "$BASE/$s.zip" || { echo "FAILED: wget $s"; continue; }
  rm -rf $DS/_tmp_$s; mkdir -p $DS/_tmp_$s
  unzip -q -o $DS/$s.zip -d $DS/_tmp_$s || { echo "FAILED: unzip $s"; continue; }
  T=$(find $DS/_tmp_$s -name transforms_train.json | head -1)
  [ -n "$T" ] || { echo "FAILED: no transforms_train.json inside $s.zip; contents:"; find $DS/_tmp_$s -maxdepth 3 | head -20; continue; }
  mv "$(dirname $T)" $DS/$s && rm -rf $DS/_tmp_$s $DS/$s.zip
  printf "%-10s train %3s  test %3s  val %3s\n" $s "$(ls $DS/$s/train 2>/dev/null | grep -c '\.png$')" "$(ls $DS/$s/test 2>/dev/null | grep -c '\.png$')" "$(ls $DS/$s/val 2>/dev/null | grep -c '\.png$')"
done
echo "--- summary ---"
for s in chair drums ficus hotdog lego materials mic ship; do
  printf "%-10s train %3s  test %3s  %s\n" $s "$(ls $DS/$s/train 2>/dev/null | grep -c '\.png$')" "$(ls $DS/$s/test 2>/dev/null | grep -c '\.png$')" "$([ -f $DS/$s/transforms_test.json ] && echo transforms_ok || echo MISSING_transforms_test)"
done
du -sh $DS
SHEOF
bash $DS/fetch_hf_v1.sh
```

期望：每场景 train 100、test 200（原版 test 目录含 depth/normal png，计数可能 600）、transforms_ok。某个场景 FAILED 就把那几行贴给我。

## §1 检查 upstream 是否认 Blender 格式（cpu-torch，只读，不改代码）

```bash
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1
echo "--- Scene 入口如何选 loader ---"; grep -n "sparse\|transforms_train\|Blender\|Colmap" $LAB/upstream/scene/__init__.py
echo "--- dataset_readers 里的 Blender 读取函数 ---"; grep -n "def readNerfSyntheticInfo\|def readCamerasFromTransforms\|white_background\|bg = \|normal\|depth" $LAB/upstream/scene/dataset_readers.py | head -40
echo "--- Camera 对象需要哪些先验字段 ---"; grep -n "normal\|depth" $LAB/upstream/scene/cameras.py $LAB/upstream/utils/camera_utils.py | head -30
echo "--- train.py 怎么用法线/深度先验 ---"; grep -n "normal\|depth" $LAB/upstream/train.py | head -30
echo "--- extract_normals.py 的输入假设 ---"; grep -n "add_argument\|sparse\|images\|glob\|listdir" $LAB/upstream/extract_normals.py | head -20
```

把这段输出贴给我。要看的点：(a) `scene/__init__.py` 有 `transforms_train.json` 分支（3DGS 原版有，MeshSplatting 大概率保留）；(b) Blender 的 CameraInfo 是否带 normal/depth 字段，缺了 train.py 会不会崩；(c) extract_normals.py 是否只认 COLMAP 的 `-s/-i` 目录。若 (b)(c) 有问题，smoke test 的 train.log 会直接报出来，我再决定是写一个 Blender 版的法线提取（新文件）还是关掉法线项。

## §2 lego smoke test（1 个 job，任意节点整块粘贴）

```bash
AN=/scratch/zr2245/analysis
cat > $AN/originals-blender-v1.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=origb
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/zr2245/analysis/origb-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/origb-%x-%j.err
# originals-blender-v1.sbatch (2026-11-04, VERSION v1): = originals-v2.sbatch for NeRF-Synthetic (Blender transforms_*.json) scenes.
# Differences: DATA=/scratch/zr2245/datasets/nerf_synthetic/$SCENE, no COLMAP/IMAGES checks (IMAGES=train is only passed through),
# no --indoor, BLACK background throughout (delivery chain hardcodes bg=0; see analysis/commands/nerf-synthetic-v1.md), normals stage
# only if extract_normals.py accepts the blender dir (NORMALS=1), output /scratch/zr2245/$SCENE-blender, table originals-blender.txt.
#   sbatch --export=ALL,SCENE=lego,XYZ=q14 --job-name=lego-origb /scratch/zr2245/analysis/originals-blender-v1.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-train}; NORMALS=${NORMALS:-0}
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; G=${G:-/scratch/zr2245/$SCENE-blender}
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12
K=${K:-4096}; R=${R:-256}; DC=${DC:-q7}; XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; TAB=$AN/originals-blender.txt
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
C=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
echo "scene $SCENE | DATA $DATA | G $G | blender, black bg | XYZ $XYZ | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $DATA/transforms_train.json $DATA/transforms_test.json $DATA/train/. $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do
  [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
ls $PKG/native/_native_qem_vis*.so >/dev/null 2>&1 || { echo "FAILED: native module missing in $PKG/native"; exit 1; }
grep -q "transforms_train.json" $LAB/upstream/scene/__init__.py || { echo "FAILED: upstream Scene has no Blender (transforms_train.json) branch"; exit 1; }
mkdir -p $G
# --- stage -1: normal priors. Off by default for blender (extract_normals.py was written for COLMAP dirs); NORMALS=1 tries it.
if [ "$NORMALS" = "1" ]; then
  echo "=== normals ($(date)) ==="
  (cd $LAB/upstream && python extract_normals.py -s $DATA -i $IMAGES > $G.normals.log 2>&1) || { echo "FAILED: NORMALS (log $G.normals.log)"; grep -n -m 6 -i 'error\|No module' $G.normals.log; exit 1; }
fi
# --- stage 0: original, black background (NO --white_background: matches the delivery chain)
if [ ! -f $C ]; then
  echo "=== train original -> $G/original ($(date)) ==="
  (cd $LAB/upstream && python train.py -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 > $G/train.log 2>&1) || { echo "FAILED: TRAIN"; grep -n -m 8 -i 'error\|Traceback\|No such\|KeyError\|AttributeError' $G/train.log; tail -20 $G/train.log; exit 1; }
  [ -f $C ] || { echo "FAILED: TRAIN produced no checkpoint"; tail -20 $G/train.log; exit 1; }
else echo "exists: $C"; fi
TEST30=$(grep -h "\[ITER 30000\] Evaluating test" $G/train.log | tail -1); echo "$TEST30"
grep -h "\[ITER 7000\] Evaluating test" $G/train.log | tail -1
# --- level-1.0 protocol (footprint + dense anchor finetune + K4096 R256 delivery)
echo "=== level-1.0 protocol ($(date)) ==="
G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 || { echo "FAILED: LEVEL-1.0"; grep -n -m 8 -i 'error\|Traceback' $G/level.log; tail -12 $G/level.log; exit 1; }
NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery"; tail -12 $G/level.log; exit 1; }
python - <<PY | tee -a $TAB
import json, re
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']
m=re.search(r'PSNR ([0-9.]+) SSIM ([0-9.]+) LPIPS ([0-9.]+)', '''$TEST30''')
print('$SCENE ORIGINAL-BLENDER-blackbg | vertices', b['vertices'], 'faces', b['faces'], '| original test', m.group(1)[:6] if m else 'n/a', m.group(2)[:6] if m else '', m.group(3)[:6] if m else '',
      '| level-1.0: finetune test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test', round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | $IMAGES')
PY
echo "originals-blender $SCENE exit 0  $(date)"
SBEOF
j=$(sbatch --parsable --export=ALL,SCENE=lego,XYZ=q14 --job-name=lego-origb $AN/originals-blender-v1.sbatch) && echo "submitted lego-origb $j" || echo "FAILED: sbatch"
squeue -u $USER -o "%.10i %.16j %.28a %.2t %.10M %.6D %R"
```

看结果：

```bash
AN=/scratch/zr2245/analysis
ls -t $AN/origb-lego-origb-*.out | head -1 | xargs tail -25
cat $AN/originals-blender.txt 2>/dev/null
grep -h "Evaluating" /scratch/zr2245/lego-blender/train.log 2>/dev/null | tail -4
```

**通过标准**：job 打印 `exit 0`，lego original test PSNR ≥ 30（白底论文数：3DGS 35.8，MobileNeRF / NeRF2Mesh 约 34，待核对，所以 30 是很宽松的"链路正常"线，不是质量线），delivered 与 finetune test 差 ≤ 0.1 dB（说明 q14 位置量化对小物体够用；差得多就改 q15）。
**没通过**：把 `.out` 的 FAILED 行和 `train.log` 的 Traceback 贴给我。最可能的两处：Blender camera 缺 normal/depth 字段（train.py 的 lambda_normals 项），或 `run_garden.sh` 里某个脚本假设 `$DATA/sparse/0` 存在。

## §3 8 场景探针（smoke test 通过后再跑；16 个 job ≈ 10 GPU-h，全部 general）

```bash
AN=/scratch/zr2245/analysis
cat > $AN/rd-prune-blender-v1.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=rd3b
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/zr2245/analysis/rd3b-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/rd3b-%x-%j.err
# rd-prune-blender-v1.sbatch (2026-11-04, VERSION v1): = rd-prune-v3c.sbatch for NeRF-Synthetic (Blender) scenes: DATA under datasets/nerf_synthetic,
# no --indoor, no COLMAP checks, BLACK background (no --white_background; delivery chain hardcodes bg=0), output /scratch/zr2245/$SCENE-rd3b-lam$L-mu$MU,
# table rd-prune3b-$SCENE.txt. Same lambda rule, same prices (BPV 10 BPF 2), same prune schedule 12000 14000 16000 18000, same codec.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-train}; LAMS=${LAMS:?set LAMS="0.00062 0.0062" via --export}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; TAB=$AN/rd-prune3b-$SCENE.txt; TR=$AN/rdgrowth/train_rd_v3.py
echo "scene $SCENE | blender, black bg | lams $LAMS | B/v $BPV B/face $BPF | prune at $PRUNE mu $MU max_frac $MAXF | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/transforms_train.json $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-rd3b-lam$L-mu$MU; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== $SCENE lam $L mu $MU: train+prune ($(date)) ==="
  if [ ! -f $CB ]; then
    python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF > $G/train.log 2>&1 \
      || { echo "FAILED: TRAIN $SCENE lam $L"; grep -n -m 8 -i 'error\|Traceback' $G/train.log; tail -12 $G/train.log; exit 1; }
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
print('$SCENE lam $L RD-GROWTH+PRUNE-mu$MU-BLENDER-blackbg | vertices', b['vertices'], 'faces', b['faces'], '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3),
      'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | grow: V@delaunay', dl[0]['vertices'] if dl else None,
      '| prune rounds', [(e['iteration'], e['vertices_before'], e['vertices_after'], 'pairs', e['n_pairs'], 'cap', e['max_pairs']) for e in pr])
PY
done
echo "===== $TAB ====="; cat $TAB
echo "rd-prune3b $SCENE exit 0  $(date)"
SBEOF

cat > $AN/submit-blender-v1.sh <<'SHEOF'
#!/usr/bin/env bash
# submit-blender-v1.sh (2026-11-04): 8 NeRF-Synthetic scenes x (1 original job + 1 lambda job of 2 lambdas 6.2e-4 / 6.2e-3) = 16 jobs, general, black bg.
# lego's original is skipped automatically if the smoke test already produced it.
AN=/scratch/zr2245/analysis; XYZ=${XYZ:-q14}
for f in $AN/originals-blender-v1.sbatch $AN/rd-prune-blender-v1.sbatch $AN/rdgrowth/train_rd_v3.py; do [ -f $f ] || { echo "FAILED: missing $f"; exit 1; }; done
n=0
for S in chair drums ficus hotdog lego materials mic ship; do
  [ -f /scratch/zr2245/datasets/nerf_synthetic/$S/transforms_train.json ] || { echo "FAILED: no dataset for $S"; continue; }
  [ -f /scratch/zr2245/$S-blender/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] && echo "note: $S original exists, job will skip its training"
  j=$(sbatch --parsable --time=02:30:00 --export=ALL,SCENE=$S,XYZ=$XYZ --job-name=$S-origb $AN/originals-blender-v1.sbatch) \
    && { echo "submitted $S-origb $j  XYZ $XYZ"; n=$((n+1)); } || echo "FAILED: sbatch $S original"
  LAMS="0.00062 0.0062"
  j=$(sbatch --parsable --time=03:00:00 --export=ALL,SCENE=$S,XYZ=$XYZ,LAMS="$LAMS",BPV=10.0,BPF=2.0,MU=1.0,MAXF=0.15 --job-name=$S-b62 $AN/rd-prune-blender-v1.sbatch) \
    && { echo "submitted $S-b62 $j  lams $LAMS  XYZ $XYZ"; n=$((n+1)); } || echo "FAILED: sbatch $S lams"
done
echo "submitted $n jobs (expected 16)"
squeue -u $USER -o "%.10i %.16j %.28a %.2t %.10M %.6D %R"
SHEOF
bash $AN/submit-blender-v1.sh
```

汇总（全部跑完后贴给我）：

```bash
AN=/scratch/zr2245/analysis
cat $AN/originals-blender.txt
cat $AN/rd-prune3b-*.txt
grep -l "exit 0" $AN/origb-*.out $AN/rd3b-*.out 2>/dev/null | wc -l; grep -l "FAILED" $AN/origb-*.out $AN/rd3b-*.out 2>/dev/null
```

## §4 决策门（plan/object-dataset-decision-2026-11-04.md §5）

8 场景均值：MeshSplatting original ≥ 31 dB 且 λ 6.2e-4 交付点 ≤ −0.3 dB → P3（加 object-centric 一节，再出白底 `-v2` 交付链重跑）；30-31 → P3 但以图为主；< 30 或链路跑不通 → P2。
对照（白底，论文数）：3DGS 33.32；HAC/HAC++ 约 33.2-33.5 在 1-2 MB；MobileNeRF 30.90；NeRF2Mesh 约 30。

## 附：已知问题（2026-11-04 smoke test 第一次提交）

- `upstream/scene/dataset_readers.py` 第 315 行 `np.array(arr*255.0, dtype=np.byte)` 与新版 Pillow 不兼容（`TypeError: Cannot handle this data type: (1, 1, 3), |i1`）。修法：改成 `dtype=np.uint8`（一行，备份为 `dataset_readers.py.bak-20261104`）。COLMAP 路径不经过这个函数，Mip-360 结果不受影响。
- Blender 分支不填 `normal_map` / `invdepthmap`，train.py 第 170 / 237 行有守卫：探针**无法线先验**。P3 时再补 Blender 版法线提取（新文件）。

## §2b lego smoke test 结果（2026-11-04 04:48）与 train-only 对照 sbatch

结果：链路通（exit 0，训练 24 min + 交付 6 min）。lego 黑底、无法线先验、室外默认超参：594k V / 1.58M F；test PSNR 7k **31.79** → 29k 29.48 → 30k **28.79**（train 32.36）；level-1.0 交付 7.98 MB / 28.55（codec −0.24 dB）。对照：3DGS lego 白底 35.8。
解读：7k→30k 掉 3 dB 而 train 还在涨，是 soft→opaque 阶段的代价（与 +++ P2 的 room −2.0 / kitchen −3.6 dB 同一现象），不是 loader 问题。

train-only 对照（不做交付链，只看 30k test PSNR 曲线；FLAGS 任意传给 train.py）：

```bash
AN=/scratch/zr2245/analysis
cat > $AN/train-only-blender-v1.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=trb
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/zr2245/analysis/trb-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/trb-%x-%j.err
# train-only-blender-v1.sbatch (2026-11-04, VERSION v1): upstream train.py on a NeRF-Synthetic scene with extra FLAGS (e.g. "--indoor", "--white_background"),
# no delivery protocol; output /scratch/zr2245/$SCENE-blender-$TAG/original; prints the full test-PSNR curve at the end.
#   sbatch --export=ALL,SCENE=lego,TAG=indoor,FLAGS="--indoor" --job-name=lego-trb-indoor /scratch/zr2245/analysis/train-only-blender-v1.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE}; TAG=${TAG:?set TAG}; FLAGS=${FLAGS:-}
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; G=/scratch/zr2245/$SCENE-blender-$TAG; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1
echo "scene $SCENE | tag $TAG | flags '$FLAGS' | G $G | job $SLURM_JOB_ID on $(hostname) $(date)"
[ -f $DATA/transforms_train.json ] || { echo "FAILED: MISSING $DATA/transforms_train.json"; exit 1; }
mkdir -p $G
(cd $LAB/upstream && python train.py -s $DATA -i train -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --test_iterations 7000 10000 12000 15000 18000 20000 22000 25000 28000 30000 $FLAGS > $G/train.log 2>&1) \
  || { echo "FAILED: TRAIN"; grep -n -m 8 -i 'error\|Traceback' $G/train.log; tail -20 $G/train.log; exit 1; }
echo "--- test curve ($SCENE $TAG '$FLAGS') ---"; grep -h "Evaluating test" $G/train.log | sed 's/\[ITER \([0-9]*\)\] Evaluating test: L1 [0-9.]* PSNR \([0-9.]*\) SSIM \([0-9.]*\) LPIPS \([0-9.]*\).*/\1 \2 \3 \4/'
python -c "import torch; sd=torch.load('$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt',map_location='cpu',weights_only=False); print('vertices', len(sd['triangles_points']), 'faces', len(sd['_triangle_indices']))"
echo "train-only $SCENE $TAG exit 0  $(date)"
SBEOF
j=$(sbatch --parsable --export=ALL,SCENE=lego,TAG=indoor,FLAGS="--indoor" --job-name=lego-trb-indoor $AN/train-only-blender-v1.sbatch) && echo "submitted lego-trb-indoor $j" || echo "FAILED: sbatch"
squeue -u $USER -o "%.10i %.16j %.28a %.2t %.10M %.6D %R"
```

## §5 val 选点版（2026-11-04 晚）：originals-blender-v2 / rd-prune-blender-v2 / submit-blender-v2

**为什么**：lego 完整曲线 test 峰值 34.08 @ 11k（soft）、opaque 阶段 32.89 @ 14k，之后 100 张训练视角过拟合单调掉到 28.79 @ 30k（train 仍在涨）。Blender 分支无法线/深度先验。用官方 val split（transforms_val.json，100 张，upstream 不读）选 checkpoint 是合法的通用 early stopping，不碰 test，8 场景同一规则。
**工具**：`analysis/blender/val_select_v1.py`（新文件；通过 upstream 自己的 `Scene(load_iteration=N)` 加载 checkpoint，shadow 目录把 transforms_val.json 当 test 读，渲染 val，选 val PSNR 最高的 iteration ≥ 12000，生成 `$G-best/original`，其 `point_cloud/iteration_30000` 是指向选中 checkpoint 的 symlink，交付链零改动）。
**代价**：每个 run 多存 19 个 checkpoint（~2.5 GB，选完即删）+ val 渲染 ~3-5 min。

### §5a 先在 lego 上验证 val_select（1 个 job，整块粘贴；INDOOR=1 已写在提交行里）

```bash
AN=/scratch/zr2245/analysis
mkdir -p $AN/blender
# ---- 把 val_select_v1.py 放到 $AN/blender/ （从 /mnt/project-files/analysis/blender/val_select_v1.py 复制内容；下面用 heredoc 写入）
cat > $AN/blender/val_select_v1.py <<'PYEOF'
"""val_select_v1.py (VERSION val_select_v1 2026-11-04): checkpoint selection by the OFFICIAL NeRF-Synthetic validation split.

Why: on NeRF-Synthetic (100 training views, no normal/depth priors in the Blender loader) MeshSplatting's test PSNR peaks
after the mesh stage and then drifts down while train PSNR keeps rising (lego: 32.9 @ 14k -> 28.8 @ 30k).  Early stopping
on the test split would be cheating; the dataset ships transforms_val.json (100 views), which upstream never reads.

What: builds a shadow dataset dir whose transforms_test.json is transforms_val.json (symlinks, nothing copied), loads each
saved checkpoint through upstream's own Scene(load_iteration=N) so no loading code is duplicated, renders the val views,
reports val PSNR/SSIM per iteration, picks the best, and (optionally) materialises <model>-best/original whose
point_cloud/iteration_30000 symlinks to the chosen checkpoint so the unchanged level-1.0 delivery chain can consume it.
Background is black (white_background=False), matching the delivery chain.

Usage (on a GPU node):
  python val_select_v1.py --lab $LAB --dataset $DATA --model $G/original --best $G-best/original [--delete_unselected]
"""
import argparse, json, os, re, shutil, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

VERSION = 'val_select_v1 2026-11-04'


def log(*a):
    print('[val_select]', *a, flush=True)


def make_shadow(dataset: Path, shadow: Path):
    """dataset dir with transforms_test.json -> transforms_val.json; everything else symlinked."""
    if shadow.exists():
        shutil.rmtree(shadow)
    shadow.mkdir(parents=True)
    for p in dataset.iterdir():
        if p.name == 'transforms_test.json':
            continue
        os.symlink(p.resolve(), shadow / p.name)
    os.symlink((dataset / 'transforms_val.json').resolve(), shadow / 'transforms_test.json')
    return shadow


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lab', required=True, help='root containing upstream/')
    ap.add_argument('--dataset', required=True, help='NeRF-Synthetic scene dir (transforms_{train,val,test}.json)')
    ap.add_argument('--model', required=True, help='model dir holding point_cloud/iteration_*/ (e.g. $G/original)')
    ap.add_argument('--iters', nargs='*', type=int, default=None, help='iterations to score (default: every saved iteration)')
    ap.add_argument('--min_iter', type=int, default=12000, help='ignore checkpoints before this (soft stage; not deliverable as an opaque mesh)')
    ap.add_argument('--supersample', type=str, default='20000:2,25000:4', help='render supersampling by iteration threshold, mirrors training (1 below the first threshold)')
    ap.add_argument('--best', type=str, default=None, help='write <best>/point_cloud/iteration_30000 -> chosen checkpoint (+ symlinks to the other model files)')
    ap.add_argument('--delete_unselected', action='store_true', help='after choosing, delete the other saved iterations except the chosen one and iteration_30000')
    ap.add_argument('--out', type=str, default=None, help='summary json (default <model>/val_select.json)')
    a = ap.parse_args()
    if not torch.cuda.is_available():
        ap.error('CUDA required')
    lab = Path(a.lab).resolve(); upstream = lab / 'upstream'
    sys.path.insert(0, str(upstream / 'submodules/effrdel/src')); sys.path.insert(0, str(upstream))
    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render
    from utils.loss_utils import ssim

    model = Path(a.model).resolve(); dataset = Path(a.dataset).resolve()
    pc = model / 'point_cloud'
    saved = sorted(int(m.group(1)) for d in pc.iterdir() for m in [re.fullmatch(r'iteration_(\d+)', d.name)] if m and (d / 'point_cloud_state_dict.pt').exists())
    iters = a.iters if a.iters else [i for i in saved if i >= a.min_iter]
    missing = [i for i in iters if i not in saved]
    if missing:
        sys.exit(f'FAILED: checkpoints missing for iterations {missing}; saved = {saved}')
    if not iters:
        sys.exit(f'FAILED: no checkpoint >= {a.min_iter}; saved = {saved}')
    sched = sorted((int(k), int(v)) for k, v in (kv.split(':') for kv in a.supersample.split(',') if kv))

    def ss_of(it):
        s = 1
        for thr, v in sched:
            if it >= thr:
                s = v
        return s

    shadow = make_shadow(dataset, model.parent / f'{model.name}-valshadow')
    data = SimpleNamespace(source_path=str(shadow), model_path=str(model), images='train', resolution=1, eval=True,
                           white_background=False, data_device='cpu', sh_degree=3)
    bg = torch.zeros(3, device='cuda'); pipe = SimpleNamespace(debug=False, convert_SHs_python=False, depth_ratio=1.)
    rows = []
    for it in iters:
        mesh = TriangleModel(3)
        scene = Scene(data, mesh, None, None, load_iteration=it, shuffle=False)
        cams = sorted(scene.getTestCameras(), key=lambda c: (c.image_name, c.uid))
        if it == iters[0]:
            log(f'val views: {len(cams)} (from transforms_val.json via shadow dir {shadow})')
        mesh.scaling = ss_of(it)
        ps, ss = [], []
        with torch.no_grad():
            for c in cams:
                im = render(c, mesh, pipe, bg)['render'].clamp(0, 1); gt = c.original_image[:3].cuda().clamp(0, 1)
                ps.append(float(-10 * torch.log10((im - gt).square().mean().clamp_min(1e-12)))); ss.append(float(ssim(im, gt)))
        nv = int(mesh.vertices.shape[0]) if hasattr(mesh, 'vertices') else -1
        nf = int(mesh._triangle_indices.shape[0])
        row = dict(iteration=it, supersample=ss_of(it), val_psnr=float(np.mean(ps)), val_ssim=float(np.mean(ss)), worst_view_psnr=float(min(ps)), vertices=nv, faces=nf)
        rows.append(row)
        log(f"iter {it:6d}  ss {row['supersample']}  val PSNR {row['val_psnr']:.3f}  SSIM {row['val_ssim']:.4f}  worst {row['worst_view_psnr']:.2f}  V {nv} F {nf}")
        del mesh, scene; torch.cuda.empty_cache()
    best = max(rows, key=lambda r: r['val_psnr'])
    log(f"BEST iteration {best['iteration']}  val PSNR {best['val_psnr']:.3f}  (last saved {rows[-1]['iteration']}: {rows[-1]['val_psnr']:.3f})")
    summary = dict(version=VERSION, model=str(model), dataset=str(dataset), min_iter=a.min_iter, supersample=a.supersample, rows=rows,
                   best_iteration=best['iteration'], best_val_psnr=best['val_psnr'])
    out = Path(a.out) if a.out else model / 'val_select.json'
    out.write_text(json.dumps(summary, indent=1))
    log(f'wrote {out}')
    if a.best:
        b = Path(a.best).resolve()
        if b.exists():
            shutil.rmtree(b)
        b.mkdir(parents=True)
        for p in model.iterdir():
            if p.name in ('point_cloud', 'val_select.json'):
                continue
            os.symlink(p.resolve(), b / p.name)          # cfg_args, cameras.json, input.ply, train.log, rd_log.jsonl, ...
        (b / 'point_cloud').mkdir()
        os.symlink((pc / f"iteration_{best['iteration']}").resolve(), b / 'point_cloud' / 'iteration_30000')
        (b / 'VAL_SELECTED').write_text(f"{VERSION}\nsource {model}\niteration {best['iteration']}\nval_psnr {best['val_psnr']:.4f}\n")
        log(f"best model dir {b}: point_cloud/iteration_30000 -> iteration_{best['iteration']}")
    if a.delete_unselected:
        keep = {best['iteration'], 30000}
        for it in saved:
            if it not in keep:
                shutil.rmtree(pc / f'iteration_{it}')
        log(f'deleted unselected checkpoints; kept {sorted(keep)}')
    shutil.rmtree(shadow, ignore_errors=True)
    print(f"VAL_SELECT_RESULT {model.parent.name} best_iter {best['iteration']} val_psnr {best['val_psnr']:.3f} last_iter {rows[-1]['iteration']} last_val_psnr {rows[-1]['val_psnr']:.3f}")


if __name__ == '__main__':
    main()
PYEOF
python -c "import ast,sys; ast.parse(open('$AN/blender/val_select_v1.py').read()); print('val_select_v1.py ok')" || echo "FAILED: val_select_v1.py did not paste cleanly"

cat > $AN/originals-blender-v2.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=origb2
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/zr2245/analysis/origb2-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/origb2-%x-%j.err
# originals-blender-v2.sbatch (2026-11-04, VERSION v2): = originals-blender-v1 + checkpoints every 1000 its from 12000, val_select_v1.py picks the
# iteration by the official val split, the level-1.0 protocol runs on $G-best (point_cloud/iteration_30000 -> chosen checkpoint), unselected
# checkpoints deleted. Black background, no --indoor unless INDOOR=1. Output /scratch/zr2245/$SCENE-blender2 (+ -best), table originals-blender-val.txt.
#   sbatch --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1 --job-name=lego-origb2 /scratch/zr2245/analysis/originals-blender-v2.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=train; INDOOR=${INDOOR:-0}
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; G=${G:-/scratch/zr2245/$SCENE-blender2}; GB=$G-best
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; VS=$AN/blender/val_select_v1.py
K=${K:-4096}; R=${R:-256}; DC=${DC:-q7}; XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; TAB=$AN/originals-blender-val.txt
SAVES="12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000 30000"
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
C=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
echo "scene $SCENE | DATA $DATA | G $G | blender, black bg, val-selected | flag '$FLAG' | XYZ $XYZ | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $DATA/transforms_train.json $DATA/transforms_val.json $DATA/transforms_test.json $VS $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do
  [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
mkdir -p $G
if [ ! -f $C ]; then
  echo "=== train original (saves every 1k from 12k) -> $G/original ($(date)) ==="
  (cd $LAB/upstream && python train.py -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --test_iterations 7000 11000 14000 20000 25000 30000 --save_iterations $SAVES $FLAG > $G/train.log 2>&1) \
    || { echo "FAILED: TRAIN"; grep -n -m 8 -i 'error\|Traceback' $G/train.log; tail -20 $G/train.log; exit 1; }
  [ -f $C ] || { echo "FAILED: TRAIN produced no checkpoint"; tail -20 $G/train.log; exit 1; }
else echo "exists: $C"; fi
echo "--- test curve ---"; grep -h "Evaluating test" $G/train.log | sed 's/\[ITER \([0-9]*\)\] Evaluating test: L1 [0-9.]* PSNR \([0-9.]*\) SSIM \([0-9.]*\) LPIPS \([0-9.]*\).*/\1 \2 \3 \4/'
echo "=== val_select ($(date)) ==="
python $VS --lab $LAB --dataset $DATA --model $G/original --best $GB/original --delete_unselected > $G/val_select.log 2>&1 || { echo "FAILED: VAL_SELECT"; grep -n -m 8 -i 'error\|Traceback\|FAILED' $G/val_select.log; tail -15 $G/val_select.log; exit 1; }
grep -h "^\[val_select\] iter\|BEST\|VAL_SELECT_RESULT" $G/val_select.log
BEST=$(grep -h "VAL_SELECT_RESULT" $G/val_select.log | awk '{print $4}'); [ -n "$BEST" ] || { echo "FAILED: no VAL_SELECT_RESULT"; exit 1; }
[ -L $GB/original/point_cloud/iteration_30000 ] || { echo "FAILED: $GB/original/point_cloud/iteration_30000 symlink missing"; exit 1; }
echo "=== level-1.0 protocol on $GB (iteration $BEST) ($(date)) ==="
G=$GB DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $GB/level.log 2>&1 || { echo "FAILED: LEVEL-1.0"; grep -n -m 8 -i 'error\|Traceback' $GB/level.log; tail -12 $GB/level.log; exit 1; }
NA=$GB/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery"; tail -12 $GB/level.log; exit 1; }
python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; v=json.load(open('$G/original/val_select.json'))
last=[r for r in v['rows'] if r['iteration']==30000]
print('$SCENE ORIGINAL-BLENDER-VALSEL | best iter', v['best_iteration'], 'val', round(v['best_val_psnr'],3), '(30k val', round(last[0]['val_psnr'],3) if last else 'n/a', ') | vertices', b['vertices'], 'faces', b['faces'],
      '| level-1.0: finetune test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test', round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | blackbg $FLAG')
PY
echo "originals-blender-v2 $SCENE exit 0  $(date)"
SBEOF
j=$(sbatch --parsable --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1 --job-name=lego-origb2 $AN/originals-blender-v2.sbatch) && echo "submitted lego-origb2 (indoor) $j" || echo "FAILED: sbatch"
squeue -u $USER -o "%.10i %.16j %.28a %.2t %.10M %.6D %R"
```

看结果：

```bash
AN=/scratch/zr2245/analysis
ls -t $AN/origb2-lego-origb2-*.out | head -1 | xargs tail -40
cat $AN/originals-blender-val.txt 2>/dev/null
```

**通过标准**：val_select 打印 19 行 val PSNR，BEST 落在 13k-20k 附近，交付行 test ≈ 32-33。通过后再写 §5b（λ 版 rd-prune-blender-v2 + submit-blender-v2，8 场景 × 3 run）。

### §5b λ 版 + 8 场景提交（2026-11-04 05:20；`--indoor` 对照结果后定稿）

`--indoor` lego 曲线：峰 34.41 @ 11k，opaque 33.42 @ 15k，2× supersample 后 **34.12 @ 21k**，30k 30.38；457k V（无 indoor：594k V，30k 28.79）。bounded 场景超参全面更好 → **所有 Blender run 统一 `--indoor` + val 选点**。预期 lego val-best ≈ 34，黑底、无先验下距 3DGS 35.8 约 1.7 dB。
提交前提：§5a 的 lego-origb2（INDOOR=1）打印 exit 0 且 BEST 合理。

```bash
AN=/scratch/zr2245/analysis
cat > $AN/rd-prune-blender-v2.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=rd3b2
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/zr2245/analysis/rd3b2-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/rd3b2-%x-%j.err
# rd-prune-blender-v2.sbatch (2026-11-04, VERSION v2): = rd-prune-blender-v1 + --indoor (INDOOR=1 default) + checkpoints every 1k from 12k +
# val_select_v1.py (official val split) + level-1.0 protocol on $G-best. Black background. Output /scratch/zr2245/$SCENE-rd3b2-lam$L-mu$MU (+ -best),
# table rd-prune3b2-$SCENE.txt. Same lambda rule, prices BPV 10 BPF 2, prune schedule 12000 14000 16000 18000, same codec.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=train; LAMS=${LAMS:?set LAMS="0.00062 0.0062" via --export}; INDOOR=${INDOOR:-1}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}
SAVES="12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000 30000"
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; VS=$AN/blender/val_select_v1.py
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; TAB=$AN/rd-prune3b2-$SCENE.txt; TR=$AN/rdgrowth/train_rd_v3.py
echo "scene $SCENE | blender, black bg, val-selected | flag '$FLAG' | lams $LAMS | B/v $BPV B/face $BPF | prune at $PRUNE mu $MU max_frac $MAXF | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $VS $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/transforms_train.json $DATA/transforms_val.json $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
grep -q "'--indoor'" $TR || { echo "FAILED: $TR has no --indoor flag"; exit 1; }
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-rd3b2-lam$L-mu$MU; GB=$G-best; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== $SCENE lam $L mu $MU $FLAG: train+prune (saves every 1k from 12k) ($(date)) ==="
  if [ ! -f $CB ]; then
    python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF --test_iterations 7000 11000 14000 20000 25000 30000 --save_iterations $SAVES $FLAG > $G/train.log 2>&1 \
      || { echo "FAILED: TRAIN $SCENE lam $L"; grep -n -m 8 -i 'error\|Traceback' $G/train.log; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE lam $L"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  grep -h "^\[RD prune" $G/train.log
  echo "--- test curve ---"; grep -h "Evaluating test" $G/train.log | sed 's/\[ITER \([0-9]*\)\] Evaluating test: L1 [0-9.]* PSNR \([0-9.]*\) SSIM \([0-9.]*\) LPIPS \([0-9.]*\).*/\1 \2 \3 \4/'
  echo "=== $SCENE lam $L: val_select ($(date)) ==="
  python $VS --lab $LAB --dataset $DATA --model $G/original --best $GB/original --delete_unselected > $G/val_select.log 2>&1 || { echo "FAILED: VAL_SELECT $SCENE lam $L"; grep -n -m 8 -i 'error\|Traceback\|FAILED' $G/val_select.log; tail -15 $G/val_select.log; exit 1; }
  grep -h "BEST\|VAL_SELECT_RESULT" $G/val_select.log
  [ -L $GB/original/point_cloud/iteration_30000 ] || { echo "FAILED: $GB/original/point_cloud/iteration_30000 symlink missing"; exit 1; }
  echo "=== $SCENE lam $L mu $MU: level-1.0 protocol on $GB ($(date)) ==="
  G=$GB DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $GB/level.log 2>&1 \
    || { echo "FAILED: LEVEL-1.0 $SCENE lam $L"; tail -8 $GB/level.log; exit 1; }
  NA=$GB/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE lam $L"; tail -8 $GB/level.log; exit 1; }
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; h=f['history']; v=json.load(open('$G/original/val_select.json'))
ev=[json.loads(l) for l in open('$G/original/rd_log.jsonl')]; dl=[e for e in ev if e['event']=='delaunay']; pr=[e for e in ev if e['event']=='prune']
last=[r for r in v['rows'] if r['iteration']==30000]
print('$SCENE lam $L RD-GROWTH+PRUNE-mu$MU-BLENDER-VALSEL | best iter', v['best_iteration'], 'val', round(v['best_val_psnr'],3), '(30k val', round(last[0]['val_psnr'],3) if last else 'n/a', ') | vertices', b['vertices'], 'faces', b['faces'],
      '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3), 'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | grow: V@delaunay', dl[0]['vertices'] if dl else None,
      '| prune rounds', [(e['iteration'], e['vertices_before'], e['vertices_after'], 'pairs', e['n_pairs'], 'cap', e['max_pairs']) for e in pr], '| blackbg $FLAG')
PY
done
echo "===== $TAB ====="; cat $TAB
echo "rd-prune3b2 $SCENE exit 0  $(date)"
SBEOF

cat > $AN/submit-blender-v2.sh <<'SHEOF'
#!/usr/bin/env bash
# submit-blender-v2.sh (2026-11-04): 8 NeRF-Synthetic scenes x (1 original job + 1 lambda job of 2 lambdas 6.2e-4 / 6.2e-3), all --indoor + val-selected, black bg, general.
# ~ 8 x 0.6 h + 8 x 1.4 h ~ 16 GPU-h. A scene whose -blender2 original already exists (lego from §5a, if it was run with INDOOR=1) skips its training.
AN=/scratch/zr2245/analysis; XYZ=${XYZ:-q14}
for f in $AN/originals-blender-v2.sbatch $AN/rd-prune-blender-v2.sbatch $AN/blender/val_select_v1.py $AN/rdgrowth/train_rd_v3.py; do [ -f $f ] || { echo "FAILED: missing $f"; exit 1; }; done
n=0
for S in chair drums ficus hotdog lego materials mic ship; do
  [ -f /scratch/zr2245/datasets/nerf_synthetic/$S/transforms_val.json ] || { echo "FAILED: no dataset for $S"; continue; }
  [ -f /scratch/zr2245/$S-blender2/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] && echo "note: $S original exists, job will skip its training (check it was INDOOR=1)"
  j=$(sbatch --parsable --time=02:30:00 --export=ALL,SCENE=$S,XYZ=$XYZ,INDOOR=1 --job-name=$S-origb2 $AN/originals-blender-v2.sbatch) \
    && { echo "submitted $S-origb2 $j  XYZ $XYZ  indoor"; n=$((n+1)); } || echo "FAILED: sbatch $S original"
  LAMS="0.00062 0.0062"
  j=$(sbatch --parsable --time=03:30:00 --export=ALL,SCENE=$S,XYZ=$XYZ,INDOOR=1,LAMS="$LAMS",BPV=10.0,BPF=2.0,MU=1.0,MAXF=0.15 --job-name=$S-b2-62 $AN/rd-prune-blender-v2.sbatch) \
    && { echo "submitted $S-b2-62 $j  lams $LAMS  XYZ $XYZ  indoor"; n=$((n+1)); } || echo "FAILED: sbatch $S lams"
done
echo "submitted $n jobs (expected 16)"
squeue -u $USER -o "%.10i %.16j %.28a %.2t %.10M %.6D %R"
SHEOF
bash $AN/submit-blender-v2.sh
```

汇总：

```bash
AN=/scratch/zr2245/analysis
cat $AN/originals-blender-val.txt
cat $AN/rd-prune3b2-*.txt
grep -l "exit 0" $AN/origb2-*.out $AN/rd3b2-*.out 2>/dev/null | wc -l; grep -l "FAILED" $AN/origb2-*.out $AN/rd3b2-*.out 2>/dev/null
```

## §6 最终版（2026-11-04 07:30）：upstream 不存中间 checkpoint → train_ckpt_v1.py 包装 + v3 sbatch

**§5a 结果**：链路通，但 `--save_iterations` 在 upstream train.py 里是摆设（只有第 396 行最后一次 `scene.save`），train_rd_v3 同样。所以 val_select 只看到 30k。
**修法（不改 upstream）**：`analysis/blender/train_ckpt_v1.py` 包装原脚本，hook `TriangleModel.update_learning_rate(iteration)`（两个训练循环每个 iteration 都调一次），在指定 iteration 通过 `Scene.save` 存成交付链能读的格式。存的是第 N 步开始前的状态（差一步，可忽略）；中间 checkpoint 没做最后那次 importance prune（面数多几 %）。
**§5a / §5b 的 v2 文件作废**，用下面 v3（输出目录 `-blender3` / `-rd3b3-`，表 `originals-blender-val3.txt` / `rd-prune3b3-*.txt`）。

### §6a lego 一个 job（整块粘贴；INDOOR=1 已在提交行）

```bash
AN=/scratch/zr2245/analysis
mkdir -p $AN/blender
cat > $AN/blender/train_ckpt_v1.py <<'PYEOF'
"""train_ckpt_v1.py (VERSION train_ckpt_v1 2026-11-04): run an unmodified training script and save checkpoints at chosen iterations.

Why: upstream train.py accepts --save_iterations but never uses it (its only scene.save() is the final one), and our
train_rd_v3.py inherited that. The NeRF-Synthetic probe needs checkpoints every 1k from 12k for validation-split selection
(val_select_v1.py). No upstream file is edited: this wrapper hooks TriangleModel.update_learning_rate, which both training
loops call once per iteration with the iteration index, and saves through Scene.save (so the on-disk format is exactly the
one the delivery chain loads: point_cloud/iteration_N/point_cloud_state_dict.pt).

Note: the hook fires at the START of iteration N (before that iteration's optimizer step), so iteration_N holds the state after
N-1 steps; the "[ITER N] Evaluating" line in the log is after step N. Intermediate checkpoints also skip the final
importance-based triangle prune that the scripts apply before their own last save (a few % more faces; rendering unchanged).

Usage:
  python train_ckpt_v1.py --saves 12000 13000 ... --script /path/upstream/train.py      -- -s DATA -i train -r 1 --eval ... -m MODEL ...
  python train_ckpt_v1.py --saves 12000 13000 ... --script /path/rdgrowth/train_rd_v3.py -- --lab LAB -s DATA ... -m MODEL --lam ...
"""
import argparse, inspect, os, runpy, sys

VERSION = 'train_ckpt_v1 2026-11-04'


def main():
    if '--' not in sys.argv:
        sys.exit('FAILED: usage: train_ckpt_v1.py --saves N... --script PATH -- <script args>')
    k = sys.argv.index('--')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--saves', nargs='+', type=int, required=True)
    ap.add_argument('--script', required=True, help='upstream/train.py or analysis/rdgrowth/train_rd_v3.py')
    a = ap.parse_args(sys.argv[1:k]); rest = sys.argv[k + 1:]
    script = os.path.abspath(a.script)
    if not os.path.isfile(script):
        sys.exit(f'FAILED: script not found: {script}')
    # model path and upstream dir from the script's own arguments
    mp = None
    for i, t in enumerate(rest):
        if t in ('-m', '--model_path') and i + 1 < len(rest):
            mp = os.path.abspath(rest[i + 1])
    if mp is None:
        sys.exit('FAILED: no -m/--model_path in the script arguments')
    if '--lab' in rest:
        up = os.path.join(os.path.abspath(rest[rest.index('--lab') + 1]), 'upstream')
    else:
        up = os.path.dirname(script)
    if not os.path.isfile(os.path.join(up, 'train.py')):
        sys.exit(f'FAILED: upstream train.py not found under {up}')
    src = open(script).read()
    if 'update_learning_rate(iteration)' not in src:
        sys.exit(f'FAILED: {script} does not call update_learning_rate(iteration) once per iteration; hook would never fire')
    sys.path.insert(0, os.path.join(up, 'submodules/effrdel/src')); sys.path.insert(0, up); os.chdir(up)
    import scene.triangle_model as tm
    from scene import Scene
    if not hasattr(tm.TriangleModel, 'update_learning_rate'):
        sys.exit('FAILED: TriangleModel has no update_learning_rate')
    ssrc = inspect.getsource(Scene.save)
    print(f'[ckpt] {VERSION} | saves {sorted(a.saves)} | model_path {mp} | script {script}\n[ckpt] Scene.save source:\n{ssrc}', flush=True)
    if 'model_path' not in ssrc:
        sys.exit('FAILED: Scene.save does not use self.model_path; hook cannot reuse it')

    class _Fake:  # stands in for the Scene object inside Scene.save: model_path + whatever attribute holds the model
        def __init__(self, model_path, model):
            self.model_path = model_path; self._model = model
        def __getattr__(self, name):
            return self._model

    saves = set(a.saves); done = set(); orig = tm.TriangleModel.update_learning_rate

    def hooked(self, iteration):
        r = orig(self, iteration)
        if iteration in saves and iteration not in done:
            done.add(iteration)
            os.makedirs(os.path.join(mp, 'point_cloud', f'iteration_{iteration}'), exist_ok=True)
            out = Scene.save(_Fake(mp, self), iteration)
            nv = int(self.vertices.shape[0]) if hasattr(self, 'vertices') else -1
            print(f'[ckpt] saved iteration {iteration} (state after {iteration - 1} steps) vertices {nv} -> {out}', flush=True)
        return r

    tm.TriangleModel.update_learning_rate = hooked
    sys.argv = [script] + rest
    runpy.run_path(script, run_name='__main__')
    missing = sorted(i for i in saves if not os.path.isfile(os.path.join(mp, 'point_cloud', f'iteration_{i}', 'point_cloud_state_dict.pt')))
    if missing:
        sys.exit(f'FAILED: checkpoints missing after training: {missing}')
    print(f'[ckpt] all {len(saves)} checkpoints present', flush=True)


if __name__ == '__main__':
    main()
PYEOF
python -c "import ast; ast.parse(open('$AN/blender/train_ckpt_v1.py').read()); print('train_ckpt_v1.py ok')" || echo "FAILED: train_ckpt_v1.py did not paste cleanly"
[ -f $AN/blender/val_select_v1.py ] && echo "val_select_v1.py present" || echo "FAILED: val_select_v1.py missing (paste §5a's first heredoc)"

cat > $AN/originals-blender-v3.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=origb3
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/zr2245/analysis/origb3-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/origb3-%x-%j.err
# originals-blender-v3.sbatch (2026-11-04, VERSION v3): = v2 but checkpoints are written by train_ckpt_v1.py (upstream ignores --save_iterations), every 1000 its from 12000; val_select_v1.py picks the
# iteration by the official val split, the level-1.0 protocol runs on $G-best (point_cloud/iteration_30000 -> chosen checkpoint), unselected
# checkpoints deleted. Black background, no --indoor unless INDOOR=1. Output /scratch/zr2245/$SCENE-blender3 (+ -best), table originals-blender-val3.txt.
#   sbatch --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1 --job-name=lego-origb3 /scratch/zr2245/analysis/originals-blender-v3.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=train; INDOOR=${INDOOR:-0}
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; G=${G:-/scratch/zr2245/$SCENE-blender3}; GB=$G-best
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; VS=$AN/blender/val_select_v1.py; CK=$AN/blender/train_ckpt_v1.py
K=${K:-4096}; R=${R:-256}; DC=${DC:-q7}; XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; TAB=$AN/originals-blender-val3.txt
SAVES="12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000"
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
C=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
echo "scene $SCENE | DATA $DATA | G $G | blender, black bg, val-selected | flag '$FLAG' | XYZ $XYZ | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $DATA/transforms_train.json $DATA/transforms_val.json $DATA/transforms_test.json $VS $CK $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do
  [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
mkdir -p $G
if [ ! -f $C ]; then
  echo "=== train original (saves every 1k from 12k) -> $G/original ($(date)) ==="
  (cd $LAB/upstream && python $CK --saves $SAVES --script $LAB/upstream/train.py -- -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --test_iterations 7000 11000 14000 20000 25000 30000 $FLAG > $G/train.log 2>&1) \
    || { echo "FAILED: TRAIN"; grep -n -m 8 -i 'error\|Traceback' $G/train.log; tail -20 $G/train.log; exit 1; }
  [ -f $C ] || { echo "FAILED: TRAIN produced no checkpoint"; tail -20 $G/train.log; exit 1; }
else echo "exists: $C"; fi
grep -h "^\[ckpt\] saved\|^\[ckpt\] all\|FAILED" $G/train.log | tail -3
echo "--- test curve ---"; grep -h "Evaluating test" $G/train.log | sed 's/\[ITER \([0-9]*\)\] Evaluating test: L1 [0-9.]* PSNR \([0-9.]*\) SSIM \([0-9.]*\) LPIPS \([0-9.]*\).*/\1 \2 \3 \4/'
echo "=== val_select ($(date)) ==="
python $VS --lab $LAB --dataset $DATA --model $G/original --best $GB/original --delete_unselected > $G/val_select.log 2>&1 || { echo "FAILED: VAL_SELECT"; grep -n -m 8 -i 'error\|Traceback\|FAILED' $G/val_select.log; tail -15 $G/val_select.log; exit 1; }
grep -h "^\[val_select\] iter\|BEST\|VAL_SELECT_RESULT" $G/val_select.log
BEST=$(grep -h "VAL_SELECT_RESULT" $G/val_select.log | awk '{print $4}'); [ -n "$BEST" ] || { echo "FAILED: no VAL_SELECT_RESULT"; exit 1; }
[ -L $GB/original/point_cloud/iteration_30000 ] || { echo "FAILED: $GB/original/point_cloud/iteration_30000 symlink missing"; exit 1; }
echo "=== level-1.0 protocol on $GB (iteration $BEST) ($(date)) ==="
G=$GB DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $GB/level.log 2>&1 || { echo "FAILED: LEVEL-1.0"; grep -n -m 8 -i 'error\|Traceback' $GB/level.log; tail -12 $GB/level.log; exit 1; }
NA=$GB/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery"; tail -12 $GB/level.log; exit 1; }
python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; v=json.load(open('$G/original/val_select.json'))
last=[r for r in v['rows'] if r['iteration']==30000]
print('$SCENE ORIGINAL-BLENDER-VALSEL | best iter', v['best_iteration'], 'val', round(v['best_val_psnr'],3), '(30k val', round(last[0]['val_psnr'],3) if last else 'n/a', ') | vertices', b['vertices'], 'faces', b['faces'],
      '| level-1.0: finetune test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test', round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | blackbg $FLAG')
PY
echo "originals-blender-v3 $SCENE exit 0  $(date)"
SBEOF

rm -rf /scratch/zr2245/lego-blender3 /scratch/zr2245/lego-blender3-best
j=$(sbatch --parsable --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1 --job-name=lego-origb3 $AN/originals-blender-v3.sbatch) && echo "submitted lego-origb3 (indoor) $j" || echo "FAILED: sbatch"
squeue -u $USER -o "%.10i %.16j %.28a %.2t %.10M %.6D %R"
```

看结果：

```bash
AN=/scratch/zr2245/analysis
ls -t $AN/origb3-lego-origb3-*.out | head -1 | xargs tail -45
cat $AN/originals-blender-val3.txt 2>/dev/null
```

通过标准：train.log 里 `[ckpt] all 18 checkpoints present`；val_select 打印 19 行；BEST 在 13k-22k；交付 test ≈ 33-34。

### §6b 8 场景（§6a 通过后整块粘贴；16 job ≈ 16 GPU-h，general）

```bash
AN=/scratch/zr2245/analysis
cat > $AN/rd-prune-blender-v3.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=rd3b3
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/zr2245/analysis/rd3b3-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/rd3b3-%x-%j.err
# rd-prune-blender-v3.sbatch (2026-11-04, VERSION v3): = v2 but checkpoints written by train_ckpt_v1.py (train_rd_v3 ignores --save_iterations); --indoor (INDOOR=1 default), every 1k from 12k,
# val_select_v1.py (official val split) + level-1.0 protocol on $G-best. Black background. Output /scratch/zr2245/$SCENE-rd3b3-lam$L-mu$MU (+ -best),
# table rd-prune3b3-$SCENE.txt. Same lambda rule, prices BPV 10 BPF 2, prune schedule 12000 14000 16000 18000, same codec.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=train; LAMS=${LAMS:?set LAMS="0.00062 0.0062" via --export}; INDOOR=${INDOOR:-1}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}
SAVES="12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000"
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; VS=$AN/blender/val_select_v1.py; CK=$AN/blender/train_ckpt_v1.py
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; TAB=$AN/rd-prune3b3-$SCENE.txt; TR=$AN/rdgrowth/train_rd_v3.py
echo "scene $SCENE | blender, black bg, val-selected | flag '$FLAG' | lams $LAMS | B/v $BPV B/face $BPF | prune at $PRUNE mu $MU max_frac $MAXF | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $VS $CK $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/transforms_train.json $DATA/transforms_val.json $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
grep -q "'--indoor'" $TR || { echo "FAILED: $TR has no --indoor flag"; exit 1; }
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-rd3b3-lam$L-mu$MU; GB=$G-best; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== $SCENE lam $L mu $MU $FLAG: train+prune (saves every 1k from 12k) ($(date)) ==="
  if [ ! -f $CB ]; then
    python $CK --saves $SAVES --script $TR -- --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF --test_iterations 7000 11000 14000 20000 25000 30000 $FLAG > $G/train.log 2>&1 \
      || { echo "FAILED: TRAIN $SCENE lam $L"; grep -n -m 8 -i 'error\|Traceback' $G/train.log; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE lam $L"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  grep -h "^\[RD prune" $G/train.log; grep -h "^\[ckpt\] all\|FAILED" $G/train.log | tail -2
  echo "--- test curve ---"; grep -h "Evaluating test" $G/train.log | sed 's/\[ITER \([0-9]*\)\] Evaluating test: L1 [0-9.]* PSNR \([0-9.]*\) SSIM \([0-9.]*\) LPIPS \([0-9.]*\).*/\1 \2 \3 \4/'
  echo "=== $SCENE lam $L: val_select ($(date)) ==="
  python $VS --lab $LAB --dataset $DATA --model $G/original --best $GB/original --delete_unselected > $G/val_select.log 2>&1 || { echo "FAILED: VAL_SELECT $SCENE lam $L"; grep -n -m 8 -i 'error\|Traceback\|FAILED' $G/val_select.log; tail -15 $G/val_select.log; exit 1; }
  grep -h "BEST\|VAL_SELECT_RESULT" $G/val_select.log
  [ -L $GB/original/point_cloud/iteration_30000 ] || { echo "FAILED: $GB/original/point_cloud/iteration_30000 symlink missing"; exit 1; }
  echo "=== $SCENE lam $L mu $MU: level-1.0 protocol on $GB ($(date)) ==="
  G=$GB DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $GB/level.log 2>&1 \
    || { echo "FAILED: LEVEL-1.0 $SCENE lam $L"; tail -8 $GB/level.log; exit 1; }
  NA=$GB/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE lam $L"; tail -8 $GB/level.log; exit 1; }
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; h=f['history']; v=json.load(open('$G/original/val_select.json'))
ev=[json.loads(l) for l in open('$G/original/rd_log.jsonl')]; dl=[e for e in ev if e['event']=='delaunay']; pr=[e for e in ev if e['event']=='prune']
last=[r for r in v['rows'] if r['iteration']==30000]
print('$SCENE lam $L RD-GROWTH+PRUNE-mu$MU-BLENDER-VALSEL | best iter', v['best_iteration'], 'val', round(v['best_val_psnr'],3), '(30k val', round(last[0]['val_psnr'],3) if last else 'n/a', ') | vertices', b['vertices'], 'faces', b['faces'],
      '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3), 'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | grow: V@delaunay', dl[0]['vertices'] if dl else None,
      '| prune rounds', [(e['iteration'], e['vertices_before'], e['vertices_after'], 'pairs', e['n_pairs'], 'cap', e['max_pairs']) for e in pr], '| blackbg $FLAG')
PY
done
echo "===== $TAB ====="; cat $TAB
echo "rd-prune3b2 $SCENE exit 0  $(date)"
SBEOF

cat > $AN/submit-blender-v3.sh <<'SHEOF'
#!/usr/bin/env bash
# submit-blender-v3.sh (2026-11-04): 8 NeRF-Synthetic scenes x (1 original job + 1 lambda job of 2 lambdas 6.2e-4 / 6.2e-3), all --indoor + val-selected, black bg, general.
# ~ 8 x 0.6 h + 8 x 1.4 h ~ 16 GPU-h. A scene whose -blender2 original already exists (lego from §6a) skips its training.
AN=/scratch/zr2245/analysis; XYZ=${XYZ:-q14}
for f in $AN/originals-blender-v3.sbatch $AN/rd-prune-blender-v3.sbatch $AN/blender/val_select_v1.py $AN/blender/train_ckpt_v1.py $AN/rdgrowth/train_rd_v3.py; do [ -f $f ] || { echo "FAILED: missing $f"; exit 1; }; done
n=0
for S in chair drums ficus hotdog lego materials mic ship; do
  [ -f /scratch/zr2245/datasets/nerf_synthetic/$S/transforms_val.json ] || { echo "FAILED: no dataset for $S"; continue; }
  [ -f /scratch/zr2245/$S-blender3/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] && echo "note: $S original exists, job will skip its training (check it was INDOOR=1)"
  j=$(sbatch --parsable --time=02:30:00 --export=ALL,SCENE=$S,XYZ=$XYZ,INDOOR=1 --job-name=$S-origb3 $AN/originals-blender-v3.sbatch) \
    && { echo "submitted $S-origb3 $j  XYZ $XYZ  indoor"; n=$((n+1)); } || echo "FAILED: sbatch $S original"
  LAMS="0.00062 0.0062"
  j=$(sbatch --parsable --time=03:30:00 --export=ALL,SCENE=$S,XYZ=$XYZ,INDOOR=1,LAMS="$LAMS",BPV=10.0,BPF=2.0,MU=1.0,MAXF=0.15 --job-name=$S-b3-62 $AN/rd-prune-blender-v3.sbatch) \
    && { echo "submitted $S-b3-62 $j  lams $LAMS  XYZ $XYZ  indoor"; n=$((n+1)); } || echo "FAILED: sbatch $S lams"
done
echo "submitted $n jobs (expected 16)"
squeue -u $USER -o "%.10i %.16j %.28a %.2t %.10M %.6D %R"
SHEOF

bash $AN/submit-blender-v3.sh
```

汇总：

```bash
AN=/scratch/zr2245/analysis
cat $AN/originals-blender-val3.txt
cat $AN/rd-prune3b3-*.txt
grep -l "exit 0" $AN/origb3-*.out $AN/rd3b3-*.out 2>/dev/null | wc -l; grep -l "FAILED" $AN/origb3-*.out $AN/rd3b3-*.out 2>/dev/null
```

## §7 v4（2026-11-04 08:10）：中间 checkpoint 缺最后的 importance prune → train_ckpt_v2 + val_select_v2

**§6a 结果**：全链通。lego `--indoor` val 选 21k：val 34.81，交付 test **33.02**（3DGS 35.8，黑底无先验差 2.8 dB）。但 21k 的 checkpoint 有 2.98 M 面 / 508k 顶点，30k 最终只有 1.32 M / 457k：训练脚本在最后一次 save 前做了 importance ≤ 0.5 的面剪枝（删 56 % 的面，都是看不见的背面/被遮挡面），中间 checkpoint 没做 → 交付 10.0 MB 而不是 ~6.5 MB。
**修法**：train_ckpt_v2 每个 checkpoint 旁边存 `importance_score.pt`；val_select_v2 把选中的 checkpoint 按同一规则（importance > 0.5 保留，删掉无面顶点）写成真正的 state dict（不再是 symlink），再渲一次 val 自检（掉 > 0.5 dB 就 FAILED，防 mask 反了）。
**提交策略**：直接跑 8 场景（v4，含 lego 重训 0.4 GPU-h）。val_select 失败不浪费 GPU：训练产物保留，修好脚本后重跑同一 sbatch 会跳过训练。

### §7a 写两个脚本（整块粘贴，不提交任何 job）

```bash
AN=/scratch/zr2245/analysis
mkdir -p $AN/blender
cat > $AN/blender/train_ckpt_v2.py <<'PYEOF'
"""train_ckpt_v2.py (VERSION train_ckpt_v2 2026-11-04): run an unmodified training script and save checkpoints at chosen iterations.

v2: also saves the model's per-face importance_score next to each checkpoint (importance_score.pt), so val_select_v2 can apply the
same final importance prune the training scripts apply before their own last save (lego: 2.98 M -> 1.32 M faces).

Why: upstream train.py accepts --save_iterations but never uses it (its only scene.save() is the final one), and our
train_rd_v3.py inherited that. The NeRF-Synthetic probe needs checkpoints every 1k from 12k for validation-split selection
(val_select_v1.py). No upstream file is edited: this wrapper hooks TriangleModel.update_learning_rate, which both training
loops call once per iteration with the iteration index, and saves through Scene.save (so the on-disk format is exactly the
one the delivery chain loads: point_cloud/iteration_N/point_cloud_state_dict.pt).

Note: the hook fires at the START of iteration N (before that iteration's optimizer step), so iteration_N holds the state after
N-1 steps; the "[ITER N] Evaluating" line in the log is after step N. Intermediate checkpoints also skip the final
importance-based triangle prune that the scripts apply before their own last save (a few % more faces; rendering unchanged).

Usage:
  python train_ckpt_v1.py --saves 12000 13000 ... --script /path/upstream/train.py      -- -s DATA -i train -r 1 --eval ... -m MODEL ...
  python train_ckpt_v1.py --saves 12000 13000 ... --script /path/rdgrowth/train_rd_v3.py -- --lab LAB -s DATA ... -m MODEL --lam ...
"""
import argparse, inspect, os, runpy, sys

VERSION = 'train_ckpt_v2 2026-11-04'


def main():
    if '--' not in sys.argv:
        sys.exit('FAILED: usage: train_ckpt_v2.py --saves N... --script PATH -- <script args>')
    k = sys.argv.index('--')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--saves', nargs='+', type=int, required=True)
    ap.add_argument('--script', required=True, help='upstream/train.py or analysis/rdgrowth/train_rd_v3.py')
    a = ap.parse_args(sys.argv[1:k]); rest = sys.argv[k + 1:]
    script = os.path.abspath(a.script)
    if not os.path.isfile(script):
        sys.exit(f'FAILED: script not found: {script}')
    # model path and upstream dir from the script's own arguments
    mp = None
    for i, t in enumerate(rest):
        if t in ('-m', '--model_path') and i + 1 < len(rest):
            mp = os.path.abspath(rest[i + 1])
    if mp is None:
        sys.exit('FAILED: no -m/--model_path in the script arguments')
    if '--lab' in rest:
        up = os.path.join(os.path.abspath(rest[rest.index('--lab') + 1]), 'upstream')
    else:
        up = os.path.dirname(script)
    if not os.path.isfile(os.path.join(up, 'train.py')):
        sys.exit(f'FAILED: upstream train.py not found under {up}')
    src = open(script).read()
    if 'update_learning_rate(iteration)' not in src:
        sys.exit(f'FAILED: {script} does not call update_learning_rate(iteration) once per iteration; hook would never fire')
    sys.path.insert(0, os.path.join(up, 'submodules/effrdel/src')); sys.path.insert(0, up); os.chdir(up)
    import scene.triangle_model as tm
    from scene import Scene
    if not hasattr(tm.TriangleModel, 'update_learning_rate'):
        sys.exit('FAILED: TriangleModel has no update_learning_rate')
    ssrc = inspect.getsource(Scene.save)
    print(f'[ckpt] {VERSION} | saves {sorted(a.saves)} | model_path {mp} | script {script}\n[ckpt] Scene.save source:\n{ssrc}', flush=True)
    if 'model_path' not in ssrc:
        sys.exit('FAILED: Scene.save does not use self.model_path; hook cannot reuse it')

    class _Fake:  # stands in for the Scene object inside Scene.save: model_path + whatever attribute holds the model
        def __init__(self, model_path, model):
            self.model_path = model_path; self._model = model
        def __getattr__(self, name):
            return self._model

    saves = set(a.saves); done = set(); orig = tm.TriangleModel.update_learning_rate

    def hooked(self, iteration):
        r = orig(self, iteration)
        if iteration in saves and iteration not in done:
            done.add(iteration)
            os.makedirs(os.path.join(mp, 'point_cloud', f'iteration_{iteration}'), exist_ok=True)
            out = Scene.save(_Fake(mp, self), iteration)
            nv = int(self.vertices.shape[0]) if hasattr(self, 'vertices') else -1
            imp = getattr(self, 'importance_score', None)
            if imp is None:
                sys.exit('FAILED: model has no importance_score attribute; cannot reproduce the final importance prune')
            import torch
            torch.save(imp.detach().cpu().clone(), os.path.join(mp, 'point_cloud', f'iteration_{iteration}', 'importance_score.pt'))
            nf = int(self._triangle_indices.shape[0]); nkeep = int((imp.detach().cpu().reshape(-1) > 0.5).sum())
            print(f'[ckpt] saved iteration {iteration} (state after {iteration - 1} steps) vertices {nv} faces {nf} (importance>0.5: {nkeep}) -> {out}', flush=True)
        return r

    tm.TriangleModel.update_learning_rate = hooked
    sys.argv = [script] + rest
    runpy.run_path(script, run_name='__main__')
    missing = sorted(i for i in saves if not os.path.isfile(os.path.join(mp, 'point_cloud', f'iteration_{i}', 'point_cloud_state_dict.pt')))
    if missing:
        sys.exit(f'FAILED: checkpoints missing after training: {missing}')
    missing = sorted(i for i in saves if not os.path.isfile(os.path.join(mp, 'point_cloud', f'iteration_{i}', 'importance_score.pt')))
    if missing:
        sys.exit(f'FAILED: importance_score.pt missing after training: {missing}')
    print(f'[ckpt] all {len(saves)} checkpoints present', flush=True)


if __name__ == '__main__':
    main()
PYEOF
cat > $AN/blender/val_select_v2.py <<'PYEOF'
"""val_select_v2.py (VERSION val_select_v2 2026-11-04): checkpoint selection by the OFFICIAL NeRF-Synthetic validation split.

v2: the chosen checkpoint is materialised as a REAL pruned state dict (not a symlink): faces with importance_score <= thresh
(importance_score.pt written by train_ckpt_v2) and the vertices they leave unused are dropped, reproducing the final prune the
training scripts apply before their last save. The pruned mesh is re-rendered on val and must stay within --prune_tol dB of
the unpruned one (self-check against an inverted mask).

Why: on NeRF-Synthetic (100 training views, no normal/depth priors in the Blender loader) MeshSplatting's test PSNR peaks
after the mesh stage and then drifts down while train PSNR keeps rising (lego: 32.9 @ 14k -> 28.8 @ 30k).  Early stopping
on the test split would be cheating; the dataset ships transforms_val.json (100 views), which upstream never reads.

What: builds a shadow dataset dir whose transforms_test.json is transforms_val.json (symlinks, nothing copied), loads each
saved checkpoint through upstream's own Scene(load_iteration=N) so no loading code is duplicated, renders the val views,
reports val PSNR/SSIM per iteration, picks the best, and (optionally) materialises <model>-best/original whose
point_cloud/iteration_30000 symlinks to the chosen checkpoint so the unchanged level-1.0 delivery chain can consume it.
Background is black (white_background=False), matching the delivery chain.

Usage (on a GPU node):
  python val_select_v2.py --lab $LAB --dataset $DATA --model $G/original --best $G-best/original [--delete_unselected]
"""
import argparse, json, os, re, shutil, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

VERSION = 'val_select_v2 2026-11-04'


def log(*a):
    print('[val_select]', *a, flush=True)


def make_shadow(dataset: Path, shadow: Path):
    """dataset dir with transforms_test.json -> transforms_val.json; everything else symlinked."""
    if shadow.exists():
        shutil.rmtree(shadow)
    shadow.mkdir(parents=True)
    for p in dataset.iterdir():
        if p.name == 'transforms_test.json':
            continue
        os.symlink(p.resolve(), shadow / p.name)
    os.symlink((dataset / 'transforms_val.json').resolve(), shadow / 'transforms_test.json')
    return shadow


def prune_state(sd, imp, thresh):
    """drop faces with importance <= thresh and vertices no kept face uses; every per-face / per-vertex tensor is filtered by shape."""
    faces = sd['_triangle_indices']; nf = int(faces.shape[0]); nv = int(sd['triangles_points'].shape[0])
    imp = imp.reshape(-1)
    if int(imp.shape[0]) != nf:
        sys.exit(f'FAILED: importance_score has {int(imp.shape[0])} entries, mesh has {nf} faces')
    if nf == nv:
        sys.exit('FAILED: faces == vertices; shape-based filtering is ambiguous')
    keep = imp > thresh
    used = torch.zeros(nv, dtype=torch.bool); used[faces[keep].reshape(-1).long()] = True
    remap = torch.full((nv,), -1, dtype=torch.long); remap[used] = torch.arange(int(used.sum()))
    out = {}
    for k, t in sd.items():
        if torch.is_tensor(t) and t.dim() >= 1 and k != '_triangle_indices':
            if int(t.shape[0]) == nf:
                out[k] = t[keep].clone()
            elif int(t.shape[0]) == nv:
                out[k] = t[used].clone()
            else:
                out[k] = t
        else:
            out[k] = t
    out['_triangle_indices'] = remap[faces[keep].long()].to(faces.dtype)
    assert int(out['_triangle_indices'].min()) >= 0
    return out, int(keep.sum()), int(used.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lab', required=True, help='root containing upstream/')
    ap.add_argument('--dataset', required=True, help='NeRF-Synthetic scene dir (transforms_{train,val,test}.json)')
    ap.add_argument('--model', required=True, help='model dir holding point_cloud/iteration_*/ (e.g. $G/original)')
    ap.add_argument('--iters', nargs='*', type=int, default=None, help='iterations to score (default: every saved iteration)')
    ap.add_argument('--min_iter', type=int, default=12000, help='ignore checkpoints before this (soft stage; not deliverable as an opaque mesh)')
    ap.add_argument('--supersample', type=str, default='20000:2,25000:4', help='render supersampling by iteration threshold, mirrors training (1 below the first threshold)')
    ap.add_argument('--best', type=str, default=None, help='write <best>/point_cloud/iteration_30000 -> chosen checkpoint (+ symlinks to the other model files)')
    ap.add_argument('--delete_unselected', action='store_true', help='after choosing, delete the other saved iterations except the chosen one and iteration_30000')
    ap.add_argument('--out', type=str, default=None, help='summary json (default <model>/val_select.json)')
    ap.add_argument('--importance_thresh', type=float, default=0.5, help='keep faces with importance_score > thresh (training scripts: <= 0.5 pruned)')
    ap.add_argument('--prune_tol', type=float, default=0.5, help='FAIL if the pruned mesh loses more than this on val (dB)')
    a = ap.parse_args()
    if not torch.cuda.is_available():
        ap.error('CUDA required')
    lab = Path(a.lab).resolve(); upstream = lab / 'upstream'
    sys.path.insert(0, str(upstream / 'submodules/effrdel/src')); sys.path.insert(0, str(upstream))
    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render
    from utils.loss_utils import ssim

    model = Path(a.model).resolve(); dataset = Path(a.dataset).resolve()
    pc = model / 'point_cloud'
    saved = sorted(int(m.group(1)) for d in pc.iterdir() for m in [re.fullmatch(r'iteration_(\d+)', d.name)] if m and (d / 'point_cloud_state_dict.pt').exists())
    iters = a.iters if a.iters else [i for i in saved if i >= a.min_iter]
    missing = [i for i in iters if i not in saved]
    if missing:
        sys.exit(f'FAILED: checkpoints missing for iterations {missing}; saved = {saved}')
    if not iters:
        sys.exit(f'FAILED: no checkpoint >= {a.min_iter}; saved = {saved}')
    sched = sorted((int(k), int(v)) for k, v in (kv.split(':') for kv in a.supersample.split(',') if kv))

    def ss_of(it):
        s = 1
        for thr, v in sched:
            if it >= thr:
                s = v
        return s

    shadow = make_shadow(dataset, model.parent / f'{model.name}-valshadow')
    data = SimpleNamespace(source_path=str(shadow), model_path=str(model), images='train', resolution=1, eval=True,
                           white_background=False, data_device='cpu', sh_degree=3)
    bg = torch.zeros(3, device='cuda'); pipe = SimpleNamespace(debug=False, convert_SHs_python=False, depth_ratio=1.)
    def eval_model(model_dir, it, ss):
        d = SimpleNamespace(**{**vars(data), 'model_path': str(model_dir)})
        mesh = TriangleModel(3)
        scene = Scene(d, mesh, None, None, load_iteration=it, shuffle=False)
        cams = sorted(scene.getTestCameras(), key=lambda c: (c.image_name, c.uid))
        mesh.scaling = ss
        ps, ss_ = [], []
        with torch.no_grad():
            for c in cams:
                im = render(c, mesh, pipe, bg)['render'].clamp(0, 1); gt = c.original_image[:3].cuda().clamp(0, 1)
                ps.append(float(-10 * torch.log10((im - gt).square().mean().clamp_min(1e-12)))); ss_.append(float(ssim(im, gt)))
        nv = int(mesh.vertices.shape[0]) if hasattr(mesh, 'vertices') else -1
        nf = int(mesh._triangle_indices.shape[0])
        del mesh, scene; torch.cuda.empty_cache()
        return dict(val_psnr=float(np.mean(ps)), val_ssim=float(np.mean(ss_)), worst_view_psnr=float(min(ps)), vertices=nv, faces=nf, views=len(cams))

    rows = []
    for it in iters:
        r = eval_model(model, it, ss_of(it))
        if it == iters[0]:
            log(f"val views: {r['views']} (from transforms_val.json via shadow dir {shadow})")
        row = dict(iteration=it, supersample=ss_of(it), **{k: r[k] for k in ('val_psnr', 'val_ssim', 'worst_view_psnr', 'vertices', 'faces')})
        rows.append(row)
        log(f"iter {it:6d}  ss {row['supersample']}  val PSNR {row['val_psnr']:.3f}  SSIM {row['val_ssim']:.4f}  worst {row['worst_view_psnr']:.2f}  V {row['vertices']} F {row['faces']}")
    best = max(rows, key=lambda r: r['val_psnr'])
    log(f"BEST iteration {best['iteration']}  val PSNR {best['val_psnr']:.3f}  (last saved {rows[-1]['iteration']}: {rows[-1]['val_psnr']:.3f})")
    summary = dict(version=VERSION, model=str(model), dataset=str(dataset), min_iter=a.min_iter, supersample=a.supersample, rows=rows,
                   best_iteration=best['iteration'], best_val_psnr=best['val_psnr'])
    out_path = Path(a.out) if a.out else model / 'val_select.json'
    out_path.write_text(json.dumps(summary, indent=1))
    log(f'wrote {out_path}')
    if a.best:
        b = Path(a.best).resolve()
        if b.exists():
            shutil.rmtree(b)
        b.mkdir(parents=True)
        for p in model.iterdir():
            if p.name in ('point_cloud', 'val_select.json'):
                continue
            os.symlink(p.resolve(), b / p.name)          # cfg_args, cameras.json, input.ply, train.log, rd_log.jsonl, ...
        src_dir = pc / f"iteration_{best['iteration']}"
        dst_dir = b / 'point_cloud' / 'iteration_30000'; dst_dir.mkdir(parents=True)
        imp_path = src_dir / 'importance_score.pt'
        if best['iteration'] == 30000 or not imp_path.exists():
            if best['iteration'] != 30000:
                sys.exit(f'FAILED: {imp_path} missing; train with train_ckpt_v2 so the final importance prune can be reproduced')
            for p in src_dir.iterdir():
                os.symlink(p.resolve(), dst_dir / p.name)
            log('best is the final checkpoint (already pruned by the training script): linked as is')
            pruned = None
        else:
            sd = torch.load(src_dir / 'point_cloud_state_dict.pt', map_location='cpu', weights_only=False)
            imp = torch.load(imp_path, map_location='cpu', weights_only=False)
            out, nf2, nv2 = prune_state(sd, imp, a.importance_thresh)
            torch.save(out, dst_dir / 'point_cloud_state_dict.pt')
            for p in src_dir.iterdir():
                if p.name not in ('point_cloud_state_dict.pt', 'importance_score.pt'):
                    os.symlink(p.resolve(), dst_dir / p.name)
            log(f"pruned best checkpoint: faces {best['faces']} -> {nf2}, vertices {best['vertices']} -> {nv2} (importance > {a.importance_thresh})")
            pruned = eval_model(b, 30000, ss_of(best['iteration']))
            log(f"pruned val PSNR {pruned['val_psnr']:.3f} (unpruned {best['val_psnr']:.3f})  V {pruned['vertices']} F {pruned['faces']}")
            if pruned['val_psnr'] < best['val_psnr'] - a.prune_tol:
                sys.exit(f"FAILED: pruned mesh lost {best['val_psnr'] - pruned['val_psnr']:.2f} dB on val (> {a.prune_tol}); check importance mask semantics")
        summary['pruned'] = pruned
        out_path.write_text(json.dumps(summary, indent=1))
        (b / 'VAL_SELECTED').write_text(f"{VERSION}\nsource {model}\niteration {best['iteration']}\nval_psnr {best['val_psnr']:.4f}\npruned {json.dumps(pruned)}\n")
        log(f"best model dir {b}: point_cloud/iteration_30000 = pruned iteration_{best['iteration']}")
    if a.delete_unselected:
        keep = {best['iteration'], 30000}
        for it in saved:
            if it not in keep:
                shutil.rmtree(pc / f'iteration_{it}')
        log(f'deleted unselected checkpoints; kept {sorted(keep)}')
    shutil.rmtree(shadow, ignore_errors=True)
    pr = summary.get('pruned')
    print(f"VAL_SELECT_RESULT {model.parent.name} best_iter {best['iteration']} val_psnr {best['val_psnr']:.3f} last_iter {rows[-1]['iteration']} last_val_psnr {rows[-1]['val_psnr']:.3f} pruned_val {pr['val_psnr']:.3f} pruned_faces {pr['faces']}" if pr else f"VAL_SELECT_RESULT {model.parent.name} best_iter {best['iteration']} val_psnr {best['val_psnr']:.3f} last_iter {rows[-1]['iteration']} last_val_psnr {rows[-1]['val_psnr']:.3f} pruned_val n/a pruned_faces {best['faces']}")


if __name__ == '__main__':
    main()
PYEOF
python -c "import ast; ast.parse(open('$AN/blender/train_ckpt_v2.py').read()); ast.parse(open('$AN/blender/val_select_v2.py').read()); print('train_ckpt_v2.py + val_select_v2.py ok')" || echo "FAILED: a script did not paste cleanly"
```

### §7b 三个 sbatch/sh + 提交 8 场景（整块粘贴；16 job ≈ 16 GPU-h，general）

```bash
AN=/scratch/zr2245/analysis
cat > $AN/originals-blender-v4.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=origb4
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/zr2245/analysis/origb4-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/origb4-%x-%j.err
# originals-blender-v4.sbatch (2026-11-04, VERSION v4): = v3 + train_ckpt_v2 saves importance_score.pt per checkpoint and val_select_v2 materialises the chosen checkpoint with the final importance prune applied (real file, not a symlink); checkpoints written by the wrapper (upstream ignores --save_iterations), every 1000 its from 12000; val_select_v2.py picks the
# iteration by the official val split, the level-1.0 protocol runs on $G-best (point_cloud/iteration_30000 -> chosen checkpoint), unselected
# checkpoints deleted. Black background, no --indoor unless INDOOR=1. Output /scratch/zr2245/$SCENE-blender4 (+ -best), table originals-blender-val4.txt.
#   sbatch --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1 --job-name=lego-origb4 /scratch/zr2245/analysis/originals-blender-v4.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=train; INDOOR=${INDOOR:-0}
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; G=${G:-/scratch/zr2245/$SCENE-blender4}; GB=$G-best
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; VS=$AN/blender/val_select_v2.py; CK=$AN/blender/train_ckpt_v2.py
K=${K:-4096}; R=${R:-256}; DC=${DC:-q7}; XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; TAB=$AN/originals-blender-val4.txt
SAVES="12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000"
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
C=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
echo "scene $SCENE | DATA $DATA | G $G | blender, black bg, val-selected | flag '$FLAG' | XYZ $XYZ | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $DATA/transforms_train.json $DATA/transforms_val.json $DATA/transforms_test.json $VS $CK $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do
  [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
mkdir -p $G
if [ ! -f $C ]; then
  echo "=== train original (saves every 1k from 12k) -> $G/original ($(date)) ==="
  (cd $LAB/upstream && python $CK --saves $SAVES --script $LAB/upstream/train.py -- -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --test_iterations 7000 11000 14000 20000 25000 30000 $FLAG > $G/train.log 2>&1) \
    || { echo "FAILED: TRAIN"; grep -n -m 8 -i 'error\|Traceback' $G/train.log; tail -20 $G/train.log; exit 1; }
  [ -f $C ] || { echo "FAILED: TRAIN produced no checkpoint"; tail -20 $G/train.log; exit 1; }
else echo "exists: $C"; fi
grep -h "^\[ckpt\] saved\|^\[ckpt\] all\|FAILED" $G/train.log | tail -3
echo "--- test curve ---"; grep -h "Evaluating test" $G/train.log | sed 's/\[ITER \([0-9]*\)\] Evaluating test: L1 [0-9.]* PSNR \([0-9.]*\) SSIM \([0-9.]*\) LPIPS \([0-9.]*\).*/\1 \2 \3 \4/'
echo "=== val_select ($(date)) ==="
python $VS --lab $LAB --dataset $DATA --model $G/original --best $GB/original --delete_unselected > $G/val_select.log 2>&1 || { echo "FAILED: VAL_SELECT"; grep -n -m 8 -i 'error\|Traceback\|FAILED' $G/val_select.log; tail -15 $G/val_select.log; exit 1; }
grep -h "^\[val_select\] iter\|BEST\|VAL_SELECT_RESULT" $G/val_select.log
BEST=$(grep -h "VAL_SELECT_RESULT" $G/val_select.log | awk '{print $4}'); [ -n "$BEST" ] || { echo "FAILED: no VAL_SELECT_RESULT"; exit 1; }
[ -L $GB/original/point_cloud/iteration_30000 ] || { echo "FAILED: $GB/original/point_cloud/iteration_30000 symlink missing"; exit 1; }
echo "=== level-1.0 protocol on $GB (iteration $BEST) ($(date)) ==="
G=$GB DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $GB/level.log 2>&1 || { echo "FAILED: LEVEL-1.0"; grep -n -m 8 -i 'error\|Traceback' $GB/level.log; tail -12 $GB/level.log; exit 1; }
NA=$GB/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery"; tail -12 $GB/level.log; exit 1; }
python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; v=json.load(open('$G/original/val_select.json'))
last=[r for r in v['rows'] if r['iteration']==30000]
print('$SCENE ORIGINAL-BLENDER-VALSEL | best iter', v['best_iteration'], 'val', round(v['best_val_psnr'],3), '(30k val', round(last[0]['val_psnr'],3) if last else 'n/a', ') | vertices', b['vertices'], 'faces', b['faces'],
      '| level-1.0: finetune test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test', round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | blackbg $FLAG')
PY
echo "originals-blender-v4 $SCENE exit 0  $(date)"
SBEOF

cat > $AN/rd-prune-blender-v4.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=rd3b4
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/zr2245/analysis/rd3b4-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/rd3b4-%x-%j.err
# rd-prune-blender-v4.sbatch (2026-11-04, VERSION v4): = v3 + importance_score.pt per checkpoint + pruned materialisation in val_select_v2 (train_rd_v3 ignores --save_iterations); --indoor (INDOOR=1 default), every 1k from 12k,
# val_select_v2.py (official val split) + level-1.0 protocol on $G-best. Black background. Output /scratch/zr2245/$SCENE-rd3b4-lam$L-mu$MU (+ -best),
# table rd-prune3b4-$SCENE.txt. Same lambda rule, prices BPV 10 BPF 2, prune schedule 12000 14000 16000 18000, same codec.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=train; LAMS=${LAMS:?set LAMS="0.00062 0.0062" via --export}; INDOOR=${INDOOR:-1}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}
SAVES="12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000"
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; VS=$AN/blender/val_select_v2.py; CK=$AN/blender/train_ckpt_v2.py
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; TAB=$AN/rd-prune3b4-$SCENE.txt; TR=$AN/rdgrowth/train_rd_v3.py
echo "scene $SCENE | blender, black bg, val-selected | flag '$FLAG' | lams $LAMS | B/v $BPV B/face $BPF | prune at $PRUNE mu $MU max_frac $MAXF | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $VS $CK $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/transforms_train.json $DATA/transforms_val.json $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
grep -q "'--indoor'" $TR || { echo "FAILED: $TR has no --indoor flag"; exit 1; }
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-rd3b4-lam$L-mu$MU; GB=$G-best; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== $SCENE lam $L mu $MU $FLAG: train+prune (saves every 1k from 12k) ($(date)) ==="
  if [ ! -f $CB ]; then
    python $CK --saves $SAVES --script $TR -- --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF --test_iterations 7000 11000 14000 20000 25000 30000 $FLAG > $G/train.log 2>&1 \
      || { echo "FAILED: TRAIN $SCENE lam $L"; grep -n -m 8 -i 'error\|Traceback' $G/train.log; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE lam $L"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  grep -h "^\[RD prune" $G/train.log; grep -h "^\[ckpt\] all\|FAILED" $G/train.log | tail -2
  echo "--- test curve ---"; grep -h "Evaluating test" $G/train.log | sed 's/\[ITER \([0-9]*\)\] Evaluating test: L1 [0-9.]* PSNR \([0-9.]*\) SSIM \([0-9.]*\) LPIPS \([0-9.]*\).*/\1 \2 \3 \4/'
  echo "=== $SCENE lam $L: val_select ($(date)) ==="
  python $VS --lab $LAB --dataset $DATA --model $G/original --best $GB/original --delete_unselected > $G/val_select.log 2>&1 || { echo "FAILED: VAL_SELECT $SCENE lam $L"; grep -n -m 8 -i 'error\|Traceback\|FAILED' $G/val_select.log; tail -15 $G/val_select.log; exit 1; }
  grep -h "BEST\|VAL_SELECT_RESULT" $G/val_select.log
  [ -L $GB/original/point_cloud/iteration_30000 ] || { echo "FAILED: $GB/original/point_cloud/iteration_30000 symlink missing"; exit 1; }
  echo "=== $SCENE lam $L mu $MU: level-1.0 protocol on $GB ($(date)) ==="
  G=$GB DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $GB/level.log 2>&1 \
    || { echo "FAILED: LEVEL-1.0 $SCENE lam $L"; tail -8 $GB/level.log; exit 1; }
  NA=$GB/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE lam $L"; tail -8 $GB/level.log; exit 1; }
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; h=f['history']; v=json.load(open('$G/original/val_select.json'))
ev=[json.loads(l) for l in open('$G/original/rd_log.jsonl')]; dl=[e for e in ev if e['event']=='delaunay']; pr=[e for e in ev if e['event']=='prune']
last=[r for r in v['rows'] if r['iteration']==30000]
print('$SCENE lam $L RD-GROWTH+PRUNE-mu$MU-BLENDER-VALSEL | best iter', v['best_iteration'], 'val', round(v['best_val_psnr'],3), '(30k val', round(last[0]['val_psnr'],3) if last else 'n/a', ') | vertices', b['vertices'], 'faces', b['faces'],
      '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3), 'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | grow: V@delaunay', dl[0]['vertices'] if dl else None,
      '| prune rounds', [(e['iteration'], e['vertices_before'], e['vertices_after'], 'pairs', e['n_pairs'], 'cap', e['max_pairs']) for e in pr], '| blackbg $FLAG')
PY
done
echo "===== $TAB ====="; cat $TAB
echo "rd-prune3b2 $SCENE exit 0  $(date)"
SBEOF

cat > $AN/submit-blender-v4.sh <<'SHEOF'
#!/usr/bin/env bash
# submit-blender-v4.sh (2026-11-04): 8 NeRF-Synthetic scenes x (1 original job + 1 lambda job of 2 lambdas 6.2e-4 / 6.2e-3), all --indoor + val-selected, black bg, general.
# ~ 8 x 0.6 h + 8 x 1.4 h ~ 16 GPU-h. A scene whose -blender2 original already exists (none expected for v4) skips its training.
AN=/scratch/zr2245/analysis; XYZ=${XYZ:-q14}
for f in $AN/originals-blender-v4.sbatch $AN/rd-prune-blender-v4.sbatch $AN/blender/val_select_v2.py $AN/blender/train_ckpt_v2.py $AN/rdgrowth/train_rd_v3.py; do [ -f $f ] || { echo "FAILED: missing $f"; exit 1; }; done
n=0
for S in chair drums ficus hotdog lego materials mic ship; do
  [ -f /scratch/zr2245/datasets/nerf_synthetic/$S/transforms_val.json ] || { echo "FAILED: no dataset for $S"; continue; }
  [ -f /scratch/zr2245/$S-blender4/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] && echo "note: $S original exists, job will skip its training (check it was INDOOR=1)"
  j=$(sbatch --parsable --time=02:30:00 --export=ALL,SCENE=$S,XYZ=$XYZ,INDOOR=1 --job-name=$S-origb4 $AN/originals-blender-v4.sbatch) \
    && { echo "submitted $S-origb4 $j  XYZ $XYZ  indoor"; n=$((n+1)); } || echo "FAILED: sbatch $S original"
  LAMS="0.00062 0.0062"
  j=$(sbatch --parsable --time=03:30:00 --export=ALL,SCENE=$S,XYZ=$XYZ,INDOOR=1,LAMS="$LAMS",BPV=10.0,BPF=2.0,MU=1.0,MAXF=0.15 --job-name=$S-b4-62 $AN/rd-prune-blender-v4.sbatch) \
    && { echo "submitted $S-b4-62 $j  lams $LAMS  XYZ $XYZ  indoor"; n=$((n+1)); } || echo "FAILED: sbatch $S lams"
done
echo "submitted $n jobs (expected 16)"
squeue -u $USER -o "%.10i %.16j %.28a %.2t %.10M %.6D %R"
SHEOF

bash $AN/submit-blender-v4.sh
```

看进度 / 汇总：

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.16j %.28a %.2t %.10M %.6D %R"
grep -h "VAL_SELECT_RESULT\|FAILED\|exit 0" $AN/origb4-*.out $AN/rd3b4-*.out 2>/dev/null
cat $AN/originals-blender-val4.txt 2>/dev/null
cat $AN/rd-prune3b4-*.txt 2>/dev/null
```

## §8 渲染 lego 看质量（2026-11-04 08:05）：render_views_v1.py + render-blender-v1.sbatch（~5 min GPU）

```bash
AN=/scratch/zr2245/analysis
mkdir -p $AN/blender
cat > $AN/blender/render_views_v1.py <<'PYEOF'
"""render_views_v1.py (VERSION render_views_v1 2026-11-04): render chosen TEST views of a MeshSplatting model and write render / GT / error images.

Loads the model through upstream's own Scene(load_iteration=N) (black background, like the delivery chain), renders the
requested test views at a given supersampling, and writes per view: <out>/view-<name>-render.png, -gt.png, -err.png
(|render-gt| summed over RGB, x4, clipped) plus a montage.png (one row per view: render | GT | error) and psnr.txt.

Usage (GPU node):
  python render_views_v1.py --lab $LAB --dataset $DATA --model $G-best/original --iteration 30000 --supersample 2 --views 0 25 50 75 100 125 150 175 --out $G-best/renders
"""
import argparse, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

VERSION = 'render_views_v1 2026-11-04'


def to_u8(x):
    return (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lab', required=True); ap.add_argument('--dataset', required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--iteration', type=int, default=30000); ap.add_argument('--supersample', type=int, default=2)
    ap.add_argument('--views', nargs='+', type=int, default=[0, 25, 50, 75, 100, 125, 150, 175], help='indices into the sorted test cameras')
    ap.add_argument('--err_gain', type=float, default=4.0); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit('FAILED: CUDA required')
    lab = Path(a.lab).resolve(); up = lab / 'upstream'
    sys.path.insert(0, str(up / 'submodules/effrdel/src')); sys.path.insert(0, str(up))
    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render
    from PIL import Image
    data = SimpleNamespace(source_path=str(Path(a.dataset).resolve()), model_path=str(Path(a.model).resolve()), images='train', resolution=1, eval=True,
                           white_background=False, data_device='cpu', sh_degree=3)
    mesh = TriangleModel(3); scene = Scene(data, mesh, None, None, load_iteration=a.iteration, shuffle=False)
    cams = sorted(scene.getTestCameras(), key=lambda c: (c.image_name, c.uid))
    print(f'[render] {VERSION} | model {a.model} iter {a.iteration} | vertices {int(mesh.vertices.shape[0])} faces {int(mesh._triangle_indices.shape[0])} | test views {len(cams)} | ss {a.supersample}', flush=True)
    mesh.scaling = a.supersample
    bg = torch.zeros(3, device='cuda'); pipe = SimpleNamespace(debug=False, convert_SHs_python=False, depth_ratio=1.)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, lines = [], []
    with torch.no_grad():
        for i in a.views:
            if i >= len(cams):
                print(f'[render] skip view index {i} (only {len(cams)} test views)'); continue
            c = cams[i]
            im = render(c, mesh, pipe, bg)['render'].clamp(0, 1); gt = c.original_image[:3].cuda().clamp(0, 1)
            psnr = float(-10 * torch.log10((im - gt).square().mean().clamp_min(1e-12)))
            err = ((im - gt).abs().sum(0, keepdim=True) * a.err_gain / 3).clamp(0, 1).repeat(3, 1, 1)
            r, g, e = to_u8(im), to_u8(gt), to_u8(err)
            for tag, x in (('render', r), ('gt', g), ('err', e)):
                Image.fromarray(x).save(out / f'view-{c.image_name}-{tag}.png')
            rows.append(np.concatenate([r, g, e], axis=1)); lines.append(f'{c.image_name} psnr {psnr:.2f}')
            print(f'[render] {c.image_name}: PSNR {psnr:.2f}', flush=True)
    if not rows:
        sys.exit('FAILED: no views rendered')
    Image.fromarray(np.concatenate(rows, axis=0)).save(out / 'montage.png')
    (out / 'psnr.txt').write_text('\n'.join(lines) + '\n')
    print(f'[render] wrote {out}/montage.png ({len(rows)} rows: render | GT | error x{a.err_gain:g}) and per-view PNGs', flush=True)


if __name__ == '__main__':
    main()
PYEOF
python -c "import ast; ast.parse(open('$AN/blender/render_views_v1.py').read()); print('render_views_v1.py ok')" || echo "FAILED: render_views_v1.py did not paste cleanly"

cat > $AN/render-blender-v1.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=rendb
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/zr2245/analysis/rendb-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/rendb-%x-%j.err
# render-blender-v1.sbatch (2026-11-04, VERSION v1): render test views of MODEL (a model dir with point_cloud/iteration_$ITER) -> $OUT/montage.png + per-view PNGs.
#   sbatch --export=ALL,SCENE=lego,MODEL=/scratch/zr2245/lego-blender3-best/original,OUT=/scratch/zr2245/lego-blender3-best/renders --job-name=lego-rend /scratch/zr2245/analysis/render-blender-v1.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
SCENE=${SCENE:?set SCENE}; MODEL=${MODEL:?set MODEL}; OUT=${OUT:?set OUT}; ITER=${ITER:-30000}; SS=${SS:-2}; VIEWS=${VIEWS:-"0 25 50 75 100 125 150 175"}
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis
echo "scene $SCENE | model $MODEL iter $ITER | ss $SS | views $VIEWS | out $OUT | job $SLURM_JOB_ID on $(hostname) $(date)"
for f in $MODEL/point_cloud/iteration_$ITER/point_cloud_state_dict.pt $DATA/transforms_test.json $AN/blender/render_views_v1.py; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
python $AN/blender/render_views_v1.py --lab $LAB --dataset $DATA --model $MODEL --iteration $ITER --supersample $SS --views $VIEWS --out $OUT > $OUT.render.log 2>&1 || { echo "FAILED: RENDER"; grep -n -m 8 -i 'error\|Traceback' $OUT.render.log; tail -10 $OUT.render.log; exit 1; }
grep -h "^\[render\]" $OUT.render.log
ls -la $OUT | head -30
echo "render $SCENE exit 0  $(date)"
SBEOF
j=$(sbatch --parsable --export=ALL,SCENE=lego,MODEL=/scratch/zr2245/lego-blender3-best/original,OUT=/scratch/zr2245/lego-blender3-best/renders --job-name=lego-rend $AN/render-blender-v1.sbatch) && echo "submitted lego-rend $j" || echo "FAILED: sbatch"
squeue -u $USER -o "%.10i %.16j %.28a %.2t %.10M %.6D %R"
```

结果：`/scratch/zr2245/lego-blender3-best/renders/montage.png`（每行一个 test 视角：render | GT | 误差 ×4），VS Code 里直接打开；`psnr.txt` 是每视角 PSNR。

## §9 几何统计（CPU，cpu-torch 即可；2026-11-04 12:30）

```bash
AN=/scratch/zr2245/analysis
mkdir -p $AN/diag
cat > $AN/diag/mesh_stats_v1.py <<'PYEOF'
"""mesh_stats_v1.py (VERSION mesh_stats_v1 2026-11-04): CPU geometry statistics of a MeshSplatting checkpoint (triangle soup after Delaunay/growth/prune).

Answers "how ill-formed is the mesh": per-edge face incidence (manifold = 2, boundary = 1, non-manifold >= 3), sliver share
(min angle < 5 / 10 deg), degenerate/zero-area faces, duplicate faces, connected components (by shared vertices), dihedral angle
distribution across manifold edges, face-area distribution, unused vertices. Pure numpy, no CUDA; a 3 M-face mesh takes ~1 min.

Usage:
  python mesh_stats_v1.py --ckpt .../point_cloud/iteration_30000/point_cloud_state_dict.pt [--label lego-30k] [--out stats.json]
"""
import argparse, json, sys
import numpy as np
import torch

VERSION = 'mesh_stats_v1 2026-11-04'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True); ap.add_argument('--label', default=None); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    sd = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    V = sd['triangles_points'].double().numpy(); F = sd['_triangle_indices'].long().numpy()
    nv, nf = len(V), len(F)
    label = a.label or a.ckpt
    # --- faces: area, angles
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(B - A, C - A); area2 = np.linalg.norm(n, axis=1); area = 0.5 * area2
    def ang(P, Q, R):
        u, w = Q - P, R - P
        c = (u * w).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-30)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))
    amin = np.minimum.reduce([ang(A, B, C), ang(B, C, A), ang(C, A, B)])
    degenerate = (area2 <= 1e-12 * np.median(area2)) | (F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 0] == F[:, 2])
    # --- edges
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); E = np.sort(E, axis=1)
    key = E[:, 0] * (nv + 1) + E[:, 1]
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inc = cnt[inv]  # per half-edge: how many faces share this edge
    ne = len(uk); n_bd = int((cnt == 1).sum()); n_man = int((cnt == 2).sum()); n_nm = int((cnt >= 3).sum())
    # faces touching a non-manifold / boundary edge
    inc_f = inc.reshape(3, nf)
    f_nm = (inc_f >= 3).any(0); f_bd = (inc_f == 1).any(0)
    # duplicate faces
    Fs = np.sort(F, axis=1); fkey = (Fs[:, 0] * (nv + 1) + Fs[:, 1]) * (nv + 1) + Fs[:, 2]
    n_dup = int(nf - len(np.unique(fkey)))
    # --- connected components via union-find on vertices (faces connect their vertices)
    parent = np.arange(nv)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    # vectorised-ish union: iterate edges in chunks with python loop over unique edges (ne up to ~5M -> fine in numpy loop? use scipy if available)
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        e0, e1 = uk // (nv + 1), uk % (nv + 1)
        g = coo_matrix((np.ones(ne), (e0, e1)), shape=(nv, nv))
        ncomp, lab = connected_components(g, directed=False)
        used = np.zeros(nv, bool); used[F.reshape(-1)] = True
        comp_sizes = np.bincount(lab[used]); comp_sizes = np.sort(comp_sizes[comp_sizes > 0])[::-1]
        ncomp_used = int(len(comp_sizes)); largest_frac = float(comp_sizes[0] / used.sum()); n_tiny = int((comp_sizes <= 10).sum())
    except ImportError:
        ncomp_used, largest_frac, n_tiny = -1, -1.0, -1
        used = np.zeros(nv, bool); used[F.reshape(-1)] = True
    # --- dihedral angles across manifold edges
    fn = n / np.maximum(area2, 1e-30)[:, None]
    order = np.argsort(inv, kind='stable'); inv_s = inv[order]; fid = (np.arange(3 * nf) % nf)[order]
    man_edges = np.flatnonzero(cnt == 2)
    starts = np.searchsorted(inv_s, man_edges)
    fa, fb = fid[starts], fid[starts + 1]
    cosd = np.clip((fn[fa] * fn[fb]).sum(1), -1, 1); dih = np.degrees(np.arccos(np.abs(cosd)))  # 0 = coplanar (orientation-agnostic)
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else float('nan')
    stats = dict(version=VERSION, label=label, ckpt=a.ckpt, vertices=nv, faces=nf, unused_vertices=int((~used).sum()),
                 edges=ne, boundary_edges=n_bd, manifold_edges=n_man, nonmanifold_edges=n_nm,
                 boundary_edge_frac=n_bd / ne, nonmanifold_edge_frac=n_nm / ne,
                 faces_touching_nonmanifold_edge_frac=float(f_nm.mean()), faces_touching_boundary_edge_frac=float(f_bd.mean()),
                 duplicate_faces=n_dup, degenerate_faces=int(degenerate.sum()),
                 sliver_minangle_lt5_frac=float((amin < 5).mean()), sliver_minangle_lt10_frac=float((amin < 10).mean()), min_angle_median_deg=float(np.median(amin)),
                 dihedral_median_deg=q(dih, 50), dihedral_p90_deg=q(dih, 90), dihedral_gt30_frac=float((dih > 30).mean()) if len(dih) else float('nan'),
                 area_median=float(np.median(area)), area_p10=q(area, 10), area_p90=q(area, 90), area_max_over_median=float(area.max() / max(np.median(area), 1e-30)),
                 components=ncomp_used, largest_component_vertex_frac=largest_frac, components_le10_vertices=n_tiny)
    print(f"MESH_STATS {label} | V {nv} F {nf} unused_v {stats['unused_vertices']} | edges {ne}: boundary {100*stats['boundary_edge_frac']:.1f} % manifold {100*n_man/ne:.1f} % non-manifold {100*stats['nonmanifold_edge_frac']:.2f} % "
          f"| faces on non-manifold edge {100*stats['faces_touching_nonmanifold_edge_frac']:.1f} %, on boundary edge {100*stats['faces_touching_boundary_edge_frac']:.1f} % | dup {n_dup} degenerate {stats['degenerate_faces']} "
          f"| sliver <5deg {100*stats['sliver_minangle_lt5_frac']:.1f} % <10deg {100*stats['sliver_minangle_lt10_frac']:.1f} % (median min angle {stats['min_angle_median_deg']:.1f}) "
          f"| dihedral median {stats['dihedral_median_deg']:.1f} p90 {stats['dihedral_p90_deg']:.1f} >30deg {100*stats['dihedral_gt30_frac']:.1f} % "
          f"| components {ncomp_used} (largest {100*largest_frac:.1f} % of used vertices, {n_tiny} with <=10 v) | area p10/med/p90 {stats['area_p10']:.2e}/{stats['area_median']:.2e}/{stats['area_p90']:.2e}")
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(stats, f, indent=1)


if __name__ == '__main__':
    main()
PYEOF
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
python -c "import ast; ast.parse(open('$AN/diag/mesh_stats_v1.py').read()); print('mesh_stats_v1.py ok')" || echo "FAILED: mesh_stats_v1.py did not paste cleanly"
for pair in "lego-21k-unpruned:/scratch/zr2245/lego-blender3-best/original/point_cloud/iteration_30000/point_cloud_state_dict.pt" \
            "lego-30k-pruned:/scratch/zr2245/lego-blender3/original/point_cloud/iteration_30000/point_cloud_state_dict.pt" \
            "room-indoor-30k:/scratch/zr2245/room-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt" \
            "room-rd3i-lam0.0062:/scratch/zr2245/room-rd3i-lam0.0062-mu1.0/original/point_cloud/iteration_30000/point_cloud_state_dict.pt"; do
  L=${pair%%:*}; C=${pair#*:}
  [ -f $C ] || { echo "skip $L: missing $C"; continue; }
  python $AN/diag/mesh_stats_v1.py --ckpt $C --label $L --out $AN/diag/mesh-stats-$L.json 2>&1 | grep -v "^$" | tail -3
done | tee -a $AN/diag/mesh-stats-v1.txt
```

把 `MESH_STATS` 行贴我。cpu-torch 只有 8 GB：3 M 面的 lego 约需 1-2 GB，够用；若 Killed 就在 GPU 节点上跑同一段。

### §9b mesh_stats_v2（v1 的 checkpoint 张量带 requires_grad，加 detach；整块粘贴）

```bash
AN=/scratch/zr2245/analysis
mkdir -p $AN/diag
cat > $AN/diag/mesh_stats_v2.py <<'PYEOF'
"""mesh_stats_v2.py (VERSION mesh_stats_v2 2026-11-04; v1 + detach: checkpoint tensors carry requires_grad): CPU geometry statistics of a MeshSplatting checkpoint (triangle soup after Delaunay/growth/prune).

Answers "how ill-formed is the mesh": per-edge face incidence (manifold = 2, boundary = 1, non-manifold >= 3), sliver share
(min angle < 5 / 10 deg), degenerate/zero-area faces, duplicate faces, connected components (by shared vertices), dihedral angle
distribution across manifold edges, face-area distribution, unused vertices. Pure numpy, no CUDA; a 3 M-face mesh takes ~1 min.

Usage:
  python mesh_stats_v2.py --ckpt .../point_cloud/iteration_30000/point_cloud_state_dict.pt [--label lego-30k] [--out stats.json]
"""
import argparse, json, sys
import numpy as np
import torch

VERSION = 'mesh_stats_v2 2026-11-04'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True); ap.add_argument('--label', default=None); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    sd = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    V = sd['triangles_points'].detach().cpu().double().numpy(); F = sd['_triangle_indices'].detach().cpu().long().numpy()
    nv, nf = len(V), len(F)
    label = a.label or a.ckpt
    # --- faces: area, angles
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(B - A, C - A); area2 = np.linalg.norm(n, axis=1); area = 0.5 * area2
    def ang(P, Q, R):
        u, w = Q - P, R - P
        c = (u * w).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-30)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))
    amin = np.minimum.reduce([ang(A, B, C), ang(B, C, A), ang(C, A, B)])
    degenerate = (area2 <= 1e-12 * np.median(area2)) | (F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 0] == F[:, 2])
    # --- edges
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); E = np.sort(E, axis=1)
    key = E[:, 0] * (nv + 1) + E[:, 1]
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inc = cnt[inv]  # per half-edge: how many faces share this edge
    ne = len(uk); n_bd = int((cnt == 1).sum()); n_man = int((cnt == 2).sum()); n_nm = int((cnt >= 3).sum())
    # faces touching a non-manifold / boundary edge
    inc_f = inc.reshape(3, nf)
    f_nm = (inc_f >= 3).any(0); f_bd = (inc_f == 1).any(0)
    # duplicate faces
    Fs = np.sort(F, axis=1); fkey = (Fs[:, 0] * (nv + 1) + Fs[:, 1]) * (nv + 1) + Fs[:, 2]
    n_dup = int(nf - len(np.unique(fkey)))
    # --- connected components via union-find on vertices (faces connect their vertices)
    parent = np.arange(nv)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    # vectorised-ish union: iterate edges in chunks with python loop over unique edges (ne up to ~5M -> fine in numpy loop? use scipy if available)
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        e0, e1 = uk // (nv + 1), uk % (nv + 1)
        g = coo_matrix((np.ones(ne), (e0, e1)), shape=(nv, nv))
        ncomp, lab = connected_components(g, directed=False)
        used = np.zeros(nv, bool); used[F.reshape(-1)] = True
        comp_sizes = np.bincount(lab[used]); comp_sizes = np.sort(comp_sizes[comp_sizes > 0])[::-1]
        ncomp_used = int(len(comp_sizes)); largest_frac = float(comp_sizes[0] / used.sum()); n_tiny = int((comp_sizes <= 10).sum())
    except ImportError:
        ncomp_used, largest_frac, n_tiny = -1, -1.0, -1
        used = np.zeros(nv, bool); used[F.reshape(-1)] = True
    # --- dihedral angles across manifold edges
    fn = n / np.maximum(area2, 1e-30)[:, None]
    order = np.argsort(inv, kind='stable'); inv_s = inv[order]; fid = (np.arange(3 * nf) % nf)[order]
    man_edges = np.flatnonzero(cnt == 2)
    starts = np.searchsorted(inv_s, man_edges)
    fa, fb = fid[starts], fid[starts + 1]
    cosd = np.clip((fn[fa] * fn[fb]).sum(1), -1, 1); dih = np.degrees(np.arccos(np.abs(cosd)))  # 0 = coplanar (orientation-agnostic)
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else float('nan')
    stats = dict(version=VERSION, label=label, ckpt=a.ckpt, vertices=nv, faces=nf, unused_vertices=int((~used).sum()),
                 edges=ne, boundary_edges=n_bd, manifold_edges=n_man, nonmanifold_edges=n_nm,
                 boundary_edge_frac=n_bd / ne, nonmanifold_edge_frac=n_nm / ne,
                 faces_touching_nonmanifold_edge_frac=float(f_nm.mean()), faces_touching_boundary_edge_frac=float(f_bd.mean()),
                 duplicate_faces=n_dup, degenerate_faces=int(degenerate.sum()),
                 sliver_minangle_lt5_frac=float((amin < 5).mean()), sliver_minangle_lt10_frac=float((amin < 10).mean()), min_angle_median_deg=float(np.median(amin)),
                 dihedral_median_deg=q(dih, 50), dihedral_p90_deg=q(dih, 90), dihedral_gt30_frac=float((dih > 30).mean()) if len(dih) else float('nan'),
                 area_median=float(np.median(area)), area_p10=q(area, 10), area_p90=q(area, 90), area_max_over_median=float(area.max() / max(np.median(area), 1e-30)),
                 components=ncomp_used, largest_component_vertex_frac=largest_frac, components_le10_vertices=n_tiny)
    print(f"MESH_STATS {label} | V {nv} F {nf} unused_v {stats['unused_vertices']} | edges {ne}: boundary {100*stats['boundary_edge_frac']:.1f} % manifold {100*n_man/ne:.1f} % non-manifold {100*stats['nonmanifold_edge_frac']:.2f} % "
          f"| faces on non-manifold edge {100*stats['faces_touching_nonmanifold_edge_frac']:.1f} %, on boundary edge {100*stats['faces_touching_boundary_edge_frac']:.1f} % | dup {n_dup} degenerate {stats['degenerate_faces']} "
          f"| sliver <5deg {100*stats['sliver_minangle_lt5_frac']:.1f} % <10deg {100*stats['sliver_minangle_lt10_frac']:.1f} % (median min angle {stats['min_angle_median_deg']:.1f}) "
          f"| dihedral median {stats['dihedral_median_deg']:.1f} p90 {stats['dihedral_p90_deg']:.1f} >30deg {100*stats['dihedral_gt30_frac']:.1f} % "
          f"| components {ncomp_used} (largest {100*largest_frac:.1f} % of used vertices, {n_tiny} with <=10 v) | area p10/med/p90 {stats['area_p10']:.2e}/{stats['area_median']:.2e}/{stats['area_p90']:.2e}")
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(stats, f, indent=1)


if __name__ == '__main__':
    main()
PYEOF
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
python -c "import ast; ast.parse(open('$AN/diag/mesh_stats_v2.py').read()); print('mesh_stats_v2.py ok')" || echo "FAILED: mesh_stats_v2.py did not paste cleanly"
cat > $AN/diag/run_mesh_stats_v2.sh <<'SHEOF'
#!/usr/bin/env bash
AN=/scratch/zr2245/analysis
for pair in "lego-21k-unpruned:/scratch/zr2245/lego-blender3-best/original/point_cloud/iteration_30000/point_cloud_state_dict.pt" \
            "lego-30k-pruned:/scratch/zr2245/lego-blender3/original/point_cloud/iteration_30000/point_cloud_state_dict.pt" \
            "room-indoor-30k:/scratch/zr2245/room-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt" \
            "room-rd3i-lam0.0062:/scratch/zr2245/room-rd3i-lam0.0062-mu1.0/original/point_cloud/iteration_30000/point_cloud_state_dict.pt"; do
  L=${pair%%:*}; C=${pair#*:}
  [ -f $C ] || { echo "skip $L: missing $C"; continue; }
  python $AN/diag/mesh_stats_v2.py --ckpt $C --label $L --out $AN/diag/mesh-stats-$L.json 2>&1 | tail -3
done | tee -a $AN/diag/mesh-stats-v2.txt
SHEOF
bash $AN/diag/run_mesh_stats_v2.sh
```

## §10 supersample 2/4/8 对比（lego 21k，200 个 test 视角，~0.1 GPU-h；2026-11-04 12:40）

```bash
AN=/scratch/zr2245/analysis
mkdir -p $AN/blender
cat > $AN/blender/render_views_v2.py <<'PYEOF'
"""render_views_v2.py (VERSION render_views_v2 2026-11-04): render chosen TEST views of a MeshSplatting model and write render / GT / error images.

v2: --all_views (every test camera), --no_png (metrics only), SSIM per view, mean PSNR/SSIM line "RENDER_MEAN ..." for supersampling comparisons.

Loads the model through upstream's own Scene(load_iteration=N) (black background, like the delivery chain), renders the
requested test views at a given supersampling, and writes per view: <out>/view-<name>-render.png, -gt.png, -err.png
(|render-gt| summed over RGB, x4, clipped) plus a montage.png (one row per view: render | GT | error) and psnr.txt.

Usage (GPU node):
  python render_views_v1.py --lab $LAB --dataset $DATA --model $G-best/original --iteration 30000 --supersample 2 --views 0 25 50 75 100 125 150 175 --out $G-best/renders
"""
import argparse, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

VERSION = 'render_views_v2 2026-11-04'


def to_u8(x):
    return (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lab', required=True); ap.add_argument('--dataset', required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--iteration', type=int, default=30000); ap.add_argument('--supersample', type=int, default=2)
    ap.add_argument('--views', nargs='+', type=int, default=[0, 25, 50, 75, 100, 125, 150, 175], help='indices into the sorted test cameras')
    ap.add_argument('--err_gain', type=float, default=4.0); ap.add_argument('--out', required=True)
    ap.add_argument('--all_views', action='store_true'); ap.add_argument('--no_png', action='store_true', help='metrics only, no images')
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit('FAILED: CUDA required')
    lab = Path(a.lab).resolve(); up = lab / 'upstream'
    sys.path.insert(0, str(up / 'submodules/effrdel/src')); sys.path.insert(0, str(up))
    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render
    from PIL import Image
    from utils.loss_utils import ssim
    data = SimpleNamespace(source_path=str(Path(a.dataset).resolve()), model_path=str(Path(a.model).resolve()), images='train', resolution=1, eval=True,
                           white_background=False, data_device='cpu', sh_degree=3)
    mesh = TriangleModel(3); scene = Scene(data, mesh, None, None, load_iteration=a.iteration, shuffle=False)
    cams = sorted(scene.getTestCameras(), key=lambda c: (c.image_name, c.uid))
    print(f'[render] {VERSION} | model {a.model} iter {a.iteration} | vertices {int(mesh.vertices.shape[0])} faces {int(mesh._triangle_indices.shape[0])} | test views {len(cams)} | ss {a.supersample}', flush=True)
    mesh.scaling = a.supersample
    bg = torch.zeros(3, device='cuda'); pipe = SimpleNamespace(debug=False, convert_SHs_python=False, depth_ratio=1.)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, lines, ps, ss_ = [], [], [], []
    views = list(range(len(cams))) if a.all_views else a.views
    with torch.no_grad():
        for i in views:
            if i >= len(cams):
                print(f'[render] skip view index {i} (only {len(cams)} test views)'); continue
            c = cams[i]
            im = render(c, mesh, pipe, bg)['render'].clamp(0, 1); gt = c.original_image[:3].cuda().clamp(0, 1)
            psnr = float(-10 * torch.log10((im - gt).square().mean().clamp_min(1e-12))); s_ = float(ssim(im, gt))
            ps.append(psnr); ss_.append(s_); lines.append(f'{c.image_name} psnr {psnr:.2f} ssim {s_:.4f}')
            if not a.no_png:
                err = ((im - gt).abs().sum(0, keepdim=True) * a.err_gain / 3).clamp(0, 1).repeat(3, 1, 1)
                r, g, e = to_u8(im), to_u8(gt), to_u8(err)
                for tag, x in (('render', r), ('gt', g), ('err', e)):
                    Image.fromarray(x).save(out / f'view-{c.image_name}-{tag}.png')
                rows.append(np.concatenate([r, g, e], axis=1))
            if not a.all_views or i % 25 == 0:
                print(f'[render] {c.image_name}: PSNR {psnr:.2f} SSIM {s_:.4f}', flush=True)
    if not ps:
        sys.exit('FAILED: no views rendered')
    if rows:
        Image.fromarray(np.concatenate(rows, axis=0)).save(out / 'montage.png')
    (out / f'psnr-ss{a.supersample}.txt').write_text('\n'.join(lines) + '\n')
    print(f'RENDER_MEAN model {a.model} iter {a.iteration} ss {a.supersample} views {len(ps)} psnr {np.mean(ps):.3f} ssim {np.mean(ss_):.4f} worst {min(ps):.2f}', flush=True)


if __name__ == '__main__':
    main()
PYEOF
python -c "import ast; ast.parse(open('$AN/blender/render_views_v2.py').read()); print('render_views_v2.py ok')" || echo "FAILED: render_views_v2.py did not paste cleanly"
cat > $AN/ss-compare-blender-v1.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=sscmp
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/zr2245/analysis/sscmp-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/sscmp-%x-%j.err
# ss-compare-blender-v1.sbatch (2026-11-04, VERSION v1): test PSNR/SSIM of one model at eval supersampling 2 / 4 / 8 on ALL test views (metrics only).
#   sbatch --export=ALL,SCENE=lego,MODEL=/scratch/zr2245/lego-blender3-best/original --job-name=lego-sscmp /scratch/zr2245/analysis/ss-compare-blender-v1.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
SCENE=${SCENE:?set SCENE}; MODEL=${MODEL:?set MODEL}; ITER=${ITER:-30000}; SSL=${SSL:-"2 4 8"}
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; OUT=$MODEL/../ss-compare
echo "scene $SCENE | model $MODEL iter $ITER | ss list $SSL | job $SLURM_JOB_ID on $(hostname) $(date)"
for f in $MODEL/point_cloud/iteration_$ITER/point_cloud_state_dict.pt $DATA/transforms_test.json $AN/blender/render_views_v2.py; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
mkdir -p $OUT
for SS in $SSL; do
  echo "=== ss $SS ($(date)) ==="
  python $AN/blender/render_views_v2.py --lab $LAB --dataset $DATA --model $MODEL --iteration $ITER --supersample $SS --all_views --no_png --out $OUT > $OUT/ss$SS.log 2>&1 || { echo "FAILED: ss $SS"; grep -n -m 4 -i 'error\|Traceback\|out of memory' $OUT/ss$SS.log; continue; }
  grep -h "RENDER_MEAN" $OUT/ss$SS.log | tee -a $AN/ss-compare-v1.txt
done
echo "ss-compare $SCENE exit 0  $(date)"
SBEOF
j=$(sbatch --parsable --export=ALL,SCENE=lego,MODEL=/scratch/zr2245/lego-blender3-best/original --job-name=lego-sscmp $AN/ss-compare-blender-v1.sbatch) && echo "submitted lego-sscmp $j" || echo "FAILED: sbatch"
squeue -u $USER -o "%.10i %.16j %.28a %.2t %.10M %.6D %R"
```

结果：`cat /scratch/zr2245/analysis/ss-compare-v1.txt`（三行 RENDER_MEAN）。

## §11 Blender 法线先验（lego，~0.5 GPU-h；train_ckpt_v3 + make_normals_input_v1 + originals-blender-v4n）

流程：train PNG 合成到黑底存 jpg → upstream extract_normals.py（Metric3D）→ 用 PNG alpha 把背景法线置零 → train_ckpt_v3 把 normals_masked/<name>.png 挂到 Blender train 相机的 normal_map（与 COLMAP 路径同一编码）→ 其余与 v4 相同（--indoor、每 1k 存、val 选点 + 剪枝、交付）。输出 lego-blender4n，表 originals-blender-val4n.txt；与 v4 批里的 lego 行（无先验）同协议可比。

```bash
AN=/scratch/zr2245/analysis
mkdir -p $AN/blender
cat > $AN/blender/train_ckpt_v3.py <<'PYEOF'
"""train_ckpt_v3.py (VERSION train_ckpt_v3 2026-11-04): run an unmodified training script and save checkpoints at chosen iterations.

v3: optional --normals DIR attaches monocular normal maps (DIR/<image_name>.png, same encoding as the COLMAP path:
RGB/255*2-1) to the Blender-loaded TRAIN cameras by wrapping sceneLoadTypeCallbacks["Blender"], so the authors' Metric3D
normal prior also applies on NeRF-Synthetic (upstream's Blender reader never fills normal_map). Fails before training if no
camera received a map.

v2: also saves the model's per-face importance_score next to each checkpoint (importance_score.pt), so val_select_v2 can apply the
same final importance prune the training scripts apply before their own last save (lego: 2.98 M -> 1.32 M faces).

Why: upstream train.py accepts --save_iterations but never uses it (its only scene.save() is the final one), and our
train_rd_v3.py inherited that. The NeRF-Synthetic probe needs checkpoints every 1k from 12k for validation-split selection
(val_select_v1.py). No upstream file is edited: this wrapper hooks TriangleModel.update_learning_rate, which both training
loops call once per iteration with the iteration index, and saves through Scene.save (so the on-disk format is exactly the
one the delivery chain loads: point_cloud/iteration_N/point_cloud_state_dict.pt).

Note: the hook fires at the START of iteration N (before that iteration's optimizer step), so iteration_N holds the state after
N-1 steps; the "[ITER N] Evaluating" line in the log is after step N. Intermediate checkpoints also skip the final
importance-based triangle prune that the scripts apply before their own last save (a few % more faces; rendering unchanged).

Usage:
  python train_ckpt_v1.py --saves 12000 13000 ... --script /path/upstream/train.py      -- -s DATA -i train -r 1 --eval ... -m MODEL ...
  python train_ckpt_v1.py --saves 12000 13000 ... --script /path/rdgrowth/train_rd_v3.py -- --lab LAB -s DATA ... -m MODEL --lam ...
"""
import argparse, inspect, os, runpy, sys

VERSION = 'train_ckpt_v3 2026-11-04'


def main():
    if '--' not in sys.argv:
        sys.exit('FAILED: usage: train_ckpt_v3.py --saves N... --script PATH [--normals DIR] -- <script args>')
    k = sys.argv.index('--')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--saves', nargs='+', type=int, required=True)
    ap.add_argument('--script', required=True, help='upstream/train.py or analysis/rdgrowth/train_rd_v3.py')
    ap.add_argument('--normals', default=None, help='dir of <image_name>.png normal maps to attach to Blender train cameras')
    a = ap.parse_args(sys.argv[1:k]); rest = sys.argv[k + 1:]
    script = os.path.abspath(a.script)
    if not os.path.isfile(script):
        sys.exit(f'FAILED: script not found: {script}')
    # model path and upstream dir from the script's own arguments
    mp = None
    for i, t in enumerate(rest):
        if t in ('-m', '--model_path') and i + 1 < len(rest):
            mp = os.path.abspath(rest[i + 1])
    if mp is None:
        sys.exit('FAILED: no -m/--model_path in the script arguments')
    if '--lab' in rest:
        up = os.path.join(os.path.abspath(rest[rest.index('--lab') + 1]), 'upstream')
    else:
        up = os.path.dirname(script)
    if not os.path.isfile(os.path.join(up, 'train.py')):
        sys.exit(f'FAILED: upstream train.py not found under {up}')
    src = open(script).read()
    if 'update_learning_rate(iteration)' not in src:
        sys.exit(f'FAILED: {script} does not call update_learning_rate(iteration) once per iteration; hook would never fire')
    sys.path.insert(0, os.path.join(up, 'submodules/effrdel/src')); sys.path.insert(0, up); os.chdir(up)
    import scene.triangle_model as tm
    from scene import Scene
    if not hasattr(tm.TriangleModel, 'update_learning_rate'):
        sys.exit('FAILED: TriangleModel has no update_learning_rate')
    ssrc = inspect.getsource(Scene.save)
    print(f'[ckpt] {VERSION} | saves {sorted(a.saves)} | model_path {mp} | script {script}\n[ckpt] Scene.save source:\n{ssrc}', flush=True)
    if 'model_path' not in ssrc:
        sys.exit('FAILED: Scene.save does not use self.model_path; hook cannot reuse it')

    class _Fake:  # stands in for the Scene object inside Scene.save: model_path + whatever attribute holds the model
        def __init__(self, model_path, model):
            self.model_path = model_path; self._model = model
        def __getattr__(self, name):
            return self._model

    saves = set(a.saves); done = set(); orig = tm.TriangleModel.update_learning_rate

    def hooked(self, iteration):
        r = orig(self, iteration)
        if iteration in saves and iteration not in done:
            done.add(iteration)
            os.makedirs(os.path.join(mp, 'point_cloud', f'iteration_{iteration}'), exist_ok=True)
            out = Scene.save(_Fake(mp, self), iteration)
            nv = int(self.vertices.shape[0]) if hasattr(self, 'vertices') else -1
            imp = getattr(self, 'importance_score', None)
            if imp is None:
                sys.exit('FAILED: model has no importance_score attribute; cannot reproduce the final importance prune')
            import torch
            torch.save(imp.detach().cpu().clone(), os.path.join(mp, 'point_cloud', f'iteration_{iteration}', 'importance_score.pt'))
            nf = int(self._triangle_indices.shape[0]); nkeep = int((imp.detach().cpu().reshape(-1) > 0.5).sum())
            print(f'[ckpt] saved iteration {iteration} (state after {iteration - 1} steps) vertices {nv} faces {nf} (importance>0.5: {nkeep}) -> {out}', flush=True)
        return r

    tm.TriangleModel.update_learning_rate = hooked

    if a.normals:
        import numpy as np
        from PIL import Image
        import scene.dataset_readers as dr
        ndir = os.path.abspath(a.normals)
        if not os.path.isdir(ndir):
            sys.exit(f'FAILED: normals dir not found: {ndir}')
        if 'Blender' not in dr.sceneLoadTypeCallbacks:
            sys.exit('FAILED: upstream has no Blender loader callback')
        orig_blender = dr.sceneLoadTypeCallbacks['Blender']

        def _with(obj, **kw):  # NamedTuple or dataclass
            if hasattr(obj, '_replace'):
                return obj._replace(**kw)
            import dataclasses
            return dataclasses.replace(obj, **kw)

        def blender_with_normals(*args, **kwargs):
            info = orig_blender(*args, **kwargs)
            cams, hit, miss = [], 0, []
            for ci in info.train_cameras:
                p = os.path.join(ndir, ci.image_name + '.png')
                if os.path.isfile(p):
                    nm = (np.array(Image.open(p).convert('RGB')).astype(np.float32) / 255.0) * 2.0 - 1.0
                    W, H = ci.image.size if hasattr(ci.image, 'size') else (ci.width, ci.height)
                    if nm.shape[0] != H or nm.shape[1] != W:
                        sys.exit(f'FAILED: normal map {p} is {nm.shape[1]}x{nm.shape[0]} but image is {W}x{H}')
                    cams.append(_with(ci, normal_map=nm)); hit += 1
                else:
                    cams.append(ci); miss.append(ci.image_name)
            if hit == 0:
                sys.exit(f'FAILED: no normal map matched any train camera in {ndir} (expected <image_name>.png, e.g. r_0.png)')
            print(f'[ckpt] normal maps attached to {hit}/{len(cams)} train cameras from {ndir}' + (f'; missing {miss[:5]}...' if miss else ''), flush=True)
            return _with(info, train_cameras=cams)

        dr.sceneLoadTypeCallbacks['Blender'] = blender_with_normals
        cu_src = open(os.path.join(up, 'utils', 'camera_utils.py')).read()
        if 'normal_map' not in cu_src:
            sys.exit('FAILED: utils/camera_utils.py never passes normal_map into Camera; the hook would be dropped')
        print(f'[ckpt] Blender loader wrapped: normal_map from {ndir}', flush=True)
    sys.argv = [script] + rest
    runpy.run_path(script, run_name='__main__')
    missing = sorted(i for i in saves if not os.path.isfile(os.path.join(mp, 'point_cloud', f'iteration_{i}', 'point_cloud_state_dict.pt')))
    if missing:
        sys.exit(f'FAILED: checkpoints missing after training: {missing}')
    missing = sorted(i for i in saves if not os.path.isfile(os.path.join(mp, 'point_cloud', f'iteration_{i}', 'importance_score.pt')))
    if missing:
        sys.exit(f'FAILED: importance_score.pt missing after training: {missing}')
    print(f'[ckpt] all {len(saves)} checkpoints present', flush=True)


if __name__ == '__main__':
    main()
PYEOF
cat > $AN/blender/make_normals_input_v1.py <<'PYEOF'
"""make_normals_input_v1.py (VERSION v1 2026-11-04): NeRF-Synthetic train PNGs (RGBA) -> <scene>/images/<name>.jpg (composited over black,
what upstream extract_normals.py expects: jpg files in an 'images' folder), and after Metric3D ran: mask <scene>/normals/<name>.png with the
PNG alpha -> <scene>/normals_masked/<name>.png (background = encoded zero vector 128,128,128, so the normal prior is silent there).

  python make_normals_input_v1.py --scene DIR --stage jpg      # before extract_normals.py
  python make_normals_input_v1.py --scene DIR --stage mask     # after  extract_normals.py -s DIR -i images
"""
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image

VERSION = 'make_normals_input_v1 2026-11-04'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--scene', required=True); ap.add_argument('--stage', choices=['jpg', 'mask'], required=True)
    ap.add_argument('--alpha_thresh', type=int, default=1)
    a = ap.parse_args(); scene = Path(a.scene); train = scene / 'train'
    pngs = sorted(p for p in train.glob('*.png') if '_depth' not in p.name and '_normal' not in p.name)
    if not pngs:
        sys.exit(f'FAILED: no train PNGs in {train}')
    if a.stage == 'jpg':
        out = scene / 'images'; out.mkdir(exist_ok=True); n = 0
        for p in pngs:
            im = np.array(Image.open(p).convert('RGBA')).astype(np.float32) / 255.0
            rgb = im[..., :3] * im[..., 3:4]  # over black
            Image.fromarray((rgb * 255).round().astype(np.uint8)).save(out / (p.stem + '.jpg'), quality=97); n += 1
        print(f'[normals-input] {VERSION}: wrote {n} jpg to {out}')
    else:
        src = scene / 'normals'; out = scene / 'normals_masked'; out.mkdir(exist_ok=True); n, miss = 0, []
        for p in pngs:
            q = src / (p.stem + '.png')
            if not q.is_file():
                miss.append(p.stem); continue
            alpha = np.array(Image.open(p).convert('RGBA'))[..., 3]
            nm = np.array(Image.open(q).convert('RGB'))
            if nm.shape[:2] != alpha.shape:
                sys.exit(f'FAILED: {q} is {nm.shape[1]}x{nm.shape[0]}, image is {alpha.shape[1]}x{alpha.shape[0]}')
            nm[alpha < a.alpha_thresh] = 128
            Image.fromarray(nm).save(out / (p.stem + '.png')); n += 1
        if miss:
            sys.exit(f'FAILED: {len(miss)} normal maps missing in {src}, e.g. {miss[:5]}')
        print(f'[normals-input] {VERSION}: wrote {n} masked normal maps to {out}')


if __name__ == '__main__':
    main()
PYEOF
python -c "import ast; ast.parse(open('$AN/blender/train_ckpt_v3.py').read()); ast.parse(open('$AN/blender/make_normals_input_v1.py').read()); print('train_ckpt_v3.py + make_normals_input_v1.py ok')" || echo "FAILED: a script did not paste cleanly"
[ -f $AN/blender/val_select_v2.py ] && echo "val_select_v2.py present" || echo "FAILED: val_select_v2.py missing (paste §7a)"

cat > $AN/originals-blender-v4n.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=origb4n
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/zr2245/analysis/origb4n-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/origb4n-%x-%j.err
# originals-blender-v4n.sbatch (2026-11-04, VERSION v4n): = v4 + Metric3D normal prior on the Blender scene: stage A makes jpg inputs, runs upstream extract_normals.py, masks the
# background with the PNG alpha (normals_masked/); stage B trains through train_ckpt_v3 --normals so the Blender cameras carry normal_map like the COLMAP path, every 1000 its from 12000; val_select_v2.py picks the
# iteration by the official val split, the level-1.0 protocol runs on $G-best (point_cloud/iteration_30000 -> chosen checkpoint), unselected
# checkpoints deleted. Black background, no --indoor unless INDOOR=1. Output /scratch/zr2245/$SCENE-blender4n (+ -best), table originals-blender-val4n.txt.
#   sbatch --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1 --job-name=lego-origb4n /scratch/zr2245/analysis/originals-blender-v4n.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=train; INDOOR=${INDOOR:-0}
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; G=${G:-/scratch/zr2245/$SCENE-blender4n}; GB=$G-best
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; VS=$AN/blender/val_select_v2.py; CK=$AN/blender/train_ckpt_v3.py; MI=$AN/blender/make_normals_input_v1.py
K=${K:-4096}; R=${R:-256}; DC=${DC:-q7}; XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; TAB=$AN/originals-blender-val4n.txt
SAVES="12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000"
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
C=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
echo "scene $SCENE | DATA $DATA | G $G | blender, black bg, val-selected | flag '$FLAG' | XYZ $XYZ | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $DATA/transforms_train.json $DATA/transforms_val.json $DATA/transforms_test.json $VS $CK $MI $LAB/upstream/extract_normals.py $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do
  [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
mkdir -p $G
# --- stage A: normal prior inputs (skipped when normals_masked already has one png per train image)
n_tr=$(ls $DATA/train | grep -c '^r_[0-9]*\.png$'); n_nm=$(ls $DATA/normals_masked 2>/dev/null | grep -c '\.png$')
echo "train pngs $n_tr | masked normals $n_nm"
if [ "$n_nm" -lt "$n_tr" ]; then
  echo "=== stage A: jpg -> Metric3D -> mask ($(date)) ==="
  python $MI --scene $DATA --stage jpg > $G/normals.log 2>&1 || { echo "FAILED: NORMALS jpg stage"; tail -5 $G/normals.log; exit 1; }
  (cd $LAB/upstream && python extract_normals.py -s $DATA -i images >> $G/normals.log 2>&1) || { echo "FAILED: NORMALS extract_normals.py"; grep -n -m 6 -i 'error\|Traceback\|No module' $G/normals.log; tail -8 $G/normals.log; exit 1; }
  python $MI --scene $DATA --stage mask >> $G/normals.log 2>&1 || { echo "FAILED: NORMALS mask stage"; tail -5 $G/normals.log; exit 1; }
  grep -h "normals-input\|Processing" $G/normals.log | tail -3
  n_nm=$(ls $DATA/normals_masked | grep -c '\.png$'); [ "$n_nm" -ge "$n_tr" ] || { echo "FAILED: masked normals incomplete: $n_nm / $n_tr"; exit 1; }
fi
if [ ! -f $C ]; then
  echo "=== train original (saves every 1k from 12k) -> $G/original ($(date)) ==="
  (cd $LAB/upstream && python $CK --saves $SAVES --script $LAB/upstream/train.py --normals $DATA/normals_masked -- -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --test_iterations 7000 11000 14000 20000 25000 30000 $FLAG > $G/train.log 2>&1) \
    || { echo "FAILED: TRAIN"; grep -n -m 8 -i 'error\|Traceback' $G/train.log; tail -20 $G/train.log; exit 1; }
  [ -f $C ] || { echo "FAILED: TRAIN produced no checkpoint"; tail -20 $G/train.log; exit 1; }
else echo "exists: $C"; fi
grep -h "^\[ckpt\] normal\|^\[ckpt\] Blender\|^\[ckpt\] all\|FAILED" $G/train.log | tail -4
echo "--- test curve ---"; grep -h "Evaluating test" $G/train.log | sed 's/\[ITER \([0-9]*\)\] Evaluating test: L1 [0-9.]* PSNR \([0-9.]*\) SSIM \([0-9.]*\) LPIPS \([0-9.]*\).*/\1 \2 \3 \4/'
echo "=== val_select ($(date)) ==="
python $VS --lab $LAB --dataset $DATA --model $G/original --best $GB/original --delete_unselected > $G/val_select.log 2>&1 || { echo "FAILED: VAL_SELECT"; grep -n -m 8 -i 'error\|Traceback\|FAILED' $G/val_select.log; tail -15 $G/val_select.log; exit 1; }
grep -h "^\[val_select\] iter\|BEST\|VAL_SELECT_RESULT" $G/val_select.log
BEST=$(grep -h "VAL_SELECT_RESULT" $G/val_select.log | awk '{print $4}'); [ -n "$BEST" ] || { echo "FAILED: no VAL_SELECT_RESULT"; exit 1; }
[ -L $GB/original/point_cloud/iteration_30000 ] || { echo "FAILED: $GB/original/point_cloud/iteration_30000 symlink missing"; exit 1; }
echo "=== level-1.0 protocol on $GB (iteration $BEST) ($(date)) ==="
G=$GB DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $GB/level.log 2>&1 || { echo "FAILED: LEVEL-1.0"; grep -n -m 8 -i 'error\|Traceback' $GB/level.log; tail -12 $GB/level.log; exit 1; }
NA=$GB/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery"; tail -12 $GB/level.log; exit 1; }
python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; v=json.load(open('$G/original/val_select.json'))
last=[r for r in v['rows'] if r['iteration']==30000]
print('$SCENE ORIGINAL-BLENDER-VALSEL-NORMALS | best iter', v['best_iteration'], 'val', round(v['best_val_psnr'],3), '(30k val', round(last[0]['val_psnr'],3) if last else 'n/a', ') | vertices', b['vertices'], 'faces', b['faces'],
      '| level-1.0: finetune test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test', round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | blackbg $FLAG')
PY
echo "originals-blender-v4n $SCENE exit 0  $(date)"
SBEOF

rm -rf /scratch/zr2245/lego-blender4n /scratch/zr2245/lego-blender4n-best
j=$(sbatch --parsable --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1 --job-name=lego-origb4n $AN/originals-blender-v4n.sbatch) && echo "submitted lego-origb4n $j" || echo "FAILED: sbatch"
squeue -u $USER -o "%.10i %.16j %.28a %.2t %.10M %.6D %R"
```

看结果：

```bash
AN=/scratch/zr2245/analysis
ls -t $AN/origb4n-lego-origb4n-*.out | head -1 | xargs tail -45
cat $AN/originals-blender-val4n.txt 2>/dev/null; grep -h "^lego " $AN/originals-blender-val4.txt 2>/dev/null
```

对比的是两行：`lego ORIGINAL-BLENDER-VALSEL`（v4，无先验）vs `lego ORIGINAL-BLENDER-VALSEL-NORMALS`（v4n）。之后可以用 §8 的 render-blender-v1.sbatch 把 `MODEL=/scratch/zr2245/lego-blender4n-best/original` 渲出来看履带下的缝隙。

## §12 v5：修 v4 sbatch 的 symlink 检查（2026-11-04 13:40）

**问题**：val_select_v2 把选中的 checkpoint 写成真实目录（剪枝后的 state dict），v4/v4n sbatch 却还在检查 `-L ...iteration_30000`（symlink），于是 val_select 成功后被自己的检查拦下（`FAILED: ... symlink missing`）。训练和 val_select 产物都在。
**修法**：用 sed 从 v4 生成 v5（只改那一行 + 头注释），输出目录/表名不变（同一批 run 的续跑）；重跑时 sbatch 会跳过训练（30k 已存在），val_select 只剩 best + 30k 两个 checkpoint（其余已删），结论不变，再接交付链。

```bash
AN=/scratch/zr2245/analysis
for v in originals-blender-v4 originals-blender-v4n rd-prune-blender-v4; do
  w=${v/v4/v5}
  sed -e 's|\[ -L \$GB/original/point_cloud/iteration_30000 \] \|\| { echo "FAILED: \$GB/original/point_cloud/iteration_30000 symlink missing"; exit 1; }|[ -f $GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] \|\| { echo "FAILED: $GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt missing"; exit 1; }|' \
      -e "s|^# $v.sbatch (2026-11-04, VERSION v4|# $w.sbatch (2026-11-04, VERSION v5 = v4 + real-dir check instead of symlink check|" $AN/$v.sbatch > $AN/$w.sbatch
  n=$(grep -c "point_cloud_state_dict.pt \] ||" $AN/$w.sbatch); echo "$w.sbatch: fixed checks = $n (expect 1)"
done
sed -e 's/submit-blender-v4/submit-blender-v5/g' -e 's/originals-blender-v4.sbatch/originals-blender-v5.sbatch/g' -e 's/rd-prune-blender-v4.sbatch/rd-prune-blender-v5.sbatch/g' $AN/submit-blender-v4.sh > $AN/submit-blender-v5.sh
grep -c "v5.sbatch" $AN/submit-blender-v5.sh
# --- lego 法线先验 run：续跑（训练已存在，只重做 val_select + 交付）
rm -rf /scratch/zr2245/lego-blender4n-best
j=$(sbatch --parsable --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1 --job-name=lego-origb5n $AN/originals-blender-v5n.sbatch) && echo "submitted lego-origb5n $j" || echo "FAILED: sbatch"
squeue -u $USER -o "%.10i %.16j %.28a %.2t %.10M %.6D %R"
```

8 场景批（v4 提交过的话）：排队中（PD）的用 `scancel` 掉再 `bash $AN/submit-blender-v5.sh`；已在跑或已 FAILED 的等它们结束后 `bash $AN/submit-blender-v5.sh`（跳过训练，只补 val_select + 交付）。没提交过就直接 `bash $AN/submit-blender-v5.sh`。

另外请确认法线真的挂上了：
```bash
grep -h "\[ckpt\] normal\|\[ckpt\] Blender\|normal maps attached" /scratch/zr2245/lego-blender4n/train.log | head -3
ls /scratch/zr2245/datasets/nerf_synthetic/lego/normals_masked 2>/dev/null | wc -l
```

## §13 直接写 v5 文件（2026-11-04 13:58）：§12 的 sed 读的是不存在的 v4 文件，产出 0 字节

**问题**：§7b（写 originals-blender-v4 / rd-prune-blender-v4 / submit-blender-v4）从未粘贴过，§12 的 sed 从空输入生成了 0 字节的 v5 文件（`fixed checks = 0` 本该拦下，我没写成硬失败）。lego 的 v5n 不受影响（v4n 文件存在）。
**修法**：下面一段直接写三个完整的 v5 文件（= §7b 内容 + 真实目录检查），写完自检非空、bash -n 通过，再提交 16 个 job。

```bash
AN=/scratch/zr2245/analysis
cat > $AN/originals-blender-v5.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=origb4
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/zr2245/analysis/origb4-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/origb4-%x-%j.err
# originals-blender-v5.sbatch (2026-11-04, VERSION v5 (= v4 + real-dir check instead of symlink check)): = v3 + train_ckpt_v2 saves importance_score.pt per checkpoint and val_select_v2 materialises the chosen checkpoint with the final importance prune applied (real file, not a symlink); checkpoints written by the wrapper (upstream ignores --save_iterations), every 1000 its from 12000; val_select_v2.py picks the
# iteration by the official val split, the level-1.0 protocol runs on $G-best (point_cloud/iteration_30000 -> chosen checkpoint), unselected
# checkpoints deleted. Black background, no --indoor unless INDOOR=1. Output /scratch/zr2245/$SCENE-blender4 (+ -best), table originals-blender-val4.txt.
#   sbatch --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1 --job-name=lego-origb4 /scratch/zr2245/analysis/originals-blender-v5.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=train; INDOOR=${INDOOR:-0}
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; G=${G:-/scratch/zr2245/$SCENE-blender4}; GB=$G-best
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; VS=$AN/blender/val_select_v2.py; CK=$AN/blender/train_ckpt_v2.py
K=${K:-4096}; R=${R:-256}; DC=${DC:-q7}; XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; TAB=$AN/originals-blender-val4.txt
SAVES="12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000"
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
C=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
echo "scene $SCENE | DATA $DATA | G $G | blender, black bg, val-selected | flag '$FLAG' | XYZ $XYZ | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $DATA/transforms_train.json $DATA/transforms_val.json $DATA/transforms_test.json $VS $CK $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do
  [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
mkdir -p $G
if [ ! -f $C ]; then
  echo "=== train original (saves every 1k from 12k) -> $G/original ($(date)) ==="
  (cd $LAB/upstream && python $CK --saves $SAVES --script $LAB/upstream/train.py -- -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --test_iterations 7000 11000 14000 20000 25000 30000 $FLAG > $G/train.log 2>&1) \
    || { echo "FAILED: TRAIN"; grep -n -m 8 -i 'error\|Traceback' $G/train.log; tail -20 $G/train.log; exit 1; }
  [ -f $C ] || { echo "FAILED: TRAIN produced no checkpoint"; tail -20 $G/train.log; exit 1; }
else echo "exists: $C"; fi
grep -h "^\[ckpt\] saved\|^\[ckpt\] all\|FAILED" $G/train.log | tail -3
echo "--- test curve ---"; grep -h "Evaluating test" $G/train.log | sed 's/\[ITER \([0-9]*\)\] Evaluating test: L1 [0-9.]* PSNR \([0-9.]*\) SSIM \([0-9.]*\) LPIPS \([0-9.]*\).*/\1 \2 \3 \4/'
echo "=== val_select ($(date)) ==="
python $VS --lab $LAB --dataset $DATA --model $G/original --best $GB/original --delete_unselected > $G/val_select.log 2>&1 || { echo "FAILED: VAL_SELECT"; grep -n -m 8 -i 'error\|Traceback\|FAILED' $G/val_select.log; tail -15 $G/val_select.log; exit 1; }
grep -h "^\[val_select\] iter\|BEST\|VAL_SELECT_RESULT" $G/val_select.log
BEST=$(grep -h "VAL_SELECT_RESULT" $G/val_select.log | awk '{print $4}'); [ -n "$BEST" ] || { echo "FAILED: no VAL_SELECT_RESULT"; exit 1; }
[ -f $GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "FAILED: $GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt missing"; exit 1; }
echo "=== level-1.0 protocol on $GB (iteration $BEST) ($(date)) ==="
G=$GB DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $GB/level.log 2>&1 || { echo "FAILED: LEVEL-1.0"; grep -n -m 8 -i 'error\|Traceback' $GB/level.log; tail -12 $GB/level.log; exit 1; }
NA=$GB/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery"; tail -12 $GB/level.log; exit 1; }
python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; v=json.load(open('$G/original/val_select.json'))
last=[r for r in v['rows'] if r['iteration']==30000]
print('$SCENE ORIGINAL-BLENDER-VALSEL | best iter', v['best_iteration'], 'val', round(v['best_val_psnr'],3), '(30k val', round(last[0]['val_psnr'],3) if last else 'n/a', ') | vertices', b['vertices'], 'faces', b['faces'],
      '| level-1.0: finetune test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test', round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | blackbg $FLAG')
PY
echo "originals-blender-v4 $SCENE exit 0  $(date)"
SBEOF

cat > $AN/rd-prune-blender-v5.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=rd3b4
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/zr2245/analysis/rd3b4-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/rd3b4-%x-%j.err
# rd-prune-blender-v5.sbatch (2026-11-04, VERSION v5 (= v4 + real-dir check instead of symlink check)): = v3 + importance_score.pt per checkpoint + pruned materialisation in val_select_v2 (train_rd_v3 ignores --save_iterations); --indoor (INDOOR=1 default), every 1k from 12k,
# val_select_v2.py (official val split) + level-1.0 protocol on $G-best. Black background. Output /scratch/zr2245/$SCENE-rd3b4-lam$L-mu$MU (+ -best),
# table rd-prune3b4-$SCENE.txt. Same lambda rule, prices BPV 10 BPF 2, prune schedule 12000 14000 16000 18000, same codec.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=train; LAMS=${LAMS:?set LAMS="0.00062 0.0062" via --export}; INDOOR=${INDOOR:-1}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}
SAVES="12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000"
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; VS=$AN/blender/val_select_v2.py; CK=$AN/blender/train_ckpt_v2.py
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; TAB=$AN/rd-prune3b4-$SCENE.txt; TR=$AN/rdgrowth/train_rd_v3.py
echo "scene $SCENE | blender, black bg, val-selected | flag '$FLAG' | lams $LAMS | B/v $BPV B/face $BPF | prune at $PRUNE mu $MU max_frac $MAXF | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $VS $CK $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/transforms_train.json $DATA/transforms_val.json $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
grep -q "'--indoor'" $TR || { echo "FAILED: $TR has no --indoor flag"; exit 1; }
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-rd3b4-lam$L-mu$MU; GB=$G-best; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== $SCENE lam $L mu $MU $FLAG: train+prune (saves every 1k from 12k) ($(date)) ==="
  if [ ! -f $CB ]; then
    python $CK --saves $SAVES --script $TR -- --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF --test_iterations 7000 11000 14000 20000 25000 30000 $FLAG > $G/train.log 2>&1 \
      || { echo "FAILED: TRAIN $SCENE lam $L"; grep -n -m 8 -i 'error\|Traceback' $G/train.log; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE lam $L"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  grep -h "^\[RD prune" $G/train.log; grep -h "^\[ckpt\] all\|FAILED" $G/train.log | tail -2
  echo "--- test curve ---"; grep -h "Evaluating test" $G/train.log | sed 's/\[ITER \([0-9]*\)\] Evaluating test: L1 [0-9.]* PSNR \([0-9.]*\) SSIM \([0-9.]*\) LPIPS \([0-9.]*\).*/\1 \2 \3 \4/'
  echo "=== $SCENE lam $L: val_select ($(date)) ==="
  python $VS --lab $LAB --dataset $DATA --model $G/original --best $GB/original --delete_unselected > $G/val_select.log 2>&1 || { echo "FAILED: VAL_SELECT $SCENE lam $L"; grep -n -m 8 -i 'error\|Traceback\|FAILED' $G/val_select.log; tail -15 $G/val_select.log; exit 1; }
  grep -h "BEST\|VAL_SELECT_RESULT" $G/val_select.log
  [ -f $GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "FAILED: $GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt missing"; exit 1; }
  echo "=== $SCENE lam $L mu $MU: level-1.0 protocol on $GB ($(date)) ==="
  G=$GB DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $GB/level.log 2>&1 \
    || { echo "FAILED: LEVEL-1.0 $SCENE lam $L"; tail -8 $GB/level.log; exit 1; }
  NA=$GB/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE lam $L"; tail -8 $GB/level.log; exit 1; }
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; h=f['history']; v=json.load(open('$G/original/val_select.json'))
ev=[json.loads(l) for l in open('$G/original/rd_log.jsonl')]; dl=[e for e in ev if e['event']=='delaunay']; pr=[e for e in ev if e['event']=='prune']
last=[r for r in v['rows'] if r['iteration']==30000]
print('$SCENE lam $L RD-GROWTH+PRUNE-mu$MU-BLENDER-VALSEL | best iter', v['best_iteration'], 'val', round(v['best_val_psnr'],3), '(30k val', round(last[0]['val_psnr'],3) if last else 'n/a', ') | vertices', b['vertices'], 'faces', b['faces'],
      '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3), 'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | grow: V@delaunay', dl[0]['vertices'] if dl else None,
      '| prune rounds', [(e['iteration'], e['vertices_before'], e['vertices_after'], 'pairs', e['n_pairs'], 'cap', e['max_pairs']) for e in pr], '| blackbg $FLAG')
PY
done
echo "===== $TAB ====="; cat $TAB
echo "rd-prune3b2 $SCENE exit 0  $(date)"
SBEOF

cat > $AN/submit-blender-v5.sh <<'SHEOF'
#!/usr/bin/env bash
# submit-blender-v5.sh (2026-11-04, v5 = v4 sbatch with real-dir check): 8 NeRF-Synthetic scenes x (1 original job + 1 lambda job of 2 lambdas 6.2e-4 / 6.2e-3), all --indoor + val-selected, black bg, general.
# ~ 8 x 0.6 h + 8 x 1.4 h ~ 16 GPU-h. A scene whose -blender2 original already exists (none expected for v4) skips its training.
AN=/scratch/zr2245/analysis; XYZ=${XYZ:-q14}
for f in $AN/originals-blender-v5.sbatch $AN/rd-prune-blender-v5.sbatch $AN/blender/val_select_v2.py $AN/blender/train_ckpt_v2.py $AN/rdgrowth/train_rd_v3.py; do [ -f $f ] || { echo "FAILED: missing $f"; exit 1; }; done
n=0
for S in chair drums ficus hotdog lego materials mic ship; do
  [ -f /scratch/zr2245/datasets/nerf_synthetic/$S/transforms_val.json ] || { echo "FAILED: no dataset for $S"; continue; }
  [ -f /scratch/zr2245/$S-blender4/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] && echo "note: $S original exists, job will skip its training (check it was INDOOR=1)"
  j=$(sbatch --parsable --time=02:30:00 --export=ALL,SCENE=$S,XYZ=$XYZ,INDOOR=1 --job-name=$S-origb4 $AN/originals-blender-v5.sbatch) \
    && { echo "submitted $S-origb4 $j  XYZ $XYZ  indoor"; n=$((n+1)); } || echo "FAILED: sbatch $S original"
  LAMS="0.00062 0.0062"
  j=$(sbatch --parsable --time=03:30:00 --export=ALL,SCENE=$S,XYZ=$XYZ,INDOOR=1,LAMS="$LAMS",BPV=10.0,BPF=2.0,MU=1.0,MAXF=0.15 --job-name=$S-b4-62 $AN/rd-prune-blender-v5.sbatch) \
    && { echo "submitted $S-b4-62 $j  lams $LAMS  XYZ $XYZ  indoor"; n=$((n+1)); } || echo "FAILED: sbatch $S lams"
done
echo "submitted $n jobs (expected 16)"
squeue -u $USER -o "%.10i %.16j %.28a %.2t %.10M %.6D %R"
SHEOF

cat > $AN/check-v5.sh <<'CHEOF'
AN=/scratch/zr2245/analysis; ok=1
for f in originals-blender-v5.sbatch rd-prune-blender-v5.sbatch submit-blender-v5.sh; do
  s=$(stat -c %s $AN/$f); bash -n $AN/$f && echo "$f: $s bytes, syntax OK" || { echo "FAILED: $f syntax"; ok=0; }
  [ "$s" -gt 1000 ] || { echo "FAILED: $f is empty/short"; ok=0; }
done
n=$(grep -c "point_cloud_state_dict.pt \] ||" $AN/originals-blender-v5.sbatch $AN/rd-prune-blender-v5.sbatch | awk -F: '{s+=$2} END{print s}'); echo "fixed checks = $n (expect 2)"; [ "$n" = 2 ] || ok=0
grep -c "symlink missing" $AN/originals-blender-v5.sbatch $AN/rd-prune-blender-v5.sbatch
[ $ok = 1 ] && echo "CHECK OK" || echo "CHECK FAILED - do not submit"
CHEOF
bash $AN/check-v5.sh
```

`CHECK OK` 之后再提交：

```bash
AN=/scratch/zr2245/analysis
bash $AN/submit-blender-v5.sh
```

## §14 lego 高码率点（目标交付 25-30 MB）：train_ckpt_v4 --override + rd-hirate-blender-v1.sbatch（2026-11-04 16:00，≈0.8-1 GPU-h）

**为什么不能只把 λ 调小**：train_rd_v3 的增长受 upstream 限速器 `min(max_points, add_percentage·V) − V`（每个 densify 间隔），`--indoor` 预设 add_percentage=1.27，并且 `update_indoor(ops)` 在解析之后才应用，所以命令行 `--add_percentage` 会被它覆盖。lego λ6.2e-4 停在 540k V（和原模型 498k 差不多）很可能是限速器而不是 λ 在起作用。
**做法**：train_ckpt_v4 = v2 + `--override add_percentage=1.5 max_points=1500000`，在 extract() 和 update_indoor() 之后各重放一次，并在训练开始前打印/核验有效值（键名打错 = FAILED，不花 GPU）。λ=6.2e-5（几乎不设价），prune 轮保留但基本不剪。预计 V@delaunay ≈ 1.2-1.5 M，交付 ≈ 20 B/顶点 → 25-30 MB。
**防过拟合**（不调参、不加新正则）：--indoor 预设 + 官方 val split 选 checkpoint（顶点越多峰值越早，这就是它的作用）+ importance 剪枝；其余任何手段都是 per-scene 调参或 +++ 的新正则。

### §14a 先看两件事（CPU，不花 GPU）

```bash
U=/scratch/zr2245/meshsplatting-original-lossless-v1/upstream
echo "=== update_indoor 到底改了什么（看 add_percentage / max_points / prune_size）==="; grep -n "def update_indoor" -A 25 $U/arguments/__init__.py | grep -n "add_percentage\|max_points\|prune_size\|def update_indoor"
echo "=== 默认值 ==="; grep -n "add_percentage\|max_points" $U/arguments/__init__.py | head -6
echo "=== lego λ6.2e-4：限速器是否在起作用（split == limit 就是）==="; grep -h "^\[RD [0-9]*\] V" /scratch/zr2245/lego-rd3b4-lam0.00062-mu1.0/original/../train.log 2>/dev/null | head -3; grep -h "^\[RD [0-9]*\] V" /scratch/zr2245/lego-rd3b4-lam0.00062-mu1.0/train.log 2>/dev/null | tail -3
```

### §14b 写文件 + 提交（一段）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/blender $AN/viewer-v2/msv
cat > $AN/blender/train_ckpt_v4.py <<'PYEOF_CK4'
"""train_ckpt_v4.py (VERSION train_ckpt_v4 2026-11-04): train_ckpt_v2 + --override KEY=VALUE for OptimizationParams that survive --indoor.

v4: train_rd_v3 / train.py apply `update_indoor(ops)` AFTER parsing, so a CLI value such as --add_percentage is overwritten by the
indoor preset (add_percentage 1.27, prune_size 1300, ...). --override add_percentage=1.5 max_points=1500000 patches
arguments.OptimizationParams.extract AND arguments.update_indoor so the given values are re-applied after both; each key must
already exist on the params object (typo = FAILED before training) and the effective values are printed and verified before
training starts.

v2: also saves the model's per-face importance_score next to each checkpoint (importance_score.pt), so val_select_v2 can apply the
same final importance prune the training scripts apply before their own last save (lego: 2.98 M -> 1.32 M faces).

Why: upstream train.py accepts --save_iterations but never uses it (its only scene.save() is the final one), and our
train_rd_v3.py inherited that. The NeRF-Synthetic probe needs checkpoints every 1k from 12k for validation-split selection
(val_select_v1.py). No upstream file is edited: this wrapper hooks TriangleModel.update_learning_rate, which both training
loops call once per iteration with the iteration index, and saves through Scene.save (so the on-disk format is exactly the
one the delivery chain loads: point_cloud/iteration_N/point_cloud_state_dict.pt).

Note: the hook fires at the START of iteration N (before that iteration's optimizer step), so iteration_N holds the state after
N-1 steps; the "[ITER N] Evaluating" line in the log is after step N. Intermediate checkpoints also skip the final
importance-based triangle prune that the scripts apply before their own last save (a few % more faces; rendering unchanged).

Usage:
  python train_ckpt_v1.py --saves 12000 13000 ... --script /path/upstream/train.py      -- -s DATA -i train -r 1 --eval ... -m MODEL ...
  python train_ckpt_v4.py --saves 12000 ... --override add_percentage=1.5 max_points=1500000 --script /path/rdgrowth/train_rd_v3.py -- --lab LAB -s DATA ... -m MODEL --lam ... --indoor
"""
import argparse, inspect, os, runpy, sys

VERSION = 'train_ckpt_v4 2026-11-04'


def main():
    if '--' not in sys.argv:
        sys.exit('FAILED: usage: train_ckpt_v4.py --saves N... [--override K=V ...] --script PATH -- <script args>')
    k = sys.argv.index('--')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--saves', nargs='+', type=int, required=True)
    ap.add_argument('--script', required=True, help='upstream/train.py or analysis/rdgrowth/train_rd_v3.py')
    ap.add_argument('--override', nargs='*', default=[], help='KEY=VALUE OptimizationParams values applied after extract() and after update_indoor()')
    a = ap.parse_args(sys.argv[1:k]); rest = sys.argv[k + 1:]
    script = os.path.abspath(a.script)
    if not os.path.isfile(script):
        sys.exit(f'FAILED: script not found: {script}')
    # model path and upstream dir from the script's own arguments
    mp = None
    for i, t in enumerate(rest):
        if t in ('-m', '--model_path') and i + 1 < len(rest):
            mp = os.path.abspath(rest[i + 1])
    if mp is None:
        sys.exit('FAILED: no -m/--model_path in the script arguments')
    if '--lab' in rest:
        up = os.path.join(os.path.abspath(rest[rest.index('--lab') + 1]), 'upstream')
    else:
        up = os.path.dirname(script)
    if not os.path.isfile(os.path.join(up, 'train.py')):
        sys.exit(f'FAILED: upstream train.py not found under {up}')
    src = open(script).read()
    if 'update_learning_rate(iteration)' not in src:
        sys.exit(f'FAILED: {script} does not call update_learning_rate(iteration) once per iteration; hook would never fire')
    sys.path.insert(0, os.path.join(up, 'submodules/effrdel/src')); sys.path.insert(0, up); os.chdir(up)
    import scene.triangle_model as tm
    from scene import Scene
    import arguments as _args
    overrides = {}
    for kv in a.override:
        if '=' not in kv:
            sys.exit(f'FAILED: --override expects KEY=VALUE, got {kv!r}')
        key, val = kv.split('=', 1)
        overrides[key] = val
    if overrides:
        if not hasattr(_args, 'update_indoor') or not hasattr(_args, 'OptimizationParams'):
            sys.exit('FAILED: upstream arguments module lacks update_indoor / OptimizationParams; cannot apply --override')
        def _apply(ops, where):
            for key, val in overrides.items():
                if not hasattr(ops, key):
                    sys.exit(f'FAILED: OptimizationParams has no attribute {key!r} ({where}); known: {sorted(k for k in vars(ops) if not k.startswith("_"))}')
                cur = getattr(ops, key); typ = type(cur) if cur is not None else float
                try:
                    newv = typ(val) if typ is not bool else (val.lower() in ('1', 'true', 'yes'))
                except Exception as e:
                    sys.exit(f'FAILED: cannot cast override {key}={val!r} to {typ.__name__}: {e}')
                setattr(ops, key, newv)
            eff = {k: getattr(ops, k) for k in overrides}
            print(f'[ckpt] overrides applied after {where}: {eff}', flush=True)
            for key, val in overrides.items():
                if str(getattr(ops, key)) != str(type(getattr(ops, key))(val)) and float(getattr(ops, key)) != float(val):
                    sys.exit(f'FAILED: override {key}={val} not effective after {where}: {getattr(ops, key)}')
            return ops
        _orig_extract = _args.OptimizationParams.extract
        def _extract(self, args):
            return _apply(_orig_extract(self, args), 'extract')
        _args.OptimizationParams.extract = _extract
        _orig_indoor = _args.update_indoor
        def _indoor(ops):
            ops = _orig_indoor(ops)
            return _apply(ops, 'update_indoor')
        _args.update_indoor = _indoor
        print(f'[ckpt] --override active: {overrides} (patched OptimizationParams.extract and update_indoor)', flush=True)
    if not hasattr(tm.TriangleModel, 'update_learning_rate'):
        sys.exit('FAILED: TriangleModel has no update_learning_rate')
    ssrc = inspect.getsource(Scene.save)
    print(f'[ckpt] {VERSION} | saves {sorted(a.saves)} | model_path {mp} | script {script}\n[ckpt] Scene.save source:\n{ssrc}', flush=True)
    if 'model_path' not in ssrc:
        sys.exit('FAILED: Scene.save does not use self.model_path; hook cannot reuse it')

    class _Fake:  # stands in for the Scene object inside Scene.save: model_path + whatever attribute holds the model
        def __init__(self, model_path, model):
            self.model_path = model_path; self._model = model
        def __getattr__(self, name):
            return self._model

    saves = set(a.saves); done = set(); orig = tm.TriangleModel.update_learning_rate

    def hooked(self, iteration):
        r = orig(self, iteration)
        if iteration in saves and iteration not in done:
            done.add(iteration)
            os.makedirs(os.path.join(mp, 'point_cloud', f'iteration_{iteration}'), exist_ok=True)
            out = Scene.save(_Fake(mp, self), iteration)
            nv = int(self.vertices.shape[0]) if hasattr(self, 'vertices') else -1
            imp = getattr(self, 'importance_score', None)
            if imp is None:
                sys.exit('FAILED: model has no importance_score attribute; cannot reproduce the final importance prune')
            import torch
            torch.save(imp.detach().cpu().clone(), os.path.join(mp, 'point_cloud', f'iteration_{iteration}', 'importance_score.pt'))
            nf = int(self._triangle_indices.shape[0]); nkeep = int((imp.detach().cpu().reshape(-1) > 0.5).sum())
            print(f'[ckpt] saved iteration {iteration} (state after {iteration - 1} steps) vertices {nv} faces {nf} (importance>0.5: {nkeep}) -> {out}', flush=True)
        return r

    tm.TriangleModel.update_learning_rate = hooked
    sys.argv = [script] + rest
    runpy.run_path(script, run_name='__main__')
    missing = sorted(i for i in saves if not os.path.isfile(os.path.join(mp, 'point_cloud', f'iteration_{i}', 'point_cloud_state_dict.pt')))
    if missing:
        sys.exit(f'FAILED: checkpoints missing after training: {missing}')
    missing = sorted(i for i in saves if not os.path.isfile(os.path.join(mp, 'point_cloud', f'iteration_{i}', 'importance_score.pt')))
    if missing:
        sys.exit(f'FAILED: importance_score.pt missing after training: {missing}')
    print(f'[ckpt] all {len(saves)} checkpoints present', flush=True)


if __name__ == '__main__':
    main()
PYEOF_CK4
cat > $AN/rd-hirate-blender-v1.sbatch <<'SBEOF_HI1'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=rd3hi
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/zr2245/analysis/rd3hi-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/rd3hi-%x-%j.err
# rd-hirate-blender-v1.sbatch (2026-11-04, VERSION v1): HIGH-RATE point for one NeRF-Synthetic scene = rd-prune-blender-v5 with ONE lambda and the
# upstream growth limiter raised through train_ckpt_v4 --override (add_percentage, max_points survive --indoor). Everything else identical:
# --indoor, black bg, checkpoints every 1k from 12k, val_select_v2 (official val split, importance prune), level-1.0 delivery, then viewer .msv export.
# Output /scratch/zr2245/$SCENE-rd3hi-lam$LAM-ap$AP-mp$MP (+ -best), table rd-hirate-blender-v1.txt, msv $AN/viewer-v2/msv/$SCENE-hi-lam$LAM.msv.
# 48G host RAM because the mesh is ~3x the lambda 6.2e-4 one (val_select + delivery hold several copies).
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=train; LAM=${LAM:?set LAM=0.000062 via --export}; AP=${AP:?set AP=1.5 (add_percentage) via --export}; MP=${MP:?set MP=1500000 (max_points) via --export}; INDOOR=${INDOOR:-1}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}
SAVES="12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000"
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; VS=$AN/blender/val_select_v2.py; CK=$AN/blender/train_ckpt_v4.py; EXP=$AN/viewer-v1/export_viewer_v1.py; MSV=$AN/viewer-v2/msv
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; TAB=$AN/rd-hirate-blender-v1.txt; TR=$AN/rdgrowth/train_rd_v3.py
echo "scene $SCENE | HIGH-RATE blender, black bg, val-selected | flag '$FLAG' | lam $LAM add_percentage $AP max_points $MP | B/v $BPV B/face $BPF | prune at $PRUNE mu $MU max_frac $MAXF | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $VS $CK $EXP $PKG/stage2_run.sh $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/transforms_train.json $DATA/transforms_val.json $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
grep -q "'--indoor'" $TR || { echo "FAILED: $TR has no --indoor flag"; exit 1; }
grep -q "update_indoor(ops)" $TR || { echo "FAILED: $TR does not call update_indoor(ops); override wrapper assumption broken"; exit 1; }
grep -q "opt.max_points, int(opt.add_percentage \* V)" $TR || { echo "FAILED: growth limiter line not found in $TR (expected min(opt.max_points, add_percentage*V))"; exit 1; }
grep -q "'--override'" $CK || { echo "FAILED: $CK is not train_ckpt_v4 (no --override)"; exit 1; }
mkdir -p $MSV
for L in $LAM; do
  G=/scratch/zr2245/$SCENE-rd3hi-lam$L-ap$AP-mp$MP; GB=$G-best; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== $SCENE lam $L ap $AP mp $MP $FLAG: train+prune (saves every 1k from 12k) ($(date)) ==="
  if [ ! -f $CB ]; then
    python $CK --saves $SAVES --override add_percentage=$AP max_points=$MP --script $TR -- --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF --test_iterations 7000 11000 14000 20000 25000 30000 $FLAG > $G/train.log 2>&1 \
      || { echo "FAILED: TRAIN $SCENE lam $L"; grep -n -m 8 -i 'error\|Traceback' $G/train.log; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE lam $L"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  grep -h "overrides applied after update_indoor" $G/train.log | tail -1 | grep -q "'add_percentage': $AP" || { echo "FAILED: override not confirmed in train.log (expected add_percentage $AP after update_indoor)"; grep -h "\[ckpt\] override" $G/train.log; exit 1; }
  echo "--- growth (first 2 / last 3 densify lines; split == limit means the limiter, not lambda, stopped growth) ---"; grep -h "^\[RD [0-9]*\] V" $G/train.log | head -2; grep -h "^\[RD [0-9]*\] V" $G/train.log | tail -3
  grep -h "^\[RD prune" $G/train.log; grep -h "^\[ckpt\] all\|FAILED" $G/train.log | tail -2
  echo "--- test curve ---"; grep -h "Evaluating test" $G/train.log | sed 's/\[ITER \([0-9]*\)\] Evaluating test: L1 [0-9.]* PSNR \([0-9.]*\) SSIM \([0-9.]*\) LPIPS \([0-9.]*\).*/\1 \2 \3 \4/'
  echo "=== $SCENE lam $L: val_select ($(date)) ==="
  python $VS --lab $LAB --dataset $DATA --model $G/original --best $GB/original --delete_unselected > $G/val_select.log 2>&1 || { echo "FAILED: VAL_SELECT $SCENE lam $L"; grep -n -m 8 -i 'error\|Traceback\|FAILED' $G/val_select.log; tail -15 $G/val_select.log; exit 1; }
  grep -h "BEST\|VAL_SELECT_RESULT" $G/val_select.log
  [ -f $GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "FAILED: $GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt missing"; exit 1; }
  echo "=== $SCENE lam $L mu $MU: level-1.0 protocol on $GB ($(date)) ==="
  G=$GB DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $GB/level.log 2>&1 \
    || { echo "FAILED: LEVEL-1.0 $SCENE lam $L"; tail -8 $GB/level.log; exit 1; }
  NA=$GB/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE lam $L"; tail -8 $GB/level.log; exit 1; }
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; h=f['history']; v=json.load(open('$G/original/val_select.json'))
ev=[json.loads(l) for l in open('$G/original/rd_log.jsonl')]; dl=[e for e in ev if e['event']=='delaunay']; pr=[e for e in ev if e['event']=='prune']
last=[r for r in v['rows'] if r['iteration']==30000]
print('$SCENE lam $L HIGH-RATE ap$AP mp$MP RD-GROWTH+PRUNE-mu$MU-BLENDER-VALSEL | best iter', v['best_iteration'], 'val', round(v['best_val_psnr'],3), '(30k val', round(last[0]['val_psnr'],3) if last else 'n/a', ') | vertices', b['vertices'], 'faces', b['faces'],
      '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3), 'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | grow: V@delaunay', dl[0]['vertices'] if dl else None,
      '| prune rounds', [(e['iteration'], e['vertices_before'], e['vertices_after'], 'pairs', e['n_pairs'], 'cap', e['max_pairs']) for e in pr], '| blackbg $FLAG')
PY
  echo "=== $SCENE lam $L: viewer .msv export ($(date)) ==="
  A=$NA/deliver/delivery/asset.mslb; [ -f $A ] || A=$(find $NA/deliver -name asset.mslb 2>/dev/null | head -1)
  [ -n "$A" ] && [ -f "$A" ] || { echo "FAILED: no asset.mslb under $NA/deliver"; exit 1; }
  CAM=$G/original/cameras.json; [ -f $CAM ] || CAM=""
  bash $PKG/stage2_run.sh $EXP --asset $A ${CAM:+--cameras $CAM} --out $MSV/$SCENE-hi-lam$L.msv --label "$SCENE hi lam$L" --native-cache $AN/viewer-v2/native-cache > $G/export_msv.log 2>&1 \
    || { echo "FAILED: msv export"; tail -8 $G/export_msv.log; exit 1; }
  grep -h "^VIEWER" $G/export_msv.log
done
echo "===== $TAB ====="; cat $TAB
echo "rd-hirate-blender-v1 $SCENE exit 0  $(date)"
SBEOF_HI1
python -m py_compile $AN/blender/train_ckpt_v4.py && bash -n $AN/rd-hirate-blender-v1.sbatch && grep -c "override" $AN/blender/train_ckpt_v4.py && echo "files OK" || echo "FAILED: file check"
j=$(sbatch --parsable --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1,LAM=0.000062,AP=1.5,MP=1500000 --job-name=lego-hi $AN/rd-hirate-blender-v1.sbatch) && echo "submitted lego-hi $j" || echo "FAILED: sbatch"
squeue -u $USER -o "%.10i %.16j %.28a %.2t %.10M %.6D %R"
```

### §14c 看结果

```bash
AN=/scratch/zr2245/analysis
ls -t $AN/rd3hi-lego-hi-*.out | head -1 | xargs tail -40
cat $AN/rd-hirate-blender-v1.txt
ls -la $AN/viewer-v2/msv/lego-hi-*.msv
```
笔记本：`scp 'torch:/scratch/zr2245/analysis/viewer-v2/msv/lego-hi-lam0.000062.msv' ~/msv2/`，用 viewer-v3.html 打开。对照行：rd-prune3b4-lego.txt 里的 λ6.2e-4（9.26 MB / 33.50）。

## §15 网格质量正则（train_rd_v4 = v3 + mesh_reg_v1）：lego λ6.2e-4 三个配置（2026-11-05 03:30，≈1.5 GPU-h，general）

**目的**（用户 11/05 决定，一次定胜负）：表面 smooth、不要跳跃连接（jump triangles）。做法 = 改训练 loss + 硬过滤，不动 upstream，不动 v3（v4 是 v3 的 patch 副本，所有 `--w_* 0 --filter_aspect 0` 时行为 = v3）。
**三项（全部无量纲，一组权重跨场景通用）**：
- `sliver`：每个面 aspect = 最长边² / (2·面积)（尖刺三角形 ≈ 长/底），loss = mean relu(log(aspect/20))，从第 500 步起全程作用（soup 阶段就开始）；> a0·e⁸ 的退化面截断（不给 1/面积 的梯度尖峰）。
- `lap`：均匀 Laplacian |mean(邻居) − v| / 平均入射边长 的平方，**只在 Delaunay 重建（11000）之后**（之前顶点不共享），开边界顶点排除（否则只会往里缩）。
- `normal`：相邻面 1 − nᵢ·nⱼ（face_adjacency 每次拓扑变化重算，缓存），同样 Delaunay 之后。
- **硬过滤**（jump filter）：aspect > 20 且最长边 > 6×中位边 的面在 Delaunay 重建后、每轮 λ-prune 后、最终 importance 剪枝后直接删（`prune_triangles(keep)`，面数断言）。大而平整的面不会被删（aspect≈1-2）。
**日志**：每 500 步 `[REG it] main … reg … | |g_main| |g_reg| ratio`（顶点梯度范数比 = 正则到底有没有“说话”），事件 `[REG jumpfilter …]`、`[REG mesh …]`（sliver %、lap、normal、二面角>60° 比例）写进 rd_log.jsonl。
**启动自检（不花 GPU 就失败）**：sbatch 先跑 `test_mesh_reg_v1.py`（CPU），train_rd_v4 开训前断言 `triangles.vertices` 是 leaf 参数、`prune_triangles` 存在；训练后 grep `[REG] mesh_reg_v1` 横幅 + Delaunay 后的 jumpfilter 事件。
**三个配置**（都在 lego λ6.2e-4，对照 = rd-prune3b4-lego 的 9.26 MB / 33.50 + §9b 网格统计；sbatch 会顺手对 v3 的 best checkpoint 再跑一次 mesh_stats_v2 放同一张表）：

| CFG | w_sliver | w_lap | w_normal | filter | 问的问题 |
|---|---|---|---|---|---|
| a | 0.05 | 0 | 0 | 20 / 6× | 只去尖刺 + 硬过滤，PSNR 掉多少 |
| b | 0.05 | 0.05 | 0.05 | 20 / 6× | 加 smooth（Laplacian + 法线） |
| c | 0.2 | 0.2 | 0.2 | 20 / 6× | 4× 权重：正则压过渲染梯度时的上限/代价 |

**判据**（看三样）：(1) viewer-v3 近看是否 smooth、无跳跃面（msv 自动导出）；(2) 表里 `mesh@final` 的 sliver % / dih>60 % 和 MESH_STATS 的 non-manifold / sliver<10° 对比 v3-ref 明显下降；(3) PSNR 代价（接受，但要写进表）。“几乎成功”= (1)(2) 成立且 PSNR 代价 ≲ 1 dB → 进 CVPR；否则 → +++ 前期探索。

### §15a 写 5 个文件 + 自检（整块粘贴，不提交 job；`CHECK OK` 才往下）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/rdgrowth $AN/blender $AN/viewer-v2/msv
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
cat > $AN/rdgrowth/mesh_reg_v1.py <<'PYEOF_MR1'
"""mesh_reg_v1.py (VERSION mesh_reg_v1 2026-11-05): mesh-quality regularisers for MeshSplatting training (pure torch, CPU-testable).

Three scale-free terms (all dimensionless, so one weight works across scenes):
  sliver_loss(pos, faces, a0)   per-face aspect = longest_edge^2 / (2 * area)  (= 1/height ratio); loss = mean(relu(log(aspect / a0)))
                                  -> penalises long thin "jump" triangles only; large flat well-shaped triangles are untouched.
  laplacian_loss(pos, faces)    uniform Laplacian delta_v = mean(neighbours) - v, normalised by the vertex's mean incident edge length
                                  -> mean(|delta_v| / s_v)^2 over interior vertices (open-boundary vertices excluded, they would only shrink inward);
                                  only meaningful once vertices are shared (after the restricted Delaunay).
  normal_loss(pos, faces, adj)  1 - n_i . n_j over interior edges (faces i, j share the edge); adj from face_adjacency(faces) (cached by caller).
  jump_face_mask(pos, faces, a1, k)  hard filter: faces with aspect > a1 AND longest edge > k * median edge length -> drop (post-Delaunay).
                                  aspect ~ length/base for a spike, so a1=20 = spikes at least 20x longer than their base; big flat well-shaped faces survive.
"""
import torch

VERSION = 'mesh_reg_v1 2026-11-05b'


def _edges_of(pos, faces):
    t = pos[faces.long()]                           # [T,3,3]
    e = torch.stack([t[:, 1] - t[:, 0], t[:, 2] - t[:, 1], t[:, 0] - t[:, 2]], dim=1)   # [T,3,3]
    return t, e


def face_aspect(pos, faces, eps=1e-12):
    """aspect = longest_edge^2 / (2*area) >= ~1.15 (equilateral); long thin slivers -> large. Returns (aspect [T], longest [T], area [T])."""
    t, e = _edges_of(pos, faces)
    l2 = (e * e).sum(-1)                            # [T,3] squared lengths
    longest2 = l2.max(dim=1).values
    area = 0.5 * torch.linalg.norm(torch.cross(e[:, 0], -e[:, 2], dim=1), dim=1)
    aspect = longest2 / (2.0 * area + eps)
    return aspect, torch.sqrt(longest2), area


def sliver_loss(pos, faces, a0=20.0, max_log=8.0):
    """mean over faces of min(relu(log(aspect/a0)), max_log); the cap gives numerically degenerate faces (aspect > a0*e^8) no gradient
    (they render nothing and are removed by the upstream importance prune) instead of a 1/area gradient spike into Adam's moments."""
    aspect, _, _ = face_aspect(pos, faces)
    return torch.relu(torch.log(aspect.clamp_min(1e-6) / a0)).clamp_max(max_log).mean()


@torch.no_grad()
def jump_face_mask(pos, faces, a1=20.0, k=6.0):
    """keep-mask [T] (True = keep). Drops faces that are both very thin (aspect > a1) and long (longest edge > k * median edge)."""
    aspect, longest, _ = face_aspect(pos, faces)
    med = torch.median(longest)
    drop = (aspect > a1) & (longest > k * med)
    return ~drop, dict(median_edge=float(med), dropped=int(drop.sum()), aspect_q50=float(aspect.median()), aspect_q99=float(torch.quantile(aspect, 0.99)) if aspect.numel() < 16_000_000 else -1.0)


@torch.no_grad()
def boundary_vertices(faces, V):
    """bool [V]: vertex lies on an edge used by exactly one face (open boundary)."""
    f = faces.long()
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]], dim=0)
    e, _ = torch.sort(e, dim=1)
    key = e[:, 0] * V + e[:, 1]
    uk, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
    bnd_edge = cnt[inv] == 1
    out = torch.zeros(V, dtype=torch.bool, device=faces.device)
    out[e[bnd_edge].flatten()] = True
    return out


def laplacian_loss(pos, faces, exclude_boundary=True, boundary=None):
    """mean over vertices of (|mean(neighbours) - v| / mean incident edge length)^2 (dimensionless). Vertices with no edges contribute 0;
    with exclude_boundary, vertices on an open boundary edge are excluded too (the uniform Laplacian would only pull them inward);
    boundary = precomputed boundary_vertices(faces, V) mask (cached by the caller per topology), else computed here."""
    f = faces.long()
    i = torch.cat([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    j = torch.cat([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    V = pos.shape[0]
    nbr_sum = torch.zeros_like(pos).index_add_(0, i, pos[j])
    cnt = torch.zeros(V, device=pos.device, dtype=pos.dtype).index_add_(0, i, torch.ones_like(i, dtype=pos.dtype))
    elen = torch.linalg.norm(pos[j] - pos[i], dim=1)
    esum = torch.zeros(V, device=pos.device, dtype=pos.dtype).index_add_(0, i, elen)
    has = cnt > 0
    if exclude_boundary:
        has = has & ~(boundary if boundary is not None else boundary_vertices(faces, V))
    c = cnt.clamp_min(1.0)
    delta = nbr_sum / c[:, None] - pos
    s = (esum / c).clamp_min(1e-12)
    r = torch.linalg.norm(delta, dim=1) / s
    r = torch.where(has, r, torch.zeros_like(r))
    return (r * r).sum() / has.sum().clamp_min(1)


@torch.no_grad()
def face_adjacency(faces):
    """Pairs of faces sharing an edge (interior edges only; edges with >2 faces contribute their first two). Returns long [P,2]."""
    f = faces.long()
    T = f.shape[0]
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]], dim=0)
    e, _ = torch.sort(e, dim=1)
    fid = torch.arange(T, device=f.device).repeat(3)
    key = e[:, 0] * (int(f.max()) + 1) + e[:, 1]
    order = torch.argsort(key)
    key, fid = key[order], fid[order]
    same = key[1:] == key[:-1]
    first = torch.ones_like(same)
    first[1:] = ~same[:-1]                      # pair only the first two faces of each edge
    sel = same & first
    return torch.stack([fid[:-1][sel], fid[1:][sel]], dim=1)


def face_normals(pos, faces):
    t, e = _edges_of(pos, faces)
    n = torch.cross(e[:, 0], -e[:, 2], dim=1)
    return n / torch.linalg.norm(n, dim=1, keepdim=True).clamp_min(1e-12)


def normal_loss(pos, faces, adj):
    if adj.numel() == 0:
        return pos.sum() * 0.0
    n = face_normals(pos, faces)
    return (1.0 - (n[adj[:, 0]] * n[adj[:, 1]]).sum(-1)).mean()


@torch.no_grad()
def mesh_report(pos, faces, adj=None, a0=20.0):
    aspect, longest, area = face_aspect(pos, faces)
    rep = dict(faces=int(faces.shape[0]), vertices=int(pos.shape[0]), aspect_q50=float(aspect.median()), sliver_frac=float((aspect > a0).float().mean()),
               lap=float(laplacian_loss(pos, faces)))
    if adj is not None and adj.numel():
        n = face_normals(pos, faces); c = (n[adj[:, 0]] * n[adj[:, 1]]).sum(-1)
        rep.update(normal=float((1 - c).mean()), dihedral_gt60_frac=float((c < 0.5).float().mean()))
    return rep


# ---- training-loop helpers (duck-typed: triangles has .vertices/._triangle_indices/.prune_triangles(keep); acc has .reset(n); log has .write(**kw)) ----
class AdjCache:
    """face_adjacency recomputed only when the topology signature (V, T) changes; callers call .invalidate() after any topology event."""
    def __init__(self):
        self.sig = None; self.adj = None; self.boundary = None

    def invalidate(self):
        self.sig = None

    def _refresh(self, triangles):
        sig = (int(triangles.vertices.shape[0]), int(triangles._triangle_indices.shape[0]))
        if self.sig != sig:
            self.adj = face_adjacency(triangles._triangle_indices)
            self.boundary = boundary_vertices(triangles._triangle_indices, sig[0])
            self.sig = sig

    def get(self, triangles):
        self._refresh(triangles); return self.adj

    def get_boundary(self, triangles):
        self._refresh(triangles); return self.boundary


def apply_jump_filter(triangles, acc, log, a1, k, tag, iteration, adj_cache=None):
    """Hard jump-triangle filter on the live model: drop faces with aspect > a1 AND longest edge > k * median edge; asserts the face count after
    prune_triangles equals keep.sum() (fails loudly if the upstream prune semantics ever change); resets the RD accumulator. Returns dropped count."""
    if a1 <= 0:
        return 0
    T0 = int(triangles._triangle_indices.shape[0])
    keep, st = jump_face_mask(triangles.vertices.detach(), triangles._triangle_indices, a1, k)
    n_keep = int(keep.sum()); n_drop = T0 - n_keep
    if n_drop > 0:
        triangles.prune_triangles(keep)
        T1 = int(triangles._triangle_indices.shape[0])
        if T1 != n_keep:
            raise RuntimeError(f'jump filter: prune_triangles(keep) left {T1} faces, expected keep.sum() = {n_keep} (keep-semantics assumption broken)')
        if acc is not None:
            acc.reset(T1)
        if adj_cache is not None:
            adj_cache.invalidate()
    ev = dict(event='jumpfilter', tag=tag, iteration=int(iteration), faces_before=T0, faces_after=T0 - n_drop, dropped=n_drop, a1=a1, k=k)
    ev.update({kk: vv for kk, vv in st.items() if kk != 'dropped'})
    log.write(**ev)
    print(f"\n[REG jumpfilter {tag} {iteration}] faces {T0}->{T0 - n_drop} dropped {n_drop} ({100.0 * n_drop / max(1, T0):.3f} %) | median edge {st['median_edge']:.4g} aspect q50/q99 {st['aspect_q50']:.2f}/{st['aspect_q99']:.1f}")
    return n_drop


def write_report(triangles, log, tag, iteration, a0=20.0, adj_cache=None):
    adj = adj_cache.get(triangles) if adj_cache is not None else None
    rep = mesh_report(triangles.vertices.detach(), triangles._triangle_indices, adj, a0)
    log.write(event='mesh', tag=tag, iteration=int(iteration), **rep)
    print(f"\n[REG mesh {tag} {iteration}] V {rep['vertices']} F {rep['faces']} | aspect q50 {rep['aspect_q50']:.2f} sliver(>{a0:g}) {100 * rep['sliver_frac']:.2f} % | lap {rep['lap']:.4f}"
          + (f" | normal {rep['normal']:.4f} dihedral>60deg {100 * rep['dihedral_gt60_frac']:.2f} %" if 'normal' in rep else ''))
    return rep
PYEOF_MR1
cat > $AN/rdgrowth/test_mesh_reg_v1.py <<'PYEOF_TMR1'
"""CPU test for mesh_reg_v1 (no GPU, no upstream): term values and gradient flow on a sphere / noisy sphere, Laplacian = 0 on a flat grid
interior (boundary excluded), jump filter + training helpers (AdjCache, apply_jump_filter, write_report) on a duck-typed TriangleModel,
degenerate-face clamp.  python test_mesh_reg_v1.py  -> prints 'MESH_REG_TEST OK' (any assertion = failure)."""
import json, math, sys, os
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_reg_v1 import (sliver_loss, laplacian_loss, normal_loss, face_adjacency, jump_face_mask, mesh_report, boundary_vertices,
                         AdjCache, apply_jump_filter, write_report, VERSION)


class Fake:
    def __init__(self, x, f):
        self.vertices = torch.tensor(x, dtype=torch.float32).requires_grad_(True)
        self._triangle_indices = torch.tensor(f, dtype=torch.int32)
        self.importance_score = torch.zeros(len(f))

    def prune_triangles(self, mask):
        self._triangle_indices = self._triangle_indices[mask].to(torch.int32); self.importance_score = self.importance_score[mask]


class Acc:
    def __init__(self): self.n = None
    def reset(self, n): self.n = int(n)


class Log:
    def __init__(self): self.rows = []
    def write(self, **kw): json.dumps(kw); self.rows.append(kw)


def grid(n):
    x = [[i, j, 0.0] for i in range(n) for j in range(n)]; f = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j; b = a + 1; c = a + n; d = c + 1; f += [[a, b, c], [b, d, c]]
    return x, f


def sphere(nu=24, nv=12, noise=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = [[0, 0, 1.0]]; f = []
    for i in range(1, nv):
        th = math.pi * i / nv
        for j in range(nu):
            ph = 2 * math.pi * j / nu
            x.append([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
    x.append([0, 0, -1.0]); S = len(x) - 1
    for j in range(nu):
        f.append([0, 1 + j, 1 + (j + 1) % nu])
    for i in range(nv - 2):
        for j in range(nu):
            a = 1 + i * nu + j; b = 1 + i * nu + (j + 1) % nu; c = a + nu; d = b + nu
            f += [[a, c, b], [b, c, d]]
    base = 1 + (nv - 2) * nu
    for j in range(nu):
        f.append([S, base + (j + 1) % nu, base + j])
    pos = torch.tensor(x); pos = pos + noise * torch.randn(pos.shape, generator=g)
    return pos, torch.tensor(f, dtype=torch.int32)


# 1 sphere: smooth -> small terms; noisy -> larger; gradients finite
pos, faces = sphere(); pos.requires_grad_(True)
adj = face_adjacency(faces)
s0, l0, n0 = float(sliver_loss(pos, faces)), float(laplacian_loss(pos, faces)), float(normal_loss(pos, faces, adj))
assert not boundary_vertices(faces, pos.shape[0]).any(), 'closed sphere has no boundary'
posn, _ = sphere(noise=0.05); posn.requires_grad_(True)
s1, l1, n1 = float(sliver_loss(posn, faces)), float(laplacian_loss(posn, faces)), float(normal_loss(posn, faces, adj))
assert l1 > 2 * l0 and n1 > 2 * n0, (l0, l1, n0, n1)
tot = sliver_loss(posn, faces) + laplacian_loss(posn, faces) + normal_loss(posn, faces, adj)
g = torch.autograd.grad(tot, posn)[0]
assert torch.isfinite(g).all() and float(g.abs().sum()) > 0
# 2 flat grid: interior Laplacian exactly 0 once boundary vertices are excluded; with boundary included it is > 0
x, f = grid(6); gp = torch.tensor(x); gf = torch.tensor(f, dtype=torch.int32)
assert float(laplacian_loss(gp, gf)) < 1e-10, float(laplacian_loss(gp, gf))
assert float(laplacian_loss(gp, gf, exclude_boundary=False)) > 0.05
assert int(boundary_vertices(gf, gp.shape[0]).sum()) == 20
assert float(normal_loss(gp, gf, face_adjacency(gf))) < 1e-6
# 3 jump filter + helpers on a grid with one spike face
x, f = grid(5); x.append([40.0, 0.5, 0.0]); f.append([0, 1, len(x) - 1])
m = Fake(x, f); acc = Acc(); log = Log(); cache = AdjCache(); T0 = len(f)
keep, st = jump_face_mask(m.vertices.detach(), m._triangle_indices, 20.0, 6.0)
assert st['dropped'] == 1 and not bool(keep[-1])
d = apply_jump_filter(m, acc, log, 20.0, 6.0, 'delaunay', 11001, cache)
assert d == 1 and m._triangle_indices.shape[0] == T0 - 1 and acc.n == T0 - 1 and log.rows[-1]['event'] == 'jumpfilter'
rep = write_report(m, log, 'delaunay', 11001, 20.0, cache)
assert rep['faces'] == T0 - 1 and abs(rep['normal']) < 1e-6 and rep['lap'] < 1e-10, rep
assert apply_jump_filter(m, acc, log, 20.0, 6.0, 'prune', 12000, cache) == 0          # nothing left to drop
assert apply_jump_filter(m, acc, log, 0.0, 6.0, 'final', 30000, cache) == 0            # filter off
a1 = cache.get(m); assert cache.get(m) is a1 and cache.sig == (26, 32) and a1.shape[0] == 40       # adjacency cached; 4x4-cell grid has 40 interior edges
assert int(cache.get_boundary(m).sum()) == 16 and abs(float(laplacian_loss(m.vertices, m._triangle_indices, boundary=cache.get_boundary(m)))) < 1e-10
# 4 degenerate face: finite, capped loss and finite gradient
m2 = Fake([[0, 0, 0], [1, 0, 0], [2, 0, 1e-9]], [[0, 1, 2]]); l2 = sliver_loss(m2.vertices, m2._triangle_indices); l2.backward()
assert math.isfinite(float(l2)) and float(l2) <= 8.0 + 1e-6 and torch.isfinite(m2.vertices.grad).all()
print(f'MESH_REG_TEST OK ({VERSION}) | sphere sliver {s0:.3f} lap {l0:.4f} normal {n0:.4f} | noisy lap {l1:.3f} normal {n1:.3f} | grid lap 0 | jump dropped 1 | degenerate loss {float(l2):.1f}')
PYEOF_TMR1
cat > $AN/rdgrowth/train_rd_v4.py <<'PYEOF_TR4'
"""train_rd_v4.py (VERSION train_rd_v4 2026-11-05) = train_rd_v3 + mesh-quality regularisers (mesh_reg_v1): sliver loss on every face
from --reg_from, Laplacian + neighbour-normal loss after the restricted Delaunay (shared vertices), hard jump-triangle filter after the
Delaunay rebuild / each prune round / the final prune (prune_triangles keep-mask, asserted), reg and mesh statistics in rd_log.jsonl.
v3 behaviour is unchanged when all --w_* are 0 and --filter_aspect 0.  Original v3 header follows.

train_rd_v3.py (v3, 2026-10-19 = v2m + phase-B rate-controlled edge-collapse pruning; v2m kept as train_rd_v2m.py): MeshSplatting training with rate-distortion controlled growth AND pruning.

v3 adds, at each iteration in --prune_iters (default 12000 14000 16000 18000: after the restricted Delaunay rebuild, before
supersampling), one round of rd_prune.select_collapses / collapse_edges: collapse edge (u,v) iff the residual mass of the
faces touching u or v, per byte the collapse saves (one vertex + the 1/2/3+ faces that disappear), is below lam * prune_mu.
Same lam and same per-view per-megapixel residual units as the split rule; the residual keeps being attributed through
rend_ids in phase B.  At most prune_max_frac of the vertices go per round (pruning twin of the growth limiter).
--prune_iters with no values disables pruning (then v3 == v2m).

v2m adds ONE option, --split_schedule <rd_log.jsonl of a reference run>: at every densification interval the number of
triangles chosen by the selection rule is capped at the number the reference run chose at the same iteration (its
"n_split" field).  With --dd_mode importance this gives the upstream importance-sampled selection on exactly the
growth schedule of a lambda run, so the two runs differ only in WHICH triangles split (ranking), not in how many or
when (the stop).  Without --split_schedule v2m behaves exactly like v2.

A copy of upstream `train.training` with ONE change: the densification call `add_new_gs(cap)` is replaced by the
Lagrangian split rule of rd_growth.py (dD_t > lam * dR_t).  Everything else (losses, pruning, opacity schedule,
restricted Delaunay at densify_until+1000, final importance cleaning) is upstream code, imported, not edited.

Run from anywhere:
  python train_rd.py --lab /scratch/zr2245/meshsplatting-original-lossless-v1 -s <scene> -i images_2 -r 1 --eval \
      --data_device cpu -m <out>/original --iterations 30000 --lam 1e-3 --bytes_per_vertex 10.0 --dd_mode error
Outputs the usual <out>/original/point_cloud/iteration_30000/point_cloud_state_dict.pt plus <out>/original/rd_log.jsonl
(one line per densification interval: vertices, triangles, candidates passing, splits, ratio quantiles).
"""
import argparse
import json
import os
import sys
import time
from random import randint

import torch
import torch.nn.functional as F

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--lab', required=True, help='root containing upstream/ (train.py)')
_lab, _rest = _pre.parse_known_args()
UP = os.path.join(os.path.abspath(_lab.lab), 'upstream')
if not os.path.isdir(UP):
    sys.exit(f'upstream dir not found: {UP}')
sys.path.insert(0, UP)
os.chdir(UP)  # upstream resolves relative paths (output/, submodules) from its own dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train as up  # noqa: E402  (upstream train.py; its __main__ block is guarded)
from arguments import ModelParams, PipelineParams, OptimizationParams, update_indoor  # noqa: E402
from scene import Scene, TriangleModel  # noqa: E402
from triangle_renderer import render  # noqa: E402
from utils.loss_utils import l1_loss, ssim, vertex_depth_loss_hr  # noqa: E402
from utils.general_utils import safe_state, get_expon_lr_func  # noqa: E402
from tqdm import tqdm  # noqa: E402
import lpips  # noqa: E402
from rd_growth_v2 import RDAccumulator, RDLog, rd_select, rd_split  # noqa: E402
from rd_prune import select_collapses, collapse_edges  # noqa: E402
from mesh_reg_v1 import sliver_loss, laplacian_loss, normal_loss, AdjCache, apply_jump_filter, write_report, VERSION as MESH_REG_VERSION  # noqa: E402


def load_split_schedule(path):
    """iteration -> n_split (triangles chosen by the rule, before the largest-area top-k) of a reference rd_log.jsonl."""
    sched = {}
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e.get('event') == 'densify':
                sched[int(e['iteration'])] = int(e['n_split'])
    if not sched:
        raise RuntimeError(f'no densify events in split schedule {path}')
    return sched


def training(dataset, opt, pipe, testing_iterations, checkpoint, debug_from, scene_name, rd):
    tb_writer = up.prepare_output_and_logger(dataset)
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma)
    triangles.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init)
    triangles.add_percentage = opt.add_percentage
    first_iter = 0
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        triangles.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = scene.getTrainCameras().copy()
    number_of_training_views = len(viewpoint_stack)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc='Training progress (RD growth)')
    first_iter += 1

    initial_sigma = opt.set_sigma
    final_sigma = 0.0001
    sigma_start = opt.sigma_start
    total_iters = opt.sigma_until
    init_opacity = 0.1
    final_opacity = .9999
    total_iters_opacity = opt.final_opacity_iter
    prune_triangles = opt.prune_triangles_threshold
    prune_size = opt.prune_size
    start_upsampling = opt.start_upsampling
    splitt_large_triangles = opt.splitt_large_triangles
    triangles.size_probs_zero = opt.size_probs_zero
    triangles.size_probs_zero_image_space = opt.size_probs_zero_image_space
    need_delaunay = False
    run_restricted_delaunay = opt.densify_until_iter + 1000
    depth_l1_weight = get_expon_lr_func(opt.depth_lambda_init, opt.depth_lambda_final, max_steps=opt.iterations)

    # --- RD growth state ---------------------------------------------------------------------------------------
    acc = RDAccumulator(triangles._triangle_indices.shape[0], mode=rd['dd_mode'], norm=rd['dd_norm'])
    log = RDLog(os.path.join(dataset.model_path, 'rd_log.jsonl'))
    bytes_per_split = 3.0 * rd['bytes_per_vertex']  # three edge midpoints per split (upper bound; shared midpoints cost less)
    sched = load_split_schedule(rd['split_schedule']) if rd['split_schedule'] else None
    prune_set = sorted(set(int(i) for i in rd['prune_iters']))
    if prune_set:
        if rd['dd_mode'] != 'error':
            raise SystemExit('pruning needs --dd_mode error (residual attribution); got ' + rd['dd_mode'])
        bad = [i for i in prune_set if not (run_restricted_delaunay + 1 < i < start_upsampling)]
        if bad and not rd['allow_any_prune_iter']:
            raise SystemExit(f'prune_iters {bad} must lie after the Delaunay rebuild ({run_restricted_delaunay}) and before supersampling ({start_upsampling}); --allow_any_prune_iter overrides (smoke only)')
    last_prune = max(prune_set) if prune_set else -1
    log.write(event='config', version='v4', lam=rd['lam'], dd_mode=rd['dd_mode'], dd_norm=rd['dd_norm'], bytes_per_vertex=rd['bytes_per_vertex'],
              bytes_per_split=bytes_per_split, max_points=opt.max_points, add_percentage=opt.add_percentage,
              densify_from=opt.densify_from_iter, densify_until=opt.densify_until_iter, interval=opt.densification_interval,
              split_schedule=rd['split_schedule'], schedule_total=(sum(sched.values()) if sched else None),
              prune_iters=prune_set, prune_mu=rd['prune_mu'], prune_max_frac=rd['prune_max_frac'], bytes_per_face=rd['bytes_per_face'],
              reg=rd['reg'], mesh_reg_version=MESH_REG_VERSION)
    if prune_set:
        print('[RD] phase-B lambda-prune at', prune_set, '| mu', rd['prune_mu'], 'max_frac/round', rd['prune_max_frac'], 'B/face', rd['bytes_per_face'])
    if sched:
        print('[RD] matched-count control: schedule from', rd['split_schedule'], '|', len(sched), 'intervals, total chosen', sum(sched.values()))
    t0 = time.time()

    # --- v4 mesh-quality regularisers (mesh_reg_v1) ---------------------------------------------------------------
    reg = rd['reg']
    vp = triangles.vertices
    if not (isinstance(vp, torch.Tensor) and vp.requires_grad and vp.is_leaf):
        raise SystemExit(f'v4 assumption broken: triangles.vertices must be a leaf tensor with requires_grad (got {type(vp)}, requires_grad={getattr(vp, "requires_grad", None)}, is_leaf={getattr(vp, "is_leaf", None)})')
    if not hasattr(triangles, 'prune_triangles'):
        raise SystemExit('v4 assumption broken: TriangleModel has no prune_triangles(keep_mask)')
    adj_cache = AdjCache()
    reg_on = reg['w_sliver'] > 0 or reg['w_lap'] > 0 or reg['w_normal'] > 0
    print(f"[REG] {MESH_REG_VERSION} | w_sliver {reg['w_sliver']} (a0 {reg['sliver_a0']}, from {reg['reg_from']}) | w_lap {reg['w_lap']} w_normal {reg['w_normal']} (after Delaunay {run_restricted_delaunay}) "
          f"| jump filter aspect>{reg['filter_aspect']} & longest>{reg['filter_k']}*median at delaunay/prune/final | log every {reg['reg_log_every']}")
    write_report(triangles, log, 'start', first_iter - 1, reg['sliver_a0'], None)

    for iteration in range(first_iter, opt.iterations + 1):
        if need_delaunay:
            with torch.no_grad():
                triangles.run_restricted_delaunay()
            need_delaunay = False
            log.write(event='delaunay', iteration=iteration, vertices=int(triangles.vertices.shape[0]),
                      triangles=int(triangles._triangle_indices.shape[0]), elapsed_s=round(time.time() - t0))
            acc.reset(triangles._triangle_indices.shape[0])  # new face set after the rebuild
            adj_cache.invalidate()
            apply_jump_filter(triangles, acc, log, reg['filter_aspect'], reg['filter_k'], 'delaunay', iteration, adj_cache)
            write_report(triangles, log, 'delaunay', iteration, reg['sliver_a0'], adj_cache)
        if iteration == start_upsampling:
            triangles.scaling = opt.upscaling_factor
        if iteration == start_upsampling + 5000:
            triangles.scaling = 4

        iter_start.record()
        triangles.update_learning_rate(iteration)
        if iteration < sigma_start:
            current_sigma = initial_sigma
        else:
            progress = min((iteration - sigma_start) / (total_iters - sigma_start), 1.0)
            current_sigma = initial_sigma - (initial_sigma - final_sigma) * progress
        triangles.set_sigma(current_sigma)
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()
        if (iteration - 1) == debug_from:
            pipe.debug = True
        bg = torch.rand((3), device='cuda') if opt.random_background else background

        if not viewpoint_stack or len(scene.getTrainCameras()) + iteration == opt.iterations:
            viewpoint_stack = scene.getTrainCameras().copy()
            if len(scene.getTrainCameras()) + iteration == opt.iterations:
                triangles.importance_score = torch.zeros((triangles._triangle_indices.shape[0]), dtype=torch.float, device='cuda')
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, triangles, pipe, bg)
        image = render_pkg['render']
        gt_image = viewpoint_cam.original_image.cuda()
        if getattr(viewpoint_cam, 'normal_map', None) is not None:
            gt_normal = viewpoint_cam.normal_map.cuda()
            gt_normal = F.interpolate(gt_normal.unsqueeze(0), size=(gt_image.shape[1], gt_image.shape[2]), mode='area').squeeze(0)
        else:
            gt_normal = None

        pixel_loss = l1_loss(image, gt_image)

        image_size = render_pkg['scaling'].detach()
        mask = image_size > triangles.image_size
        triangles.image_size[mask] = image_size[mask]
        importance_score = render_pkg['max_blending'].detach()
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]
        pixel_count = render_pkg['triangle_was_rendered'].detach()
        mask = pixel_count > triangles.pixel_count
        triangles.pixel_count[mask] = pixel_count[mask]

        if up.FUSED_SSIM_AVAILABLE:
            ssim_value = up.fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim_value)

        lambda_weight = opt.lambda_weight if iteration < opt.start_opacity_floor else 0
        if lambda_weight > 0:
            mask_out = triangles.vertices.shape[0]
            vertex_weights = triangles.get_vertex_weight[:mask_out][triangles._triangle_indices]
            loss += lambda_weight * vertex_weights.mean()

        lambda_vertex = opt.lambda_vertex if iteration > opt.start_vertex_opt else 0
        if lambda_vertex > 0:
            loss += lambda_vertex * vertex_depth_loss_hr(render_pkg['vertex_depth_out'], render_pkg['image_2D'], render_pkg['vertex_rendered'],
                                                         render_pkg['surf_depth'], max_diff_threshold=opt.max_diff_threshold)

        if depth_l1_weight(iteration) > 0 and getattr(viewpoint_cam, 'invdepthmap', None) is not None:
            invDepth = 1.0 / (render_pkg['expected_depth'] + 1e-6)
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            loss += depth_l1_weight(iteration) * torch.abs((invDepth - mono_invdepth) * depth_mask).mean()

        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        lambda_normal = opt.lambda_normals if iteration > opt.iteration_mesh else 0
        if lambda_normal > 0:
            loss += lambda_normal * (1 - (rend_normal * surf_normal).sum(dim=0))[None].mean()
        if gt_normal is not None:
            lambda_normals_super = opt.lambda_normals_super if iteration > opt.iteration_mesh else 0
            loss += lambda_normals_super * (1 - (rend_normal * gt_normal).sum(dim=0))[None].mean()

        # --- v4 regularisers: dimensionless mesh-quality terms on the vertex positions ---------------------------------
        if reg_on:
            reg_terms = {}
            if reg['w_sliver'] > 0 and iteration >= reg['reg_from']:
                reg_terms['sliver'] = sliver_loss(triangles.vertices, triangles._triangle_indices, reg['sliver_a0'])
            if iteration > run_restricted_delaunay:  # vertices are shared only after the restricted Delaunay rebuild
                if reg['w_lap'] > 0:
                    reg_terms['lap'] = laplacian_loss(triangles.vertices, triangles._triangle_indices, boundary=adj_cache.get_boundary(triangles))
                if reg['w_normal'] > 0:
                    reg_terms['normal'] = normal_loss(triangles.vertices, triangles._triangle_indices, adj_cache.get(triangles))
            if reg_terms:
                reg_total = sum(reg['w_' + k] * v for k, v in reg_terms.items())
                if iteration % reg['reg_log_every'] == 0:
                    g_pix = torch.autograd.grad(loss, triangles.vertices, retain_graph=True, allow_unused=True)[0]
                    g_reg = torch.autograd.grad(reg_total, triangles.vertices, retain_graph=True, allow_unused=True)[0]
                    npix = float(g_pix.norm()) if g_pix is not None else 0.0
                    nreg = float(g_reg.norm()) if g_reg is not None else 0.0
                    ev = dict(event='reg', iteration=iteration, loss_main=float(loss), reg_total=float(reg_total), grad_main=npix, grad_reg=nreg,
                              grad_ratio=(nreg / npix if npix > 0 else None), faces=int(triangles._triangle_indices.shape[0]))
                    ev.update({k: float(v) for k, v in reg_terms.items()})
                    log.write(**ev)
                    print(f"\n[REG {iteration}] main {float(loss):.5f} reg {float(reg_total):.5f} (" + ' '.join(f'{k} {float(v):.4f}' for k, v in reg_terms.items())
                          + f") | |g_main| {npix:.3e} |g_reg| {nreg:.3e} ratio {(nreg / npix if npix > 0 else float('nan')):.3f}")
                loss = loss + reg_total
        if iteration % 5000 == 0 and iteration >= reg['reg_from']:
            write_report(triangles, log, 'periodic', iteration, reg['sliver_a0'], adj_cache if iteration > run_restricted_delaunay else None)

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # RD: attribute this view's residual to triangles (phase A for growth; phase B up to the last prune round)
            if iteration < run_restricted_delaunay or iteration <= last_prune:
                acc.accumulate(render_pkg, image, gt_image, triangles._triangle_indices,
                               vertex_grad=triangles.vertices.grad if rd['dd_mode'] == 'grad' else None)

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({'Loss': f'{ema_loss_for_log:.5f}', 'V': int(triangles.vertices.shape[0])})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            up.training_report(tb_writer, scene_name, iteration, pixel_loss, loss, l1_loss, iter_start.elapsed_time(iter_end),
                               testing_iterations, scene, render, (pipe, background))

            if iteration % 500 == 0 and iteration < run_restricted_delaunay:
                triangle_vertex_weights = triangles.opacity_activation(triangles.vertex_weight[triangles._triangle_indices])
                min_weights = triangle_vertex_weights.min(dim=1).values
                mask_opacity = (min_weights <= prune_triangles).squeeze()
                mask_importance = (triangles.importance_score <= prune_triangles).squeeze()
                mask_size = (triangles.image_size > prune_size).squeeze()
                delete_mask = mask_opacity | mask_size
                if number_of_training_views < 500:
                    delete_mask = delete_mask | mask_importance
                keep_mask = ~delete_mask
                if iteration > opt.start_pruning:
                    triangles.prune_triangles(keep_mask)
                    acc.apply_mask(keep_mask)

                device = triangles.vertices.device
                used_vertex_mask = torch.zeros(triangles.vertices.shape[0], dtype=torch.bool, device=device)
                if triangles._triangle_indices.numel() > 0:
                    used_vertex_mask[triangles._triangle_indices.flatten()] = True
                weight_mask = (triangles.get_vertex_weight.squeeze() >= prune_triangles)
                mask_out = triangles.vertices.shape[0]
                vertex_mask = weight_mask[:mask_out] | used_vertex_mask
                valid_tris = vertex_mask[triangles._triangle_indices.long()].all(dim=1) if triangles._triangle_indices.numel() > 0 else None
                triangles._prune_vertices(vertex_mask)
                if valid_tris is not None:
                    acc.apply_mask(valid_tris)

                needs_densification = (iteration < opt.densify_until_iter and iteration % opt.densification_interval == 0
                                       and iteration > opt.densify_from_iter)
                if needs_densification:
                    V = int(triangles.vertices.shape[0])
                    max_split = max(0, min(opt.max_points, int(opt.add_percentage * V)) - V)  # upstream growth limiter
                    k_ref = None
                    if sched is not None:  # matched-count control: never choose more than the reference run did here
                        k_ref = sched.get(iteration, 0)
                        max_split = min(max_split, k_ref)
                    idx, st = rd_select(acc, triangles, rd['lam'], bytes_per_split, max_split,
                                        triangles.size_probs_zero, triangles.size_probs_zero_image_space)
                    n_split = rd_split(triangles, idx, iteration, splitt_large_triangles)
                    st.update(event='densify', iteration=iteration, vertices_before=V, vertices_after=int(triangles.vertices.shape[0]),
                              triangles_after=int(triangles._triangle_indices.shape[0]), n_split_total=n_split, k_ref=k_ref, elapsed_s=round(time.time() - t0))
                    log.write(**st)
                    print(f"\n[RD {iteration}] V {V}->{st['vertices_after']} pass {st['n_pass']}/{st['n_tri']} split {n_split} "
                          f"(limit {max_split}) ratio q50/q90/q99 {st['ratio_q50']:.2e}/{st['ratio_q90']:.2e}/{st['ratio_q99']:.2e} "
                          f"lam {rd['lam']:.2e} valid {st['valid_frac']} k_ref {k_ref}")
                acc.reset(triangles._triangle_indices.shape[0])

                if iteration > opt.start_opacity_floor:
                    a = min(1.0, max(0.0, (iteration - opt.start_opacity_floor) / max(1, total_iters_opacity - opt.start_opacity_floor)))
                    triangles.update_min_weight(min(init_opacity + (final_opacity - init_opacity) * a, final_opacity))
                    prune_triangles += 0.01
            elif iteration == run_restricted_delaunay:
                need_delaunay = True
            elif iteration % 500 == 0 and iteration > run_restricted_delaunay + 1000:
                if iteration > opt.start_opacity_floor:
                    a = min(1.0, max(0.0, (iteration - opt.start_opacity_floor) / max(1, total_iters_opacity - opt.start_opacity_floor)))
                    triangles.update_min_weight(min(init_opacity + (final_opacity - init_opacity) * a, final_opacity))
                    prune_triangles += 0.01

            if iteration in prune_set:
                V0, T0 = int(triangles.vertices.shape[0]), int(triangles._triangle_indices.shape[0])
                dD = acc.distortion()
                if dD.numel() != T0:
                    raise RuntimeError(f'accumulator/face mismatch at prune {iteration}: {dD.numel()} vs {T0}')
                n_views = acc.views
                pairs, ps = select_collapses(triangles.vertices, triangles._triangle_indices, dD, rd['lam'], rd['prune_mu'],
                                             rd['bytes_per_vertex'], rd['bytes_per_face'], rd['prune_max_frac'])
                cs = collapse_edges(triangles, pairs)
                acc.reset(triangles._triangle_indices.shape[0])
                ev = dict(event='prune', iteration=iteration, views=n_views, elapsed_s=round(time.time() - t0))
                ev.update(ps)
                ev.update(cs)
                log.write(**ev)
                print(f"\n[RD prune {iteration}] V {V0}->{cs['vertices_after']} T {T0}->{cs['faces_after']} | cand {ps['n_cand']}/{ps['edges']} edges "
                      f"({ps['cand_frac']:.3f}) pairs {ps['n_pairs']} (cap {ps['max_pairs']}, flip-rejected {ps['n_flip_rejected']}, passes {ps['passes']}) "
                      f"| score q10/q50/q90 {ps['score_q10']:.2e}/{ps['score_q50']:.2e}/{ps['score_q90']:.2e} thr {ps['thr']:.2e} "
                      f"| faces-removed share sel {ps['faces_removed_share_sel']} all {ps['faces_removed_share_all']} | views {n_views}")
                adj_cache.invalidate()
                apply_jump_filter(triangles, acc, log, reg['filter_aspect'], reg['filter_k'], 'prune', iteration, adj_cache)
                write_report(triangles, log, 'prune', iteration, reg['sliver_a0'], adj_cache)
            if iteration < opt.iterations:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none=True)

    # final cleaning, verbatim upstream
    viewpoint_stack = scene.getTrainCameras().copy()
    triangles.importance_score = torch.zeros((triangles._triangle_indices.shape[0]), dtype=torch.float, device='cuda')
    while viewpoint_stack:
        viewpoint_cam = viewpoint_stack.pop(0)
        render_pkg = render(viewpoint_cam, triangles, pipe, bg)
        importance_score = render_pkg['max_blending'].detach()
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]
    mask_importance = (triangles.importance_score <= 0.5).squeeze()
    triangles.prune_triangles(~mask_importance)
    adj_cache.invalidate()
    apply_jump_filter(triangles, None, log, reg['filter_aspect'], reg['filter_k'], 'final', opt.iterations, adj_cache)
    used_vertex_mask = torch.zeros(triangles.vertices.shape[0], dtype=torch.bool, device=triangles.vertices.device)
    if triangles._triangle_indices.numel() > 0:
        used_vertex_mask[triangles._triangle_indices.flatten()] = True
    triangles._prune_vertices(used_vertex_mask)
    adj_cache.invalidate()
    write_report(triangles, log, 'final', opt.iterations, reg['sliver_a0'], adj_cache)
    scene.save(iteration)
    log.write(event='final', vertices=int(triangles.vertices.shape[0]), triangles=int(triangles._triangle_indices.shape[0]),
              elapsed_s=round(time.time() - t0))
    with open(os.path.join(dataset.model_path, 'rd_summary.json'), 'w') as f:
        json.dump(dict(version='v4', reg=rd['reg'], lam=rd['lam'], dd_mode=rd['dd_mode'], dd_norm=rd['dd_norm'], bytes_per_vertex=rd['bytes_per_vertex'], split_schedule=rd['split_schedule'],
                       prune_iters=prune_set, prune_mu=rd['prune_mu'], prune_max_frac=rd['prune_max_frac'], bytes_per_face=rd['bytes_per_face'],
                       vertices=int(triangles.vertices.shape[0]), triangles=int(triangles._triangle_indices.shape[0])), f, indent=1)
    print('Training is done; vertices', int(triangles.vertices.shape[0]), 'triangles', int(triangles._triangle_indices.shape[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MeshSplatting training with RD-controlled growth', parents=[_pre])
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--test_iterations', nargs='+', type=int, default=[7_000, 30_000])
    parser.add_argument('--save_iterations', nargs='+', type=int, default=[7_000, 30_000])
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--checkpoint_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--start_checkpoint', type=str, default=None)
    parser.add_argument('--scene_name', default='Garden', type=str)
    parser.add_argument('--indoor', action='store_true', default=False)
    parser.add_argument('--lam', type=float, required=True, help='price of a byte in per-view pixel-L1 units (0 = error-ranked growth, no price)')
    parser.add_argument('--bytes_per_vertex', type=float, default=10.0, help='measured delivered bytes per vertex for this scene/precision')
    parser.add_argument('--dd_mode', choices=['error', 'grad', 'importance'], default='error')
    parser.add_argument('--dd_norm', choices=['mpx', 'none'], default='mpx', help='mpx: dD per megapixel of the loss image (resolution-independent lambda); none: v1 raw pixel units')
    parser.add_argument('--split_schedule', type=str, default=None, help='rd_log.jsonl of a reference run: cap the per-interval split count at its n_split (matched-count control)')
    parser.add_argument('--prune_iters', nargs='*', type=int, default=[12000, 14000, 16000, 18000], help='phase-B lambda-prune rounds (no values = no pruning)')
    parser.add_argument('--prune_mu', type=float, default=1.0, help='collapse iff ring-residual / bytes-saved < lam * prune_mu')
    parser.add_argument('--prune_max_frac', type=float, default=0.15, help='at most this fraction of vertices collapsed per round')
    parser.add_argument('--bytes_per_face', type=float, default=2.0, help='topology bytes per face in the delivered file (bytes_per_vertex - 2*bytes_per_face = attribute bytes per vertex)')
    parser.add_argument('--allow_any_prune_iter', action='store_true', help='smoke only: allow prune rounds outside (Delaunay, supersampling)')
    parser.add_argument('--w_sliver', type=float, default=0.0, help='v4: weight of the per-face sliver loss mean(relu(log(aspect/a0))) (0 = off)')
    parser.add_argument('--sliver_a0', type=float, default=20.0, help='v4: aspect (longest^2 / 2 area) above which a face is a sliver')
    parser.add_argument('--w_lap', type=float, default=0.0, help='v4: weight of the normalised uniform-Laplacian loss (after the Delaunay rebuild)')
    parser.add_argument('--w_normal', type=float, default=0.0, help='v4: weight of the neighbour-face normal loss 1 - n_i.n_j (after the Delaunay rebuild)')
    parser.add_argument('--filter_aspect', type=float, default=20.0, help='v4: hard jump filter: drop faces with aspect (~ length/base) > this AND longest edge > filter_k * median edge (0 = off)')
    parser.add_argument('--filter_k', type=float, default=6.0)
    parser.add_argument('--reg_from', type=int, default=500, help='v4: first iteration the sliver loss is applied')
    parser.add_argument('--reg_log_every', type=int, default=500, help='v4: log reg terms and vertex-gradient norms every N iterations')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print('Optimizing ' + args.model_path, '| RD growth+prune v4 lam', args.lam, 'mode', args.dd_mode, 'norm', args.dd_norm, 'B/v', args.bytes_per_vertex,
          'schedule', args.split_schedule, '| prune', args.prune_iters, 'mu', args.prune_mu, 'max_frac', args.prune_max_frac, 'B/face', args.bytes_per_face,
          '| v4 reg w_sliver', args.w_sliver, 'a0', args.sliver_a0, 'w_lap', args.w_lap, 'w_normal', args.w_normal, 'filter', args.filter_aspect, args.filter_k, 'from', args.reg_from)
    for nm in ('w_sliver', 'w_lap', 'w_normal', 'sliver_a0', 'filter_aspect', 'filter_k'):
        if getattr(args, nm) < 0:
            sys.exit(f'--{nm} must be >= 0')
    if args.split_schedule and not os.path.isfile(args.split_schedule):
        sys.exit(f'split schedule not found: {args.split_schedule}')
    up.lpips_fn = lpips.LPIPS(net='vgg').to(device='cuda')  # training_report reads this module global
    safe_state(args.quiet)
    lps, ops, pps = lp.extract(args), op.extract(args), pp.extract(args)
    if args.indoor:
        ops = update_indoor(ops)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lps, ops, pps, args.test_iterations, args.start_checkpoint, args.debug_from, args.scene_name,
             rd=dict(lam=args.lam, bytes_per_vertex=args.bytes_per_vertex, dd_mode=args.dd_mode, dd_norm=args.dd_norm, split_schedule=args.split_schedule,
                     prune_iters=args.prune_iters, prune_mu=args.prune_mu, prune_max_frac=args.prune_max_frac, bytes_per_face=args.bytes_per_face,
                     allow_any_prune_iter=args.allow_any_prune_iter,
                     reg=dict(w_sliver=args.w_sliver, sliver_a0=args.sliver_a0, w_lap=args.w_lap, w_normal=args.w_normal, filter_aspect=args.filter_aspect,
                              filter_k=args.filter_k, reg_from=args.reg_from, reg_log_every=max(1, args.reg_log_every))))
PYEOF_TR4
cat > $AN/rd-reg-blender-v1.sbatch <<'SBEOF_REG1'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=rd4reg
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/zr2245/analysis/rd4reg-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/rd4reg-%x-%j.err
# rd-reg-blender-v1.sbatch (2026-11-05, VERSION v1): ONE NeRF-Synthetic scene, ONE lambda, ONE regulariser config (CFG) with train_rd_v4
# (= train_rd_v3 + mesh_reg_v1: sliver loss WS, Laplacian WL + neighbour-normal WN after the Delaunay rebuild, hard jump filter FA/FK).
# Everything else = rd-prune-blender-v5: --indoor, black bg, checkpoints every 1k from 12k (train_ckpt_v4, no --override), val_select_v2,
# level-1.0 delivery, mesh_stats_v2 on the selected checkpoint, viewer .msv export.
# Output /scratch/zr2245/$SCENE-rd4reg-$CFG-lam$LAM (+ -best), table rd-reg-blender-v1.txt, msv $AN/viewer-v2/msv/$SCENE-reg-$CFG-lam$LAM.msv.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=train; LAM=${LAM:?set LAM=0.00062 via --export}; CFG=${CFG:?set CFG=a|b|c (name only) via --export}; INDOOR=${INDOOR:-1}
WS=${WS:?set WS (w_sliver) via --export}; WL=${WL:?set WL (w_lap) via --export}; WN=${WN:?set WN (w_normal) via --export}; FA=${FA:-20}; FK=${FK:-6}; A0=${A0:-20}; RFROM=${RFROM:-500}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}
SAVES="12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000"
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; VS=$AN/blender/val_select_v2.py; CK=$AN/blender/train_ckpt_v4.py; EXP=$AN/viewer-v1/export_viewer_v1.py; MSV=$AN/viewer-v2/msv
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; TAB=$AN/rd-reg-blender-v1.txt; TR=$AN/rdgrowth/train_rd_v4.py; MR=$AN/rdgrowth/mesh_reg_v1.py; MT=$AN/rdgrowth/test_mesh_reg_v1.py; MS=$AN/diag/mesh_stats_v2.py
REF=${REF:-/scratch/zr2245/$SCENE-rd3b4-lam$LAM-mu1.0-best/original/point_cloud/iteration_30000/point_cloud_state_dict.pt}
echo "scene $SCENE | REG cfg $CFG: w_sliver $WS (a0 $A0 from $RFROM) w_lap $WL w_normal $WN filter aspect>$FA & longest>$FK*median | blender black bg val-selected | flag '$FLAG' | lam $LAM | B/v $BPV B/face $BPF | prune at $PRUNE mu $MU max_frac $MAXF | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $MR $MT $MS $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $VS $CK $EXP $PKG/stage2_run.sh $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/transforms_train.json $DATA/transforms_val.json $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
grep -q "VERSION train_rd_v4" $TR || { echo "FAILED: $TR is not train_rd_v4"; exit 1; }
grep -q "from mesh_reg_v1 import" $TR || { echo "FAILED: $TR does not import mesh_reg_v1"; exit 1; }
for a in w_sliver w_lap w_normal filter_aspect filter_k sliver_a0 reg_from; do grep -q "'--$a'" $TR || { echo "FAILED: $TR has no --$a"; exit 1; }; done
grep -q "update_indoor(ops)" $TR || { echo "FAILED: $TR does not call update_indoor(ops)"; exit 1; }
grep -q "'--saves'" $CK || { echo "FAILED: $CK is not a train_ckpt wrapper (no --saves)"; exit 1; }
echo "=== mesh_reg_v1 CPU self-test (fails here, before any GPU time, if the module is broken) ==="
python $MT 2>&1 | tail -1 | tee $AN/rd4reg-$SCENE-$CFG-selftest.txt | grep -q "MESH_REG_TEST OK" || { echo "FAILED: mesh_reg_v1 self-test"; python $MT 2>&1 | tail -15; exit 1; }
mkdir -p $MSV
L=$LAM
G=/scratch/zr2245/$SCENE-rd4reg-$CFG-lam$L; GB=$G-best; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
echo "=== $SCENE cfg $CFG lam $L $FLAG: train_rd_v4 + prune (saves every 1k from 12k) ($(date)) ==="
if [ ! -f $CB ]; then
  python $CK --saves $SAVES --script $TR -- --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
    --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF --test_iterations 7000 11000 14000 20000 25000 30000 $FLAG \
    --w_sliver $WS --sliver_a0 $A0 --w_lap $WL --w_normal $WN --filter_aspect $FA --filter_k $FK --reg_from $RFROM --reg_log_every 500 > $G/train.log 2>&1 \
    || { echo "FAILED: TRAIN $SCENE cfg $CFG"; grep -n -m 8 -i 'error\|Traceback\|assumption broken' $G/train.log; tail -12 $G/train.log; exit 1; }
  [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE cfg $CFG"; tail -8 $G/train.log; exit 1; }
else echo "exists: $CB"; fi
grep -q "^\[REG\] mesh_reg_v1" $G/train.log || { echo "FAILED: train.log has no '[REG] mesh_reg_v1' banner (v4 regularisers not active)"; exit 1; }
grep -q "^\[REG jumpfilter delaunay" $G/train.log || { echo "FAILED: no jump-filter event after the Delaunay rebuild in train.log"; exit 1; }
echo "--- growth (first 2 / last 2 densify lines) ---"; grep -h "^\[RD [0-9]*\] V" $G/train.log | head -2; grep -h "^\[RD [0-9]*\] V" $G/train.log | tail -2
echo "--- reg terms (first 3 / last 3; ratio = |grad_reg| / |grad_main| on the vertices) ---"; grep -h "^\[REG [0-9]*\]" $G/train.log | head -3; grep -h "^\[REG [0-9]*\]" $G/train.log | tail -3
echo "--- jump filter + mesh reports ---"; grep -h "^\[REG jumpfilter\|^\[REG mesh" $G/train.log
grep -h "^\[RD prune" $G/train.log; grep -h "^\[ckpt\] all\|FAILED" $G/train.log | tail -2
echo "--- test curve ---"; grep -h "Evaluating test" $G/train.log | sed 's/\[ITER \([0-9]*\)\] Evaluating test: L1 [0-9.]* PSNR \([0-9.]*\) SSIM \([0-9.]*\) LPIPS \([0-9.]*\).*/\1 \2 \3 \4/'
echo "=== $SCENE cfg $CFG: val_select ($(date)) ==="
python $VS --lab $LAB --dataset $DATA --model $G/original --best $GB/original --delete_unselected > $G/val_select.log 2>&1 || { echo "FAILED: VAL_SELECT $SCENE cfg $CFG"; grep -n -m 8 -i 'error\|Traceback\|FAILED' $G/val_select.log; tail -15 $G/val_select.log; exit 1; }
grep -h "BEST\|VAL_SELECT_RESULT" $G/val_select.log
BCK=$GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; [ -f $BCK ] || { echo "FAILED: $BCK missing"; exit 1; }
echo "=== $SCENE cfg $CFG: mesh statistics on the selected checkpoint (+ v3 reference if present) ($(date)) ==="
python $MS --ckpt $BCK --label "$SCENE reg-$CFG lam$L" --out $GB/mesh_stats.json > $GB/mesh_stats.log 2>&1 || { echo "FAILED: MESH_STATS"; tail -8 $GB/mesh_stats.log; exit 1; }
grep -h "^MESH_STATS" $GB/mesh_stats.log | tee -a $TAB
if [ -f $REF ]; then python $MS --ckpt $REF --label "$SCENE v3-ref lam$L" > $G/mesh_stats_ref.log 2>&1 && grep -h "^MESH_STATS" $G/mesh_stats_ref.log | tee -a $TAB; else echo "(no v3 reference checkpoint at $REF)"; fi
echo "=== $SCENE cfg $CFG mu $MU: level-1.0 protocol on $GB ($(date)) ==="
G=$GB DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $GB/level.log 2>&1 \
  || { echo "FAILED: LEVEL-1.0 $SCENE cfg $CFG"; tail -8 $GB/level.log; exit 1; }
NA=$GB/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE cfg $CFG"; tail -8 $GB/level.log; exit 1; }
python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; h=f['history']; v=json.load(open('$G/original/val_select.json'))
ev=[json.loads(l) for l in open('$G/original/rd_log.jsonl')]; dl=[e for e in ev if e['event']=='delaunay']; pr=[e for e in ev if e['event']=='prune']
jf=[e for e in ev if e['event']=='jumpfilter']; me=[e for e in ev if e['event']=='mesh']; rg=[e for e in ev if e['event']=='reg']
last=[r for r in v['rows'] if r['iteration']==30000]
m0=[e for e in me if e['tag']=='delaunay']; m1=[e for e in me if e['tag']=='final']
fm=lambda e: (f"sliver {100*e['sliver_frac']:.2f}% lap {e['lap']:.4f}" + (f" normal {e['normal']:.4f} dih>60 {100*e['dihedral_gt60_frac']:.2f}%" if 'normal' in e else '')) if e else 'n/a'
print('$SCENE lam $L REG-$CFG ws$WS wl$WL wn$WN fa$FA fk$FK RD-GROWTH+PRUNE-mu$MU-BLENDER-VALSEL | best iter', v['best_iteration'], 'val', round(v['best_val_psnr'],3), '(30k val', round(last[0]['val_psnr'],3) if last else 'n/a', ') | vertices', b['vertices'], 'faces', b['faces'],
      '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3), 'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | V@delaunay', dl[0]['vertices'] if dl else None,
      '| jump dropped', [(e['tag'], e['iteration'], e['dropped']) for e in jf], '| mesh@delaunay', fm(m0[0] if m0 else None), '| mesh@final', fm(m1[-1] if m1 else None),
      '| grad ratio reg/main first/last', (round(rg[0]['grad_ratio'],3) if rg and rg[0]['grad_ratio'] is not None else None, round(rg[-1]['grad_ratio'],3) if rg and rg[-1]['grad_ratio'] is not None else None),
      '| prune rounds', [(e['iteration'], e['vertices_before'], e['vertices_after']) for e in pr], '| blackbg $FLAG')
PY
echo "=== $SCENE cfg $CFG: viewer .msv export ($(date)) ==="
A=$NA/deliver/delivery/asset.mslb; [ -f $A ] || A=$(find $NA/deliver -name asset.mslb 2>/dev/null | head -1)
[ -n "$A" ] && [ -f "$A" ] || { echo "FAILED: no asset.mslb under $NA/deliver"; exit 1; }
CAM=$G/original/cameras.json; [ -f $CAM ] || CAM=""
bash $PKG/stage2_run.sh $EXP --asset $A ${CAM:+--cameras $CAM} --out $MSV/$SCENE-reg-$CFG-lam$L.msv --label "$SCENE reg-$CFG lam$L" --native-cache $AN/viewer-v2/native-cache > $G/export_msv.log 2>&1 \
  || { echo "FAILED: msv export"; tail -8 $G/export_msv.log; exit 1; }
grep -h "^VIEWER" $G/export_msv.log
echo "===== $TAB ====="; cat $TAB
echo "rd-reg-blender-v1 $SCENE cfg $CFG exit 0  $(date)"
SBEOF_REG1
cat > $AN/check-reg-v1.sh <<'CHEOF_REG1'
AN=/scratch/zr2245/analysis; ok=1
chk() { f=$1; want=$2; [ -s $f ] || { echo "FAILED: $f empty/missing"; ok=0; return; }; n=$(wc -l < $f); [ "$n" = "$want" ] && echo "$f: $n lines OK" || { echo "FAILED: $f has $n lines, expected $want (paste truncated?)"; ok=0; }; }
chk $AN/rdgrowth/mesh_reg_v1.py 182
chk $AN/rdgrowth/test_mesh_reg_v1.py 94
chk $AN/rdgrowth/train_rd_v4.py 452
chk $AN/rd-reg-blender-v1.sbatch 95
python -m py_compile $AN/rdgrowth/mesh_reg_v1.py $AN/rdgrowth/test_mesh_reg_v1.py $AN/rdgrowth/train_rd_v4.py && echo "py_compile OK" || { echo "FAILED: py_compile"; ok=0; }
bash -n $AN/rd-reg-blender-v1.sbatch && echo "sbatch syntax OK" || { echo "FAILED: sbatch syntax"; ok=0; }
grep -q "VERSION train_rd_v4" $AN/rdgrowth/train_rd_v4.py && grep -q "from mesh_reg_v1 import" $AN/rdgrowth/train_rd_v4.py && echo "v4 markers OK" || { echo "FAILED: v4 markers"; ok=0; }
d=$(diff <(grep -v "^#\|^\"\"\"\|^from mesh_reg_v1\|^    parser.add_argument('--w_\|^    parser.add_argument('--sliver\|^    parser.add_argument('--filter\|^    parser.add_argument('--reg_" $AN/rdgrowth/train_rd_v3.py) <(grep -v "^#\|^\"\"\"\|^from mesh_reg_v1\|^    parser.add_argument('--w_\|^    parser.add_argument('--sliver\|^    parser.add_argument('--filter\|^    parser.add_argument('--reg_" $AN/rdgrowth/train_rd_v4.py) | grep -c "^<"); echo "v3 lines removed/changed in v4 (expect 6: header, config version x2, print x2, rd dict tail): $d"
python $AN/rdgrowth/test_mesh_reg_v1.py 2>&1 | tail -1 | grep "MESH_REG_TEST OK" || { echo "FAILED: mesh_reg_v1 CPU self-test"; python $AN/rdgrowth/test_mesh_reg_v1.py 2>&1 | tail -12; ok=0; }
[ $ok = 1 ] && echo "CHECK OK" || echo "CHECK FAILED - do not submit"
CHEOF_REG1
bash $AN/check-reg-v1.sh
```

### §15b 提交 3 个 lego job（`CHECK OK` 之后整块粘贴；≈1.5 GPU-h，general）

```bash
AN=/scratch/zr2245/analysis; n=0
for cfg in "a 0.05 0 0" "b 0.05 0.05 0.05" "c 0.2 0.2 0.2"; do set -- $cfg
  j=$(sbatch --parsable --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1,LAM=0.00062,CFG=$1,WS=$2,WL=$3,WN=$4,FA=20,FK=6 --job-name=lego-reg$1 $AN/rd-reg-blender-v1.sbatch) \
    && { echo "submitted lego-reg$1 $j  ws $2 wl $3 wn $4"; n=$((n+1)); } || echo "FAILED: sbatch cfg $1"
done
echo "submitted $n jobs (expected 3)"
squeue -u $USER -o "%.10i %.16j %.28a %.2t %.10M %.6D %R"
```

### §15c 看结果

```bash
AN=/scratch/zr2245/analysis
for f in $(ls -t $AN/rd4reg-lego-reg*.out | head -3); do echo "===== $f"; grep -h "^\[REG\|FAILED\|exit 0\|BEST\|VAL_SELECT_RESULT\|MESH_STATS\|^VIEWER" $f; done
echo "===== table"; cat $AN/rd-reg-blender-v1.txt
ls -la $AN/viewer-v2/msv/lego-reg-*.msv
```
笔记本：`scp 'torch:/scratch/zr2245/analysis/viewer-v2/msv/lego-reg-*.msv' ~/msv2/`，用 viewer-v3.html 打开，和 lego-lam0.00062.msv（v3）同一视角近看。
