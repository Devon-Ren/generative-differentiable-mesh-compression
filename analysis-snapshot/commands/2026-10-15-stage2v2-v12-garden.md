# 2026-10-15：stage2-v2 补丁 + vis_collapse v12 + garden 一次 GPU session

先把 `analysis/stage2-v2-patch-v1.zip` 和 `analysis/vis_collapse-v12.zip` 传到 `/scratch/zr2245/analysis/`。
GPU 申请建议：**6 h、--mem=96G、8 CPU**（coarse_vis 在 Room 上要 ≥48 GB 内存，garden 顶点可能更多；训练 30k 步约 1–1.5 h，
三个 level 各约 30–40 min，Room 两个小测量约 20 min）。

## A. 安装 + 自检（CPU 即可，~2 min；GPU 节点上直接做）

```bash
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
cd /scratch/zr2245/analysis
unzip -q stage2-v2-patch-v1.zip && bash stage2-v2-patch-v1/apply_stage2_v2.sh        # 复制 v1 -> shared-collapse-stage2-v2 并打补丁，v1 不动
unzip -q vis_collapse-v12.zip && cp vis_collapse-v11/native/_native_qem_vis*.so vis_collapse-v12/native/ && cat vis_collapse-v12/VERSION | head -1
cd /scratch/zr2245/shared-collapse-stage2-v2 && python /scratch/zr2245/analysis/stage2-v2-patch-v1/roundtrip_check.py . /scratch/zr2245/stage2-v2-check
```
期望：4 行，`audit True`、`colors finite True`，`big200-default` 的 file 字节明显小于 `big200-weight=keep-index_codec=zstd`（本地 395,820 vs 536,977），
opacity err ~1e-6。

## B. Room 两个小测量（GPU，~20 min）

```bash
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
PKG=/scratch/zr2245/analysis/vis_collapse-v12; W=/scratch/zr2245/room-vis-collapse
bash $PKG/scripts/run_codec_variants.sh 2>&1 | tail -8        # 2x2：vertex_weight 常量 / ctx 各省多少 MB，ctx 行 PSNR 必须与 zstd 行完全一致
bash $PKG/scripts/run_default_room.sh 2>&1 | tail -4          # 候选默认点：v0.8 + K4096 R256 + q14/q7 + stage2-v2 -> $W/default.txt
```

## C. garden 全流程（GPU，~4 h，可中断重跑：每一步有产物就跳过）

```bash
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
PKG=/scratch/zr2245/analysis/vis_collapse-v12; G=/scratch/zr2245/garden-vis-collapse
ls /scratch/zr2245/datasets/m360/garden/images_4 | head -2      # 数据集路径不对就 DATA=... 覆盖
bash $PKG/scripts/run_garden.sh 2>&1 | tee -a $G.log
cat $G/garden.txt
```
产物：`$G/original/`（原模型，train.log）、`$G/footprint-sens.npz`、`$G/axisB/v{1.0,0.8,0.7}/{report.json,finetune/,deliver/}`、`$G/garden.txt`。
把 `garden.txt`、`$W/default.txt`、`$W/axisB/v0.8/deliver/codec-variants/sweep.txt` 贴回来即可。

## C'. garden 改为非交互 sbatch（10/15 补充，代替 C）

`garden-v1.sbatch`（同目录）：L40S ×1、8 CPU、96 GB、6 h，日志 `/scratch/zr2245/analysis/garden-<jobid>.out`。
```bash
cp /scratch/zr2245/analysis/garden-v1.sbatch /scratch/zr2245/analysis/ 2>/dev/null; sbatch /scratch/zr2245/analysis/garden-v1.sbatch
squeue -u zr2245; tail -f /scratch/zr2245/analysis/garden-<jobid>.out
```
中途被杀就再 `sbatch` 一次，已完成的阶段自动跳过。

## C''. garden 数据集准备（10/15：`/scratch/zr2245/datasets/m360/` 只有 bonsai、room）

Room 是用 `images_2 + normals_2`（Metric3D，upstream/extract_normals.py）训的；garden 要同样的协议：`images_4 + normals_4`。
没有 normals 时 train.py 不报错、静默不用 normal 监督，所以 `garden-v2.sbatch` 开头会检查并自动补 normals_4。
```bash
cd /scratch/zr2245/datasets/m360 && wget -c http://storage.googleapis.com/gresearch/refraw360/360_v2.zip     # ~12 GB，断点续传
unzip -q 360_v2.zip 'garden/images_4/*' 'garden/sparse/*' -d /scratch/zr2245/datasets/m360/ && ls /scratch/zr2245/datasets/m360/garden && ls /scratch/zr2245/datasets/m360/garden/images_4 | wc -l
sbatch /scratch/zr2245/analysis/garden-v2.sbatch
```
