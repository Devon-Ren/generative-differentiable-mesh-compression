# hold 版本的完整流程（训练 → 编码 → 解码）

本目录 = `meshsplatting-rd-complete-v1`（2026-10-26 的完整包：upstream MeshSplatting、stage2 codec、编码器、解码器）
加上 hold 版本新增的文件。包内原有文件与 11 月的最新版本逐一比对过，内容相同（rd_growth_v2.py、rd_prune.py、
train_rd_v3.py、coarse_finetune.py、vis_collapse v12）。

## 新增文件（相对 10-26 的包）

| 文件 | 作用 |
|---|---|
| `rdgrowth/train_rd_adapt_v9b_abl7.py` | hold 的训练脚本（1083 行，md5 1eeb64ab5032e903f69ad1a21466d99f）。按字节精确删除的 `select_feasible` 已内联在其中 |
| `rdgrowth/probe_bytes_v1.py` | 训练中调用真实编码器测文件大小 |
| `blender/train_ckpt_v2.py`, `blender/val_select_v3.py` | 物体数据集：存 checkpoint、按验证集选择 |
| `jobs/ablation-b24-r8.sbatch`, `jobs/ablation-b24-r8-blender.sbatch` | hold 的 Slurm 任务（真实场景 / 物体），`ABL=hold` |
| `jobs/submit-batch31-v1.sh` | batch31（63 次 hold 运行）的提交脚本 |
| `jobs/capjob-v1.sh`, `jobs/budget-sweep-*.sbatch`, `jobs/submit-batch34-*.sh` | vertex cap 对照实验 |
| `jobs/generic-rd-v1.sbatch`, `jobs/generic_rd_v1.py` | Draco / TFAN / zstd 的 RD 曲线 |
| `analysis-tools/` | 结果分析脚本和 CPU 测试 |

## 三个阶段

**1. 训练（生成网格，带尺寸控制）**：`rdgrowth/train_rd_adapt_v9b_abl7.py`

hold 的参数（来自 `jobs/ablation-b24-r8.sbatch` 的 `hold)` 分支）：

    --trim_exact 5 --trim_hold --allow_any_prune_iter
    --phase_a_overshoot 1.35
    --prune_iters 12500 13000 ... 29500      （每 500 步一轮，共 35 轮）
    --probe_extra_iters （空）

每一轮：用真实编码器测当前文件大小；高于目标就按字节数删掉多出的部分；不高于就不动。
完整命令行见 sbatch 第 189-192 行。输入只有场景和 `--target_mb`。

**2. 编码**：`vis_collapse/scripts/run_garden.sh`（`FRACS="1.0"`）
→ `coarse_finetune.py`（3000 步微调）→ `vis_collapse/deliver_coarse.sh`（K4096 R256 字典）→ `stage2/` codec。
独立入口：`encode_v2.py --shared <deliver>/shared-final.pt --output <dir> --xyz qN --dc q7`，输出唯一文件 `asset.mslb`。

**3. 解码**：`decode.py --asset asset.mslb --output decoded.pt`（只需要这一个文件，CPU 即可）。
`verify_isolation.py` 在隔离进程里解码并核对；`render_asset.py` 从解码结果渲染并算 PSNR/SSIM/LPIPS。

## 与 HPC 路径的对应

| 本目录 | HPC |
|---|---|
| `upstream/` | `/scratch/zr2245/meshsplatting-original-lossless-v1/upstream` |
| `rdgrowth/` | `/scratch/zr2245/analysis/rdgrowth` |
| `vis_collapse/` | `/scratch/zr2245/analysis/vis_collapse-v12` |
| `stage2/` | `/scratch/zr2245/shared-collapse-stage2-v2` |
| `coarse_finetune.py` | `/scratch/zr2245/analysis/coarse_finetune.py` |
| `blender/` | `/scratch/zr2245/analysis/blender` |
| `jobs/*.sbatch`, `jobs/*.sh` | `/scratch/zr2245/analysis/` |

sbatch 里写的是 HPC 的绝对路径；换机器运行需要改 `LAB`、`AN`、`PKG`、`STAGE2`、`DATA` 这几个变量。

## 本备份里验证过和没验证过的

- 验证过：所有 Python 文件能编译，所有 shell / sbatch 语法正确；`rdgrowth/test_rd_prune_cpu.py` 通过；
  `analysis-tools/test_select_feasible_cpu.py` 对 abl7 通过；abl7 的 md5 与 batch31 提交时核对的一致。
- 没验证：没有在这里做过 GPU 上的端到端运行（这里没有 GPU 和数据集）。端到端结果是 HPC 上 batch31 的 61 次运行。
- 没法在这里确认：HPC 上的 `stage2` 和 `vis_collapse-v12` 是否与本目录逐字节相同。用 `../hpc/verify_hpc_md5.sh` 在登录节点核对。
