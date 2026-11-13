# BudgetMesh 实验备份（2026-11-13）

仓库提交：de8af56（分支 claude/zealous-lovelace-7h69j6）。

## 内容

| 目录 | 内容 |
|---|---|
| `hold-pipeline/` | hold 版本的完整代码：训练、编码、解码，加任务脚本。先读 `hold-pipeline/HOLD-PIPELINE.md` |
| `repo/` | 仓库的完整快照：所有 release-* 目录（每批实验的提交脚本、收集到的原始结果、分析）、论文草稿、图、给 Yao 的说明、历史文档 |
| `raw-uploads/` | 从 HPC 贴回来的原始 collect 文件（18 个），未经处理 |
| `hpc/` | 两个在 HPC 登录节点上运行的脚本，见下 |
| `MANIFEST-md5.txt` | 本备份里每个文件的 md5 |

## 结果在哪里

| 实验 | 位置（`repo/` 下） |
|---|---|
| hold 控制器，63 次运行（61 次完成） | `release-batch31-hold/results/` |
| hold 对 vertex cap，同文件大小的 PSNR | `release-batch31-hold/results/hold-vs-cap-2026-11-13.txt` |
| 上一版控制器和 cap 基线（21 个场景） | `release-batch30-overnight/results/` |
| Draco / TFAN / zstd 的 RD 曲线 | `release-batch33-generic-rd/results/` |
| 同一 cap（600,000）在 13 个场景上 | `release-batch34-same-cap/results/` |
| cap 的全部运行（178 次）分散在 | `release-g0-refresh-v1`, `release-batch21-v1`, `release-batch24-ablation-v1`, `release-batch28-cap-stability`, `release-batch30-overnight`, `release-batch34-same-cap` 的 `results/` |
| 给 Yao 的两页说明（LaTeX + PDF + 生成脚本） | `paper/yao-note/` |
| cap 基线报告 | `paper/cap-report/`, `paper/cap-baseline-report-2026-11-13.pdf` |
| 论文草稿 | `paper/cvpr/` |

## 这份备份不包含的东西

这些只在 HPC 的 `/scratch/zr2245/` 上，体积大，没法从这里取：

- 训练好的模型（每次运行的 `original/point_cloud/...`）
- 编码后的文件（`asset.mslb`）
- 每次运行的日志（`train.log`、`rd_log.jsonl`）
- 数据集

`/scratch` 通常有清理策略。`hpc/backup_hpc_runs.sh` 把代码、结果表和 hold 运行的日志打成一个 tar，
并报告编码文件的总大小，由你决定是否一并打包。

## 还没回来的结果

- batch31：ship 的 4 MB 和 1.5 MB 两次 hold 运行当时仍在跑。
- batch32：cap 的 1% 网格、目标的 ±5% 网格、无同类先验的 cap，已提交，结果未收集。
- batch34：cap = 3,000,000 的 13 次运行，脚本已准备，是否提交未定。
