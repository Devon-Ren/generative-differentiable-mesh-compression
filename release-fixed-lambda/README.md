# release-fixed-lambda — 固定 λ 定稿版本（technical report 里的版本，可回滚）

## 这是什么
`meshsplatting-rd-complete-v1.zip`（18.6 MB，2507 文件，`cat VERSION` → `meshsplatting-rd-complete-v1 2026-10-26`）= 一条命令跑完整链路：
train_rd_v3（rate-controlled growth + phase-B λ-prune）→ xyz_rule_v1（XYZ 精度）→ vertex_footprint → coarse_vis --target-vertices V0（level 1.0）→ coarse_finetune 3000 → deliver_coarse K4096 R256 400 步 fit → encode_v2.py（**只输出一个文件 asset.mslb**，MSLB001）→ verify_isolation.py（子进程 decode.py 在洗净的环境里只看得到那个文件；`sys.addaudithook` 审计每个 open/Popen；VIOLATION → exit 1；与 restore(prepare(shared-final.pt)) 精确一致）→ render_asset.py → viewer 导出。`result.json` 含 isolation_pass、encoder_deterministic、file_MB、ratio、dB。阶段可重启（`completed/<stage>.json`，`--resume`）。

包内 README 有完整用法（`bash setup.sh`；`bash run_rd.sh --dataset S --images images_2 --lam 0.0062 --output OUT`；`sbatch submit_rd.sh ...`，account torch_pr_38_general，48G，4 h）。λ 0.00185 ≈ 32× 点，0.0062 ≈ 73× 点。**注意：这个包只在本地验证过（verify.py、tests/、dry-run）；HPC 上的 setup.sh / run_rd.sh 从未跑过。** 论文里的数字全部来自 `protocol-paste-blocks/` 里的 sbatch 链（同一代码，分阶段跑）。

## 代码版本（与 analysis-snapshot 逐字节一致）
rdgrowth/train_rd_v3.py（2026-10-19）、rd_prune.py、rd_growth_v2.py、coarse_finetune.py、vis_collapse v12、stage2 = stage2-v2（2026-10-15）、xyz_rule_v1.py。价格常数 BPV=10（= B_a 6 + 2·B_f）、BPF=2、MU=1、MAXF=0.15；prune 12000/14000/16000/18000；K4096 R256；DC q7；XYZ 按场景（见 03-HPC §5 / memory/sweep-scene-settings-nov2026.md）。

## 结果数据
- 包内 `results/all-results-v1.csv`（287 点，10/26 之前的所有系列）+ `summary-by-scene-v1.md`。
- 10/26 之后：`../analysis-snapshot/rd/`：rd-prune3-hull2-2026-11-01.csv（六点 hull）、rd-prune3-indoor-2026-11-01.csv + originals-indoor-2026-11-01.csv（indoor 重跑）、rd-11scenes-v3.csv（11 场景总表）、lambda-k-fan-v1/v2.csv（λ×K fan）、ba-sens-v1.csv、indoor-diag-v1-results.md、`../analysis-snapshot/blender/results-blender8-v1.md`（NeRF-Synthetic 8 场景）、redeliver-v3-blender-2026-11-06.txt（val_select_v3 重交付的 Blender 参考点）。
- 固定 λ 网格：6.2e-4 / 1.85e-3 / 6.2e-3 / 1.85e-2（v2 mpx 单位）。

## protocol-paste-blocks/
每张表怎么来的（一次粘贴的完整块，含 sbatch heredoc）：rd-sweep-v1/v2(-min)（Room / 六场景 λ sweep）、rd-ctrl-v1（匹配计数控制）、rd-prune-v3（phase-B prune Room）、batchA-originals-v1 / batchA-resume-v1 / batchB-rd3-v1（11 场景）、rd-hull6-v1/v1b（六点 hull）、rd-indoor-v1（indoor 重跑）、nerf-synthetic-v1（Blender §1–§15）、lambda-k-fan-v1(-garden) + lambda-k-fan-v1/v2.sbatch + node-batch1-v1（fan 11 场景 + ba-sens）、ba-sens-v1.sbatch、indoor-diag-v1(-stages) / indoor-xyz-v1 + sbatch、generic-codecs-v1 / o3dgc-v3（通用 codec 基线）、g0-budget / budget-sweep / budget-control / scene-v1 / scene-axisA-v1（早期 axis / budget）、redeliver-v3-blender.sbatch。

## 报告
`report-latex-v5.zip` + `report-v5.pdf`（16 页，给 Yao 的 technical report v5，含 Table 6/7、Fig 7 fan、§2.10 ba-sens、adapt-v1 节）；`paper/`：technical-report-v1.md（10/30 分阶段说明）、related-work-v1.md、baselines-v2.md（文献数字、Table 2 行集）。
