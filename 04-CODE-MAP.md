# 04 — 代码地图（`analysis-snapshot/` = `/mnt/project-files/analysis/`，HPC 上是 `/scratch/zr2245/analysis/`）

原则：**上游（Wenhao 的 repo、upstream/）从不改**；我们的每个工具都是 `analysis/` 下的新文件，import 他的模块（`--lab <LAB>` 指向 `meshsplatting-original-lossless-v1`，脚本 chdir 到 `upstream/`）。每个文件名带版本；旧版本保留不删（HPC 上也是）。**现行版本**在表里加粗。

## 1. rdgrowth/ — 训练时的 rate control

| 文件 | 版本 / 状态 | 作用 |
|---|---|---|
| rd_growth.py → **rd_growth_v2.py** | v2 = `--dd_norm mpx` | RDAccumulator（按 rend_ids 把像素 L1 残差归到三角形）、rd_select（ΔD/ΔR > λ）、rd_split |
| train_rd.py / train_rd_v2.py / train_rd_v2m.py | v1 / v2 / v2m（匹配计数控制 `--split_schedule`） | upstream train.training 的拷贝 + λ-split |
| **train_rd_v3.py** | **固定 λ 定稿**（10/19） | = v2m + phase-B λ-prune（`--prune_iters 12000 14000 16000 18000 --prune_mu 1 --prune_max_frac 0.15 --bytes_per_face 2`）；rd_log.jsonl / rd_summary.json |
| **rd_prune.py** | 定稿 | select_collapses（score = ring 残差 / ΔR，独立集匹配，foldover 过滤，cap）、collapse_edges（中点、属性/Adam 平均、重映射、退化/重复面删除、tracker 切片）；CPU 测试 test_rd_prune_cpu.py |
| train_rd_v4.py（文件名被复用了三次！） | 10/26 方向 A 版（rd_rebuild_v1）→ 10/31 速度版（render_v4）→ 11/05 mesh_reg 版；**当前内容 = mesh_reg 版**（VERSION train_rd_v4 2026-11-05，import mesh_reg_v1）；前两版内容只在 HPC 上（若未被覆盖）和 commands/rd-topo-v4.md / speed-v4-v1.md 的 heredoc 里 | 都不是现行；见 02 E6 / F1 / B12 |
| train_rd_v5.py | 方向 A v5（rd_rebuild_v2，FixedSoupRestriction，`--snap_only`） | 死线 |
| rd_rebuild_v1/v2.py, delaunay_canon_v1.py, delaunay_check_v1.py, test_rd_rebuild_cpu*.py | 方向 A | 死线（保留作消融） |
| render_v4.py | 速度版渲染 | 搁置 |
| mesh_reg_v1.py, test_mesh_reg_v1.py | VERSION 2026-11-05b | +++ 起点 |
| train_rd_adapt_v1 … v6.py | 自适应 λ 历史 | 见 02 §H |
| train_rd_adapt_v7.py（824 行，HPC 上有） | adapt-v7 | 有 bug（最终 probe 解包）→ v7b |
| train_rd_adapt_v7b.py（827 行，md5 24f478ca…，HPC 上有） | adapt-v7b = **跑完 47 job sweep 的版本**（results 在 release-adapt-v7/results/） | v7 + 一行修复；phase A 仍是常数先验 |
| train_rd_adapt_v8.py（887 行，md5 a150276d…；HPC 上有） | adapt-v8（测过：b_v 标定成立，落点退步，根因在计划） | v7b + phase A 现场标定：`SizeModel.update_attributes`（soup probe → b_v + overhead，拓扑先验不动）、`probe_bytes(..., min_vertices=)`、densify 块里的 probe_a 事件、`--probe_a_every`/`--probe_a_smoke`；由 release-adapt-v8/build/make_v8.py 从 v7b 精确替换生成（17 处） |
| **train_rd_adapt_v9.py**（910 行，md5 a120351f…；HPC 上有） | **adapt-v9 = 现行** | v8 + phase-A 计划两处修正：densify 块 `remaining = max(1, -(-(until − it)//interval))`、`plan_splits` 的 limiter 用三角形数（`add_pct * v - v`，不再 /3）、`max_split_up` / `limiter_max` / `cap N limiter M` 打印；由 release-adapt-v9/build/make_v9.py 从 v8 精确替换生成（11 处），test_v9_cpu.py 重放 batch11 的 6 条轨迹 |
| **probe_bytes_v1.py**（66 行，PROBE_VERSION 'probe-v1'） | 现行 | 训练中对 cleaning 后拷贝跑真 encoder（stage2_run.sh 子进程），返回 file_bytes / payload_bytes / vertices / faces |

自适应 λ 的 train_rd_adapt_v7(b)/v8 结构（给接手的人）：`SizeModel`（b_v/b_f/overhead 由 probe stream 拟合，`mb(V,F)`、`vertices_for(target)`；**v8 加 `update_attributes`：phase A 的 soup probe 只替换 b_v 和 overhead，`probes_a` 列表**）；**v8 phase A**：`iteration > opt.start_pruning` 且距上次 ≥ `probe_a_every` 的 densify interval 上，先 `probe_bytes(check=False, min_vertices=max(K,R))` → `size.update_attributes` → 再算 `V_star`（事件 `probe_a` / `probe_a_skipped`，densify 事件带 `size_source`）；`SurvivalMeter`（σ = 上一轮新顶点在 upstream prune 后的存活率）；`plan_splits`（几何计划：剩余 interval 内到 V* 要 split 多少）；`rank_lambda`（λ = 计划计数的 rank 价格，按 yield 修正）；phase A 每 500 it（densify_from…10000）；phase B 在 `--prune_iters`（Blender 12000 起每 500 到 19500；Mip 每 1000 到 19000）：`probe_bytes` → `extrapolate_probe`（LS 到 24000）→ excess → wanted = min(max(desired, n_lam), excess_pairs) → `edge_scores` / `rank_threshold` / `select_collapses` / `faces_removed_by` / `collapse_edges` → 精确 removed_mb → mode；`--probe_extra_iters` 诊断 probe；最终 cleaning（upstream 逐字）→ final probe → rd_summary.json（growth_capped 等）。

## 2. blender/ — NeRF-Synthetic 协议

| 文件 | 状态 | 作用 |
|---|---|---|
| train_ckpt_v1 → **train_ckpt_v2.py**（VERSION train_ckpt_v2 2026-11-04） | 现行 | 包装任意训练脚本（`--script`），hook TriangleModel.update_learning_rate 在 `--saves N…` 存 checkpoint（含 importance_score.pt） |
| train_ckpt_v3.py（--normals）/ v4（--override add_percentage/max_points） | 已试，无增益 / 高码率 dominated | |
| val_select_v1/v2 → **val_select_v3.py**（VERSION val_select_v3，recompute_importance） | 现行 | 影子数据集（transforms_val 当 test）、每个 checkpoint 评 val、选最佳（≥ 12k）、在选中 checkpoint **重算** importance（floor 0.999 / 4×）并 prune、写 `$G-best/original/point_cloud/iteration_30000`（symlink）、`--delete_unselected` |
| make_normals_input_v1.py, render_views_v1/v2.py | 诊断 | |
| rd-hirate-blender-v1.sbatch, rd-reg-blender-v1.sbatch | 死线 sbatch | |
| results-blender8-v1.md, results-reg-v1.md | 结果 | 8 场景表（含文献 Synthetic 数字）、正则化结果 |

## 3. vis_collapse/（v12，zip v3…v12 并存）+ coarse_finetune.py — level 协议

`vertex_footprint.py`（每顶点渲染灵敏度）→ `coarse_vis.py --target-vertices V0`（render-aware QEM，native/native_qem_vis.cpp）→ `../coarse_finetune.py`（3000 步，Blender 黑底）→ `deliver_coarse.py/.sh --k 4096 --r 256 --xyz qN --dc q7 --weight const --index-codec ctx`（k-means 字典 + 400 步 fit + stage2-v2 编码）；`scripts/run_garden.sh`（整条，`FRACS="1.0"` = level-1.0；表行标签仍叫 "garden frac"）、`run_axisA/axisB/rate_aware/index_rd/draco.sh`；`stage2_run.sh`（在 STAGE2_ENV 里跑 stage2 编码器，probe 也用它）；`codec_variants.py`（重导出变体 q13–q16、DC q6–q8、K×R）；`precision_sweep.py`、`index_rd.py`、`index_context_codec.py`。

## 4. stage2-v2-patch-v1/ — 交付 codec 补丁

patch + `apply_stage2_v2.sh`（在 stage2-v1 的 COPY 上建 shared-collapse-stage2-v2）+ roundtrip_check.py。格式 MSLB001：strip-fetch 顶点重排、edge-cache 算术 connectivity（64 条有向边缓存，新顶点 varint 0）、XYZ 每轴网格 qN + 最小索引邻居预测（mode 2）、DC q7 shuffle+zstd（无预测）、index ctx（先序 1-ring 不同码 ≤ 32）、常数 vertex_weight。spec：baselines/stage2-v2-bitstream-spec-v1.md。

## 5. rd/ — 结果与 sbatch（225 个文件）

- sbatch：adapt-v1/v1b/v2/v3/v4/v5/v5-blender(-v2)/v6-blender/v7/v7-blender/v7b/v7b-blender/v8/v8-blender（release-adapt-v8/build/make_sbatch_v8.py 从 v7b 生成）/**v9/v9-blender**（release-adapt-v9/build/make_sbatch_v9.py 从 v8 生成：名字 adapt9、表 adapt-v9*.txt、smoke 断言 `remaining` == 按 config 算的 interval 数且有 `limiter_max`）、ba-sens-v1、indoor-diag-v1、indoor-xyz-v1、lambda-k-fan-v1/v2、redeliver-v3-blender。
- 贴回的表 `*-<date>.txt`、整理 csv、分析脚本 `adapt_analysis_v*.py`（→ `adapt-v*-analysis.txt`）、`adapt_replay_v7.py`（CPU 回放 v6 log）、`plot_*.py`、`build_11scenes_v*.py`、`allocator_greedy.py`、notes md。关键汇总：`rd-11scenes-v3.csv`（11 场景六点 hull）、`rd-hull6-v2-table.md`、`lambda-k-fan-v1-results.md`、`indoor-diag-v1-results.md`、`adapt-v6-blender-analysis.txt`、`adapt-v7-replay-on-v6-logs.txt`、**`adapt-v7b-blender-analysis.txt`（batch8-v2）、`adapt-v7b-sweep-analysis.txt` / `adapt-v7b-sweep.csv` / `adapt-v7b-sweep-table.tex`（`adapt_analysis_v7b_sweep.py`：命中表、等字节 ΔPSNR vs rd-hull6-v2 上包络、growth-capped 清单、论文表）**。
- 图：rd-11scenes-v3.png、rd-main-v2.png、room-rd-codebook-levels-v1.png、budget-vs-collapse-7scenes-v1.png、seven-scene-hulls-v2.png、lambda-vs-cap-7scenes-v1.png、pass-curves-7scenes-v1.png、three-scene-hulls-v1.png 等。

## 6. commands/ — 每一次 HPC 粘贴块（可复现每张表）

按时间：rd-sweep-v1/v2(-min).md、rd-ctrl-v1.md、rd-prune-v3.md、error-anatomy、envmap-probe、batchA-originals-v1.md、batchA-resume-v1.md、batchB-rd3-v1.md、generic-codecs-v1、o3dgc-v1/v2/v3、rd-topo-v4/v5、planarity-v1/v1b、texrefit-v1、p2b-lego-v1/v2、profile-train-v1/v1b、speed-v4-v1、viewer-v1…v3、rd-hull6-v1/v1b.md、rd-indoor-v1.md、indoor-move-to-general-v1/v2、nerf-synthetic-v1.md（§1–§15）、indoor-diag-v1(-stages).md、indoor-xyz-v1.md、lambda-k-fan-v1(-garden).md、node-batch1…9-v1.md、node-batch8-v2.md、node-batch9-v2.md、node-batch10-v1.md、**node-batch11-v1.md（v8）**、make-node-batch7/8/9-v1.py（生成器：嵌入文件为 heredoc、算行数/md5；v8 的生成器在 release-adapt-v8/build/make_batch11.py）。还有早期 sbatch（garden-v1…v3、scene-v1、scene-axisA-v1、bonsai-axisA-v1/v2、g0-budget-v1、budget-sweep/control-v1、crossbank-v1/v2、hull-v2-v1、axisA-levels-v1/v2、rd-prune-v4、texrefit-v1）和 geom_precision.py（→ xyz_rule_v1）。

## 7. diag/ — 诊断与 +++ 探针

error_anatomy_v1、envmap_fit_v1/v2、mesh_stats_v1/v2（manifold / sliver / 分量统计）、profile_train_v1/v1b、planarity_v1 / planarity_points_v1 / planarity_detail_v1（+++ P1/P1b）、texture_refit_v1/v2、surf_extract_v1/v2（P2b）、p2b-lego-v1/v2.sbatch、results-p2b-lego-v2.md。

## 8. report/ — technical report 与图

latex-v1…v5（main.tex、figures/、main.pdf）、report-latex-vN.zip、report-vN.md/pdf、tables-v1/v2.md、figstyle_v1/v2.py（统一风格）、fig_*.py → fig-*.png/pdf（adapt、ba-sens、codecs-room、fan、gauss-class、in-the-loop、mean-rd、mesh-class、split-prune）、report_tables_v1/v2.py、cross-tables-v1.tex。

## 9. baselines/、figures/、viewer-v1…v3、根目录工具

baselines：baselines-room-garden-v1.md（文献逐场景）、generic-codecs-*.md、generic-v1/（Draco 等驱动）、o3dgc-v1/v2（Open3DGC 驱动 + codec py）、stage2-v2-bitstream-spec-v1.md。viewer：export_viewer_v1 / .msv 查看器（iPhone 测试）。根目录：coarse_finetune.py、index_locality.py、redundancy_stats.py、topology_reference.py、vertex_weight_check.py、submit_room_stats.sh。

## 10. 包（release）

- `release-fixed-lambda/meshsplatting-rd-complete-v1.zip`（2507 文件，VERSION "meshsplatting-rd-complete-v1 2026-10-26"）：upstream/（含 UPSTREAM_PROVENANCE.json）+ rdgrowth（train_rd_v3）+ vis_collapse v12 + coarse_finetune + xyz_rule_v1 + stage2（v2）+ encode_v2.py / decode.py / verify_isolation.py / render_asset.py / viewer + rd_pipeline.py / run_rd.sh / submit_rd.sh / setup.sh / env.sh + results/all-results-v1.csv（287 点）+ docs/。
- `legacy-packages/meshsplatting-{dense,collapse}-complete-v1.zip`：10/09 的两条链路（pipeline.py、stage2/coarse_once.py）。
