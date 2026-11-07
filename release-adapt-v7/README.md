# release-adapt-v7 — 自适应 λ（closed-loop `--target_mb`）v7 / v7b 完整一键可跑包

## 状态（2026-11-07 08:00Z，v2 更新）
- **v7b 已跑完**：batch8-v2（lego/chair × 2.5/4，落点过）→ batch9-v2 + batch10（47 job sweep 全完）。所有表 / 原始粘贴 / 分析 / csv / LaTeX 在本目录 `results/`；提交块 `commands/node-batch8-v2.md`、`node-batch9-v2.md`、`node-batch10-v1.md`。结论：交付精确、phase B 落点误差 ≤ 3 %，Mip/T&T 欠目标 10–30 %（phase A 常数先验）→ 现行版本是 `../release-adapt-v8/`。下面是 v1 时的状态记录（保留）。

## 状态（2026-11-06 15:40Z，v1 原文）
- `code/rdgrowth/train_rd_adapt_v7.py`（824 行，md5 5aa4256eaa021ecf126787536cd27c5e）= node-batch8 写到 HPC 上的版本，**最终 probe 解包 bug**（第 747 行 `final_probe, _ = probe_bytes(...)`，v7 的 probe_bytes 返回三元组）→ 4 个 smoke 全挂。
- `code/rdgrowth/train_rd_adapt_v7b.py`（827 行，md5 24f478cab998b59d1886c2b8b956a111）= v7 + 该行修成 `final_probe, _, _ =` + version 字符串 'adapt-v7b'（config 事件、rd_summary.json、两处 print），其余逐字相同。CPU：pyflakes 干净、py_compile 过。**未上 HPC、未跑。**
- `sbatch/adapt-v7b-blender.sbatch`（139 行，VERSION v7b-r2，md5 2d110e69c79c3c2552c64fdbfef49c91）、`sbatch/adapt-v7b.sbatch`（129 行，VERSION v7-r2，md5 0649962ab5797f7f69dc39128d74a7e9）= v7 版 + TR 指向 v7b + 版本 grep/断言改 'adapt-v7b'；bash -n 过。run 目录名 / 表名 / job 名 不变（adapt7b-…，adapt-v7-blender.txt；adapt7-…，adapt-v7.txt）。
- 待做：用 `commands/make-node-batch8-v1.py` 的方式生成 node-batch8-v2.md（嵌入 v7b 两文件 + submit-batch8-v2.sh，核对行数 827/139、grep `version='adapt-v7b'` 与 `VERSION v7b-r2`），重交 lego/chair × {2.5, 4}；过了通过标准再生成 node-batch9-v2.md（F 表换成 v7b 文件名与 md5，写 `$AN/adapt-v7b.sbatch`，`$AN/submit-batch9-v2.sh`）。

## 一键跑什么（每个 sbatch 自含）
`sbatch --reservation=yw523-l40s --account=torch_pr_38_general --cpus-per-task=6 --export=ALL,SCENE=lego,XYZ=q14,TARGETS="2.5 4" --job-name=lego-2.5-adapt7b $AN/adapt-v7b-blender.sbatch`
阶段：文件/版本自检 → probe self-check（真 encoder 对 `$S-rd3b4-lam0.0062-mu1.0-best/axisB/v1.0`，probe/delivered 必须 0.92–1.08）→ smoke 1300 it（目标 0.01 MB，1250 强制 prune，`--probe_check_scaling`，断言 rd_log 的 prune/final 事件字段、rd_summary 两个 probe、growth_capped False、probe_extra_iters []）→ 每个目标 T：`train_ckpt_v2 --saves 12000…29000 --script train_rd_adapt_v7b.py -- --lab $LAB -s $DATA -i train -r 1 --eval --indoor -m $G/original --iterations 30000 --lam lam0_of(T) --target_mb T --bytes_per_vertex 10 --bytes_per_face 2 --faces_per_vertex 2.5 --dd_mode error --dd_norm mpx --phase_a_overshoot 1.0 --probe_floor 0.9999 --probe_scaling 4 --sigma_prior 0.5 --prune_iters 12000…19500 --prune_mu 1 --prune_max_frac 0.15 --probe_extra_iters 22000 26000 --probe_cmd "bash $PKG/stage2_run.sh $AN/rdgrowth/probe_bytes_v1.py" --probe_k 4096 --probe_r 256 --probe_xyz q14 --probe_dc q7` → val_select_v3 → level-1.0（run_garden.sh FRACS=1.0 on `$G-best`）→ 一行进 `$AN/adapt-v7-blender.txt`（hit、probe/delivered、test/val PSNR、growth-capped、limiter 计数、每轮 probe/est/mode、诊断 probe）。Mip-360/T&T 版 `adapt-v7b.sbatch`：IMAGES/INDOOR/XYZ/DATA 按场景，TARGETS "5 10 20"，prune 12000..19000 每 1000，无 train_ckpt/val_select（Mip 用 30k 最终模型）。

## HPC 依赖（都已在 HPC 上，提交脚本核对）
`$AN/rdgrowth/{train_rd_v3.py, rd_growth_v2.py, rd_prune.py, probe_bytes_v1.py}`、`$AN/blender/{train_ckpt_v2.py(99 行), val_select_v3.py(254 行)}`、`$AN/coarse_finetune.py`、`$AN/vis_collapse-v12/{stage2_run.sh, scripts/run_garden.sh}`、`/scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION`、`$LAB/upstream`、数据集、Blender 参考 run（self-check 用）。本目录 `code/` 放了同版本副本（vis_collapse-v12.zip、stage2-v2-patch-v1.zip 是 HPC 上那两个目录的来源）。

## 控制器设计（v7；v1–v6 的差异见 ../02-EXPERIMENTS-INDEX.md §H）
- Phase A（densify_from…10000，每 500）：SizeModel（b_v/b_f/overhead 先验 6/2/0.6 MB、faces_per_vertex 2.5，第一次 probe 后由 stream 拟合）给 V* = vertices_for(target)；SurvivalMeter σ（先验 0.5，测得后用）；plan_splits 给本轮 desired；λ_next = rank_lambda（下一轮计划计数的 rank 价格 × yield 修正）；upstream limiter（add_percentage）仍然生效 → limiter_bound 计数。
- Phase B（prune_iters）：probe_bytes（cleaning 在 floor 0.9999 + 4× 下模拟，真 encoder）→ extrapolate_probe（LS 到 24000）→ excess → wanted = min(max(desired, n_lam), excess_pairs)（门控 + 上限）→ 选 collapse（rank_threshold vs λ·μ）→ faces_removed_by 精确记账（只记 probe kept 集合内的顶点/面）→ collapse_edges（不一致 → RuntimeError）→ mode none/lambda/capped/count，λ := shadow price when cap/rank binds。
- 结尾：upstream 最终 cleaning → final probe → growth_capped = (last est < target) ∧ (limiter 卡住 > 一半 interval)。
- 通过标准 / 预期：见 ../00-README-HANDOVER.md §4。

## history/
v1–v6 代码与 sbatch、node-batch2…7 提交块、每版 .txt/.csv 结果与 `adapt-v*-analysis.txt`、v7 在 v6 log 上的 CPU 回放 `adapt-v7-replay-on-v6-logs.txt`、`redeliver-v3-blender-2026-11-06.txt`（Blender 参考点）、分析脚本 adapt_analysis_v*.py / adapt_replay_v7.py。
