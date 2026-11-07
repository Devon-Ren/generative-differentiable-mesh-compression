# release-adapt-v8 — 自适应 λ v8：phase A 由 byte probe 现场标定（已测，2026-11-07；现行 = ../release-adapt-v9）

## 状态（11/07 09:00Z，batch11 6/6 exit 0，`results/adapt-v8-batch11-2026-11-07-paste.md`）
(a)(b)(e) 过：soup 上的 b_v 与 12k 网格 b_v 相差 0.8–8 %，overhead < 1 %，probe 2–3 min。(c)(d) 不过：落点 lego 2.5 0.916、chair 2.5 0.879、lego 4 0.846、chair 4 0.797、bicycle 10 0.710、room 10 0.795，全低于 v7b。根因不在 v8 的 probe，而在 v4 起的 phase-A 计划（多数一个 densify interval + limiter 单位 /3）→ 修在 v9（../release-adapt-v9/README.md）。v8 的 probe 代码原样进入 v9。

## 内容
- `code/rdgrowth/train_rd_adapt_v8.py`（887 行，md5 a150276d61de7c8c0f1686ed4a0d3fa5）= v7b（827 行）+ phase A 标定；phase B 一字不动。由 `build/make_v8.py <v7b> <v8>` 从 v7b 精确替换生成（17 处，每处断言恰好命中一次）。其余依赖（probe_bytes_v1、rd_prune、rd_growth_v2、train_rd_v3、train_ckpt_v2、val_select_v3、coarse_finetune、vis_collapse-v12、stage2-v2）与 `../release-adapt-v7/code/` 相同，HPC 上已有。
- `sbatch/adapt-v8-blender.sbatch`（155 行，VERSION v8b-r1，md5 bd60b5c0ef7946c9ae0a84de9d8e22f0）、`sbatch/adapt-v8.sbatch`（147 行，VERSION v8-r1，md5 27905f74be3f41a188fdec554bbc5caa）：由 `build/make_sbatch_v8.py` 从 v7b 的两个 sbatch 生成；改动 = TR/表名/目录名 v8、`PAE=${PAE:-1000}` → `--probe_a_every $PAE`、smoke 1600 it（1500 强制 phase-A probe `--probe_a_smoke`，1550 强制 prune）+ 新断言、结果行加每次 phase-A probe；Mip 版 phase B 改 16 轮（12000 起每 500 到 19500）。
- `commands/node-batch11-v1.md`：一次粘贴 = 写三个文件（heredoc，逐字节与本目录一致）+ `submit-batch11-v1.sh`（行数 + md5 核对、py_compile、bash -n、数据集/依赖存在，然后 6 个 sbatch）+ §2/§3 检查块 + 通过标准 (a)–(e)。由 `build/make_batch11.py` 生成。
- `build/test_v8_cpu.py`：SizeModel（常数 → soup 更新保留拓扑先验 → 12k 网格更新）与 probe 时点规则（17 个 interval、start_pruning 3000 → 7 次 probe）的 CPU 单元测试。

## 设计（为什么这样改）
- sweep（v7b，47 job）：phase A 的 V* = (T − 0.6 MB)/11 B 是常数先验；12k 实测 B/V 5/10/20 MB = 11–12.8 / 9–11 / 7.7–10 → 大目标 V* 偏低 10–30 %，10000 之后无增长杠杆 → Mip/T&T 全部欠目标。
- v8：`iteration > start_pruning`（第一次 upstream prune 之后；之前是未训练的初始 soup）的 densify interval，每隔 `--probe_a_every`（1000）跑一次 v2–v7 的 byte probe（cleaning 在最终 floor 0.9999 + 4× 下模拟、真 encoder 子进程）。`SizeModel.update_attributes` 只取 **属性价 b_v**（文件 − 拓扑流 − 字典 − 头 / kept 顶点）和 **overhead**（字典 + 头）；**拓扑价不取**（soup 的连接流与交付的 restricted-Delaunay 网格无关），b_f / faces_per_vertex 保持初值到 12k 网格 probe（`update`，v2–v7b 逻辑）。拓扑占 35–45 % 字节 → 剩余 V* 先验误差 ≈ ±10 %。kept share（probe 顶点 / 模型顶点）只记录（12k 0.8–1.0，结束 ≈0.98 → 交付顶点 = 模型顶点）。不做 b_v 对 V 的趋势拟合（Blender 实测 b_v 对 V 基本平：8.16/8.05/8.02 @ 118k/167k/218k；V* 每 interval 重算，偏差随 V → V* 收敛）。
- probe 的 K-means 需要 ≥ K=4096 个样本：kept 顶点 < max(K,R) 时 `probe_bytes(min_vertices=)` 返回 None，事件 `probe_a_skipped`，下个 interval 再试。
- 新日志：事件 `probe_a`（probe_mb、probe_vertices、vertices_model、kept_share、b_v、overhead_mb、soup_b_f（不用）、v_star_before/after、source）；densify 事件 `size_source`；`final` / rd_summary `probe_a_count`、`probes_a`、`probe_a_every`、`probe_a_smoke`、`start_pruning`。
- 预期（按 sweep 数字推）：room 5 0.88 → ≈1.0；room 20 0.75 → ≈0.87（limiter 仍卡 → growth-capped）；indoor 20 MB 可能因 12k 后 +27–39 % 的"摊开"超目标 10–25 % → 16 轮 phase B 砍回。lego/chair（limiter 全程卡住）预期与 v7b 持平，只验机制。

## 一键跑
见 `commands/node-batch11-v1.md` §1（6 job）。单个：`sbatch --reservation=yw523-l40s --account=torch_pr_38_general --cpus-per-task=6 --time=01:45:00 --export=ALL,SCENE=lego,XYZ=q14,TARGETS=2.5 --job-name=lego-2.5-adapt8b $AN/adapt-v8-blender.sbatch`；Mip：`--export=ALL,SCENE=room,INDOOR=1,IMAGES=images_2,XYZ=q14,TARGETS=10 $AN/adapt-v8.sbatch`（T&T 加 `DATA=/scratch/zr2245/datasets/tandt/$SCENE,IMAGES=images`）。batch12（33 个 Mip/T&T）= batch9-v2 的 `m ()` 列表换成 `adapt-v8.sbatch`。

## 通过标准（batch11）
(a) 6 job exit 0，smoke 有 `smoke probe-a`；(b) 行里 `last b_v X vs 12k b_v Y` 相差 ≤ 10 %，overhead ≤ 1 %；(c) room 10 / bicycle 10 hit 0.90–1.10（v7b 0.814 / 0.775），等字节 PSNR 比 v7b 不低超过 0.05 dB；(d) lego/chair hit 0.90–1.10（capped 行除外）且 PSNR 与 v7b 相差 ≤ 0.05 dB；(e) phase-A probe 总开销 Blender ≤ 3 min、Mip ≤ 8 min。(b)(c) 过 → batch12；(b) 过 (c) 不过 → 看 probes-a 的 V* 轨迹与 limiter 计数再定。
