# release-adapt-v9 — 自适应 λ v9：v8 + phase-A 计划的两处计数修正（现行版本，2026-11-07）

## 内容
- `code/rdgrowth/train_rd_adapt_v9.py`（910 行，md5 a120351f64c9472e0353cef29d8ddb04）= v8（887 行）+ 头注释 + 代码 diff 4 行；由 `build/make_v9.py <v8> <v9>` 精确替换生成（11 处，每处断言恰好命中一次）。其余依赖同 ../release-adapt-v7/code/，HPC 上已有。
- `sbatch/adapt-v9-blender.sbatch`（162 行，VERSION v9b-r1，md5 68a42385114634c478a721a58edf50b6）、`sbatch/adapt-v9.sbatch`（154 行，VERSION v9-r1，md5 df13ce37898f7dcc34a6c39ccd5d7dba）：由 `build/make_sbatch_v9.py` 从 v8 的两个生成；改动 = 名字 adapt9 / 表 adapt-v9*.txt / 目录 `$S-adapt9b-t{T}mb(-best)`、`$S-adapt9-t{T}mb` / grep v9 关键行 / smoke 断言 densify 事件的 `remaining` == 按 config（densify_until, interval）算的 interval 数且带 `limiter_max`。
- `commands/node-batch12-v1.md`：测试块（6 job：lego/chair 2.5、chair 4、room/bicycle 10、room 20；heredoc 逐字节 = 本目录文件；由 `build/make_batch12.py` 生成，头文字在 `build/head12.txt`）。**已跑完**，结果 `results/adapt-v9-batch12-2026-11-07-paste.md`。
- `commands/node-batch13-v1.md`：43-job sweep 块（Mip/T&T 11 场景 × {5,10,20} − 已跑的 3 + Blender 8 × {2.5,4} − 已跑的 3；三个文件不改，按 md5 核对；场景参数同 batch9-v2）。**待 Wenhao 选 A 才提交。**
- `build/test_v9_cpu.py <v9.py>`：(1) intervals 公式 vs 穷举（5 种 schedule）；(2) 9500 处 6 条 batch11 轨迹重放：旧计划复现实测缺口（0.3–5 %），v9 到 V* 或 limiter；(3) 3500→9500 整段重放 6/6 到 1.000 V*。

## 两处修正（为什么）
1. `remaining = max(1, (until − it)//interval + 1)` → `max(1, ceil((until − it)/interval))`。upstream 只在 `iteration < densify_until` 时 densify：1500…9500 共 17 个 interval，旧代码在 9500 还以为剩 2 个，几何计划把缺口摊到两个 interval，最后只长到 sqrt(V·V*)。batch11 六个 run 最后一次 prune 后的模型 = 0.81–0.92 V*，= sqrt(V_9500/V*)。v4–v8 全部带着；v7b 常数先验在 Blender 上把 V* 抬高 30 % 盖住了它。
2. `plan_splits` 的前向模拟把 upstream 的允许量（`add_new_gs`：每个 interval 最多分裂 add_pct·V − V 个三角形；rd_select 的 max_split 同单位）除以 3 当顶点数 → 以为 limiter 紧 3 倍，过早 front-load 且只要允许量的 1/3。现在用三角形数。densify 行以前把 min(cap, limiter) 打成 `limit`（看起来像 limiter 卡住，其实是计划自己的上限），现在打印 `cap N limiter M`，事件带 `limiter_max`。

## batch12 结果（11/07 11:10Z，6/6 exit 0）
phase A 到位：最后一次 upstream prune 后的模型 / V* = lego 0.993、chair 0.964、bicycle 0.973、room 10 0.993、room 20 0.990（chair 4 在 9500 真被 limiter 卡：cap 57161 > limiter 46950）。落点 lego 2.5 0.991、chair 2.5 0.965、chair 4 0.947、bicycle 10 0.790、room 10 0.877、room 20 0.829（v7b 1.014 / 0.968 / 0.892 / 0.775 / 0.814 / 0.75；v8 0.916 / 0.879 / 0.797 / 0.710 / 0.795 / –）。phase B 在 Blender 上砍了（lego 4321、chair 7745 对），Mip 上门控关着。PSNR vs chord：lego −0.22（v7b −0.13）、chair +0.02、chair 4 −0.13；room 10 比 v7b +0.1 dB（+7.7 % 字节），bicycle 持平。

**Mip 剩余差的分解**：最终 cleaning 保留率 κ_end = 交付顶点 / 模型顶点 = 0.92–0.94（Blender 0.995）× 交付网格拓扑价 b_f × faces/vertex = 3.7–4.4 B/V（Blender 6.1；先验 5）。hit = κ × (b_v + 拓扑)/(b_v + 5)：room 10 0.86、bicycle 0.80、room 20 0.81（实测 0.877 / 0.790 / 0.829）。两者都要等 11k 的 restricted Delaunay 才量得到。瑕疵：计划在 9500 完成后 `rank_lambda(k=0)` 返回 lam_max → "phase A ended at lam 1e-1"，没砍的行报 `lam_final (shadow) 1e-1`（phase B 第一轮 capped 就重置为 shadow price，行为不变）。下一版可改：k=0 时保留上一轮的价。

## 之后
- A（推荐）：`commands/node-batch13-v1.md`（43 job）→ `adapt_analysis_v7b_sweep.py` 改读 adapt-v9*.txt → 19 场景表。
- v10 想法（方法改动）：11k Delaunay + 12k 网格 probe 之后再开一个增长窗口（量到的 κ / b_f / fpv 直接用），再 Delaunay 一次，phase B 砍 κ 的漂移。半天设计 + CPU 审查 + 6 job 测试。
