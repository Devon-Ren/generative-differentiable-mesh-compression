# 02 — 全部实验索引（成功 / 失败 / 放弃 都列）

格式：**ID（日期）目的** | 配置 | 结果 | 采用情况 | 文件（相对 `analysis-snapshot/` = `/mnt/project-files/analysis/`；`memory/…` = 本包 memory 目录；HPC 路径以 `/scratch/zr2245/` 开头）。
"采用"含义：**纸** = 进论文/report；**默认** = 成了后续所有 run 的默认设置；**控制/消融** = 只作对照；**死** = 方向关闭，不再做实验；**+++** = 归下一篇。

---

## A. Codec 与冗余（Room 为主，10/09–10/16）

**A1（10/09）Room 字节分解** | dense 41.4 MB asset 的 stream | topology 30.4 bit/v、XYZ 29.9、DC 15.1；near-soup（45 % 边界边、8277 分量）；coding-only 天花板 ≈ 4 MB | 纸（§2 动机） | plan/cvpr2027-research-plan.md §0.5；memory/meshsplatting-status-history-oct2026.md

**A2（10/12–10/13）C2 render-aware collapse** | vis_collapse v3–v5，native_qem_vis：score = max(位置半径×pixel_scale, 外观半径×app_scale)；exp1 纯 pixel_scale；exp2 uniform × 灵敏度权重（sens_pos/sens_dc，power 0.5，clip 0.1–10）；消融 geo-only / geo+app / app-only | exp1 更差（val 27.66 vs 28.97）= 负结果；uniform×sens step-0 val +0.26 dB / SSIM +0.022；finetune 后 PSNR ≈，SSIM/LPIPS 更好；geo-only finetune test 27.695，交付 26.695 MB / 27.624（vs 历史 26.884 / 27.347）；geo-only 同面数多留 6 % 顶点 → 等面 ≠ 等字节 | 默认 geo-only + `--target-vertices`；后被 G0 取代 | memory/c2-collapse-experiments.md；vis_collapse-v3…v12.zip

**A3（10/13）精度 sweep（事后重导出）** | geo-only Room 1.98 M v；XYZ q18→q12，DC q8→q6；STE DC-QAT | q18→q16 免费、q15 −0.016、q14 −0.056、q13 −0.19、q12 悬崖 −0.6；DC q8→q7 −1.04 MB / −0.025 dB（最便宜杠杆），q6 −0.14；QAT 比事后差 0.1 dB（负结果） | 默认 q14/q7（后按场景规则）；QAT 死 | memory/rate-aware-results.md；rd/room-rd-points-v1…v4.csv

**A4（10/13）index_rd + ctx coder** | 邻域上下文 index 重分配 λ∈{0.3,1,3}，200 步 refit；v10 index_ctx 编码器 | index 频率均匀（H(z) = log2K − 0.3，无偏斜可用）；冗余是空间的（邻居命中 0.12–0.22，lift 100–780×）；ctx 无损 −0.55 MB（2 %）；λ 重分配 0.06–0.13 dB/MB | ctx coder 进 stage2-v2（默认）；index_rd 不用 | memory/rate-aware-results.md；rd/index-locality-v1.csv；room-index-locality-v1.png；index_locality.py

**A5（10/13）axis B（顶点数 100→50 %）Room** | run_axisB.sh，geo-only，3000 步 finetune，K4096R1024 q14/q8 | 1.0 36.61 MB/28.119 … 0.8 30.12/27.931 … 0.5 19.29/27.115；0.9→0.8 平台；worst view 0.7 以下崩 | 纸（hull）；默认 v0.8（后被 budget 取代） | memory/rate-aware-results.md；rd/room-rd-hull-v1/v2.png

**A6（10/15–10/16）axis A（K×R 字典）Room** | run_axisA.sh 在 v0.8，K∈{1024,4096,16384}×R∈{1,256,1024,4096}，k-means 4 轮，400 步 fit | hull K1024R1→K4096R1→K4096R256→R1024→R4096→K16384R4096；默认 K4096R256 q14/q7 = 25.513 MB / 27.883；每层（1.0/0.8/0.7/0.6）各自 K×R（42 行）→ 全局 hull 跳过 0.9，0.7 全在 0.8 之下 | 纸（Fig codebook levels）；默认 K4096 R256 | memory/axis-a-dictionary-results.md, room-final-format-hull-v2.md；rd/room-codebook-*.csv, room-rd-codebook-levels-v1.png；rd/allocator_greedy.py（贪心复现 hull 9/9）

**A7（10/15）stage2-v2 codec patch** | const vertex_weight（一个 logit 标量）+ ctx index coder，打在 stage2-v1 的 COPY 上 | keep/zstd 字节一致；v2 能解 v1；格式 v1→v2 统一 −12.5 % 字节、−0.027 dB（= DC q8→q7），hull 形状不变 | 默认（vis_collapse v12 起 STAGE2=/scratch/zr2245/shared-collapse-stage2-v2） | memory/stage2-v2-codec-patch.md, stage2-v2-bitstream-spec.md；stage2-v2-patch-v1/；baselines/stage2-v2-bitstream-spec-v1.md

**A8（10/16）跨场景 codebook** | Room↔garden：own / xfer-frozen / xfer-adapt / random-own 控制 | frozen −0.65、adapt −0.23 dB，比随机自样本（−0.05）还差；反向 −0.23/−0.15 | 死（"universal codebook" 证伪；纸：外观 = 场景特定纹理样本） | memory/crossbank-index-locality-results.md；rd/crossbank-2026-10-16.txt；commands/crossbank-v1/v2.sbatch

**A9（10/15–10/17）garden / bonsai / counter / kitchen / bicycle / stump axis B + 精度 + K×R** | commands/scene-v1.sbatch、garden-v3、bonsai-axisA-v1/v2、scene-axisA-v1 | garden q15 默认 27.34 MB / 24.371（27×）；bonsai 无 0.8 平台（斜率 0.095 dB/MB）、operating point = 1.0；4 场景 q14 太粗（sharing+quant 0.4–1.5 dB）→ q15/q16；K×R backbone 7/7 场景同序；精度规则 b = ceil(log2(extent/(0.1·median edge))) 精确或低 1 bit | 纸；默认场景 XYZ（见 memory/sweep-scene-settings-nov2026.md） | memory/garden-scene-results.md, bonsai-scene-results.md, scenes4-results.md；rd/scenes4-*.txt/csv, three-scene-hulls-v1.png；commands/geom_precision.py → xyz_rule_v1.py

**A10（10/24, 11/01）通用 codec 基线** | 同一 Room/garden asset：PLY+zstd、Draco（quant + 纹理路线）、Open3DGC SC3DMC/TFAN（MPEG-4 Part 16；SH 当 6×8 维属性）| Draco 拆 soup 4.5× 顶点；MPEG = 最强通用 codec：Room 108 MB、garden 126 MB，geometry-only 20 MB，纹理路线 146–149 MB；我们小 3.5×（等质量）/ 7.9×（−0.28 dB） | 纸 Table 1 | memory/o3dgc-mpeg-baseline.md；baselines/generic-codecs-2026-10-24.md, -11-01-v2/v3.md, o3dgc-v1/v2；rd/generic-codecs-*.txt；commands/o3dgc-v1/v2/v3.md；report/fig-codecs-room-v2

**A11（10/16）Delaunay/拓扑可预测性** | 由点预测连接性，只编码错误？ | Delaunay 面 1.6 %、16-NN 边 21 %、Draco 拆分 → 否 | 死（后来方向 A 再确认） | memory/crossbank-index-locality-results.md（TOPOLOGY QUESTION）；topology_reference.py

## B. 生成式 / rate-controlled growth（10/17–11/01）

**B1（10/17）G0 train-at-budget vs 事后 collapse（Room）** | `--max_points 1.6M / 1.0M`，level-1.0 协议 K4096R256 q14/q7；控制 = dense collapse 到同顶点数 | 1.35 M v：13.56 MB / 27.984 vs 13.93 / 26.660（**+1.32 dB**）；967 k：10.38 / 27.711 vs 10.14 / 25.665（+2.05）；dense→1.35 M 只花 0.006 dB/MB | 纸（关键发现）；方法转向 | memory/g0-train-at-budget-results.md；rd/g0-budget-2026-10-17.txt, room-budget-vs-collapse-v1.png；commands/g0-budget-v1.sbatch

**B2（10/17–10/18）budget sweep + 控制，7 个 Mip-360 场景** | budget-sweep-v1 + budget-control-v1 链 | 14/14 点 budget > collapse，+0.3 … +4.3 dB；均值 budget 1.6 M 14.4 MB / −0.68 / 46×，1.0 M 9.8 / −0.97 / 67×，旧 collapse 默认 23.6 / −0.44 / 28× | 纸（Fig in-the-loop）| memory/scenes7-budget-vs-collapse.md；rd/budget-sweep-2026-10-17.txt, budget-vs-collapse-7scenes-v1.csv/png, seven-scene-hulls-v1/v2

**B3（10/18）rdgrowth v1 λ-split，Room 6 个 λ** | train_rd.py：split iff ΔD_t/ΔR_t > λ，ΔD 由 rend_ids 归因的像素 L1，ΔR = 3·B_v（10 B）；λ∈{0,3e-4,1e-3,3e-3,1e-2,3e-2} | 1e-3 +0.21 dB（阻尼 split/prune 振荡）；3e-3 = 1.32 M v 13.70/27.985 与 cap 打平；1e-2 494 k 5.70/27.37（+0.25 vs cap）；3e-2 226 k 3.01/26.509（246×） | 纸；→ v2 | memory/rdgrowth-v1-design.md, rdgrowth-room-lambda-results.md；rd/rd-sweep-room-2026-10-18.txt, room-lambda-vs-cap-v1/v2.png；commands/rd-sweep-v1.md

**B4（10/18）v2 mpx 单位，六场景 × {2e-3, 6e-3}** | rd_growth_v2.py / train_rd_v2.py `--dd_norm mpx` | λ ≥ cap 等字节 10/10 点（+0.0 … +0.57）；均值 2e-3 19.6 MB / −0.26 / 36×，6e-3 8.8 / −0.88 / 80×；同 λ 停在 41–100 % / 15–43 % 顶点（场景自适应停止） | 纸 | memory/rdgrowth-scenes7-results.md；rd/rd-sweep2-2026-10-18.txt, lambda-7scenes-v1.csv, lambda-vs-cap-7scenes-v1.png；commands/rd-sweep-v2-min.md

**B5（10/19）v2m 匹配计数控制** | train_rd_v2m.py `--split_schedule`（同 split 数、importance 采样） | 348 k v 4.31 MB / 26.81 vs λ 曲线 26.93 → 排序 +0.12 dB，渐进 schedule +0.3 | 控制/消融（纸 Fig 4 消融） | memory/rdgrowth-v3-prune-design.md；commands/rd-ctrl-v1.md

**B6（10/19）v3 phase-B λ-prune（Room）** | rd_prune.py select_collapses / collapse_edges；`--prune_iters 12000 14000 16000 18000 --prune_mu 1 --prune_max_frac 0.15 --bytes_per_face 2` | 6.2e-3：439 k v 5.228 MB / 27.382（无 prune 5.695 / 27.370）；1.85e-3：12.572 / 28.004；−11 % 顶点 / −8 % 字节 / PSNR 不变；foldover 过滤拒绝多于接受；"按面定价" 不成立 | 默认（v3 = 定稿训练脚本）；纸写成 λ 反向应用 | memory/rdgrowth-v3-prune-design.md；rd/rd-prune3-room-2026-10-19.txt；rdgrowth/test_rd_prune_cpu.py；commands/rd-prune-v3.md

**B7（10/19）pass-curve 分析（0 GPU）** | 从 rd_log 提取通过比例 | λ ≥ 1.85e-3 时几何衰减（半衰期 ~3000 步）；λ ≤ 6e-4 候选集增长；延长 densify 窗口预测 < 3 % 改变 → 取消 | 纸 C4b "λ-set steady state" | rd/rd-pass-curve-2026-10-19.txt, pass-curves-7scenes-v1*；commands/rd-pass-curve-v1.md

**B8（10/24）batch A/B：11 场景 v3（λ 1.85e-3 / 6.2e-3）** | rd-prune-v3b.sbatch；新 4 场景 flowers/treehill/truck/train 的 originals（batchA） | 11 场景均值 20.7 MB / 32× / −0.22；9.4 / 73× / −0.59；v3 比 v2 少 4–8 % 字节；flowers/truck/train 小 asset test ≥ 大 asset | 纸 Table 1 | memory/rdgrowth-v3-scenes7-results.md；rd/rd-prune3-7scenes-2026-10-24.csv, rd-prune3-new4-2026-10-24.csv, rd-11scenes-v1.csv/png；commands/batchA-originals-v1.md, batchA-resume-v1.md, batchB-rd3-v1.md

**B9（11/01）六点 hull（λ 6.2e-4 / 1.85e-2 加入）** | rd-hull6-v1/v1b | 11 场景均值：6.2e-4 28.2 MB / 22× / −0.11；1.85e-2 4.5 / 149× / −1.25；bicycle/flowers/truck/train/treehill 的高码率点 dominated | 纸（Pareto hull per scene） | memory/rd-hull6-results-nov2026.md；rd/rd-prune3-hull2-2026-11-01.csv, rd-hull6-v1/v2-table.md, rd-11scenes-v2/v3.*；commands/rd-hull6-v1.md

**B10（11/01）indoor 重跑（`--indoor` 预设从未用过）** | room/kitchen/counter/bonsai 原模型 + 4 λ（12 job） | 原模型与论文 ±0.09 dB（复现关闭）；绝对 hull 不变或更好（kitchen +0.7）；但 level-1.0 codec 损失 indoor 翻倍（−0.5 … −0.76）→ 诊断 D3 | 纸（Mip-360 均值更新：FP32 642 MB / 24.81；6.2e-4 26.6 MB / 24× / −0.24；1.85e-3 20.0 / 32× / −0.37；6.2e-3 9.0 / 72× / −0.92；1.85e-2 4.3 / 151× / −1.68） | rd/rd-prune3-indoor-2026-11-01.csv, originals-indoor-2026-11-01.csv, indoor-first-look-2026-11-01.md；commands/rd-indoor-v1.md

**B11（10/25）deployability** | 训练时长、单核解码、iPhone/laptop fps、viewer-v1 | train 0.57×/0.87× 原时长；解码 2.3–6.4 s；iPhone 17 Pro 145–350 fps | 纸 Table 2 | rd/table2-deployability-v1.md, decode-times-v1, fps-bench-v1, walltime-v1；viewer-v1…v3；figures/viewer-iphone17pro-bicycle-2026-10-25.jpg

**B12（10/31）训练速度审查** | profile_train_v1b；train_rd_v4（速度版：eval 只在 test_iterations、跳 identity pool、fused Adam、数据上 GPU） | backward kernel 43 %；预期省 20 %；用户搁置，v4 未跑 | 未采用 | memory/training-speed-review-oct2026.md；diag/profile_train_v1b.py；commands/speed-v4-v1.md；rdgrowth/render_v4.py（注意：rdgrowth/train_rd_v4.py 后来被 mesh_reg 版本重用了名字——见 E6）

## C. 质量差距诊断（10/19–10/20，已关闭）

**C1 error anatomy** | diag/error_anatomy_v1.py，Room dense | 背景 0.4 % px 占 9.1 % SSE（+0.74 dB 上限）、silhouette 3.3×、interior_hf 2.3×；train/test gap 2.45 dB dense vs 1.70 @440 k（dense 一半容量是记忆） | 纸（分析一句） | rd/error-anatomy-room-2026-10-19.md；commands/error-anatomy-v1/v2.md
**C2 env-map 探针** | diag/envmap_fit_v1/v2，room/garden/stump/bicycle | 事后 env map +0.002–0.121 dB → 关闭 | 死 | rd/envmap-probe-2026-10-20.md；commands/envmap-probe-v1/v2.md
结论：2.4 dB 差距 = opaque-mesh class gap（plan/quality-gap-assessment-2026-10-19.md, plan/why-mesh-2026-10-20.md）。

## D. 论文支撑（Yao 的 RD 问题，11/05–11/06）

**D1 λ×K fan（11/05–11/06）** | lambda-k-fan-v1（room/kitchen rd3i + garden × 4 λ × {4096×1, 1024×256, 4096×1024, 16384×1024} + DC q8）→ v2 其余 8 场景（node-batch1） | 固定 K4096R256/q7 距联合 hull ≤ 0.12（3 场景）/ 0.16（11 场景）dB；hull 只在两端离开（K1024 在 3–5 MB，R1024/K16384 ≥ 17 MB）→ K∝V；K4096R1 永远最差；DC q8 只 room 有效 | 纸 Table 6 / Fig 7（report v4/v5） | rd/lambda-k-fan-v1-results.md, lambda-k-fan-v1/v2.csv, fan-analysis-v1/v2.txt；commands/lambda-k-fan-v1.md, node-batch1-v1.md；report/fig-fan-v2
**D2 ba-sens（11/06）** | kitchen/garden × B_v∈{8,13} × 2 λ | 都在 B_v 10 曲线上（−0.01 … +0.07 dB）；λ_eff/λ 0.78 / 1.31 vs 预测 0.80 / 1.30 → 价格误差 = λ 重标 | 纸 §2.10 | rd/ba-sens-v1-analysis.txt, ba-sens-v1.csv；rd/ba-sens-v1.sbatch；report/fig-ba-sens-v1
**D3 indoor-diag + indoor-xyz + stages（11/05）** | room/kitchen 重导出：DC q8 / K16384R1024 / K4096R1024 / 真 weight；XYZ q15/q16；summary.json 各阶段分解 | DC q8 只恢复 0.08/0.04 dB；K16384R1024 +0.09/+0.18（+12.5 % 字节）；kitchen = SH sharing 85 %；room = sharing 0.47 + quant/coding 0.35（DC 0.08、XYZ 0.08–0.11）；b 规则在 5/8 场景低 1 bit（operating point 不变） | 纸 Table 7（损失分解）；回答 Yao "DC q7 不是天花板" | rd/indoor-diag-v1-results.md；commands/indoor-diag-v1.md, indoor-diag-v1-stages.md, indoor-xyz-v1.md；rd/indoor-diag-v1.sbatch, indoor-xyz-v1.sbatch
**D4 report-latex v1→v5** | 16 页 technical report（LaTeX） | v2 → Yao 23 条评论 → v3（W/F 修改）→ v4（fan/indoor 表）→ v5（node-batch2 结果） | 给 Yao | report/latex-v5, report-latex-v5.zip, report-v5.pdf；plan/advisor-report-comments-2026-11-05.md

## E. NeRF-Synthetic 单物体线（11/04–11/05）

**E1 探针（11/04）** | 8 场景，黑底，`--indoor`，train_ckpt_v2 每 1k 存 checkpoint（12k 起）+ val_select_v2（官方 val 选最佳、补做最终 importance prune）；FP32 + λ 6.2e-4 / 6.2e-3 | FP32 4.48 MB / 30.93（val-best 31.55）；6.2e-4 5.10 / 31.21；6.2e-3 2.63 / 30.93；3DGS 白底 33.31；lego 30k 数字是 schedule artefact（峰在 11k–21k） | 纸（单物体章节）；默认 Blender 协议 | memory/nerf-synthetic-probe-nov2026.md；blender/results-blender8-v1.md；commands/nerf-synthetic-v1.md（§1–§15 全部 paste block）
**E2 normal prior（11/04）** | train_ckpt_v3 --normals + make_normals_input_v1 | 无增益（val 34.838 vs 34.812） | 死 | 同上
**E3 supersample 2/4/8（11/04）** | 评估超采样 | 33.65 / 33.19 / 33.01 无增益 | 死 | 同上
**E4 高码率点 lego-hi（11/05）** | train_ckpt_v4 --override（λ 6.2e-5，add_percentage 1.5，max_points 1.5 M） | 10.90 MB / 33.28 vs 6.2e-4 9.26 / 33.50 → dominated；lego hull 饱和 ~9 MB（upstream 每轮 opacity/size prune 限制平衡 mesh） | 死 | blender/rd-hirate-blender-v1.sbatch
**E5 mesh 统计（11/04）** | diag/mesh_stats_v2.py，所有模型含作者 Room 原模型 | 都只有 ~1/3 manifold 边（非流形 18–28 %、边界 37–46 %）；sliver < 10° 21–30 %；我们 λ 6.2e-3 Room 与原模型每边相当、分量少 16× | 纸（E2 回答：rate control 不额外碎片化） | plan/geometry-control-review-2026-11-04.md §5
**E6 mesh 正则化 train_rd_v4 + mesh_reg_v1（11/05，一次性）** | sliver / Laplacian / normal / jump filter；lego 6.2e-4 配置 a/b/c | a 6.35 MB / 32.60 被 6.2e-3（3.10 / 32.58）dominated；b/c 塌到 19 dB；reg/main 梯度比 260–3100 | 死（归 +++，起点 w 1e-4/1e-3 记录在 memory） | memory/mesh-reg-v4-line.md；blender/results-reg-v1.md；rdgrowth/mesh_reg_v1.py, test_mesh_reg_v1.py；blender/rd-reg-blender-v1.sbatch
**E7 val_select_v3 重交付（11/06）** | 在选中 checkpoint 重算 importance（floor 0.999 / 4×） | 参考 run 字节 ×0.81–0.90，+0.00–0.02 dB（lego 7.517 MB / 33.516；2.763 / 32.583；chair 4.660 / 32.212；2.391 / 32.166） | 默认（Blender 交付链） | rd/redeliver-v3-blender-2026-11-06.txt；rd/redeliver-v3-blender.sbatch；blender/val_select_v3.py

## F. 死线 / 放弃（不再做实验）

**F1 方向 A：decoder 侧 Delaunay 重建 connectivity（10/26–10/27）** | v4 rd_rebuild_v1（逐轮重建）；v5 rd_rebuild_v2 FixedSoupRestriction（20000 / 26000 两事件）+ snap-only 控制 | v4 −1.8 dB；v5 −1.29 dB（soup 面保留 1.5 % / 11 %：训练很快离开 Delaunay 性）；snap-only +0.19 dB / −1.5 % 字节（单场景） | 死；纸作 "connectivity 不可再导出" 消融 | memory/decoder-side-topology-idea.md；rdgrowth/delaunay_canon_v1.py, delaunay_check_v1.py, rd_rebuild_v1/v2.py, train_rd_v4.py（topo 版本，与 E6/B12 同名文件注意版本）, train_rd_v5.py；commands/rd-topo-v4/v5.md
**F2 跨场景 codebook**（A8）死。 **F3 Draco/topology 编码**（A11）死。 **F4 DC QAT**（A3）死。 **F5 env map**（C2）死。 **F6 normal prior / supersample / 高码率 Blender**（E2–E4）死。 **F7 mesh 正则化**（E6）→ +++。 **F8 训练速度 v4**（B12）搁置。
**F9 evidence-priced growth（10/29）** | CPU 检查 all-results-v1.csv：Spearman(residual, test/train gain) = −0.73（n=11）；朴素 MDL 定价差 1e6 倍 | 论点成立但未做 GPU probe（Room+flowers 1–2 GPU-h 提议，用户未批） | 未实施 | memory/evidence-priced-growth-principle.md；plan/evidence-priced-growth-2026-10-29.md

## G. MeshSplat+++ 探针（下一篇；严格不进 CVPR）

**G1 P1 面级平面（10/30）** | diag/planarity_v1.py，11 场景 | 面内部 ≤ 12.7 %；失败 | +++ | commands/planarity-v1.md
**G2 P1b 顶点级平面（10/30）** | diag/planarity_points_v1.py | indoor 46–75 % 顶点在平面上 → "世界平、mesh 皱"；通过 | +++（论证 4.1+4.2） | commands/planarity-v1b.md；run_planarity_v1b.sh
**G3 P2 纹理重拟合（10/30）** | diag/texture_refit_v1.py（nvdiffrast opaque） | opaque 渲染 room −2.0 / kitchen −3.6 dB；贴图 ≈ 顶点色（~1 px/face） | +++ | commands/texrefit-v1.md
**G4 P2b lego 曲面+贴图（11/05）** | diag/surf_extract_v2.py（TSDF + marching cubes + 简化）+ texture_refit_v2 | TSDF 曲面 25.4 vs opaque soup 30.6 vs soft 33.5；贴图随分辨率变差 → 几何分层是容量 → 只剩 surface + alpha cards | +++ | diag/results-p2b-lego-v2.md；commands/p2b-lego-v2.md；plan/surface-first-thinking-2026-11-05.md, surface-plus-alpha-cards-2026-11-05.md
注意：open3d 在 HPC 不可用（libEGL）。

## H. 自适应 λ（closed-loop `--target_mb`，11/05 晚–11/07）

全部在 `release-adapt-v7/`（v1–v7b：代码 + sbatch + 提交块 + 各版分析 + results/）、`release-adapt-v8/`（v8，已测）和 `release-adapt-v9/`（现行）。每版的 sbatch 都有：文件存在/版本 grep/py_compile、probe self-check（真 encoder 对已交付 mesh：probe/delivered 必须 0.92–1.08）、1300-it smoke（1250 强制 prune，断言事件字段）、然后每个目标 T 训练 30000 + 交付 + 一行结果进表。

| 版 | 批 | 改了什么 | 结果 | 文件 |
|---|---|---|---|---|
| v1 | node-batch2（11/05 晚，16 job 含 fan/ba-sens） | phase A desired = (target − R̂)/(3 B_v·remaining)，λ *= (n_pass/desired)^0.5；phase B λ *= (R̂/target)^2 | 质量在 hull（−0.12 … +0.07），大小 52–77 % 目标（B_v 重复计连接性；最后 prune 后 upstream 收缩 0.85） | rdgrowth/train_rd_adapt_v1.py；rd/adapt-v1*.txt/csv；commands/node-batch2-v1.md |
| v2 | node-batch3（12 job） | **真 encoder 做字节表**（probe_bytes_v1.py）；b_v/b_f/overhead 由 stream 拟合；phase B count-driven + shadow-price λ | probe/delivered 0.998–1.000；hit 0.65–1.05；phase B 从不 collapse（probe 在当前 floor 读低 1.1–1.6×）；densify dump：σ 0.33–0.5 | train_rd_adapt_v2.py, probe_bytes_v1.py；rd/adapt-v2*.txt；commands/node-batch3-v1.md, adapt-v2-densify-dump.md |
| v3 | node-batch4 | SurvivalMeter σ；λ_next = 计划计数的 rank 价格（去掉增益常数）；probe 在最终 floor 0.9999；phase B = 固定 λ prune + count collapses；overshoot 1.15 | hit 0.73–1.03（均 0.87）；质量 −0.10 … +0.07 | train_rd_adapt_v3.py；rd/adapt-v3*.txt；commands/node-batch4-v1.md |
| v4 | node-batch5 | 外推 / probe 1× 问题诊断 | 见 rd/adapt-v4-analysis.txt | train_rd_adapt_v4.py；commands/node-batch5-v1.md |
| v5 | node-batch6（Blender 测试台 lego/chair，v5-blender / v5-blender-v2） | probe 用 4× scaling；LS 外推到 24000 | 缺口在交付链（val_select_v2 running-max importance）→ GPU-WASTE 规则 | train_rd_adapt_v5.py；rd/adapt-v5-blender*.txt；commands/node-batch6-v1.md |
| v6 | node-batch7（lego/chair × 2.5/4/6） | val_select_v3 修交付；probe 在 floor 0.9999 + 4× | probe/delivered 1.006–1.074，质量高于 hull（chair +0.3–0.5 dB），但 hit 0.67–0.95；三处 phase-B 故障量化 | train_rd_adapt_v6.py；rd/adapt-v6-blender-2026-11-06.txt, adapt-v6-blender-analysis.txt, adapt-v7-replay-on-v6-logs.txt；commands/node-batch7-v1.md |
| v7 | node-batch8（lego/chair × 2.5/4，11/06 15:0xZ） | 精确 kept-aware 记账（faces_removed_by 镜像 collapse_edges，不一致 → RuntimeError）；门控（est > target 才砍）+ 上限（不砍过目标）；mode none/lambda/capped/count；growth_capped；`--probe_extra_iters 22000 26000` | smoke FAILED：最终 probe 解包 bug（一行）；smoke 前半（self-check 0.999、prune 精确记账）正常 → v7b | train_rd_adapt_v7.py；rd/adapt-v7-blender.sbatch, adapt-v7.sbatch；commands/node-batch8-v1.md, node-batch9-v1.md（sweep，待 v2） |
| v7b | node-batch8-v2（lego/chair × 2.5/4，11/06 16:55Z）；**node-batch9-v2 + batch10 sweep（47 job，11/07 07:17Z 全完）** | = v7 + 一行修复 | 测试台：落点过（1.014 / 0.968 / 0.994 / 0.892-capped，est 误差 ≤ 1.1 %），门控关着的 4 MB 比 v6 低 0.32/0.53 dB（λ-prune = 正则化器）。Sweep：probe/delivered 0.999–1.001；**Mip/T&T 33 行全欠目标**（5/10/20 MB 均 0.888/0.831/0.767，±10 % 内 5/2/1 of 11，phase B 一刀没砍，growth-capped 0/2/9，等字节 vs hull 均 −0.10 dB）；Blender 16 行：2.5 MB 可行目标 0.97–1.01，4 MB 全 capped 0.49–0.99。原因 = phase A 的 B/V 常数先验（见 00 §4） | train_rd_adapt_v7b.py；rd/adapt-v7b-blender.sbatch, adapt-v7b.sbatch；results/adapt-v7b-blender-2026-11-06.txt + analysis、adapt-v7b-sweep-2026-11-07-paste.md、adapt-v7-2026-11-07.txt、adapt-v7-blender-2026-11-07.txt、redeliver-v3-blender-2026-11-07.txt、adapt_analysis_v7b_sweep.py → adapt-v7b-sweep-analysis.txt / .csv / -table.tex；commands/node-batch8-v2.md, node-batch9-v2.md, node-batch10-v1.md |
| v8 | node-batch11-v1（lego/chair × 2.5/4 + room/bicycle 10，11/07 09:00Z 6/6 exit 0） | phase A 由 byte probe 现场标定：start_pruning 之后每 1000 it 在 densify interval 上 probe soup，只取 b_v + overhead（拓扑先验到 12k）；kept < 4096 跳过；Mip phase B 16 轮；smoke 1600 it | (a)(b)(e) 过：b_v@9500 vs @12k 0.8–8 %，overhead < 1 %，probe 2–3 min。(c)(d) 不过：hit 0.916 / 0.879 / 0.846 / 0.797 / 0.710 / 0.795，全低于 v7b → 根因 = 计划多数一个 interval（v4 起）+ limiter 单位 /3（见 00 §4） | release-adapt-v8/：train_rd_adapt_v8.py（887 行）、adapt-v8(-blender).sbatch、commands/node-batch11-v1.md、build/、results/adapt-v8-batch11-2026-11-07-paste.md |
| **v9** | **node-batch12-v1（lego/chair 2.5 + chair 4 + room/bicycle 10 + room 20，11/07 11:10Z 6/6 exit 0）**；**node-batch13-v1（43-job sweep，待 Wenhao 选）** | = v8 + `remaining = ceil((until − it)/interval)` + plan_splits 的 limiter 用三角形数 + `cap/limiter` 打印、`limiter_max` 事件；probe / phase B 不动 | phase A 到 V*（最后一次 prune 后 0.96–0.99 V*）；hit lego 0.991、chair 0.965、chair 4 0.947、bicycle 10 0.790、room 10 0.877、room 20 0.829；PSNR vs chord lego −0.22 / chair +0.02 / chair 4 −0.13；Mip 剩余差 = 最终 cleaning 保留率 0.92–0.94 × 拓扑价 3.7–4.4 vs 先验 5（分解到 ±0.02）。瑕疵：没砍的行报 lam_final = lam_max | release-adapt-v9/：train_rd_adapt_v9.py（910 行，md5 a120351f…）、adapt-v9(-blender).sbatch、commands/node-batch12-v1.md + node-batch13-v1.md、build/（make_v9.py, make_sbatch_v9.py, make_batch12.py, test_v9_cpu.py）、results/adapt-v9-batch12-2026-11-07-paste.md |

通过标准（v7/v7b，已判）：lego 2.5 / chair 2.5 / lego 4 命中 0.90–1.10，probe/delivered ≈ 1.0，chair 4 growth-capped True，最后一轮 est 误差 ≤ ±5 %（全过）、PSNR ≥ v6（4 MB 两行不过：λ-prune 正则化效应）。v8 / v9 的通过标准和判定见上表和 00 §4。

## I. HPC 批次（node-batchN，11/05 → 11/07，reservation 期间）

node-batch1（fan v2 8 场景 + ba-sens，~20 GPU-h，exit 0）→ batch2（adapt-v1 16 job）→ batch3（adapt-v2 12 job）→ batch4（adapt-v3）→ batch5（adapt-v4）→ batch6（adapt-v5 Blender）→ batch7（adapt-v6 Blender 6 job）→ batch8（adapt-v7 4 job，smoke FAILED）→ batch8-v2（v7b 4 job，过）→ batch9-v2（v7b sweep 47 job，≈45 GPU-h 过夜，42 过）→ batch10-v1（5 个 Blender 2.5 SMOKE=0 重交，过）→ batch11-v1（adapt-v8 6 job，过，落点退步 → 根因）→ batch12-v1（adapt-v9 6 job，过）→ batch13-v1（adapt-v9 43-job sweep，已写好，待 Wenhao 选 A/B/C）。每个 `commands/node-batchN-v1.md` 是一次粘贴的完整块（写文件 + 核对 + 提交 + 之后的检查命令）。
