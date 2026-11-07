# 01 — 进化历程（2026-10-09 → 2026-11-07）

每一行：日期 | 发生了什么 / 决定 | 证据文件（相对 `analysis-snapshot/`、`plan/`、`memory/`）。实验细节见 `02-EXPERIMENTS-INDEX.md`。

## 阶段 0：起点（10/09–10/12）

- 10/09 Wenhao 给了两个完整包 `meshsplatting-{dense,collapse}-complete-v1.zip`（同一代码，route 不同）：COLMAP → upstream train 30k → 严格 K4096 字典 → 残差 K1024 → [collapse: stage2/coarse_once.py] → encode（q18 XYZ、q8 DC、strip-fetch、因果 XYZ AC、edge-cache AC topology）→ CPU decode → render。Room：dense 41.399 MB / 27.934 dB，collapse 26.884 MB / 27.347 dB；FP32 基线 741.7 MB（819 MB 含训练统计）。
- 10/09 room-results-evidence 包 → 研究计划 v1（plan/cvpr2027-research-plan.md §0–§9）。Room 字节分解：topology 30.4 bit/v、XYZ 29.9、DC 15.1；coding-only 天花板 ≈ 4 MB；vertex_weight 可无损丢掉。
- 10/12 C2：render-aware collapse（vis_collapse v3–v5）：pure pixel_scale 更差（负结果）；uniform × 几何灵敏度权重 = +0.26 dB / +0.022 SSIM（finetune 后 PSNR 持平、SSIM/LPIPS 更好）；geo-only 默认；"等面数 ≠ 等字节"（payload 随顶点数）→ `--target-vertices`。
- 10/12 工作习惯规则（memory/user-wenhao-working-style.md）：一个 GPU 节点跑一切、完整粘贴块、版本化交付。

## 阶段 1：冗余分析 → convex hull（10/13–10/17）

- 10/13 **方向重置**（Wenhao + Yao）：别堆方法；论点 = "opaque mesh 的 SH 高度冗余"；两条 RD 轴 + hull。rate-aware 结果：精度拐点 q14/q8（后来 q14/q7），DC QAT 负结果，index_rd λ sweep（邻域上下文重分配），上下文编码器 v10（无损 −2 %）。Draco 把 soup 拆成 4.5× 顶点 → topology 编码方向关闭。
- 10/15 axis A（K×R）在 Room v0.8：hull K1024R1→K4096R1→K4096R256→R1024→R4096→K16384R4096；默认 K4096 R256 q14/q7 = 25.51 MB / 27.883。stage2-v2 codec patch（const vertex_weight + ctx index coder，用户批准，打在 COPY 上）。
- 10/16 **论文 story 定稿**（plan/story-candidates-2026-10-16.md）："texture-mapped mesh in disguise"。跨场景 codebook 证伪；index 局部性统计；garden（q15）与 bonsai（q15，无 0.8 平台）backbone 与 Room 同序；精度规则 b = ceil(log2(extent/(0.1·median edge)))；greedy allocator 精确复现 Room 全局 hull（9/9）。
- 10/17 counter/kitchen/bicycle/stump axis B + 精度（规则精确或低 1 bit，7/7 场景）；sharing+quant 在 100 % 顶点的损失 0.4–1.5 dB 是 q14 太粗造成的。
- 10/17 **G0（决定性）**：Room `--max_points 1.6M` → 1.35 M v，13.56 MB / 27.984 vs 事后 collapse 同顶点 26.66 → **+1.3 dB**；1.0 M 预算 +2.05 dB。→ "rate 控制生成"对几何是对的，事后 collapse 是错的杠杆。方法设计 plan/rd-growth-method-design-2026-10-17.md。

## 阶段 2：rate-controlled growth（10/18–10/24）

- 10/18 7/7 场景 train-at-budget > collapse（+0.3 … +4.3 dB）。**rdgrowth v1**（train_rd.py + rd_growth.py）：split iff ΔD_t/ΔR_t > λ；Room 6 个 λ：1.3 M 处与 cap 打平、0.5 M +0.25、0.23 M 明显更好。v2 加 `--dd_norm mpx`；六场景 × {2e-3, 6e-3}：λ ≥ cap 等字节 10/10 点，一个 λ 自适应停在 41–100 % / 15–43 % 顶点；story v2（plan/story-v2-cvpr-main-2026-10-18.md）。
- 10/19 **v2m 匹配计数控制**（importance 采样、同 split 数）：λ 排序本身 +0.12 dB，渐进 schedule +0.3 → 分解清楚。**v3 phase-B λ-prune**（rd_prune.py：edge collapse，score = ring 残差 / ΔR，独立集匹配，foldover 过滤，Adam 状态平均）：Room −11 % 顶点 / −8 % 字节 / PSNR 不变（等字节 +0.11–0.16）；"按面定价"不成立 → 论文写成 λ 反向应用。pass-curve 分析：λ ≥ 1.85e-3 时通过比例几何衰减（半衰期 ~3000 步）。**定位 v3**（Wenhao）：工程需求、完整 codec、lossy 不是弱点。低算力计划（FairShare 耗尽）。
- 10/19–10/20 质量差距诊断：error anatomy（背景 9 % SSE、silhouette 3.3×）、env-map 探针（+0.002–0.12 dB → 关闭）。why-mesh 回答（plan/why-mesh-2026-10-20.md）。
- 10/22 advisor deck v1（manim）；10/23 Yao 关切（baselines、topology coding、MPEG）→ 10/24 通用 codec 表（Room/garden：PLY+zstd、Draco、texture 路线）。
- 10/24 **batch B：v3 在 11 场景**（λ 1.85e-3 / 6.2e-3）：20.7 MB / 32× / −0.22；9.4 MB / 73× / −0.59；flowers/truck/train 上小 asset 的 test PSNR ≥ 大 asset（低覆盖场景多余顶点拟合训练视角）。bitstream spec 核实（stage2-v2-bitstream-spec-v1.md）。
- 10/25 readiness 复评（weak-accept 材料）；deployability：train 0.57×/0.87× 原时长、单核解码 2.3–6.4 s、iPhone 17 Pro 145–350 fps；viewer-v1。

## 阶段 3：定稿包、死线、+++ 探针（10/26–11/02）

- 10/26 **meshsplatting-rd-complete-v1.zip**（technical report 版本）：十段链路 + 隔离证明（本地验证，HPC setup 未跑）。几何碎片问题 = 类属性（limitation 句 + supp 图）。
- 10/26–10/27 **方向 A（decoder 侧 Delaunay 重建 connectivity）死**：v4 −1.8 dB，v5 固定 soup 两事件 −1.29 dB（训练很快离开 Delaunay 性）；snap-only 控制 Room +0.19 dB / −1.5 % 字节（单场景，用户拒绝跟进）。
- 10/29 "evidence-priced growth" 论点（Spearman −0.73；朴素 MDL 被 1e6 倍否定）；**MeshSplat+++** 构想（plan/meshsplat-plus-2026-10-29.md）。10/30 +++ 探针 P1（面级平面：失败）、P1b（顶点级：通过，indoor 46–75 % 顶点在平面上 → "世界平、mesh 皱"）、P2（opaque 渲染 room −2.0 / kitchen −3.6 dB；贴图在 ~1 px/face 无用）。**10/30 Wenhao：+++ 与 CVPR 严格分开**；CVPR 待办 = baselines + technical report。related-work-v1 / technical-report-v1 / baselines-v2 起草。
- 10/31 训练速度审查（backward kernel 43 %）→ train_rd_v4 速度版，用户搁置。
- 11/01 图表审查（发现 **indoor flag 从未用过**）→ 12 个 indoor 重跑 job（用户提交，算力耗尽）；六点 hull（λ 6.2e-4 / 1.85e-2 加入 11 场景）；MPEG Open3DGC 基线（Room 108 MB，我们小 3.5×）；report-v1…v4（figstyle 统一，用户图表规则 memory/feedback-figure-style.md）。indoor 重跑后原模型与论文一致（±0.09 dB），但 level-1.0 的 codec 损失 indoor 翻倍（−0.5 … −0.76）→ 诊断。
- 11/02 HPC 申请评估（CVPR 线 60–100 GPU-h、+++ ~200）；技术 report（Claude Doc）。

## 阶段 4：Yao 的质疑、单物体、report v5（11/03–11/05）

- 11/03 Yao 看 garden 73× 渲染，质疑 mesh 方向 → plan/advisor-questions-2026-11-03.md（P1/P2/P3 投稿选项，推荐 P3 = 加单物体章节）。B_a/B_f 常数的来历与 notation（B_a 6 B、B_f 2 B；plan/split-prune-rationale-2026-11-03.md）。
- 11/04 **NeRF-Synthetic 8 场景探针通过**：黑底、--indoor、val 选 checkpoint（train_ckpt_v2 + val_select_v2）；FP32 4.48 MB / 30.93；λ 6.2e-3 同 PSNR 59 % 字节；normal prior / supersample 无增益；高码率点 dominated（lego hull 饱和 ~9 MB）。几何控制审查（plan/geometry-control-review-2026-11-04.md：加约束降 PSNR；mesh_stats_v2：所有模型含作者的只有 ~1/3 manifold 边）。单物体数据集决策：加 NeRF-Synthetic，保留 Mip-360/T&T。
- 11/05 Wenhao 允许任何训练改动求 smooth → train_rd_v4 + mesh_reg_v1 一次性（lego a/b/c）**失败**（a −0.9 dB 被 dominated，b/c 塌到 19 dB，reg/main 梯度比 ~1000）→ 归 +++。surface-first 分解（plan/surface-first-thinking-2026-11-05.md）；P2b lego：surface+texture 前提失败（几何分层才是容量；open3d 在 HPC 不可用 libEGL）→ +++ 只剩 surface + alpha cards。
- 11/05 Yao 三个 RD 问题 + 23 条 report 评论 → indoor-diag（DC q8 只恢复 0.08/0.04 dB）、indoor-xyz（room q15 +0.08）、stages 分解（kitchen = SH sharing 85 %）、**λ×K fan**（固定 K4096R256/q7 距联合 hull ≤ 0.12–0.16 dB，K∝V）、ba-sens（价格误差 = λ 重标）→ **report-latex-v4 / v5**。**节点批准**：reservation yw523-l40s（gl030 4×L40S，11/05 09:35 → 11/12 09:35）。node-batch1（fan v2 8 场景 + ba-sens）。

## 阶段 5：自适应 λ（11/05 晚 → 11/06）

- node-batch2 **adapt-v1**（16 job 含 fan/ba-sens）：质量在 hull 上，大小只有目标 52–77 %（投影 bug：B_v 重复计连接性；最后 prune 后 upstream 收缩 0.85）。
- node-batch3 **adapt-v2**：真 encoder 做字节表（probe，probe/delivered 0.998–1.000）；hit 0.65–1.05；phase B 从不 collapse（当前 floor 下 probe 读低 1.1–1.6×）；densify dump：upstream prune 杀掉每轮 30–70 % 新顶点（σ 0.33–0.5）。
- node-batch4 **adapt-v3**：SurvivalMeter σ、λ = 计划计数的 rank 价格、probe 在最终 floor 0.9999、overshoot 1.15；hit 0.73–1.03（均 0.87），质量在 hull。缺口：probe 1× 只留 0.30–0.59 面（最终 cleaning 在 4×）、固定 λ prune 在欠目标时也砍 11–16 %、20 MB 受 limiter。
- node-batch5/6 **adapt-v4/v5**（Blender 测试台 lego/chair；v5 probe 4× scaling + 外推到 24000）：剩余缺口在交付链（val_select_v2 的 running-max importance），不在控制器 → **GPU-WASTE 规则**（11/06 12:28Z："给我仔细审查，不要不停返工了"）。
- node-batch7 **adapt-v6**（val_select_v3 修交付：probe/delivered 1.006–1.074，质量高于 hull）：命中 0.67–0.95；三处 phase-B 故障量化（记账高估 1.5–1.9×；λ-prune 16 轮全开砍 14–19 % V；chair 4/6 growth-capped）。
- 11/06 14:45Z Wenhao：睡前 1.5 h 内彻底 debug phase B、定版、然后过夜**场景 sweep**（不是 λ sweep）。
- node-batch8 **adapt-v7**（精确 kept-aware 记账 + 门控/上限 + growth_capped + 诊断 probe 22000/26000；CPU 在 v6 log 上回放 est@19.5k/final 1.006–1.030）→ 15:27Z **smoke FAILED**：最终 probe 二元组解包（v7 返回三元组）。→ **v7b** 一行修复已备（本包 `release-adapt-v7/code/`），node-batch9-v1（47 job sweep）等 batch8 通过后改成 v2 再给。
- 11/06 15:23Z Wenhao 要求本交接包；15:35Z："先把 zip 资料包整理出来给我，然后再 debug"。

## 阶段 6：v7b 定版、场景 sweep、v8（11/06 16:00Z → 11/07 08:00Z）

- 11/06 16:00Z 交接包 v1 交付。16:30Z node-batch8-v2（v7b：一行修复 + smoke 断言）→ 16:55Z 四个 job 全过：lego 2.5 hit 1.014、chair 2.5 0.968、lego 4 0.994、chair 4 0.892（growth-capped True），最后一轮 est 误差 ≤ 1.1 %，probe/delivered 1.00–1.04。质量：门控关着的 4 MB 两行比 v6 低 0.32 / 0.53 dB（v6 的 λ-prune 砍了 16–18 % V12 反而 PSNR 更高 → 渐进 λ-prune 是稀疏视角上的正则化器）。决策卡：Wenhao 17:08Z 选"Sweep v7b 现在"；v7c（λ-prune 常开）留作消融。证据 `release-adapt-v7/results/adapt-v7b-blender-2026-11-06.txt`、`adapt-v7b-blender-analysis.txt`。
- 11/06 17:20Z node-batch9-v2（47 job：Mip/T&T 11 × {5,10,20} + Blender 6 × {2.5,4} + 重交付 2）由 Wenhao 提交，4 卡过夜。
- 11/07 06:14Z 结果：42/47 exit 0（5 个 Blender 2.5 在 smoke 的 1300-it 最终 probe 上挂，空 probe log，harness 问题）。06:30Z sweep 分析（`adapt_analysis_v7b_sweep.py`）：Mip/T&T 全部欠目标（0.89/0.83/0.77），phase B 在 33 run 一刀没砍，等字节 −0.10 dB；**原因 = phase A 的 B/V 常数先验**（12k 实测 11–12.8 / 9–11 / 7.7–10 随目标变大而降）+ 10k 后无增长杠杆。07:17Z node-batch10-v1（5 job SMOKE=0）全过 → 47/47。
- 11/07 07:23Z Wenhao："依旧 adaptive 不能写进论文吗？" 答：现在只能写弱 claim（交付精确、落点误差 ≤ 3 %、Mip 欠 10–30 %）；建议一轮 v8（probe 标定 phase A）。07:25Z："v8 现在开始，同时 v7b 和 v8 更新进 handover zip"。
- 11/07 07:50Z **adapt-v8 交付**（node-batch11-v1，6 job）：phase A 用 soup 上的 byte probe 标定属性价 + overhead（拓扑先验留到 12k），V* 每 interval 重算；phase B 不动；Mip sbatch phase B 16 轮；smoke 1600 it 强制走一次 phase-A probe。CPU 审查（生成脚本精确替换、单元测试、断言块/结果行在假数据上跑通、heredoc 逐字节核对）。通过标准 (a)–(e)；过 → batch12（33 个 Mip/T&T 重跑）。
- 11/07 08:00Z 交接包 v2。
- 11/07 09:00Z batch11 结果（v8，6/6）：b_v 标定成立（0.8–8 %），落点全比 v7b 低。09:40Z 根因：计划多数一个 densify interval（v4 起）+ limiter 单位 /3 → **adapt-v9**（两处修正，代码 diff 4 行）交付 node-batch12-v1（6 job），Wenhao 10:30Z 提交。
- 11/07 11:10Z batch12 结果（v9，6/6）：phase A 到 V*（0.96–0.99）；lego 0.991、chair 0.965、chair 4 0.947、bicycle 10 0.790、room 10 0.877、room 20 0.829。Mip 剩余差分解为最终 cleaning 保留率（0.92–0.94）× 拓扑先验（5 vs 3.7–4.4）。11:40Z 决策卡片：A 今晚 v9 sweep（node-batch13-v1，43 job）/ B 先 v10 / C 停。
- 11/07 12:00Z 交接包 v3（本包）。
