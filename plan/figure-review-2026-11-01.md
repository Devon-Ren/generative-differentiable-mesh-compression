# 论文图表全盘审查（2026-11-01）—— 目标 CVPR main，mean score 4

回答 11/01 的四个问题：(0) `--indoor` 到底用没用、LPIPS 怎么统一、SSIM/LPIPS 的 RD；(1) 哪些图进正文、哪些进附录；(2) 诚实的 RD 和我们有优势 / 可比的区域；(3) "演化史"要不要画；最后是按顺序的运行清单和 GPU 小时。

## 0. 三个事实问题

### 0.1 `--indoor` 从来没有用过（所有 original 和所有 λ run）

- 证据：`analysis/commands/batchA-originals-v1.md` 第 71 行 "no --indoor, as for the 7 scenes"；`rd-sweep-v1.md` 第 4 行 "不加 --indoor，与现有 cap/collapse 结果同口径"；`train_rd_v3.py` 有 `--indoor`（默认 False，走 upstream 的 `update_indoor`），但 hull-6 的 sbatch 没有传它。
- upstream `arguments/__init__.py` 的 `update_indoor` 改 14 个超参：`iteration_mesh` 5000→12000、`densify_from_iter` 500→1000、`add_percentage` 1.23→1.27、`feature_lr` 0.0016→0.004、`start_pruning` 4000→3000、`prune_size` 1400→1300、`splitt_large_triangles` 100→500、`lambda_normals` 5e-5→1e-5、`lambda_weight` 1.9e-6→0、`size_probs_zero` 7.5e-5→0、`weight_lr` 0.03→0.05、`depth_lambda_init/final` 0.01/0.001→0/0（室内关掉深度先验）。室外默认值就是论文的室外设置，所以 5 个室外场景和 T&T 没问题。
- 后果：我们的 FP32 复现 vs 论文 per-scene：room −0.26、counter −0.57、kitchen −0.97、bonsai −0.90；室外全在 ±0.26 内。"我们比 MeshSplatting 本身高一点"目前只对**我们自己的复现**成立（λ 6.2e-4：room +0.06、kitchen +0.19、bonsai +0.20、counter −0.04）；对论文表格里的数，室内我们低 0.4–0.8 dB。审稿人把两张表一对就能看到。
- 修法：4 个室内场景用 `--indoor` 重跑 original + λ 点。`train_rd_v3.py --indoor` 不用改代码：Delaunay 重建在 11000、`iteration_mesh` 12000、phase-B prune 在 12k/14k/16k/18k，顺序兼容；λ 规则仍在 upstream 的 `add_percentage` 限速器内。只需要一个新的 `rd-prune-v3c.sbatch`（在 train 命令里加 `${INDOOR:+--indoor}`，original 阶段同样），不改 v3b。
- 成本：original 约 2 GPU-h/场景（normals + 训练 1–1.5 h + 交付 40 min；batchA walltime 3.5 h）→ 8 GPU-h；λ run 约 0.9 GPU-h（hull-6 实测 11 场景 × 2 λ ≈ 16–17 h）→ 16 run ≈ 14 GPU-h。**合计 ≈ 22 GPU-h**，像 hull-6 一样 general / tandon 各一半，一天墙钟。最小版（original + λ 6.2e-4 + 6.2e-3，12 run ≈ 15 GPU-h）会让室内只有 2 个 λ 点，图上不一致，不推荐。
- 风险：室内 +0.2 dB 的"λ 高于 original"可能变；结果是什么就报什么。它是用 upstream 自己的室内协议评估已定稿的方法，不是调参。
- 顺序：这 20 个 run 必须最先交，因为 Table 1 的室内行、统一评估、定性图都要等这批资产。

### 0.2 LPIPS：evaluator 一致，差的是室内（已核实）

- 我们所有 LPIPS 都来自 `training_report` 里的 `lpips.LPIPS(net='vgg')`（pip `lpips`，输入 [0,1]）——和 upstream `train.py` 第 501 行是同一个调用；upstream `metrics.py`（= 3DGS `metrics.py`）用 `lpipsPyTorch` VGG，权重和 scaling 相同，对 [0,1] 输入给同样的值。HAC / HAC++ Table II 也是这个口径。
- 论文没写 LPIPS 用哪个网络（正文、参考文献都没有）。但论文附录 Table 6–8 的 per-scene 数和我们 FP32 复现一对（PSNR / SSIM / LPIPS，论文 → 我们）：

| scene | 论文 | 我们 FP32（无 --indoor） | 差 |
|---|---|---|---|
| garden | 24.70 / 0.762 / 0.217 | 24.73 / 0.762 / 0.249 | 0 / 0 / +0.03 |
| flowers | 19.34 / 0.480 / 0.417 | 19.45 / 0.492 / 0.409 | +0.11 / +0.012 / −0.008 |
| treehill | 20.53 / 0.540 / 0.428 | 20.39 / 0.539 / 0.427 | −0.14 / 0 / 0 |
| truck | 22.32 / 0.799 / 0.237 | 22.39 / 0.807 / 0.226 | +0.07 / +0.008 / −0.011 |
| train | 18.72 / 0.693 / 0.337 | 18.90 / 0.712 / 0.319 | +0.18 / +0.019 / −0.018 |
| room | 28.52 / 0.873 / 0.271 | 28.26 / 0.883 / 0.339 | −0.26 / +0.010 / **+0.068** |
| counter | 26.51 / 0.846 / 0.279 | 25.94 / –  / –（level 1.0: 0.820 / 0.389） | −0.57 / / **≈ +0.11** |
| kitchen | 27.42 / 0.858 / 0.227 | 26.45 / – / –（level 1.0: 0.793 / 0.363） | −0.97 / / **≈ +0.14** |
| bonsai | 28.19 / 0.876 / 0.294 | 27.29 / – / – | −0.90 / / ? |

  室外 5 个场景和 T&T 的 LPIPS 在 ±0.02 内（多数我们更低）→ **evaluator 就是同一个（VGG）**，不存在"论文用 Alex"的问题。差距全在室内：PSNR −0.6…−1.0、LPIPS +0.07…+0.14，和 0.1 的 `--indoor` 缺失一致（室内 `feature_lr` 高 2.5 倍、深度先验关掉、prune 更早）。所以 0.40 vs 0.31 的均值差 = 室内四场景 + 交付资产相对 FP32 的 codec 损失，不是 evaluator。
- 论文 Table 5 列的室内/室外差异只有 5 行（feature lr、opacity lr、start pruning、βd、βo），代码 `update_indoor` 改 14 个（含 `iteration_mesh` 12000、`add_percentage` 1.27、`prune_size` 1300、`lambda_normals`、`splitt_large_triangles` 500）；**以代码为准**，直接用 `--indoor` 开关，不要手抄表格。
- 统一评估仍然要做（但目的变了）：5 个旧 original 没记 SSIM/LPIPS、各批次的日志格式不同、室内重跑后要全部重算一遍。`eval_all_v1.py`：66 交付资产 + 11 original + 作者 RGB-only PLY，解码 → 渲染 test → PSNR / SSIM / LPIPS-VGG（`metrics.py` 口径），一次跑完冻结；Alex 不需要了。成本 ≈ 2–3 GPU-h，排在室内重跑之后。

### 0.3 SSIM / LPIPS 的 RD

- 数据已经在交付日志里：44 个 λ 点全有 SSIM+LPIPS（`rd-prune3-*.csv`），level 1.0 有 9/11，original 有 6/11（bonsai/counter/kitchen/bicycle/stump 的 original 没记 SSIM/LPIPS）。
- 临时图 `analysis/rd/rd-metrics-v1.png`（λ 曲线的 Mip-360 / T&T 均值 vs 已发表的点，PSNR/SSIM/LPIPS 三行）；读法在 `rd-metrics-v1-notes.md`。要点：SSIM 和 PSNR 讲同一个故事（同一个 knee 在 6.2e-3），LPIPS 在低码率端掉得更快（Mip-360 0.376 → 0.515，28 → 4 MB），所以 150× 那个点只能卖大小不能卖感知质量；T&T 三个指标都是 6.2e-3 最好（R9 效应跨指标一致）；对 MiLo（253 MB）我们在 20 MB 处 PSNR +0.18、SSIM +0.024、LPIPS 差 0.08——室内重跑后 LPIPS 应收窄，但 LPIPS 是我们最弱的指标，表里要如实放。正式版等统一评估后重画（附录 S2）。

## 1. 正文放什么、附录放什么（审稿人视角）

原则：每张正文图回答一个审稿人会问的问题；全文一套视觉语言；baseline 只留 2024 及之后、同协议的（BakedSDF 2023 / MobileNeRF 2023 → 附录；3DGS、Scaffold-GS 只作为参考点）；所有 RD 横轴绝对 MB、带 "better" 箭头；压缩比只作为我们点上的标签，相对图例里点名的那个 FP32 资产。

**正文（8 页：6 图 + 3 表）**

| # | 图 / 表 | 回答的问题 | 数据状态 | 还缺 |
|---|---|---|---|---|
| Fig 1 | teaser：同一场景 FP32 742 MB vs 我们 5 MB（142×）的裁剪对比 + wireframe + 手机 viewer 截图（350 fps） | "这篇在干什么、值不值" | viewer 截图有 | 渲染裁剪 <0.5 GPU-h |
| Fig 2 | 方法总览图（手画）：growth loop 里的定价 split iff ΔD > λ·ΔR → phase-B λ-prune → codec（reorder / 局部 XYZ 预测 / SH 字典 K×R / context coder）→ 单文件 → 独立解码 | "方法是什么" | 0 GPU | 画 |
| Fig 3 | 主 RD（rd-main-v2 → v3）：(a) 同一资产、每种 codec——从 Room 升级为 Mip-360 9 场景均值（Table 1 扩到 9 场景后）；(b) engine-ready mesh 的点（MeshSplatting RGB-PLY、MiLo 2025、BOG 2024、RaDe-GS/GOF mesh 2024）；(c) Gaussian 压缩器相对各自 base 的 ΔPSNR（可选，参照写在图例） | "比现成 codec 好多少；在 mesh 类里是不是最小；对 3DGS 类怎么说" | (b)(c) 有 | (a) 的 9 场景 generic codec（CPU） |
| Fig 4 | "为什么必须在训练里决定顶点"：同一 codec、三种选顶点方式（post-hoc collapse / 固定上限 cap / λ-growth+prune）的 Mip-360 均值 RD + 同顶点数下的 ΔPSNR 条 | "进环的必要性" | 7 场景全有（budget-vs-collapse、lambda-vs-cap） | 重画成一张 |
| Fig 5 | 可压缩性分析：字节分解（XYZ / 索引 / SH 字典 / 连接）+ K×R 与精度 knee（axis A） | "为什么 740 MB 里只有 25 MB" | Room/garden 全有 | 重画 |
| Fig 6 | 同字节（≈10 MB）定性对比：我们 vs post-hoc collapse vs Draco-RGB vs texture route + 误差图 | "少的那 0.5 dB 长什么样" | 无 | <0.5 GPU-h |
| Tab 1 | 同一资产的 generic codec（11 场景均值 + per-scene 在附录） | 主张 1 | room/garden 有 | 9 场景 CPU |
| Tab 2 | 跨方法 4+4（PSNR/SSIM/LPIPS/MB/FPS/是否 mesh/解码端）：MeshSplatting（论文 / 复现 / RGB-PLY）、MiLo、BOG、我们 ×2；3DGS（参考）、HAC++、RDO-Gaussian、Compact3DGS 或 LightGaussian | 主张 2、3 | 全有 | 统一评估后填我们的 SSIM/LPIPS |
| Tab 3 | 部署：训练时长比、单核解码秒、手机/笔记本 FPS、文件格式 | "mesh 的好处兑现了吗" | 全有（table2-deployability-v1） | 无 |

**附录**

- S1 11 场景 per-scene RD（rd-11scenes-v3 瘦身：我们的 hull + post-hoc + HAC per-scene 曲线）——**这就是"如实报告的真实 RD"**，绝对 PSNR，不遮。
- S2 SSIM / LPIPS 的 RD 均值（统一评估后）。
- S3 完整 baseline 表（HAC++ Table II 全部行 + BakedSDF / MobileNeRF / RaDe-GS / GOF / 2DGS / Triangle Splatting opaque）。
- S4 非单调场景（flowers / treehill / truck / train）的 val vs test 表：多出的顶点拟合了训练视图。
- S5 codec 消融：K、R、XYZ 精度、context coder、index locality、跨场景码本。
- S6 λ-vs-cap 逐场景、prune 轮次消融、μ。
- S7 几何统计 / 轨迹外视图（limitation 的证据）。
- S8 无信息泄漏协议 + bitstream 规格。

## 2. 诚实的 RD、优势区、可比区

- **如实**：同字节下绝对 PSNR 比 HAC++ 低 3.3 dB、比 RDO-Gaussian 低 2.6 dB（Mip-360）。放 S1，正文一句话带数字，并引用 MeshSplatting 未压缩时对 3DGS 的差距（24.78 vs 27.46）说明这是不透明 mesh 的表示差距，不是 codec 的。
- **优势（可辩护）**：(1) 同一资产，generic codec 5×（PLY+zstd）到 15–17×（Draco / texture）更大；丢 SH 的 codec 掉 3 dB。(2) engine-ready mesh 里比所有已发表方法小 10–100 倍且 PSNR 不低（BOG +1.1 dB 但 913 MB）。(3) 部署：标准 mesh、单核秒级解码、手机 145–350 fps，3DGS 类 codec 没有这一列。(4) 训练比原版快（0.57–0.87×）。
- **可比（不是优势）**：对 3DGS 压缩器只在"相对各自 base 保住的质量"和"压缩比"上可比：同压缩比我们比 RDO-Gaussian 保得多；HAC 家族 +0.1…+0.3 是因为 Scaffold-GS 的 anchor 本身更可压，明说。

## 3. "演化史"（no-collapse → post-collapse → quantize → λ）要不要画

不画成历史。它映射成 Fig 4 的三条曲线：level 1.0 = 只有 codec（quantize）、post-hoc collapse、λ-growth+prune，同一 codec。这是消融，不是过程叙述。

"生长进环"更直观的证据，按说服力排：
1. 同顶点数对比：train-at-budget 比 post-hoc collapse 高 +0.3…+4.3 dB（14/14 点），最差 val 视图高 3–5 dB（已有，7 场景）。
2. 顶点去了哪里：11 场景 Spearman(residual, 顶点增量) = −0.73——字节花在多视角证据多的地方。可做成一个小 inset：某场景的顶点密度图 vs 误差图（CPU，从 checkpoint 出，0 GPU）。
3. 同字节定性裁剪（Fig 6）。
4. 训练反而更快（Tab 3）。
建议 Fig 4 = 1 + 2 的 inset。

## 4. 为什么现在的图"丑"，怎么统一

- 现在的问题：12 格网格、每点都有比率标签、每张图配色不同、图例位置不固定、横轴有时比率有时 MB。
- 统一样式（所有图一个脚本模块 `figstyle_v1.py`）：我们 = 绿色实线圆点；post-hoc = 橙色；mesh 类 = 黑色空心标记；Gaussian 类 = 冷色系细线菱形；参考点 = 黑星。比率标签只标主曲线 4 个 λ 点；横轴 log MB、整数刻度；"better" 箭头固定左上；图例固定右下或图外；8 pt 无衬线；双栏宽 7.0 in，单栏 3.4 in；PDF 矢量。逐场景网格只进附录。
- Fig 3/4/5 用这套重画一次（0 GPU）。

## 5. 运行清单（按顺序）与成本

| # | 内容 | 成本 | 依赖 |
|---|---|---|---|
| R1 | 4 室内场景 `--indoor`：4 original + 16 λ run（新 `rd-prune-v3c.sbatch` + 提交脚本） | ≈ 22 GPU-h | 你拍板 |
| R2 | Table 1 generic codec 扩到其余 9 场景（PLY+zstd、Draco、texture 路线） | CPU，≈ 1 h/场景，在 GPU 节点上跑 | 室内 4 场景等 R1 |
| R3 | 统一评估 `eval_all_v1.py`：66 资产 + 11 original + RGB-PLY，VGG/Alex 并排 | ≈ 3 GPU-h | R1 |
| R4 | 作者 RGB-only PLY 自测（room/garden） | <0.5 h | 无 |
| R5 | 定性渲染（Fig 1、Fig 6）+ 顶点密度/误差 inset | <1 GPU-h | R1（室内）|
| R6 | 全部图按统一样式重画 | 0 GPU | R2、R3 |

合计 ≈ 27 GPU-h + CPU。

## 6. 对分数的预期

现在是 borderline / weak accept 材料。到 4 需要四件事都成立：对 MeshSplatting 表格没有不一致（R1）；一个 evaluator（R3）；teaser + 同字节定性图（R5）；主图一眼看出"同一资产、每种 codec"的差距（Fig 3a 用 9 场景均值）。方法新颖性的短板（RDO 进训练在 3DGS 里有 RDO-Gaussian）不靠图解决，靠正文把贡献写成"第一条完整的 MeshSplatting 编解码链 + 可压缩性分析 + 把定价放进拓扑生长"。
