# MeshSplatting Compression（CVPR 2027）— 阶段性总结 / 交接包 v3（2026-11-07）

`cat VERSION` → `meshsplatting-handover-v3 2026-11-07T12:00Z`。**v3 = v2 + 11/07 08:00Z → 12:00Z 的全部进展**：batch11（v8）结果与判定、phase-A 计划 bug 的根因（v4 起）、adapt-v9（两处修正）代码 + sbatch + 测试块、batch12（v9）结果与 Mip 剩余差的分解、43-job sweep 块 node-batch13-v1（待 Wenhao 选）。v2 = v1 + v7b 测试台 / 47-job sweep / v8。v1、v2 的内容原样保留；新东西在 §4 / §5 和 `release-adapt-v9/`。v1 的内容原样保留；新东西在 §4 / §5、`release-adapt-v7/results/`、`release-adapt-v8/`、`01` 阶段 6、`02` §H/§I、`03` §7。

这个包的目的：让一个**新的 thread / 新的 Claude / 新的合作者**一次读完就知道 Wenhao 干了什么、为什么干、Yao（导师 Prof. Yao Wang）的意见和我们的改进、现在卡在哪一步、下一步做什么，并且能**无缝接着做**（所有代码、所有实验数据、HPC 上的全部信息都在包里或有明确路径）。

## 0. 阅读顺序（30 分钟版）

1. 本文件（项目 idea、进化、Yao 反馈→改进、当前状态、下一步、工作规则）。
2. `03-HPC.md`：account、reservation、环境激活、目录、数据集、结果表、正在跑 / 待提交的 job。接手第一天必读。
3. `02-EXPERIMENTS-INDEX.md`：**每一个**做过的实验（成功的、失败的、放弃的）：目的、配置、结果、采用情况、文件。
4. `01-EVOLUTION.md`：按日期的进化历程和每次决策。
5. `04-CODE-MAP.md`：代码地图（哪个文件是哪个版本、现行版本是哪个、谁依赖谁）。
6. `release-fixed-lambda/`：固定 λ 定稿版本（technical report 里的版本）的一键可跑完整链路包 + 当时的 paste block + report v5。
7. `release-adapt-v7/`：自适应 λ（closed-loop `--target_mb`）v7 / v7b 的完整代码 + sbatch + 提交块（batch8-v2 / 9-v2 / 10）+ **results/（v7b 测试台 + 47 job sweep 的表、分析、csv、LaTeX）** + v1→v6 历史。
7b. `release-adapt-v8/`：adapt-v8（phase A 由 byte probe 现场标定；已测，probe 部分原样进入 v9）：代码、两个 sbatch、node-batch11-v1 提交块、build/、results/（batch11 全部输出）。
7c. `release-adapt-v9/`：**现行版本 adapt-v9**（v8 + phase-A 计划两处计数修正）：代码、两个 sbatch、node-batch12-v1（已跑）/ node-batch13-v1（43-job sweep，待选）提交块、build/、results/（batch12 全部输出）；README 里有根因、batch12 结果、Mip 剩余差的分解。
8. `analysis-snapshot/`、`plan/`、`paper/`、`memory/`、`uploads/`、`slides/`、`legacy-packages/`：全部原始资料快照（见 §8）。

## 1. 项目一句话

把 **MeshSplatting**（arXiv 2512.06818，可微分不透明三角 mesh 的 NVS 表示，FP32 checkpoint ≈ 740 MB/场景）压到 **几 MB 到 20 MB**，用的是一条完整的 **generative（rate-in-the-loop）压缩链**：训练时就按字节价格 λ 决定 mesh 长不长（split）/ 砍不砍（collapse），然后 SH 字典共享 + XYZ 网格量化 + 上下文算术编码，encoder 只输出一个文件，decoder 只读这个文件（隔离证明）。目标：CVPR 2027（deadline ≈ 2026-11 中旬）。

**现在的 headline（固定 λ，11 场景 Mip-360 ×9 + T&T ×2，相对自家 FP32 原模型）**：
- λ 1.85e-3：20.7 MB / 32× / −0.22 dB；λ 6.2e-3：9.4 MB / 73× / −0.59 dB；λ 1.85e-2：4.5 MB / 149× / −1.25 dB；λ 6.2e-4：28.2 MB / 22× / −0.11 dB（indoor 重跑后 Mip-360 均值：6.2e-4 = 26.6 MB / 24× / −0.24）。
- NeRF-Synthetic 8 场景（黑底，--indoor，val 选 checkpoint）：FP32 4.48 MB / 30.93 dB；λ 6.2e-3 2.63 MB / 30.93（同 PSNR 59 % 字节）；λ 6.2e-4 5.10 MB / 31.21。
- 和 3DGS 类压缩（HAC++ 等）在等字节下差 ≈ 3 dB：这是 opaque-mesh 的 class gap（MeshSplatting 自身 FP32 就比 3DGS 低 2.4–2.6 dB），不是 codec 的损失（codec ≤ 0.3 dB）。论文定位见 §3。

## 2. 核心 idea 和它是怎么长出来的

1. **（10/09–10/13）起点**：Wenhao 已有 dense（41.4 MB）/ collapse（26.9 MB）两条 Room 链路。Room 字节分解：topology 30.4 bit/v，XYZ 29.9，DC 15.1；mesh 近乎 triangle soup（45 % 边界边、8277 个连通分量）。
2. **（10/13，Yao）方向重置**："differentiable opaque mesh 不需要每顶点独立 SH，表示高度冗余"。于是做了两条 RD 轴：axis A（SH 字典 K×R）、axis B（顶点数 100→50 %），加 precision（XYZ qN / DC q7）和 index 上下文编码，拼出 convex hull；跨场景 codebook 被证伪（信息在 index 分配，不在 codebook 内容）。
3. **（10/16）故事定稿**："opaque differentiable mesh is a texture-mapped mesh in disguise；740 MB 里只有 ~25 MB 信息"，codec 和 29× 是推论。
4. **（10/17–10/18）决定性发现 G0**：训练时限制顶点上限（train-at-budget）比事后 collapse 到同顶点数**高 +1.3 … +4.3 dB**（7/7 场景）→ dense 训练过度生长、事后砍不回来 → "rate 必须在生成环里"。于是写了 rate-controlled growth：split iff ΔD_t > λ·ΔR_t（ΔD 用 `rend_ids` 把像素 L1 残差归到三角形，每 Mpx 归一；ΔR = 3·B_a + 6·B_f ≈ 30 B），phase-B 周期 λ-prune（edge collapse，分数 = ring 残差 / 字节）。λ 在等字节下 ≥ cap 控制（+0.0 … +0.57 dB），同一 λ 在各场景自适应停在 15–100 % 顶点。
5. **（10/19，Wenhao）论文定位 v3**：工程需求 → "第一条完整 MeshSplatting encode/decode pipeline + compressibility 分析 + 选定方法的性质推理和实验"；接受"不新"的批评；PSNR 低于 3DGS 类是表示继承的；lossy RD 压缩，不把 80× / −0.9 dB 当弱点。
6. **（10/24–11/01）全场景 + 六点 hull + indoor 重跑**（所有旧 run 漏了 upstream 的 `--indoor` 预设，重跑后 indoor 原模型与论文一致 ±0.09 dB）。
7. **（10/26）rd-complete-v1 包**：一条命令 train→encode（一个文件）→隔离 decode→render，含 isolation proof（`sys.addaudithook` 审计子进程每个 open/Popen）。这就是 **technical report 里的定稿版本**（`release-fixed-lambda/`）。
8. **（11/03–11/05，Yao 三个 RD 问题 + 23 条 report 评论）**：λ×K 联合 RD 曲线（fan，11 场景：固定 K4096R256/q7 距联合 hull ≤ 0.16 dB，K∝V）；DC q7 是不是高码率天花板（indoor-diag：不是，损失分散在 SH sharing 85 % / DC 0.08 / XYZ 0.08）；不是 end-to-end（B_a 灵敏度：价格误差 = λ 的重标，B_a 8/13 都在 B_a 10 曲线上）→ report-latex-v5。
9. **（11/05–11/06）Wenhao 的新要求：完全自适应的 λ**，不调常数，给目标 MB 直接命中（closed-loop `--target_mb`）。v1→v7 七轮（只在 lego/chair Blender 测试台迭代，每轮 ~25 min/run）：v2 起用**真 encoder 当字节表**（probe，probe/delivered 0.998–1.07）；v3 加生存率 σ + rank 定价；v5 probe 在最终 floor 0.9999 + 4× supersampling 下做 cleaning；v6 用 val_select_v3 修交付；v7 修 phase B 三处（精确记账、门控 + 上限、growth-capped 标记）。**v7 的 batch8 在 smoke 阶段挂了**，一行修复 v7b（batch8-v2）11/06 17:00Z 通过落点标准。
10. **（11/06 晚 → 11/07）v7b 场景 sweep（47 job，19 场景 × 目标 MB）**：交付精确（probe/delivered 0.999–1.001），phase B 落点误差 ≤ 3 %，**但 Mip-360/T&T 全部欠目标**（5/10/20 MB 均值 0.89/0.83/0.77，±10 % 内 8/33）：phase A 的 V* 来自常数先验 11 B/顶点 + 0.6 MB，而 12k 实测 B/V 随模型变大从 11–12.8 降到 7.7–10，且 10000 之后没有加字节的杠杆。Blender 可行目标命中 0.97–1.01，小物体 4 MB 被 limiter 卡住（标 growth-capped）。→ **adapt-v8**：phase A 里也跑同一个 byte probe（soup 上只取属性价 b_v 和 overhead，拓扑先验留到 12k 网格 probe），V* 每个 interval 按实测重算。已交付 node-batch11-v1（见 §4）。

## 3. Yao 的反馈 → 我们的改进（按时间）

| 日期 | Yao 说了什么 | 我们做了什么 | 结果 / 落在哪 |
|---|---|---|---|
| 10/13 | 别再堆方法抠 0.1 dB；核心是"mesh 的 SH 高度冗余"；要 (1) 顶点重排后重新想 face-index 编码 (2) 利用 codebook index 频率偏斜 (3) codebook 大小 RD + 顶点数 RD 轴 | axis A/B/precision hull；index 频率均匀、冗余是空间的 → 上下文编码器（stage2-v2 ctx，无损 −0.55 MB）；Draco/topology 编码封闭（soup 拆 4.5× 顶点） | `02` §A；paper/technical-report-v1.md §2 |
| 10/20 | 为什么要 mesh（PSNR < 3DGS） | plan/why-mesh-2026-10-20.md：PSNR 差 = opaque-mesh class gap；优势 = 标准 mesh 部署（光栅、碰撞、CPU 解码）；可证明的 claim = engine-ready mesh 里的 quality per byte | paper 定位 |
| 10/23 | baselines、topology coding、要 MPEG codec 对比 | 通用 codec 表（PLY+zstd、Draco、Open3DGC/SC3DMC）：同一 asset 上我们比 MPEG 小 3.5×、比 Draco 15–17× | rd/generic-codecs-*、baselines/ |
| 11/03 | 看到 garden 73× 渲染后质疑整个 mesh 方向：why mesh / 3DGS 同大小好得多 / 先试简单物体 / 真贡献是 bounded compact physically-correct mesh | plan/advisor-questions-2026-11-03.md 逐条回答；加 NeRF-Synthetic 单物体章节（8 场景，48× smaller than MobileNeRF at equal PSNR）；"bounded mesh" = 下一篇（MeshSplat+++，严格分开） | `02` §E/§G |
| 11/05 | (1) 要 λ×K 联合 RD 遍历 (2) DC 7 bit 有无熵编码、是否造成高码率天花板 (3) 不是端到端（B_a/B_f 固定、熵编码固定）；不加这些 CVPR 接受存疑；另 23 条 report 评论 | fan（11 场景 λ×{K,R,DC}）、indoor-diag + indoor-xyz（损失分解）、ba-sens（B_a∈{8,13}）；23 条逐条答复（plan/advisor-report-comments-2026-11-05.md）→ report-latex-v5 | analysis/report/latex-v5, rd/lambda-k-fan-v1-results.md, rd/indoor-diag-v1-results.md, rd/ba-sens-v1-analysis.txt |
| 11/05（Wenhao 转述方向） | 要"完全自适应"的 λ：不调参、过程中更新参数、直接命中目标大小 | adaptive λ v1→v7（`release-adapt-v7/`） | 进行中，见 §4 |

## 4. 当前状态（2026-11-07 ≈ 12:00Z）

**固定 λ 线：定稿。**（同 v1：`release-fixed-lambda/`、report-latex-v5。）

**自适应 λ 线：v7b 定版并跑完 19 场景 sweep；v8 已交付、待 Wenhao 提交。**
- **batch8-v2（v7b，lego/chair × 2.5/4，11/06 17:00Z）**：落点通过（hit 1.014 / 0.968 / 0.994 / 0.892-capped，最后一轮 est 误差 ≤ 1.1 %，probe/delivered 1.00–1.04），但门控关着的 4 MB 两行比 v6 低 0.32 / 0.53 dB（v6 的渐进 λ-prune 是稀疏视角场景上的正则化器：chair val-test gap 0.22 vs 0.84）。Wenhao 选"先 sweep v7b"，λ-prune 常开（v7c）留作消融。
- **batch9-v2 + batch10（v7b sweep，47/47 exit 0，11/07 07:17Z）**：`release-adapt-v7/results/adapt-v7b-sweep-analysis.txt`（+ .csv、-table.tex 论文表）。Mip/T&T 33 行：hit 5 MB 0.81–1.02（均 0.888，±10 % 内 5/11）、10 MB 0.75–0.99（0.831，2/11）、20 MB 0.70–0.92（0.767，1/11）；phase B 在 33 个 run 里一刀没砍（门控正确：落点从未高于目标）；growth-capped 标记 0/2/9。等字节 ΔPSNR vs 固定 λ hull 均值 −0.10（5 MB −0.17、10 −0.08、20 −0.04）。Blender 16 行：2.5 MB lego/chair/hotdog 1.01/0.97/0.98（phase B 工作）、drums/materials/ship 1.00/0.97/0.99、ficus/mic 0.77/0.87（capped）；4 MB 0.49–0.99 全 capped。5 个 Blender 2.5 的 smoke 在 1300-it 退化网格的最终 probe 上挂（harness 问题，空 log），SMOKE=0 重交全过。
- **原因（sweep 分析 §3）**：V* = (T − 0.6 MB)/11 B 的先验在 10/20 MB 上偏低 10–30 %（B/V@12k 11–12.8 → 9–11 → 7.7–10：overhead 摊薄 + 属性流随密度变便宜）；phase A 又只到 V* 的 0.83–0.97（20 MB limiter 卡 9–18 个 interval）；第一次 probe 在 12k，densify 10k 已停。v4 的 ×1.15 瞄准曾把这个掩住。
- **v8（11/07 07:50Z 交付，`release-adapt-v8/`，`commands/node-batch11-v1.md`）**：`train_rd_adapt_v8.py`（887 行 = v7b + 60 行，diff 84 行，phase B 一字不动）：`iteration > start_pruning`（3000，第一次 prune 砍掉初始 soup 的 70–95 %）之后的 densify interval 上每隔 `--probe_a_every`（1000）跑一次 byte probe（Blender 3500…9500 共 7 次 ≈15 s，Mip 6–7 次 ≈20–45 s），`SizeModel.update_attributes` 只取属性价 b_v 和 overhead；拓扑价（b_f × faces/vertex，占 35–45 % 字节）保持先验到 12k 网格 probe（soup 的连接流和交付的 Delaunay 网格无关），剩余 V* 先验误差 ≈ ±10 %；kept share 记录不用（12k 0.8–1.0 → 结束 0.98）；kept 顶点 < 4096 时 probe 跳过（encoder 的 K-means 要 ≥ K 个样本）。`--probe_a_every 0` = v7b。sbatch v8b-r1 / v8-r1：smoke 延长到 1600 it（1500 强制 phase-A probe，1550 强制 prune，断言 probe_a 事件），Mip 的 phase B 改 16 轮（12000 起每 500 到 19500）。CPU 审查：脚本精确替换生成、py_compile/pyflakes、SizeModel + 时点规则单元测试、smoke 断言块和结果行块在假数据上跑通、heredoc 逐字节核对。
- **batch11（v8，11/07 09:00Z，6/6 exit 0，`release-adapt-v8/results/adapt-v8-batch11-2026-11-07-paste.md`）**：(a)(b)(e) 过——soup 上量的 b_v 和 12k 网格 probe 的 b_v 相差 0.8–8 %，overhead < 1 %，probe 共 2–3 min；(c)(d) 不过——6 行全比 v7b 低（lego 2.5 0.916、chair 2.5 0.879、lego 4 0.846、chair 4 0.797、bicycle 10 0.710、room 10 0.795）。
- **根因（v4 起就有，从 densify 行里看出来）**：(1) 计划把 densify interval 多数了一个（`remaining = (until − it)//interval + 1` 在 9500 给 2，upstream 只在 `iteration < densify_until` 时 densify，1500…9500 共 17 个，计划按 18 个算）→ 几何计划最后只长到 sqrt(V·V*)，6 个 run 最后一次 prune 后的模型 = 0.81–0.92 V*（CPU 重放旧计划复现到 0.3–5 %）；sweep 分析里"phase A 只到 V* 的 0.83–0.97"就是它，v7b 的常数先验在 Blender 上把 V* 抬高 30 % 恰好盖住。(2) plan_splits 的 limiter 模拟把 upstream 的允许量（add_pct·V − V 个**三角形**）除以 3 当顶点数 → 以为 limiter 紧 3 倍，4 MB 行的"limiter-bound 12–13/17"大半是假象。
- **v9（11/07 09:45Z 交付，`release-adapt-v9/`）** = v8 + `remaining = ceil((until − it)/interval)` + limiter 用三角形数 + densify 行打印 `cap N limiter M`（事件带 `limiter_max`）；probe 和 phase B 一字不动（`train_rd_adapt_v9.py` 910 行，代码 diff 4 行，md5 a120351f…）。CPU：intervals 公式对 5 种 schedule 穷举、6 条 batch11 轨迹重放（旧计划复现缺口，v9 全到 V*）、smoke 断言正反。
- **batch12（v9，11/07 11:10Z，6/6 exit 0，`release-adapt-v9/results/adapt-v9-batch12-2026-11-07-paste.md`）**：phase A 到位（最后一次 prune 后 0.96–0.99 V*，chair 4 真 limiter 卡住）。落点 lego 2.5 0.991、chair 2.5 0.965、chair 4 0.947（不再 capped）、bicycle 10 0.790、room 10 0.877、room 20 0.829（v7b 1.014 / 0.968 / 0.892 / 0.775 / 0.814 / 0.75）。phase B 在 Blender 上砍了（lego 4321 对、chair 7745 对），Mip 上门控关着（落点从未高于目标）。PSNR vs chord：lego −0.22（v7b −0.13）、chair +0.02、chair 4 −0.13（v7b −0.20）；room 10 比 v7b +0.1 dB（+7.7 % 字节）、bicycle 持平。
- **Mip 剩余差的完全分解**：最终 cleaning 在 Mip 上只保留 92–94 % 顶点（Blender 99.5 %）× 交付网格的拓扑价 b_f × faces/vertex = 3.7–4.4 B/V（Blender 6.1，先验 5）。hit = κ × (b_v + 拓扑)/(b_v + 5) = room 10 0.86 / bicycle 0.80 / room 20 0.81（实测 0.877 / 0.790 / 0.829）。两者都是数据集类别属性，soup 上量不到，要等 11k 的 Delaunay 网格 → v9 = "soup 上能量的都量了"的版本。小瑕疵：计划在 9500 完成后 rank_lambda 返回 lam_max，没砍的行报 `lam_final (shadow) 1e-1`（phase B 第一轮 capped 就重置为 shadow price，行为不变，只是没砍的行报价退化）。
- **待 Wenhao 选（11/07 11:40Z 卡片）**：A（推荐）= 今晚跑 `node-batch13-v1`（43 job：Mip/T&T 30 + Blender 13，batch12 的 6 行留用，三个文件不改、md5 核对，≈34 GPU-h），同时 CPU 上设计 v10；B = 先做 v10 再 sweep；C = 停在 v9。v10 的想法：11k Delaunay + 12k 网格 probe 之后再开一个增长窗口（量到的 κ / b_f / fpv 直接用），再 Delaunay 一次，phase B 砍 κ 的漂移——方法改动，半天设计 + 2.5 h 测试。
- HPC 上现在：v7b 的 47 个 run 目录、v8 的 6 个（`$S-adapt8b-t{T}mb(-best)`、`$S-adapt8-t{T}mb`）、v9 的 6 个（`$S-adapt9b-t{T}mb(-best)`、`$S-adapt9-t{T}mb`）、表 adapt-v7*.txt / adapt-v8*.txt / adapt-v9*.txt；reservation 到 11/12 09:35。

## 5. 下一步（优先级）

1. **按卡片的选择走**：A → batch13 结果（`adapt-v9.txt` 33 行 + `adapt-v9-blender.txt` 16 行）→ `adapt_analysis_v7b_sweep.py` 改读 v9 的表 → 19 场景命中率 / 等字节 ΔPSNR / LaTeX 表，和 v7b 的表并排；论文 claim 以 v9 为准（phase A 精确到 V*，Blender 0.95–0.99，Mip 10/20 MB 0.79–0.88，差额解释为两项数据集先验）。B → 先设计 v10（第二增长窗口），CPU 审查 + lego/chair + room/bicycle 测试后再 sweep。任何版本都不要再碰 v7b/v8/v9 的文件（新版本新文件名）。之后 λ-prune 常开消融（4 job）。
2. 论文：qualitative 图、Table 1 其它 9 场景、统一 LPIPS pass（[0,1] 3DGS 约定）、related work 成文、limitation（soup 碎片、class gap）。Figure 风格规则见 memory `feedback-figure-style`。
3. 可选（Yao 的 E2E 关切）：DC 预测编码（bitstream 改动需 Wenhao 同意，stage2-v3）。
4. **不要做**：MeshSplat+++（下一篇）的任何东西混进 CVPR；mesh 正则化（train_rd_v4 一次性失败，归 +++）；decoder-side Delaunay（死）；高码率 Blender 点（被 dominated）。

## 6. 和 Wenhao 一起工作的规则（必须遵守，详见 memory/user-wenhao-working-style.md 等）

- 中文回复 + 英文术语；一次只给一个明确动作；先给最小 run 列表和 GPU-h。
- **GPU-WASTE 规则（11/06，他很生气）**：任何 GPU job 前在 CPU 上审完整条链（training wrapper → selection → delivery → row script），用手头 log 回放新逻辑，一次交付一个审过的版本；启动自检在长步骤之前 fail；先说通过标准。不要"跑了看"。
- 改他的代码 / push / 开 PR 必须先问；新工具一律放 `/mnt/project-files/analysis/` 新文件，import 他的模块，不改 upstream、不改旧版本。
- **版本化**：改过的文件/包永远用新名字（-vN，内含 VERSION），已经在 HPC 上的文件永远不覆盖；提交脚本按行数 + md5 核对。
- HPC 脚本一律 heredoc 写文件 + 提交，一次粘贴；每个块重定义所有变量；脚本 fail loudly；**粘贴块里绝不能有 `set -e` 或裸 `exit 1`**（会杀他的 VS Code 终端）；逻辑放进 heredoc 写的脚本用 `bash` 跑。
- 他只有一个节点时所有事都在那跑（CPU 重活也在 GPU 节点）；`--mem=32G` 默认；account `torch_pr_38_general`；永远不写 `ssh greene`（集群是 NYU Torch）；`sacct -u $USER`；`XYZ` 永远显式 qN（`XYZ=auto` 坏的）。
- 研究方法：Room 只是调试场景；策略必须通用，不针对视角/场景调参，不用数据集相关常数。
- 定稿线（CVPR）与 MeshSplat+++ 严格分开。
- 文件通过 `/mnt/project-files` 路径链接交付；plan（`plan/cvpr2027-research-plan.md`）和 memory 持续更新（memory 文件 < 4 KB）。

## 7. 关键数字速查

- Room dense FP32 ≈ 742 MB（含训练统计 819 MB）；dense codec 41.4 MB / 27.93；collapse 26.9 MB / 27.35；v0.8 K4096R256 q14/q7 25.5 MB / 27.88；train-at-budget 1.35 M v 13.56 MB / 27.98；λ 6.2e-3 + prune 5.23 MB / 27.38（142×）。
- 价格常数：B_a = 6 B/vertex 属性（代码里 BPV=10 = B_a + 2 B_f），B_f = 2 B/face；split ΔR = 30 B，collapse ΔR = 10 B；44 个 asset 实测 9.5–13.3 B/v（中位 10.8）。
- XYZ 精度规则 b = ceil(log2(extent / (0.1·median edge)))：room q14、kitchen q16、counter q15、bonsai q15、garden q15、bicycle/stump/flowers/treehill q16、truck/train q18、Blender q14；DC q7 通用。
- 固定 λ 网格：6.2e-4 / 1.85e-3 / 6.2e-3 / 1.85e-2（v2 mpx 单位）。
- Upstream 事实：densify 每 500 到 10000（add_percentage 1.23/1.27 限幅）；restricted Delaunay 11000；opacity floor 0.1→0.9999 从 5000 到 24000；supersampling 2×@20000、4×@25000；最终 cleaning 30000 只留 max_blending > 0.5 的面；`--save_iterations` 无效；`update_indoor(ops)` 覆盖 CLI 的 add_percentage/max_points；Blender 分支不加载 normal/depth prior。

## 8. 包内目录

```
00-README-HANDOVER.md  01-EVOLUTION.md  02-EXPERIMENTS-INDEX.md  03-HPC.md  04-CODE-MAP.md  VERSION
release-fixed-lambda/   README.md + meshsplatting-rd-complete-v1.zip（一键链路，technical report 版本）+ protocol-paste-blocks/ + report-latex-v5.zip + report-v5.pdf + paper/
release-adapt-v7/       README.md + code/（v7, v7b, probe, prune, growth, ckpt, val_select, coarse_finetune）+ sbatch/ + commands/（node-batch8-v1/v2、9-v1/v2、10-v1 + 生成器）+ results/（batch8-v2 行 + 分析、sweep 原始粘贴 / 三张表 / 分析 / csv / LaTeX）+ history/（v1–v6）
release-adapt-v8/       README.md + code/rdgrowth/train_rd_adapt_v8.py + sbatch/ + commands/node-batch11-v1.md + build/ + results/adapt-v8-batch11-2026-11-07-paste.md（batch11 全部输出）
release-adapt-v9/       README.md + code/rdgrowth/train_rd_adapt_v9.py（现行）+ sbatch/（adapt-v9.sbatch, adapt-v9-blender.sbatch）+ commands/（node-batch12-v1.md 测试块、node-batch13-v1.md 43-job sweep 块）+ build/（make_v9.py, make_sbatch_v9.py, make_batch12.py, test_v9_cpu.py）+ results/adapt-v9-batch12-2026-11-07-paste.md
analysis-snapshot/      /mnt/project-files/analysis 完整快照（去 __pycache__）：rd/ 全部结果表与图、commands/ 全部 HPC 粘贴块、rdgrowth/、blender/、diag/、report/、baselines/、vis_collapse/、viewer-v1..3
plan/                   /mnt/project-files/plan 全部计划/决策文档（主线：cvpr2027-research-plan.md 执行记录，最后两条 = batch11 判定 + v9 / batch12 判定 + 选项）
paper/                  technical-report-v1.md, related-work-v1.md, baselines-v2.md
memory/                 共享记忆 49 个文件 + MEMORY.md 索引（每个 < 4 KB；新 thread 先读 MEMORY.md；adaptive-lambda-nov2026 = 自适应线现状）
uploads/                Wenhao 上传过的原始文件（见 uploads/INDEX.md：upstream 源码 zip、stage2-v2 源码 zip、Room 证据包、torch_connect.sh、adapt 粘贴、截图）
slides/                 deck-v2（manim 脚本 + 视频 2.5 MB）+ deck-v1 的脚本与 notes（22 MB 视频未收录）
legacy-packages/        meshsplatting-dense-complete-v1.zip, meshsplatting-collapse-complete-v1.zip（10/09 起点的两条链路）
```
