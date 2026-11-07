# 架构/创新层面的设计空间：还有哪些"巧妙设计"能解决现在的问题 (2026-10-26)

问题（用户）：不要小提升，要像"把码率放进生长环"那样、用一个设计解决一个现存问题的方案。
下面每条都按 **现存问题的证据 → 设计 → 期望收益 → 一次实验怎么判定 → 成本** 写。三条主线合起来是同一个论点的延伸：
**"rate-controlled generation"：凡是 decoder 能自己再生的，都不该传；凡是传的字节，都该买到泛化。**

## 主线 A：拓扑由 decoder 再生（connectivity-free delivery）

* 证据：Room dense 的字节构成 = topology 30.4 bits/v、XYZ 29.9 (q18 时)、DC 15.1、SH 索引 ~20；在最终 q14/q7 格式里 topology 仍是最大的一条流（≈ 40 %）。它贵是因为交付的 mesh 是 soup：45 % 边界边、18 % 非流形、8277 个连通分量，edge-cache AC 只能做到 15 bits/face；Draco 更差（把顶点撕成 4.5 倍）。10/16 测过"从点预测连接"：最终 mesh 只有 1.6 % 的面是最终顶点集的 Delaunay 面 → 事后预测不可能（C4 DEAD）。
* 为什么事后不行但"在环里"可以：上游在 11000 步做一次 restricted Delaunay 重建，之后拓扑固定、顶点继续移动 19000 步，所以最终拓扑和最终顶点早已脱钩。设计：**在训练末尾、在量化后的顶点上再做一次 RDel 重建，然后冻结 XYZ、只训练 appearance 若干步**。交付时只传 (量化顶点, 每个 Delaunay 四面体/面 的占用标签)；decoder 对收到的顶点重跑同一个 Delaunay（确定性），再用标签选出表面。四面体数 ≈ 6–7 V，占用标签带上下文编码预计 1–3 bits/v，对比现在 30 bits/v。
* 期望收益：同一 PSNR 下字节 −30..−35 %（相当于再往 λ 曲线右下走一档），而且是对每个 λ 点都成立的平移；decode 多一次 Delaunay（50 万点级别为秒级）。
* 风险：末尾重建改变拓扑 → PSNR 损失；量化网格会扰动 Delaunay（所以必须在量化后的点上重建并冻结 XYZ）；RDel 的"restricted"判据是否只依赖点集 + 标签，要读 effrdel 确认。
* 一次实验判定（Room，λ6.2e-3）：train_rd_v4 = v3 + 在 26000 步把 XYZ 量化到 q14、重跑 RDel、冻结 XYZ、继续训练到 30000。看三件事：test PSNR 相对 v3 的差；交付 mesh 的面有多少 % 属于 Delaunay(最终点)（目标 100 %）；标签熵估计 (bits/v)。成本 ~0.6 GPU-h（一个 λ）。若 PSNR 损失 ≤ 0.1 dB → 这是本文之后最大的一张牌，可以进 CVPR 的分析节做 "upper bound" 或进下一版正文。

## 主线 B：字节只买泛化（generalization-aware pricing）

* 证据：Room dense train/test gap 2.45 dB，440k 顶点时 1.70 dB → dense 一半的容量在记忆训练视角；flowers / treehill / truck / train 上 λ1.85e-3 多花的字节让 train PSNR +0.1..+0.6 而 test ±0.1 (R9 表)。碎片感（上一条消息）正是单视角拟合的几何表现。
* 设计：现在 split/prune 的价值信号 ΔD 是各视角残差质量的和。改为 **多视角一致的残差**：每个三角形同时累计 (残差和, 贡献视角数, 残差在视角间的方差)；只有被 ≥ m 个视角、以一致方向支持的残差才计入 ΔD（或 ΔD × 一致性权重）。单视角残差 = 记忆 = 不值得买字节。不需要额外渲染，accumulator 多两个数组。
* 期望收益：同字节下 test PSNR 上升、train PSNR 下降（gap 收窄）；低覆盖场景上 RD 曲线不再出现"多花字节 test 不涨"；碎片（单视角薄片）减少是副产品。它把 R9 从"观察"变成"方法"。
* 风险：训练视角稀疏时一致性门槛会压制合法细节；m 和权重形式要在两个场景上定一次、不许按场景调（10/13 规则）。
* 一次实验判定：Room + flowers，λ6.2e-3，各一跑（可加 λ1.85e-3 共 4 跑）。判据：等字节 test PSNR ≥ v3 + 0.1 dB 且 train−test gap 缩小；flowers 上 λ1.85e-3 的 test 增益从 −0.1 变为正。成本 1–2 GPU-h。

## 主线 C：一条码流服务所有码率（embedded / scalable bitstream）

* 证据：现在每个 λ 一个独立资产；手机用 73× 点、桌面用 32× 点要传两份；而生长过程本身就是 progressive mesh 的构造序列。
* 设计：先训 base (λ_hi)，然后 **在 base 上继续以 λ_lo 生长，base 顶点的位置和码字冻结（或只允许小范围微调并作为 refinement 传）**；enhancement layer = 新顶点 + 它们的面 + 码字（新面的连接只涉及"新顶点 ↔ 旧顶点"，可用旧顶点索引做上下文）。decoder 读到任何一层停止即得到一个可渲染资产，base 可先传先渲染。
* 期望收益：一个文件覆盖整条 RD 曲线；流式传输；论文多一个 reviewer 熟悉的系统贡献（progressive meshes × rate control）。
* 风险："可扩展性代价"：冻结 base 后 λ_lo 层的质量低于独立训练的 λ_lo 资产（视频 SVC 通常 5–15 % 码率代价）；base 不动会让 enhancement 需要更多顶点。
* 一次实验判定：Room：base = 现有 λ6.2e-3 的 checkpoint，继续 λ1.85e-3 生长 10k 步并冻结 base → 和独立 λ1.85e-3 比 (12.57 MB / 28.00)。判据：总字节 ≤ 1.15 × 独立资产且 PSNR 差 ≤ 0.1 dB。成本 ~0.5 GPU-h。

## 次级（真正的收益但属于 codec 工程，不是"设计"）

| 想法 | 解决什么 | 收益上限 | 成本 |
|---|---|---|---|
| E. 用真实上下文模型的条件概率给顶点定价（index ctx + XYZ 残差），替代常数 10 B/v + 2 B/face | 现在 rate 价是常数，生长不会偏好"可预测"的顶点；index 流 20 bits/v 接近 raw | index 流 −30..−40 %，整体 −10..−15 % | 中高（要在环里评估 ctx 概率） |
| F. DC 用邻居预测 + 残差编码（现在 DC 不预测） | DC 流 ~15–21 bits/v | −5 bits/v ≈ 整体 −7 % | 低（codec 改动，需你同意改 stage2） |
| D. prune 用 QEM/最优位置代替中点 collapse | prune 被 foldover 拒绝的比接受的多 | 字节 −3..−8 % | 中 |

## 建议

截稿前 3 周、FairShare 受限：
1. **B 先做**（1–2 GPU-h，代码改动最小，直接回应 R9 和碎片问题，若成功就是正文里的第二个"环内"设计）。
2. **A 做一次 probe**（0.6 GPU-h）：只要 PSNR 损失 ≤ 0.1 dB，就把它作为本文分析节的 "topology is regenerable: upper bound" 一段 + 下一篇的主线；若损失大，写成 negative result 一句话。
3. **C** 放到 future work，除非 A、B 都在两天内有结论。
E/F 是 codec 迭代，等论文定稿后再做；D 不做。

与 10/25 readiness 顺序的关系：qualitative figure 与 related work 仍然排在这些实验之前；A/B 的 Room 跑可以和写作并行提交。
