# "生成式压缩" 建议的评估与决定性实验（2026-10-17）

来源：PhD 同学意见——只做 RD performance 分析不够 CVPR；应把码率作为控制生成（训练/densification）的一部分，这样 primitive 数量会大幅减少。

## 1. 把他的话翻译成我们的数据语言

"码率进生成环"= rate-constrained training / compression-aware training。3DGS 压缩这两年的主流正是这条线（HAC、HAC++、ContextGS、CompGS 把熵模型放进训练；Reduced-3DGS、Mini-Splatting、Taming-3DGS 把 primitive 预算放进 densification）。审稿人对"训完再压"的 codec 论文的常见评语是 engineering / incremental。这一点和 10/13 的判断一致：hull 本身不是 CVPR，缺一个 method framing。

但"码率进环"有两处可以放，我们的数据已经把其中一处判死：

| 放进训练环的量 | 我们已有的证据 | 结论 |
|---|---|---|
| 外观（SH 码本 / 量化） | 码本内容几乎无关（随机采样 ≈ k-means，0.05 dB）；DC 的 STE QAT 比事后量化差 0.1 dB；外观 98.7 % 冗余事后几乎零代价去掉 | 进环没有收益，不碰 |
| 几何（顶点数 / 面数） | 顶点数由 densification 预算决定（三个场景都 ~3M，与场景无关）；事后 collapse 到 50 % 掉 1.2 dB；C2 证明"删谁"决定质量（同面数 +0.26 dB）；collapse 后 finetune 平台期说明损失在被删的东西里 | 这是唯一可能"大幅减少 primitive"的地方，也正是 PhD 的直觉 |

所以他的建议对一半：方向对（CVPR 要 method），机制只对几何流成立。而几何流恰好是我们字节的大头（topology + XYZ ≈ 60 %）和唯一的硬地板（topology ~4 B/顶点）。

## 2. 决定性实验 G0："训练时预算" vs "事后 collapse 到同预算"（零代码改动）

问题：同样 1.6 M 顶点，是训练一开始就只长到 1.6 M 好，还是先长到 3.2 M 再 collapse 到 1.6 M 好？这是 PhD 假设最便宜的可证伪形式。

设计（Room，复用 run_garden.sh 的流程）：
- 用 upstream train.py 的 densification 预算参数把原始训练的顶点上限设为 50 % 和 30 %（≈1.6 M、1.0 M 顶点），30k 步，--eval。
- 不做 collapse，直接 footprint → 3000 步 finetune（可省）→ deliver（K4096 R256 q14/q7，stage2-v2）。
- 对照：事后 collapse v0.5（已有：16.9 MB / 27.090 test，v2 格式）和 v0.3（补一个点，同一流程）。
- 按 bytes 和按顶点数各对齐一次。

判定规则（提前定）：
- 训练时预算在 50 % 处比事后 collapse 高 ≥ 0.3 dB，且 30 % 处仍在原版 −0.6 dB 以内 → 假设成立，值得把 λ-控制的 densification 做成方法（§3）。
- 两者相差 < 0.1 dB → 假设在这个表示上证伪；这本身是一个支持主线的结果（"事后 collapse 已经等于在环内控制"），写进分析章节。
- 训练时预算更差 → 说明 dense 训练是优化需要的脚手架，collapse 是必要的；同样写进去。

成本：2 次训练 + 2 次交付 ≈ 3–4 h L40S，一个 sbatch，不改任何代码。需要的唯一信息：upstream train.py 控制 densification 上限的参数名。

## 3. 若 G0 为正：方法 = λ-controlled mesh growth

- densification 判据改成 RD 形式：只有 ΔD > λ·ΔR 才 split/加点，ΔR 用实测的每顶点字节（~10 B，其中 topology 4 B 不可压）。
- 周期性 prune 用 C2 的 render-sensitivity 判据（native_qem_vis 已有），每 N 步一次，而不是训完做一次。
- λ 扫出 RD 曲线：分配器从"查 hull 表"变成"一个 λ"。主线的 claim 4 直接升级。
- 工程量：改 upstream 的 densification + 周期 collapse 的调用，1–2 周；需要先征得同意再改代码。

## 4. 风险

- 表示本身：MeshSplatting 的网格是 near-soup（45 % 边界边、8277 个连通块）。"少量大三角形贴墙"这种 5–10× 的减量需要一个 surface prior，而这正是候选 3（流形正则）的内容，不是本次 deadline 能做的。G0 的 30 % 点就是在测这个上限。
- 时间：假设 deadline 11 月中，G0 本周出结果，λ-方法最多 2 周，剩 1 周写。四个场景的 hull 扫描照常在队列里跑，不冲突。
- 共识冲突：10/13 定的"不再堆方法"针对的是 0.1 dB 的小 trick；λ-growth 是一个 method contribution，不是 trick，且由 G0 的数据决定做不做。

## 5. 论文结构（把老板和 PhD 的意见合在一起）

1. 分析（老板的论题）：opaque differentiable mesh 高度冗余；三条流的冗余结构不同——外观事后零代价去掉，拓扑是地板，几何是唯一值得在训练时控制的流。
2. 方法（PhD 的论题）：因此把码率只放进几何的生成环：λ-controlled growth + 周期 sensitivity prune；外观用码本事后编码。
3. 结果：hull、分配器、7 场景、baseline 表。

G0 的结果决定第 2 节是"方法"还是"负结果 + 事后 codec"。
