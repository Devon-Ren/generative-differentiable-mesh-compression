# 从"顶点上限"到 CVPR 级方法：rate-controlled mesh growth 的设计（2026-10-17）

## 0. 老实回答"现在是怎么做到的"

现在就是机械的：`train.py --max_points N`。MeshSplatting 的 `add_new_gs` 每 500 步把顶点数长到 `min(cap, 1.27 × 当前)`，默认 cap 4M 永远撞不到，所以顶点数由增长率 × 18 轮决定（3.2M）；设 cap = 1.6M / 1.0M 就撞顶。primitive 数量是**输入**，不是网络自己决定的。它证明的只是一件事：dense 训练长过头 2–3 倍，而且事后删不回来（同顶点数差 1–4 dB）。这是一个发现，不是方法。

## 1. CVPR 版本要满足的三件事

1. primitive 数量是**输出**：由一个 rate–distortion 目标决定，不是手设的 N。
2. 数量是**场景自适应**的：同一个超参在 Room 和 bonsai 上停在不同的顶点数。现在用 cap 时 Room 在 1.35M 只掉 0.1 dB，garden/bonsai 掉 0.5–0.7 dB，说明 1.6M 对 garden/bonsai 太紧、对 Room 太松——cap 需要逐场景调，λ 不需要。
3. 有一个 mesh 专属的理由，说明为什么 3DGS 的 Lagrangian 剪枝（RDO-Gaussian ECCV'24、Mini-Splatting、Taming-3DGS 的 budget）不能直接搬过来。我们的理由就是对照实验本身：Gaussian 可以事后剪掉 60 % 几乎不掉点（LightGaussian），而 mesh 事后 collapse 同顶点数掉 1–4 dB，因为顶点被拓扑耦合，不能独立删。**所以 mesh 必须在生长时控制**。

## 2. 方法：ΔD > λ·ΔR 的生长与剪枝

### 2.1 生长（split）
- 现状：按 importance_score 采样三角形 split + 强制 split 最大的 500 个。
- 改为：对每个三角形在一个 densification interval 内累积 ΔD_t（顶点位置梯度范数 × 屏幕面积，和 3DGS 的 densify_grad 同源，但按三角形聚合），只有 ΔD_t > λ·ΔR_t 才 split。
- ΔR_t = 这次 split 新增的字节：新增顶点 × (topology + XYZ + DC + index 位) + 新增面 × topology 位。用我们 codec 的实测值：约 10 B/顶点（topology ≈ 4 B 不可压，XYZ 2–2.6 B 由精度规则定，DC ≈ 1.5 B，index ≈ 2.4 B）。
- 诚实的说明：如果 ΔR 取常数，这条规则退化成"带码率单位的梯度阈值"。它不退化的地方在 2.2 和 2.3。

### 2.2 剪枝（prune），这是方法的主体
- 把 C2 的 render-sensitivity 判据（sens_pos，已验证事后有效）搬进训练：每个 interval 对每个顶点算 ΔD_v（删掉它的几何敏感度）和 ΔR_v（删掉它省的字节），ΔD_v < λ·ΔR_v 就 collapse。
- ΔR_v 不是常数：删一个顶点省掉的面数从 1 到 6 不等（soup/边界顶点省得少），这正是我们 10/13 发现的"同面数 ≠ 同字节"。所以 λ 剪枝会优先删"贵而没用"的顶点，这是 cap 和梯度阈值都做不到的。
- 周期性做，而不是训完做一次：顶点被删后的区域还有训练步数去修复，这是"训练时预算 > 事后 collapse"那 1–4 dB 的来源。

### 2.3 和分配器的耦合
- λ 同时是精度和码本的价格：XYZ 精度由几何规则（extent / 边长）给初值，再按 λ 校正 1 bit；K/R 按 backbone 斜率选（我们已证明 7 场景同序）。
- 给定目标 MB，分配器输出 λ（用实测斜率反解），训练一次就落在目标附近；不再扫 hull 表。

### 2.4 可证伪的预测（写进论文）
- 同一个 λ 在 7 个场景上停在不同顶点数，但每个场景运行点的局部斜率 dD/dR 都 ≈ λ（这正是我们 greedy allocator 验证过的"等斜率"条件，只是从事后搬到了训练时）。
- λ-prune 在同顶点数下优于 cap-only（现有结果）≥ 0.2 dB，差距在低码率端更大。
- 去掉周期剪枝、只在训练末剪一次，就退回事后 collapse 的水平。

## 3. 实现路径（不改他的仓库：analysis/rdgrowth/ 新文件，import upstream，子类覆盖 add_new_gs / prune）

- 第 1 周：`train_rd.py` = upstream train.py 的薄包装，覆盖 densification 为 λ-split，加 λ-prune（复用 native_qem_vis 的敏感度打分，每 interval 跑一次），ΔR 查表（按场景精度）。Room 上 λ 扫 4 个值，和 cap 曲线、collapse 曲线画在一张图上。
- 第 2 周：7 场景 × 3 λ，对照（cap-only、prune-only、split-only、训末单次剪枝），FPS/三角形数。
- 第 3 周起：写作。分析章节（精度规则、backbone、拓扑地板、"为什么只控制几何"）已经齐了。

## 4. 风险
- ΔD 的估计噪声大：3DGS 的梯度阈值出了名的不稳。缓解：按 interval 累积、按三角形聚合、用 sens_pos 这种渲染敏感度而不是裸梯度。
- λ 在 split 端可能几乎不起作用（ΔR 近常数），那方法内容就全在 prune 端 + 分配器，要提前接受这个结果并如实写。
- 时间：4 周内只能做到 2.1–2.3 的最简版本；流形正则（候选 3）仍是下一篇。
