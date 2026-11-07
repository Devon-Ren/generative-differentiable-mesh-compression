# 大场景前还要不要做"抬 PSNR"的实验？（2026-10-19）

问题：能否靠微妙策略 / 小设计把 MeshSplatting 的 PSNR"大幅"抬到与 3DGS 可比（不输太多）？差距是表示本身的，还是错配？

## 0. 结论

1. **差距的主体是 opaque-mesh 这一类表示的共性，不是 MeshSplatting 的 bug 或某个可一键修复的错配。** 文献里所有 opaque-mesh NVS 方法都比 3DGS 低 2.5–3.5 dB（BakedSDF 室内 27.06、BOG 27.71、MobileNeRF 室外 23.5 vs 3DGS 室内 ≈30.5 / 室外 ≈24.6），只有改成体积式（Radiance Meshes，四面体体渲染，室内 30.61）才追上。我们测到的 7 场景均值 25.81 vs 3DGS ≈28.2，差 2.4 dB，与作者论文的 24.78 vs 27.21（差 2.4）一致：我们在复现论文，没有多丢。
2. **"可比"（差 ≤ 1 dB）靠小改做不到；0.3–1 dB 的局部收益可能存在，但要先花 10 分钟 GPU 定位，不要猜。** 已写好诊断 `analysis/diag/error_anatomy_v1.py`（粘贴块 `analysis/commands/error-anatomy-v1.md`），把 test 误差拆成五个像素桶并给每桶"清零能换回多少 dB"的上界。
3. **不建议在大场景批次前做任何方法堆叠。** 只有一种小改（环境贴图）既便宜又不动码流，且仅当诊断显示 background 桶上界 ≥ 0.5 dB 时才值得在 garden 跑一次（≈1 h）。其他路线要么是另一篇论文，要么会重做整个码流。

## 1. 代码里看到的结构事实（dense-complete-v1 包）

| 事实 | 位置 | 含义 |
|---|---|---|
| 颜色 = 每顶点 SH 在**顶点处**求值（视线 = 相机→顶点），RGB 按**屏幕空间 2D 重心**线性插值 | `forward.cu` computeVertexColorsCUDA + render loop 580–618 | Gouraud 着色：外观分辨率 = 顶点密度；非透视校正（小三角形下可忽略） |
| alpha = min_vertex_weight · phi^sigma；sigma 1.0→1e-4（30k），opacity floor→0.9999（24k）；T<1e-4 早停 | `train.py` sigma/opacity schedule；`forward.cu` 556–566 | 训练前期是软光栅，最后完全不透明；几何梯度在后期几乎只剩重心插值项 |
| **没有背景 / 天空模型**：C + T·bg，bg 固定黑 | `forward.cu` 647 | alpha<1 的像素直接显示黑色；室外天空/远景只能靠 mesh 盖住 |
| loss = 0.8 L1 + 0.2 DSSIM（与 3DGS 同）+ vertex-depth + normal 正则（12k 后） | `train.py` | 光度目标没有差异 |
| 4× 超采样渲染（20k ×2，25k ×4）再 area 下采样 | `train.py` / `triangle_renderer` | 抗锯齿已做；也是 FPS 的代价 |
| train[::8] 验证 30.87 vs test 28.37（Room λ=0，2.9M 顶点）；226k 顶点时 27.9 vs 26.5 | `rd/rd-sweep-room-2026-10-18.txt` | 2.5 dB 泛化差，且随顶点数增大：dense 模型在**记忆训练视角** |

## 2. 五个假设、诊断读数、若成立的小改、成本

| # | 假设 | 诊断里看哪一项 | 若成立的小改 | 预期 / 成本 |
|---|---|---|---|---|
| H1 | 外观分辨率被顶点密度绑死（纹理区大三角形） | footprint 表：大三角形 MSE 显著更高；`interior_hf` 上界 | 只加"颜色顶点"：每面 1→4 中点细分（几何不变），冻几何微调颜色 | **已有反证**：3.19M→1.35M 顶点只掉 0.1 dB，顶点密度不是当前瓶颈；上界估计 <0.3 dB。验证 Room 微调 ≈0.5 h，仅在诊断支持时做 |
| H2 | 泛化 / 曝光：Mip-360 曝光变化 + 每顶点 SH 过拟合 | `train/test gap`、`affine UB` | per-image 曝光仿射或 appearance embedding | 测试时需要估计每张图的参数 → 与 3DGS 不可比；**只报告，不做**。如 affine UB ≥1 dB，写进 limitation（"差距有一部分是曝光"） |
| H3 | 背景/天空没被 mesh 覆盖（alpha<0.99 显示黑） | `background` 桶像素份额与上界 | 可学习环境贴图：C + T·env(dir)，外挂在 render 之后，几 KB 字节 | 室内 ≈0；室外（garden/bicycle/stump/flowers/treehill）可能 0.3–1 dB。garden 一次训练 ≈1 h。**唯一建议考虑的小改** |
| H4 | 轮廓 / 缝：soft→hard 的几何错位，near-soup 的颜色不连续 | `silhouette` + `seam` 桶 SSE 份额 | 更长软光栅期、更晚 opacity floor（训练更慢）；缝靠共享顶点（Delaunay 后已是） | 这是表示本身；份额高只能写成 limitation |
| H5 | 低频 / 大块错（拓扑错、遮挡错） | `lowpass 2/4` 仍远低于 3DGS | 无小改 | 表示 + 优化问题，另一篇论文 |

## 3. 架构层面能不能"大幅"追上

- **(a) 体积 / 半透明化**（Radiance Meshes 路线）：能追上，但卖点"标准不透明 mesh、引擎直接加载"没了，整个 codec 也要重做。另一篇论文。
- **(b) 外观与顶点解耦**（per-face texel / neural texture）：1–2 周工程，收益不确定（H1 反证），码流从"顶点属性"变成"纹理"，主表全部重跑。不在这篇做；可写 future work："appearance resolution as a third RD axis"。
- **(c) 环境贴图**：小、安全、字节可忽略；条件是诊断支持。

## 4. 建议顺序与成本

1. 跑诊断（Room 原始 + λ=6.2e-3 两模型，≈10 min），贴回两张表。
2. 若 `background` 上界 ≥0.5 dB：在 garden 加 env-map 训练一次（≈1 h），对比 original；成立则作为所有室外场景的默认，纳入批次 A/B（不另加成本）。
3. 否则直接开批次 A（四个 original，≈4 h）。
额外总成本 ≤ 1.2 GPU 小时；诊断结果本身也进论文（"loss 在哪"是可压缩性分析的一部分，Fig. 2 旁半栏）。

## 5. 结果（2026-10-20）：两条探针都跑了，没有可做的小改，直接开批次 A

- error anatomy（`rd/error-anatomy-room-2026-10-19.md`）：误差密度 背景 23× > 轮廓 3.3× > 高频纹理 2.3× > 平坦 0.5×；没有单一桶 ≥ 0.8 dB；曝光仿射上界 +0.34；dense 比 λ 模型多的容量一半是记忆训练视角（train/test gap 2.45 vs 1.70）。
- env-map 探针（`rd/envmap-probe-2026-10-20.md`）：4/4 场景 test 增益 0.002–0.121 dB，未覆盖像素 < 0.4 %（室外也是，densification 用远三角形把天空盖住了）。**关闭**。论文分析段一句话带过。
- 结论维持 §0：差距是 opaque-mesh 类共性，不在大场景前堆方法。批次 A（flowers / treehill / truck / train 的 original + level-1.0 交付）见 `analysis/commands/batchA-originals-v1.md`。
