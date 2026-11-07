# Story v2（2026-10-18）：Rate-controlled growth —— 把码率放进 differentiable mesh 的生成环

取代 `story-candidates-2026-10-16.md` 候选 1 的"RD 解剖 + codec"主线。PhD 的判断（只做 RD 分析不够主会；码率要进生成环）已被 G0/G1 实验证实，本文件是按这个判断重写的论文口径。所有数字来自 `analysis/rd/`（七场景，10/16–10/18）。

## 1. 一句话

**Differentiable opaque mesh 的 primitive 数量应该由码率决定，而且必须在生长时决定：dense 训练把顶点长过头 2–3×，事后再删补不回来（同顶点数差 0.3–4.3 dB，7/7 场景），而把一个 Lagrangian 价格 λ 放进 densification，顶点数就成了 λ 的输出，训练一次直接落在 RD hull 上。**

标题候选：
- *Grow What You Can Afford: Rate-Controlled Mesh Growth for Compact Differentiable Meshes*
- *Rate-Controlled Growth of Opaque Differentiable Meshes*
- *Pay-as-You-Grow Meshes: Lagrangian Densification for 50× Smaller MeshSplatting*

## 2. 为什么够主会（三个新东西，每个都有对照实验）

1. **一个 mesh 专属的发现**：对 Gaussian，事后剪枝几乎免费（LightGaussian 剪 60 % 掉 <0.3 dB），所以 3DGS 压缩全是 post-hoc 的；对 opaque mesh，顶点被拓扑耦合，事后 collapse 到同顶点数掉 0.3–4.3 dB（七场景 14 个点无一例外，低码率端更差）。这解释了为什么 3DGS 压缩那套（HAC、RDO-Gaussian、Mini-Splatting）不能直接搬到 mesh 上，也是"必须在训练时控制"的论据。图：`budget-vs-collapse-7scenes-v1.png`。
2. **一个方法**：rate-controlled growth。split 判据 ΔD_t > λ·ΔR_t（ΔD_t = 该三角形在区间内承担的每 megapixel 像素残差质量，ΔR_t = 分裂新增的交付字节，按场景实测 B/vertex），顶点数由 λ 决定而非手设 cap；同一个 λ 在不同场景停在不同顶点数（dense 的 41–100 %），同字节下 ≥ cap（10/10 可比点）。不要承诺等 dB 斜率（已证伪，见 C3c）。Phase B（Delaunay 之后）的周期性 λ-prune（ΔR_v 随省掉的面数变化）是 v2。
3. **一个系统结论**：几何是唯一需要进训练环的流。外观流（SH → K4096×R256 索引）事后共享就够（跨场景码本证伪、QAT 负结果、backbone 顺序 7/7 场景相同），拓扑流是不可压缩地板（Draco 对照）。所以方法只控制几何，其他两流用 rate-aware codec + 按实测斜率的分配器。这把"为什么不做端到端"讲清楚了，审稿人问"为什么不把码本也放进训练"有实验答案。

结果口径（七场景均值，相对原 FP32 模型，10/18 λ 版）：λ=2e-3 **19.6 MB / −0.26 dB / 36×**，λ=6e-3 **8.8 MB / −0.88 dB / 80×**（cap 版 14.4 / −0.68 / 46× 与 9.8 / −0.97 / 67×；λ 在低端字节更少还更高）；旧的事后 collapse 路线只有 23.6 MB / −0.44 dB 28×，而且在 20 MB 以下够不到同 PSNR。解码后是标准 mesh（引擎可用），比 MobileNeRF / BakedSDF / BOG / MiLo 小 4–30×。

## 3. 可证伪 claims 与证据状态

| # | claim | 证据 | 状态 |
|---|---|---|---|
| C1 | dense densification 长过头：cap 到 42 % 顶点只掉 0.1–0.7 dB，事后 collapse 到同顶点数掉 1–4 dB | 七场景 budget vs collapse | **已有** |
| C2 | 差距来自拓扑耦合而不是 finetune 不够：collapse 后 appearance finetune 平台期、worst view 掉 3–5 dB | C2 实验 + 七场景 worst val | 已有，可补一个 finetune 步数 ablation |
| C3a | 同字节下 λ-split ≥ cap：七场景 10 个可比点全部 ≥ 0（+0.0～+0.57 dB），对事后 collapse +0.3～+1.6 | `rd/lambda-7scenes-v1.csv`、`lambda-vs-cap-7scenes-v1.png` | **已有（10/18）** |
| C3b | 场景自适应：同一 λ 不设 cap，各场景停在 dense 的 41–100 %（2e-3）/ 15–43 %（6e-3）；七场景均值 19.6 MB / −0.26 dB / 36× 和 8.8 MB / −0.88 / 80× | 同上 | **已有** |
| C3c | ~~等斜率：同 λ 各场景落点的 dB/MB 相同~~ **不成立**：两 λ 点之间的斜率各场景差 5×（0.019–0.100）。原因：λ 定价 L1 质量，PSNR 是 log-MSE。论文只写 C3a/C3b，等斜率作为 limitation 一句带过 | Room + 六场景 | 已证伪（10/18） |
| C4 | 选择规则的作用随预算收紧而增大：Room 1.3M 打平、0.5M +0.25、0.23M 明显赢；dense 端 λ=1e-3 比原版高 0.21（抑制 split/prune 震荡） | `rd/room-lambda-vs-cap-v1.png` | 已有；matched-count importance 对照已跑（10/19）：同日程换 importance 采样 4.31 MB / 26.81，λ 曲线同字节 26.93（排序 +0.12），cap 约 26.5（日程 +0.3）；importance 采样的分裂被剪掉多 30 %（V@delaunay 370k vs 525k） |
| C4b | 自然停止（改写，10/19）：有价格时候选集几何衰减（λ=6e-3 半衰期 ≈3000 步，4.5–7.6 个 interval；λ=2e-3 ≈5000–10000 步），无价格（λ ≤ 6e-4）候选集反而增长、只靠限速器/4M cap 停；λ=6e-3 时 7500→9500 的三角形数在 ±8 % 内（7/7 场景，增长 ≈ 剪枝的稳态），λ=2e-3 时 4/7 场景平、3 场景还在 +8～24 %。论文口径：λ 给出的是由 λ 决定的稳态三角形数，不是"pass=0"；延长窗口的预测影响 <3 %（6e-3），不跑 | `rd/pass-curves-7scenes-v1.png`、`-table.md`、`rd-pass-curve-2026-10-19.txt` | 已有（10/19，0 GPU） |
| C5 | Phase-B 周期 λ-prune：同 λ 下顶点 −11 %、字节 −8 %、PSNR 不变（6.2e-3 与 1.85e-3 两点都是 +0.01～0.02）；同字节 +0.11～0.16 dB（线性插值上界）；4 轮收敛（每轮 pairs 减半，cap 从不生效）。**ΔR 变价不成立**：被选中的边里边界边更多、3+ 面边略少（ring 残差项主导，8–12 B 的价差太小）→ 论文把剪枝写成“同一个 λ 在 phase B 的反向应用”，不 claim 变价。训末单次剪枝对照未跑 | `rd/rd-prune3-room-2026-10-19.txt` | 已有（10/19） |
| C6 | 外观流不需要进训练环：跨场景码本无用、QAT 负、backbone 7/7 同序 | 已有 | 已有 |
| C7 | 精度规则 b = ceil(log2(extent / 0.1·median edge)) 命中或低 1 bit（7/7） | 已有 | 已有 |
| C8 | 拓扑地板：Draco 顶点拆分 4.5×，总字节更大 | 已有 | 已有 |

## 4. 方法章节（论文 Sec. 3–4）

**3.1 Where the bytes go**（半页）：742 MB → 三条流各自的冗余结构，给出 B/vertex 分解（topology ≈4、XYZ 2–2.6 by precision rule、DC ≈1.5、index ≈2.4）。这就是 ΔR 的来源，让 λ 有单位。

**3.2 Post-hoc removal fails on meshes**（C1/C2，一张图一段话）。

**3.3 Rate-controlled growth**：
- 分裂：每个 densification 区间累计 ΔD_t（per-pixel L1 按 dominant triangle id 归属，渲染器已输出 `rend_ids`），ΔR_t = 3 个中点 × B/vertex；ΔD_t/ΔR_t > λ 才分裂，上限沿用 add_percentage 做日程安全阀。λ=0 退化为 error-ranked 无价格增长，`importance` 模式复现上游采样（同代码路径的对照）。
- 停止：不设 cap；当所有三角形的残差质量都低于 λ·ΔR 时自然停。
- v2 剪枝（Delaunay 之后）：每 2000 步对顶点算 ΔD_v（渲染敏感度，复用 C2 sens_pos）与 ΔR_v（删掉它实际省的字节：面数 1–6 不等），ΔD_v < λ·ΔR_v 就 collapse；因为还有上万步训练修复，和事后 collapse 本质不同。
- 诚实段落：ΔR_t 在 split 端近似常数，所以 split 规则 = 带字节单位的残差阈值；真正让 λ 非平凡的是 (a) 跨场景统一的停止点，(b) v2 的变 ΔR_v 剪枝，(c) 同一个 λ 给精度和码本定价（3.4）。

**3.4 Rate-aware codec and allocator**（现有内容压缩成一页）：const vertex_weight、拓扑上下文索引编码、精度规则、K×R backbone、贪心分配器 = hull。

## 5. 实验章节

- **主表**：9 场景 Mip-360（flowers/treehill 要补）+ T&T 两场景：original MB/PSNR → λ 两档（≈46× / 67×）MB、PSNR、SSIM、LPIPS、FPS、三角形数。对照：post-hoc collapse 同字节、作者 RGB-only PLY、MobileNeRF / BakedSDF / BOG / MiLo（mesh 类），HAC++ 只在文字里承认 PSNR/MB 不是本文战场。
- **主图 1**：七场景 budget-vs-collapse（已有）。
- **主图 2**：λ sweep：顶点数、MB、PSNR 随 λ；叠加 cap 曲线与 collapse 曲线（Room 先，七场景后）。
- **主图 3**：七场景全局 hull（已有 `seven-scene-hulls-v1.png`，headline 改成 λ 点）。
- **Ablation 表**：importance 采样 vs error-ranked（λ=0）vs λ-split；有无 phase-B 剪枝；训末单次剪枝；精度规则 vs 固定 q14；K×R 固定 vs 分配器。
- **分析图**（supp 或正文半页）：index 空间相关性、跨场景码本（证伪）、精度规则七场景、backbone 七场景同序。

## 6. 审稿人会问什么

- *"这不就是 3DGS 的 budget densification（Taming-3DGS）/ RDO-Gaussian 搬到 mesh？"* → 区别有实验：Gaussian 事后能剪、mesh 不能（C1），所以 mesh 的控制点只能在生长期；且 ΔR 用的是真实 codec 字节而不是 primitive 计数。
- *"λ 和 cap 有什么本质区别？"* → cap 是输入、逐场景要调（1.6M 对 Room 太松、对 garden 太紧）；λ 跨场景统一，落点满足等斜率（C3 的图直接回答）。
- *"为什么不端到端（码本进训练）？"* → C6 三条负结果。
- *"PSNR 比 3DGS 压缩低 3 dB"* → 不同表示类；本文比较 mesh 类，并给引擎可用性 + FPS。
- *"只有 Mip-360？"* → 补 T&T（truck/train），作者仓库自带脚本。

## 7. 排期（假设 deadline 11 月中）

- 本周（已完成 10/18）：Room λ sweep 6 点 + 六场景 × 2 λ；λ 两档定为 2e-3 / 6e-3（每 Mpx）。剩：importance 同路径对照 1 run；Room 填缝可选。
- 下周：七场景 × 2 λ + flowers/treehill/T&T 的 original 训练；v2 phase-B 剪枝（需要训练环内 edge-collapse + optimizer 状态重映射 —— 这是"重写 MeshSplatting"的部分，见 §8）。
- 第 3 周：ablation、baseline 表、FPS；写作。

## 8. 如果要动 MeshSplatting 本体

v1 不动仓库（`analysis/rdgrowth/` 新文件 import upstream）。v2 的周期剪枝需要在 Delaunay 之后的连通网格上做 edge collapse 并保持 optimizer 状态，上游没有这个操作（`_prune_vertices` 只会删掉整条三角形留洞）。做法：在 rdgrowth 里实现 `collapse_edges(triangles, pairs)`（顶点合并到中点、面重映射、去退化面、`_prune_vertex_optimizer` 风格的 Adam 状态重映射），仍是子类/外挂；只有当上游的 CUDA 光栅器需要新输出（例如 per-vertex 的像素敏感度）时才改仓库，那时再问。

## 9. 和旧口径的关系

旧主线的四条 claim（外观=纹理采样、几何 20 % 冗余、拓扑地板、杠杆切换）全部保留，但降为 Sec. 3.1 和分析章节，用来回答"为什么只控制几何"。codec 和 29× 不再是贡献列表的第一条。
