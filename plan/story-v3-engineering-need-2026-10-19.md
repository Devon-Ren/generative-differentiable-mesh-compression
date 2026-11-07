# Story v3（2026-10-19）：A Complete Codec for Differentiable Opaque Meshes —— 工程需求定位

取代 `story-v2-cvpr-main-2026-10-18.md` 的"λ vs cap 方法论文"口径。v2 的实验、claim 编号（C1–C8）和图全部保留，只改**讲法与主表结构**。定位来自用户 10/19 的判断（记录在 `contribution-review-2026-10-19.md` §6）：

> 论文解决一个工程需求：differentiable opaque mesh（MeshSplatting）的可压缩性。贡献 = 给这类表示建全链路编解码器 + 对其压缩性做全面分析 + 把码率放进生长环实现生成式压缩，50–100×。RD 口径、lossy by design，不承诺近无损；不与 3DGS 压缩方法比绝对 PSNR（差距是 MeshSplatting 自身的 NVS 质量）。

## 1. 一句话

**MeshSplatting 训出来的是 742 MB 的 FP32 三角形汤；我们证明这类 differentiable opaque mesh 的冗余按流各不相同（外观可事后共享、拓扑是地板、几何只能在训练中控制），据此建了第一条从训练到引擎可用 mesh 的完整压缩链路，7 场景均值 36×（−0.26 dB）到 80×（−0.88 dB），而所有事后路线在同字节下都低 0.3–4 dB。**

标题候选（强调对象是"表示"，不是"方法"）：
- *Compressing Differentiable Opaque Meshes: A Codec and a Compressibility Study for MeshSplatting*
- *How Compressible Is a Differentiable Mesh? Rate-Controlled Growth and a Full Codec for MeshSplatting*
- *From 742 MB to 9 MB: Rate-Aware Training and Coding of Differentiable Opaque Meshes*

## 2. 动机（intro 的顺序，智力核心前置）

1. **工程需求。** Differentiable opaque mesh（MeshSplatting 及其同类）是 NVS 里唯一直接输出标准三角网格的表示，引擎可用，但一个 Mip-360 场景 742 MB，比 3DGS 还大 2×；没有任何工作给它做过压缩，作者仓库只有一个丢掉 SH 的 RGB-only PLY 导出（≈100 MB，−2 dB）。
2. **它为什么不是"把 3DGS 压缩搬过来"。** 两条 mesh 专属事实（这是全文的智力核心，要放在 intro 第二段，否则会被归为纯工程）：
   - **事后删顶点在 mesh 上失败**（C1/C2）：Gaussian 事后剪 60 % 掉 <0.3 dB，所以 3DGS 压缩全是 post-hoc；opaque mesh 的顶点被拓扑耦合，事后 collapse 到同顶点数掉 0.3–4.3 dB，7 场景 14 个点无一例外，finetune 补不回来。
   - **只有几何流需要进训练环**（C6）：外观（SH→K4096×R256 索引）事后共享足够（跨场景码本证伪、QAT 负结果、backbone 7/7 场景同序），拓扑是不可压缩地板（Draco 对照，C8）。
3. **因此链路的形状是被这两条事实决定的**：几何在生长时按码率控制（rate-controlled growth，v2 §3.3），外观和精度用 rate-aware codec + 按实测斜率的分配器，拓扑用上下文编码到地板。这不是堆 trick，是"每条流按它的冗余结构处理"。

## 3. 三条贡献（论文 contribution list 原文草稿）

1. **A complete codec for differentiable opaque meshes.** 从 MeshSplatting 训练输出到引擎可加载的 mesh：几何精度规则（C7，b = ⌈log2(extent/0.1·median edge)⌉，7/7 场景命中或低 1 bit）、共享 SH 字典 K×R + 邻居上下文索引编码、拓扑 strip-fetch + edge-cache 算术编码（30.4 bits/vertex，原始 126）、const vertex_weight 无损去除；加一个按实测 dB/MB 斜率的贪心分配器，输出即 convex hull。
2. **A compressibility study of the representation.** 回答"冗余在哪、能不能事后去掉"：三流的 B/vertex 分解；外观流的三个负结果（跨场景码本、QAT、频率编码）说明冗余在空间邻居而非符号统计；拓扑地板；**事后删顶点失败（7/7 场景）**这一 mesh 专属发现。
3. **Generative compression: rate in the growth loop.** 把 Lagrangian 价格 λ 放进 densification（ΔD_t/ΔR_t > λ，ΔR 用真实交付字节）和 phase-B 的 λ-prune，顶点数成为 λ 的输出而非手设 cap：同一 λ 跨场景自适应停在 dense 的 15–100 %（C3b）、达到由 λ 决定的稳态（C4b）、同字节 ≥ cap ≥ 事后 collapse（C3a）。七场景均值 **19.6 MB / −0.26 dB / 36×** 和 **8.8 MB / −0.88 dB / 80×**。

不写进贡献的（降为分析段落或 limitation）：λ vs cap 的 0–0.57 dB 差（写成"cap 是 λ 的 budget 特例，λ 省掉逐场景调 cap"）；排序 +0.12 / 日程 +0.3 的分解（ablation 表一行）；等斜率（已证伪，limitation 一句）；ΔR 变价（不成立，剪枝写成 λ 在 phase B 的反向应用）。

## 4. 主表（Table 1）结构：对手 = 同一表示上的通用编解码器

| 行 | 来源 | 状态 | 备注 |
|---|---|---|---|
| MeshSplatting FP32（原始） | 作者 checkpoint 导出 | 已有 7 场景 | 742 MB 量级，PSNR 基准（相对 dB 的 0 点） |
| + zstd | 同上压一遍 | **待算（CPU，分钟级）** | 审稿人问"直接压不行吗"的答案，预计 ≈1.3–1.5× |
| Draco（qp14 + SH 当 generic attribute，8 bit） | run_draco.sh | Room 已有拓扑+XYZ 对照；**全属性版待算（CPU）** | 顶点拆 4.5×，总量比原始 XYZ+拓扑还大 |
| 作者 RGB-only PLY | 作者仓库导出脚本 | **待跑（GPU 分钟级，7 场景）** | 表里标注"no SH"，作为表示自带的导出而非公平对手 |
| 事后 collapse（C2 geo-only）+ 本 codec | 已有 | 7 场景已有 | 同字节下最强的 post-hoc 路线，23.6 MB / −0.44 / 28× |
| 本文 codec，dense 模型（λ=0） | 已有 | 7 场景已有（v0.8 / 1.0 两档） | 展示 codec 单独的贡献（29×） |
| 本文 λ=2e-3 | v3 | 7 场景已有（v2），**v3 带 prune 待跑** | 19.6 MB / −0.26 / 36× |
| 本文 λ=6e-3 | v3 | 同上 | 8.8 MB / −0.88 / 80× |

列：MB、×、PSNR、ΔPSNR vs 原始、SSIM、LPIPS、顶点数、FPS（引擎 viewer）。场景：Mip-360 全 9 + T&T truck/train（11 场景）。外部方法（HAC++、MobileNeRF、BakedSDF、BOG、MiLo）放 Table 2"跨表示参考"，只给 MB + PSNR + 是否标准 mesh，正文一句话承认 PSNR/MB 不是本文战场、差距是 MeshSplatting 自身的。

## 5. 图

- Fig. 1（teaser）：同一场景 742 MB → 19.6 MB → 8.8 MB 的渲染 + wireframe 放大，下方一条 RD 曲线（原始点、事后 collapse 曲线、λ 曲线）。
- Fig. 2：三流 B/vertex 分解 + "冗余在哪"（已有素材）。
- Fig. 3：事后 collapse vs 训练时控制，7 场景（已有 `budget-vs-collapse-7scenes-v1.png`）。
- Fig. 4：λ sweep：顶点数、MB、PSNR 随 λ；叠 cap 与 collapse（Room 已有 `room-lambda-vs-cap-v2`，七场景待 v3 跑完）。
- Fig. 5：pass curve / 稳态（已有 `pass-curves-7scenes-v1`）。
- Fig. 6：全局 hull（已有 `seven-scene-hulls-v1.png`，headline 改 λ 点）。
- Supp：index 空间相关性、跨场景码本证伪、精度规则、backbone 同序、定性图（同字节 collapse vs λ）。

## 6. 实验清单与算力（总约 20 GPU 小时，按 FairShare 分批，每批 2–3 h walltime，general 账户）

| 批 | 内容 | GPU 小时 | 产出 |
|---|---|---|---|
| A | flowers、treehill、truck、train 的 original（4 × ≈1 h） | ≈4 | 主表基准行补齐到 11 场景 |
| B | 11 场景 × λ∈{2e-3, 6e-3} 的 v3（带 phase-B prune，≈45 min/run） | ≈16 | 主表两档 + Fig. 4 七→十一场景 |
| C | 作者 RGB-only PLY 导出 + FPS 测量 + 定性图（分钟级） | <1 | 主表一行、FPS 列、Fig. 1 |
| D（CPU） | zstd、Draco 全属性、主表汇总脚本 | 0 | 主表两行 |

Ablation（Room 已有，不再跑）：importance vs λ=0 vs λ（C4 分解）；有无 phase-B prune（C5）；精度规则 vs 固定 q14；K×R 固定 vs 分配器。可选：训末单次 prune 对照（≤0.5 h）。

## 7. 审稿人会问什么（按这个定位的回答）

- *"方法不新，Taming-3DGS / RDO-Gaussian 已有 budget densification。"* → 本文不以 λ-split 为唯一贡献；mesh 上事后剪枝失败（C1）使"进训练环"成为必需而非选项，ΔR 用真实字节，phase-B collapse 保持 optimizer 状态是 mesh 独有的操作。
- *"没有和 SOTA 压缩比。"* → 没有同类（differentiable mesh → NVS → 可压缩 → 可渲染）工作；Table 1 比的是同一表示上的通用 codec（全赢），Table 2 给跨表示参考并承认差距来源。
- *"PSNR 比 HAC++ 低 3 dB。"* → 是 MeshSplatting 的 NVS 质量差距（原始模型已低 2.4 dB），本文按 RD 口径报相对损失。
- *"80× 点掉 0.9 dB 太多。"* → RD 曲线上的一个点，不是推荐工作点；36× 点只掉 0.26。
- *"只有 Mip-360？"* → 11 场景（批 A/B）。

## 8. 与 v2 文件的对应

v2 §3 的 claim 表、§4 方法章节、§8 是否改上游、§9 旧口径关系全部沿用；v2 §2 的"三个新东西"改为本文 §3 的三条贡献；v2 §5 实验章节由本文 §4–§6 取代。
