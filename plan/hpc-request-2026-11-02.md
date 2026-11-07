# HPC 资源申请：GPU-h 清单与判断（2026-11-02）

单位：L40S GPU-h。一次 λ 训练 + 协议评估 ≈ 1 GPU-h（35–60 min 训练 + 8–11 min eval，室内重跑 20 个 run ≈ 22 GPU-h 为实测基准）。
两条线严格分开：第 1 节是 CVPR 2027 投稿（generative compression of differentiable mesh），第 2 节是下一篇（MeshSplat+++）。

## 0. 判断（给自己看，不进邮件）

- 现有结果 + 已定的 storytelling，CVPR 1–5 分的预期：**3.3–3.6（borderline / weak accept 边缘），不是稳 4**。
  - 加分项：完整系统（训练内 rate 控制 + 一文件 codec + 独立解码器）、11 场景 RD 曲线、24–151×、部署数据（0.57× 训练、2–6 s 单核解码、手机 145–350 fps）、generic/MPEG codec 对照、诚实的 3DGS 对比。
  - 扣分项（审稿人必问）：① 只在一个很新的表示（MeshSplatting，2025-12 arXiv）上做，"是不是 MeshSplatting 专用技巧"；② 核心机制（ΔD/ΔR 阈值生长、SH 码本）都是已知部件的组合，新意在"放进训练环"；③ 绝对 PSNR 比 3DGS 类 codec 低 3 dB，即使解释成表示差距，部分审稿人仍会扣；④ 消融目前只有 7 场景旧参数 + Room 的 prune 消融，不够完整。
  - GPU 能改变的只有 ① 和 ④：补完整消融（§1 B1）、补一个"codec 在别的 mesh 方法上也能用"的实验（§1 B5）、补一个近无损上限点（§1 B2）。这三项做完预期到 **3.6–3.9**；要稳 4 需要 B5 结果明显正面（MiLo 等 mesh 也能 10× 以上压缩而不掉质量）。表示本身的 3 dB 差距和"niche"属性是天花板，再多 GPU 也不改。
- 结论：**CVPR 这条线最多需要 60–100 GPU-h，不需要几百小时**。大头给 MeshSplat+++ 是对的。

## 1. CVPR 2027：generative compression of differentiable mesh（必做 + 提分）

| 编号 | 项目 | run 数 | GPU-h | 作用 |
|---|---|---:|---:|---|
| A1 | 统一评估：66 份交付资产 + 11 原始 + 作者 RGB-PLY，统一到 3DGS LPIPS 口径 | 1 批 | 3 | 表格口径一致（必做） |
| A2 | 定性图 / teaser / 顶点密度–误差 inset 渲染 | – | 1 | 主图（必做） |
| A3 | 室内 codec 损失诊断：level-1.0 re-export 3 组设置 × room/kitchen（不训练） | 6 encode | 0.5 | 解释 Δ 变大（必做） |
| A4 | MPEG 行 GPU 复核 + 其余 9 场景 MPEG 行评估（编码在 CPU） | – | 0.5 | Table 1 完整 |
| B1 | 完整 in-the-loop 消融：11 场景 × 2 个 λ × {固定上限、只 split 不 prune}（post-hoc collapse 走 CPU） | 44 | 44 | 现在只有 7 场景旧参数；审稿人必查 |
| B1' | 同上缩到 6 场景（室内 2 + 室外 3 + T&T 1） | 24 | 24 | B1 的最小版 |
| B2 | 近无损上限点：λ 2e-4 + 更细 codec 档（DC q8、K16384/R1024），11 场景 | 11 | 11 | 回应"最好点也掉 0.24 dB" |
| B3 | Deep Blending（drjohnson, playroom）：FP32 + 4 λ | 10 | 10 | 与 HAC++/3DGS 表同数据集 |
| B5 | codec 通用性：在第二个 mesh 方法（MiLo，有代码）9 场景的网格上跑同一 codec | 9 训练 + encode | 12 | 回应"只对 MeshSplatting 有效" |
| B6 | prune 日程消融（轮数 2/4/6、每轮上限 10/15/25 %）× 3 场景 | 9 | 9 | 方法细节的依据 |
| B7 | 审稿前预留（复核 1–2 场景、补点） | – | 10 | |
| **合计** | 最小（A 全 + B1' + B2 + B7） | | **≈ 50** | |
| | 完整（A 全 + B1 + B2 + B3 + B5 + B6 + B7） | | **≈ 100** | |

全部可脚本化、互相独立（每个 run 一个 sbatch），适合填满多卡节点的前 1–2 天。

## 2. MeshSplat+++（下一篇：bounded / closed / compact 的端到端可微 mesh）

探索性工作的特点：时间花在调试迭代上，不是固定 sweep；按"场景 × 配置 × 迭代次数"估，取中值。

| 编号 | 项目 | 估计 GPU-h | 说明 |
|---|---|---:|---|
| C1 | P2b：P1b 平面的粗网格重建 + per-face 纹理（room / kitchen / garden，3–5 个分辨率配置） | 15 | 已设计未跑（10/30 推迟）；验证"顶点换纹理"是否成立 |
| C2 | 闭合初始化（2DGS→TSDF 或 visual hull + 远景壳）+ 只做流形操作（split/collapse/flip）的生长原型；4 场景 × ~10 轮迭代 | 60 | 核心原型，失败率最高，吃时间最多 |
| C3 | opaque 流形 + alpha cards 混合（植被/细结构），4 场景 × 5 配置 | 20 | P2 显示 soup 依赖软混合，这是替代方案 |
| C4 | 统一定价律（dD > λ dR 同时管 split/collapse/card/texture-res）的 RD 曲线，6 场景 × 4 λ | 24 | 新一代的"generative"叙事 |
| C5 | evidence-gated growth 迁移到 3DGS（Room + flowers 探针，然后 4 场景） | 10 | 无 ddl 的旁支，若正面可独立成文 |
| C6 | 对照：MiLo / 2DGS-mesh / BakedSDF 类在同数据上的复现 | 25 | 新论文的 baseline |
| C7 | 失败重跑与调试冗余（C2–C4 的 50 %） | 50 | 探索性工作必有 |
| **合计** | | **≈ 200**（宽松 250–300） | |

## 3. 关于"整节点一周"

- 若节点是 4 卡，5 天名义 = 480 GPU-h；但 reservation 从开始计时就算用，**空转也是浪费**。能真正吃满的只有脚本化 sweep（§1 的 ~100 h 和 §2 的 C1/C4/C6）；C2/C3 这类调试性工作一次只占 1 卡，且有大段空档。现实利用率 50–60 % → 5 天 ≈ 250–300 有效 GPU-h，正好覆盖 §1 完整版 + §2 的中值。
- 建议的排法：第 1–2 天 4 卡全跑 §1 的 sweep（一个简单的队列脚本按 CUDA_VISIBLE_DEVICES 喂任务），同时 §2 用其中 1 卡交互开发；第 3–5 天 §2 的 sweep 上 4 卡。
- 邮件里可以同时问两种形式：(a) 整节点 reservation 5 天；(b) 不 reserve，改为给账号一次 FairShare reset / 临时高优先 QOS，总量 ~350 GPU-h。对 HPC 团队 (b) 往往更容易批，对我们也更灵活（正常 sbatch，不怕空转）。Slurm reservation 通常按节点给，单卡 reservation 要看他们是否配置了 GRES 级别的 reservation，直接问。
- 邮件必列的两项：CVPR 2027 截稿（~11 月中）；§1 + §2 的清单与小时数（§1 是"投稿必需"，§2 是"下一阶段方法开发"，分开写，让他们看到用量结构）。

## v2（11/03，老师质疑后）：加 object-centric / human 数据集，算力仍倾向 +++，邮件里统一写成 CVPR 需求

| 组 | 项目 | 作用 | GPU-h |
|---|---|---|---:|
| A 物体/人物数据集 | NeRF-Synthetic 8 物体 × (FP32 + 2 λ) | 老师要的 object-centric 验证 | 8 |
| A | DTU 15 scans × 3（2DGS 提供的 COLMAP 格式） | 几何精度（Chamfer）+ 压缩 | 18 |
| A | T&T 物体场景（Ignatius, Caterpillar, Barn, Meetingroom, Courthouse）× 3 | 大物体/人像雕塑 | 15 |
| A | 多视角人体（THuman2.0 渲染 或 ZJU-MoCap 单帧）4–6 个 × 3 | 老师提的"a person" | 8 |
| A | 同数据上的 3DGS / 2DGS 对照 | 同字节比较 | 12 |
| B 评估与消融 | 统一评估 + 定性图 | 表格口径一致 | 4 |
| B | in-the-loop 消融 6 场景 × 2 λ × 2 变体 | 方法贡献 | 24 |
| B | 价格敏感性 B_a ∈ {4,6,9} × 2 场景 | 审稿人问题 | 6 |
| C 表示开发（+++） | 粗网格重建 + per-face 纹理（3 场景 × 3–5 配置） | 顶点换纹理 | 15 |
| C | 闭合初始化 + 流形保持生长原型（4 场景，迭代） | bounded / closed | 60 |
| C | opaque mesh + alpha card 混合（4 场景 × 5 配置） | 植被/细结构 | 20 |
| C | 统一定价律 RD 曲线（6 场景 × 4 点） | 一个 λ 管所有操作 | 24 |
| C | 对照复现（MiLo / 2DGS-mesh / SDF-mesh） | baseline | 25 |
| C | 生长准则迁移到 3DGS（4 场景） | 通用性 | 10 |
| 余量 | 失败重跑 / 调试（约 20 %） | | 50 |
| **合计** | | | **≈ 300** |

邮件版本见 thread 11/03 草稿卡（v2）。待确认：Wenhao 提到的 "adai"、"skynet" 两个数据集名。
