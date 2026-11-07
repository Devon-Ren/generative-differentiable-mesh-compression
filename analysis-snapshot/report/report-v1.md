# 汇报包 v1（2026-11-01）：给老板 / PhD / HPC 申请用的图和表

目录：`analysis/report/`。所有图一套样式（`figstyle_v1.py`：我们 = 绿色平滑曲线（PCHIP，log-MB），post-hoc = 橙，固定上限 = 灰，mesh 类 = 黑色空心圆 + 文字标签，Gaussian 类 = 冷色圆点，generic codec = 紫；横轴一律绝对 MB、log；每张 RD 图左上角 "better" 箭头；压缩比只在表里，不在图上）。

## 推荐汇报的 4 张图 + 3 张表

| # | 文件 | 一句话读法 | 用途 |
|---|---|---|---|
| 图 1 | `fig-codecs-room-v1.png/.pdf` | 同一个 742 MB 的 FP32 MeshSplatting 资产：PLY+zstd 158 MB 保住 28.09 dB，Draco（SH 当属性）456 MB，Draco/贴图路线丢掉 SH 后 62–137 MB 直接掉 3 dB；我们 26 MB 28.32、5 MB 27.38、2.8 MB 26.57。 | "现成 codec 不够用"的证据，老板关心的 MPEG 待加（见下） |
| 图 2 | `fig-mesh-class-v1.png/.pdf` | (a) engine-ready mesh 的 size–PSNR；(b) 同 PSNR 需要的体积：MiLo 253 MB → 我们 15 MB（17×）、RaDe-GS 1.1 GB → 6.6 MB（166×）、GOF 1.5 GB → 4.2 MB 且 +2.3 dB；作者 RGB-only PLY 100 MB → 我们最好点 28 MB 低 0.31 dB；BOG 913 MB 高 1.15 dB 不可达。 | "在 mesh 里极其领先"的直观图（不是 RD） |
| 图 3 | `fig-gauss-class-v1.png/.pdf` | (a) 如实的绝对图：3DGS 类高 3 dB，是表示差距；(b) 相对各自全精度模型保住的质量：同体积下我们比 RDO-Gaussian 保得多，HAC 家族 +0.1…+0.3 是因为 Scaffold-GS 本身更可压。 | 诚实 + 可比区 |
| 图 4 | `fig-in-the-loop-v1.png/.pdf` | 同一 codec、三种选顶点方式：post-hoc collapse 在 10 MB 塌到 22.7 dB，固定上限 24.9，λ-growth+prune 25.0 且能到 4 MB；(b) 同顶点数下 14/14 点 in-the-loop 胜，+0.3…+4.3 dB。 | 方法贡献的证据（这就是"演化史"的正确画法） |
| 表 1 | `tables-v1.md` T1 | 11 场景 per-scene：FP32（train.log）/ 论文 / 四个 λ 点的 MB、压缩比、PSNR/SSIM/LPIPS。 | 真实数字 |
| 表 2 | `tables-v1.md` T2 | 跨方法均值：mesh 类正常报告，3DGS 类 PSNR/SSIM/LPIPS/MB/FPS，FPS 缺的用 "–"，压缩比相对各自全精度模型。 | 老板要的表 |
| 表 3 | `tables-v1.md` T3 | 部署：训练时长比 0.57×、单核解码 2–6 s、手机 145–350 fps。 | 3DGS 类 codec 没有的一列 |

附：三指标 RD（PSNR/SSIM/LPIPS）均值图 `analysis/rd/rd-metrics-v1.png`（临时，统一评估后重画）。

## 汇报时必须带的三句话（如实）

1. 室内四场景现在的 FP32 和 λ 点都没开 upstream 的 `--indoor`，比论文低 0.3–1.0 dB；12 个 job 正在重跑，落地后图 1–4 和表 1–2 的室内行会更新（预计室内 PSNR +0.5…+1.0、LPIPS −0.05…−0.1）。
2. 我们交付资产的 LPIPS 是 pipeline 口径（输入归一到 [-1,1]），比论文/HAC++ 用的 3DGS 口径高约 0.03（garden FP32：0.249 vs 0.216，SSIM 完全一致）。统一评估脚本把所有资产换到 3DGS 口径后，表里我们的 LPIPS 会降 ~0.03。
3. 对 3DGS 类压缩器，绝对 PSNR 低 3 dB 是不透明 mesh 的表示差距（MeshSplatting 未压缩时就低 2.7 dB），不是 codec 的；我们的可比区是同体积下保住的质量和部署。

## MPEG（老板关注）

图 1 现在只有 PLY+zstd、Draco、贴图路线。MPEG 的 mesh 编码有两条：
- **SC3DMC / TFAN（MPEG-4 Part 16，静态 mesh）**：开源实现 Open3DGC（AMD，C++，含 TFAN 连接编码 + 属性量化），能直接吃我们的 mesh + per-vertex 属性，和 Draco 同类。CPU，一个场景分钟级。
- **V-DMC（MPEG-I Part 29，video-based dynamic mesh coding，2023–）**：参考软件 TMM 在 MPEG GitLab，面向动态序列，静态单帧也能编；工程量大（依赖 HM/VTM 视频编码器）。
建议：先加 Open3DGC（SC3DMC）作为"MPEG"行，V-DMC 作为可选。需要我写编码脚本时说一声（CPU，和 Table 1 扩 9 场景一起跑）。

## 这次申请资源要说的用量（按 L40S GPU-h）

| 项 | 内容 | GPU-h |
|---|---|---|
| 在跑 | 室内 4 场景 --indoor 重跑（4 original + 16 λ） | ≈ 22 |
| 待跑 | 统一评估（66 资产 + 11 original + RGB-PLY，一次） | ≈ 3 |
| 待跑 | 定性图 / teaser / 顶点密度-误差 inset | ≈ 1 |
| 待跑 | Table 1 扩 9 场景 + MPEG 行（CPU，可在 GPU 节点上顺带） | CPU |
| 预留 | 审稿前补跑（T&T 第 3 λ、消融 μ/prune 轮次、1–2 个场景复核） | ≈ 10 |
| 合计 | | ≈ 36 |

截稿 ~11 月中；现有 FairShare 跑完在跑的 12 个 job 后基本归零，所以需要额外配额或 reset。
