# 汇报包 v3（2026-11-01，v2 + MPEG garden / 几何 / 贴图行 + 几何-颜色 ablation）：给老板 / PhD / HPC 申请用的图和表

目录：`analysis/report/`。所有图一套样式（`figstyle_v1.py`：我们 = 绿色平滑曲线（PCHIP，log-MB），post-hoc = 橙，固定上限 = 灰，mesh 类 = 黑色空心圆 + 文字标签，Gaussian 类 = 冷色圆点，generic codec = 紫；横轴一律绝对 MB、log；每张 RD 图左上角 "better" 箭头；压缩比只在表里，不在图上）。

## 推荐汇报的 4 张图 + 3 张表

| # | 文件 | 一句话读法 | 用途 |
|---|---|---|---|
| 图 1 | `fig-codecs-room-v2.png/.pdf` | 同一个 742 MB 的 FP32 MeshSplatting 资产：PLY+zstd 158 MB 保住 28.09 dB，MPEG SC3DMC/TFAN（Open3DGC）108 MB 同质量，Draco（SH 当属性）456 MB，Draco/贴图路线丢掉 SH 后 62–137 MB 直接掉 3 dB；我们 26 MB 28.32、5 MB 27.38、2.8 MB 26.57。 | "现成 codec 不够用"的证据，MPEG 行已加（11/01，见下） |
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

## MPEG（老板关注）— 已加入（11/01）

MPEG 的静态 mesh 标准 SC3DMC/TFAN（MPEG-4 Part 16）用开源实现 Open3DGC 跑在同一个 Room FP32 资产上
（`analysis/baselines/o3dgc-v2/`，CPU 一个场景约 1.5 分钟）：

| 行 | MB | PSNR | 说明 |
|---|---:|---|---|
| MPEG SC3DMC/TFAN，SH 8-bit 整数属性（与 quant-PLY 行同值，无损） | 108.0 | ≈ 28.09（与 quant-PLY 行同构，GPU 复核待跑） | 3.19 M 顶点全部保留，不像 Draco 那样 4.5× 拆点 |
| MPEG SC3DMC/TFAN，SH 浮点属性 + Open3DGC 自带 8-bit 量化 | 102.5 | 待 GPU 评估 | |
| 我们，codec only | 30.9 | 28.06 | 比 MPEG 小 3.5× |
| 我们，λ = 2e-3 | 13.7 | 27.99 | 比 MPEG 小 7.9×，−0.28 dB |

| MPEG 几何（只有 topo + 坐标） | 19.9 | – | Draco 几何 32.8（拆点） |
| MPEG 几何 + 8k JPEG 贴图 | 145.7（4k：91.0） | 25.29（−3.0） | atlas mesh 17.4 M 顶点（接缝复制 5.4×），比 Draco 版 136.8 还大 |

garden 同样：MPEG 126.0 MB vs zstd 172.0 vs Draco 544.4，我们 level 1.0 32.5 MB（小 3.9×）、λ = 2e-3 24.8 MB（−0.29 dB，小 5.1×）。

一句话：MPEG 是最强的通用 codec（比 zstd 小 1.5×，比 Draco 小 4.2×），但同质量下仍比我们大 3.5×（garden 3.9×）。

**几何 / 颜色拆分（ablation，Room）**：MPEG 的 108 MB = 几何 19.9 + SH 属性 88.1（每顶点 27.6 B，每通道 4.6 bit，通用预测+算术编码几乎压不动 8-bit SH）。我们 31 MB ≈ 连接/坐标 16 + SH 15。
只换几何编码器为 MPEG：≈ 35 MB（+13 %）；只换 SH 编码器为 MPEG：≈ 104 MB。所以 108 → 31 的增益里 ≈ 73 MB 来自 SH codebook（3.1×），≈ 4 MB 来自我们的几何编码器（1.13×）。
论文可以老实写：几何可以交给标准 codec（代价 13 %），贡献在 SH 侧和 rate-in-the-loop。贴图路线换 MPEG 几何也不会变好（接缝复制的顶点 + UV 让几何反而更大），贴图层固定在 90–150 MB / −3 dB。
V-DMC（MPEG-I Part 29）只换熵编码器、不改变"每顶点 48 字节 SH 对通用编码器是噪声"这个事实，暂不做。

## 这次申请资源要说的用量（按 L40S GPU-h）

| 项 | 内容 | GPU-h |
|---|---|---|
| 在跑 | 室内 4 场景 --indoor 重跑（4 original + 16 λ） | ≈ 22 |
| 待跑 | 统一评估（66 资产 + 11 original + RGB-PLY，一次） | ≈ 3 |
| 待跑 | 定性图 / teaser / 顶点密度-误差 inset | ≈ 1 |
| 待跑 | Table 1 扩 9 场景 + MPEG 行其余场景（CPU）；MPEG 行 GPU 复核 | CPU + ~0.2 GPU-h |
| 预留 | 审稿前补跑（T&T 第 3 λ、消融 μ/prune 轮次、1–2 个场景复核） | ≈ 10 |
| 合计 | | ≈ 36 |

截稿 ~11 月中；现有 FairShare 跑完在跑的 12 个 job 后基本归零，所以需要额外配额或 reset。
