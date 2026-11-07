# 老师 11/03 看完渲染后的问题：诊断、逐条回答、对投稿的影响（2026-11-03）

## 0. 一句话判断

她看到的画面里，大约 2.7 dB 是 MeshSplatting 这个表示本身的（FP32 就有），再 1.1 dB 是我们选给她看的 73× 操作点的；codec 本身只占 ≤ 0.3 dB。她问的每一个问题都是审稿人会问的问题，而其中"同样字节 3DGS 质量好得多"我们只能承认，不能反驳。她指的方向（bounded、compact、physically correct 的可微 mesh）就是 +++，和你 11/02 想把算力倾向 +++ 的判断一致。CVPR 这篇要不要投、怎么投，建议由她要的 object-centric 测试结果来定（8 GPU-h，等节点）。

## 1. 截图是什么

viewer 状态栏：garden，F 2,157,003，K4096 R256，asset 10.17 MB，cam 6/185。对应 λ = 6.2e-3 的点：

| | MB | PSNR | 与谁比 |
|---|---:|---:|---|
| 这张图（garden λ 6.2e-3，73×） | 10.2 | 23.65 | |
| garden λ 6.2e-4（24×） | 36.4 | 24.73 | 与 FP32 相同 |
| garden FP32（MeshSplatting 原样，740 MB） | 740 | 24.75 | 论文 24.70 |
| 3DGS（1.3 GB） | 1328 | 27.30 | 表示差距 2.6 dB |
| HAC++ low（12.8 MB） | 12.8 | 27.19 | 同字节下比我们高 3.5 dB |

画面里三类东西的来源：
1. **树冠变成大片平三角形、碎片之间有裂缝**：triangle soup 在植被上的类属性。MeshSplatting 的 soup 靠软混合（soft blending）训练，变不透明后每个像素只剩一个三角形；叶子这种亚像素结构被少数大三角形代替。FP32 就有（10/26 的 geometry-fragmentation 笔记；P2 实验：把训练好的 soup 直接不透明渲染，room 掉 2.0 dB、kitchen 掉 3.6 dB）。
2. **放大后细节是块状的**：73× 点的 83 万顶点撑不起 garden 的树冠和草地；这是我们为了展示压缩比选的点。λ 6.2e-4 的点 36 MB、24.73 dB，和 FP32 看不出差别。
3. **黑斑 / 颜色跳变**：SH 3 阶在训练轨迹外的外推，也是类属性。

不是原因的：codec。level 1.0 只换 codec 不动顶点，garden 损失约 0.3 dB。

## 2. 逐条回答她

**"Why so bad, so many cracks? never seen this in 3DGS."**
对。3DGS 每个像素混合几十到几百个软 primitive，误差被平均掉；不透明 mesh 每个像素只有一个三角形，误差全露出来。这就是 2.6 dB 的来源（10/20 的 class-gap 分析），也是裂缝的来源。这个问题在我们这篇里解决不了，它是下一个表示的问题。

**"Why mesh at all? Gaussians are also fast, more software handles Gaussians now."**
mesh 能给、Gaussian 现在给不了的，只有这几样，而且要老实说每一样的代价：
- 固定管线光栅化：没有排序、没有自定义 splat 渲染器；我们的 WebGL viewer 几百行，手机 150–350 fps。Gaussian 的引擎插件是"查看器"，不是场景几何。
- 碰撞、导航网格、物理：mesh 直接可用；Gaussian 要另外抽 mesh。
- 阴影、重光照、和传统几何混排：标准 shader 就行（把 SH 烘成顶点色，掉 0.3–0.7 dB，P2 实测）。Gaussian 需要研究级方法。
- 编辑：DCC 工具直接开。
- 解码：单核 CPU 2–6 s，不需要 GPU 解码器。
代价：无界场景低 2.6–3 dB，植被明显更差。所以诚实的说法是"一个像游戏几何一样行为的重建资产，但保真度低 3 dB"，而不是"更好的 NVS"。

**"3DGS compressed to similar size has much better quality, people will ask why."**
会问，而且我们只能说是。同字节下 HAC++ 高 3.5 dB（garden），论文从来没有声称 PSNR 比 3DGS 类 codec 好；声称的是 (a) 在 engine-ready mesh 这一类里，同 PSNR 下比 MiLo 小 22×、比 BOG/RaDe-GS/GOF 小 34–352×；(b) 对 mesh，rate 放进训练环比事后简化好 0.3–4.3 dB。审稿人是否接受"只在 mesh 类里比"这个框架，就是 11/02 我估 3.3–3.6 的原因；她的反应就是审稿人看到定性图时的反应，所以下限是真的。

**"Why MeshSplatting in the first place? its contribution?"**
那是上游论文的贡献（从零训练的可微 mesh，笔记本 220 fps），不是我们要辩护的。我们继承了它的上限。

**"Look at simple object-centric data: a table, a person."**
同意，而且这和 +++ 的 P1b 结论一致（室内 50–75 % 顶点在平面上，植被只有 16 %）：不透明表面在有界实体上是对的模型，在植被上不是。可做的测试：NeRF-Synthetic（8 个物体，800×800，白底，3DGS 代码自带 Blender 加载器，MeshSplatting 建在 3DGS 代码上，需要先 smoke 一次确认 `--white_background` 路径没坏）。预期：lego / chair / hotdog / materials 这类实体应接近 3DGS（差 ≤ 1 dB），ficus / ship 这类细结构会重现碎片。费用：8 场景 × (FP32 + λ 6.2e-4 + λ 6.2e-3) = 24 run × ~0.3 h ≈ 8 GPU-h。"a person"没有带 GT 的标准小数据集，先不做。

## 3. 对 CVPR 投稿意味着什么

事实：论文里的定性图会把她看到的东西和 HAC++ 同字节的图并排放；审稿人的第一个问题就是她的问题。现有数据上我维持 3.3–3.6 的估计。

三条路：

| | 做什么 | 代价 | 预期 |
|---|---|---|---|
| P1 照原计划投 | 写作 + 统一评估 + 定性图（≈ 15 GPU-h） | 2 周写作 | 3.3–3.6；老师不背书 |
| P2 不投 CVPR | 压缩工作并入 +++（+++ 的 4.4 "one pricing law" 本来就是它）；+++ 投 2027 年中（ICCV 3 月 / NeurIPS 5 月 / SIGGRAPH Asia 5 月） | 放弃这个截稿 | 老师认可的方向 |
| P3 改定位再投 | 标题/摘要从"压缩 MeshSplatting"改成"可微不透明 mesh 有多可压、代价是什么"：主体是分析（mesh 买到什么、付出 3 dB、字节去哪、in-the-loop > post-hoc），加 object-centric 一章（她要的）和引擎 demo（Blender/Unity 里带阴影和碰撞的截图） | object 测试 8 GPU-h + demo 1–2 天 CPU | 若 object 结果接近 3DGS，能到 3.5–3.8；否则回到 P1 的分数 |

建议：**P3，但由 object-centric 结果决定**。节点一到先跑这 8 GPU-h；实体物体上接近 3DGS → 投 P3；不接近 → 走 P2，把截稿让给 +++。节点若 4 天内到不了，P3 来不及，就在 P1 / P2 之间选，我的倾向是 P2：老师不信的论文，她的签名和 HPC 申请的背书都成问题。

## 4. 回老师时可以说的要点（不是邮件）

1. 她看到的画面里，2.6 dB 是 MeshSplatting 本身的，1.1 dB 是我选的 73× 点；下次先给她看室内场景和 24× 点，并排放 HAC++ 同字节。
2. 承认：同字节 3DGS 更好；mesh 的意义只在"能当几何用"，我会做一个引擎里带阴影/碰撞的 demo 来证明这句话有没有价值。
3. 她说的 bounded / compact / physically correct 的可微 mesh，就是我 10/29 写的 +++ 计划（P1b 已经证明室内 50–75 % 顶点在平面上，但 mesh 是皱的），这是申请算力的主要用途。
4. 先按她的建议跑 object-centric（8 GPU-h），用结果决定压缩这篇投哪里。

## 5. 现在不用 GPU 就能做的

- viewer 默认改成室内场景、24× 点、测试相机（而不是 garden 73×）。
- 解码后的 mesh 烘顶点色导入 Blender，打灯、投影、碰撞各一张截图（CPU，解码器已输出标准 mesh）。
- 把 §2 的"为什么 mesh"写成一页纸给她。
- 准备 NeRF-Synthetic 的 smoke 脚本，节点一到就跑。
