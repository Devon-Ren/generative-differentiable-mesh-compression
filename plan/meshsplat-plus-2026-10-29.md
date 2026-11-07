# MeshSplat+++：从 differentiable mesh 的机制重新开始（2026-10-29）

用户问（10/29）：能不能从 MeshSplatting 的机制本身出发，找一个"完全更好的形式化"：训练快、优化快、贴近物理世界、场景闭合无裂缝、平的地方连成平面？
这份文档的立场：**可以，而且我们这个项目的数据已经指出了该改哪三处；但它是下一篇论文，不是 CVPR 这篇。** 下面先说现在的机制、我们测到的证据、同类方法的数字，再给出 +++ 的形式化、难点、以及现在就能做的两个零/低成本探针。

## 0. 一句话

MeshSplatting 把四种物理 regime（实体表面、薄片、亚像素/半透明结构、反射透射）全部塞进一种 primitive（不透明三角 soup），又让顶点同时承担形状和颜色两份工作。+++ 的形式化是：**几何与外观解耦（面上纹理），流形由构造保证（闭合初始化 + 只做流形操作），按物理 regime 分两种 primitive（不透明流形 mesh + alpha cards），三者都由同一条法则驱动：bytes 只买多视角证据。** 尺寸、平面度、LOD、card 的数量全是输出。

## 1. 现在的机制（上游代码读到的）

SfM 点 → 按 importance 采样 split 生长 → 11000 步做一次 restricted Delaunay 把点连成三角形集合（点云厚的地方天然非流形）→ 之后 19000 步顶点自由移动、继续 split（中点）→ 每顶点 SH deg 3 + 每顶点 weight → 软光栅 sigma 从 1 退火到 1e-4（边缘由软到硬），alpha 上限 0.999，20000/25000 步后 2x/4x supersampling → 结束时按 max blending 删面。结果：Room dense 45 % 边界边、38 % 边挂 3 个以上面、8277 个连通分量，轨迹外看是碎片（10/26 截图）。

## 2. 我们的数据说了三件事

1. **画质来自 soup 的自由度。** 方向 A：把拓扑换成自己顶点的 restricted Delaunay 要掉 1.3–1.8 dB（v4/v5）；训练只需几千步就离开 Delaunay 性；snap + 冻结顶点反而 +0.19 dB。拓扑是真信息（29 bits/v）。
2. **顶点在替纹理打工。** "740 MB 只带 25 MB 信息"、K4096 codebook 不掉画质、index 的冗余是空间相邻一致 —— 每顶点 SH 的信息本质是贴在面上的图像；顶点密度由颜色细节而不是形状决定。（这是 10/16 定下的论文主线"texture-mapped mesh in disguise"。）
3. **字节必须在环里买证据。** train-at-budget 比事后 collapse 高 +0.3..+4.3 dB；多买的字节在残差底高的场景只买到 train（11 场景 Spearman −0.73，plan/evidence-priced-growth-2026-10-29.md）。

## 3. 同类方法的数字说了什么（analysis/baselines/baselines-room-garden-v1.md）

Mip-360 均值：3DGS 27.21；MeshSplatting 24.78（3M 顶点）；MiLo 24.09；BakedSDF 室内 27.06 / 室外 22.47（watertight SDF mesh，434 MB）；BOG 室内 27.71 / 室外 23.94（0.85 GiB，927 FPS）；Radiance Meshes（体积四面体 mesh，CVPR26）室内 30.61 / 室外 24.38。
读法：(a) 闭合/流形并不必然掉画质 —— BakedSDF、BOG 都是 watertight 的，室内和 MeshSplatting 相当或更好；它们差在室外（植被、细结构），也就是"不透明"这个假设差。(b) 一旦允许体积（Radiance Meshes 的四面体），mesh 也能到 3DGS 水平。所以 2–3 dB 的差距是**不透明单一 primitive**的代价，不是"mesh"的代价。这正是第 4.3 节的依据。

## 4. +++ 的形式化：三条轴，一条法则

### 4.1 外观 ⟂ 几何：面上纹理，不要 UV
每个面带自己的纹理（Ptex / "mesh colors" 的思路：按面分配分辨率，不展 UV，相邻面靠邻接关系做滤波），纹素里存 RGB + 低阶视角相关项（SH1 或 2–3 个基）。顶点只为形状服务。预期：等画质顶点数 ÷3..÷10，外观字节变成图像编码问题（我们已证明它本来就是图像），平的墙面变成两个大三角形 + 一张纹理。这一条是第 2 节证据 2 的直接推论，也是"平面"能出现的前提：外观不解耦，带纹理的墙永远会为了颜色被细分。

### 4.2 流形由构造保证：闭合初始化 + 只做流形操作 + 预条件
不再用 restricted Delaunay 和自由 split。初始化给一个闭合 mesh（SfM 点 → 粗 SDF / 视觉壳 / 2DGS→TSDF，几百步小网格即可；室外加一层远景壳/天空球），之后只允许保持流形的操作：edge split / collapse / flip（Continuous Remeshing 的那套），顶点更新用 Laplacian 预条件（Large Steps），几何几百步收敛而且天然光滑。结果：无裂缝（顶点共享、流形操作不产生裂缝），闭合处永远闭合，连接关系用 Edgebreaker/Draco 类编码 ~2 bits/v（现在 29），方向 A 想解决的问题在这个形式化里不存在。

### 4.3 两种 primitive 对应两种物理 regime
实体表面 = 不透明流形 mesh（4.2）；薄片 / 亚像素 / 半透明结构（植被、毛发、栏杆）= alpha-textured cards —— 游戏引擎几十年来就是这么表示植被的，照样在任何引擎里光栅化。反射透射先不碰（SH 视角项兜底）。这一条是回收 2–3 dB class gap 的地方（第 3 节），代价是"纯 mesh"要放宽到"引擎标准资产"。

### 4.4 一条法则
split / collapse / 加 card / 纹理分辨率升级，统一定价：ΔD_consistent > λ · ΔR（多视角一致残差换字节）。平的地方没有一致残差 → 不细分 → 平面是输出；植被处不透明细分买不到证据 → card 更便宜 → card 是输出。这条法则是这篇论文的原理在新表示上的延续，不是推倒重来。

## 5. 快在哪里
三角形数量 ÷3..÷10（纹理承担细节）→ 光栅成本降；几何用预条件优化几百步收敛（不是 30000）；外观给定几何后接近线性问题（我们 400 步 shared fit 已经在用这个事实）；不再需要 4x supersampling 去逼细节（纹理 + 解析抗锯齿）。目标量级：分钟级，而不是现在的 30–50 分钟。

## 6. 难点（诚实）
- 硬边不透明 mesh 的可见性梯度只在边缘上：要么保留 MeshSplatting 的 sigma 退火，要么用 nvdiffrast 式抗锯齿梯度；两者都成熟，但和流形操作的耦合要重做。
- 拓扑变化（孔洞出现、物体分离）靠梯度做不到：闭合初始化要足够好；初始化错的拓扑后期难修 —— 这是 SDF 类方法赢的原因，也是 4.2 的最大风险。
- 没有 cards 的纯不透明版本在室外仍会有 ~2 dB 的 floor（第 3 节）。
- 无 UV 纹理的跨面滤波/压缩要自己写；一整套新的 CUDA 光栅器 + 流形操作 + 纹理：按月计。

## 7. 已有工作与我们的差异
nvdiffrec / FlexiCubes / DMTet（watertight 可微等值面 + 纹理，但网格受 grid 限制、不自适应、无定价）；Continuous Remeshing（流形操作，无外观）；Large Steps（预条件）；2DGS / GOF / Gaussian surfels（体积训练得到好表面）；SuGaR / MiLo / GaMeS（mesh 当骨架，渲染仍是高斯）；PlanarSplatting（平面 primitive，分钟级）；Adaptive Shells（表面处用壳、模糊处用体积 —— 两 regime 思想的先例）；BakedSDF / MobileNeRF / BOG（烘焙到 mesh 的部署路线）；Triangle Splatting(+) / MeshSplatting（直接优化不透明三角形）。没有人做的：**一条定价法则驱动的、流形 + 面上纹理 + cards 的引擎标准资产，尺寸和平面度是输出。** 这是 +++ 的论点。

## 8. 现在就能做的两个探针（不等下一篇）
P1 **平面度统计**（CPU，现有 checkpoint，分钟级）：把面按法向和偏移聚成平面区域，报告"落在 ε 内平面区域的顶点比例"和"这些区域若用平面 + 纹理需要多少三角形"。判据：≥ 40 % 的顶点在平面区域 → 轴 4.1 + 4.2 至少值 2x 顶点。
P2 **面上纹理重拟合**（1–2 GPU-h + 一天代码，纯 PyTorch，用现有渲染器的 id map + 投影重建重心坐标）：冻结 λ6.2e-3 的 mesh（440k v），把每顶点 SH 换成每面 4×4 纹素，拟合 1k 步；和 λ1.85e-3 的 mesh（1.3M v，每顶点 SH）比等字节 PSNR。判据：粗 mesh + 纹理把两者之间 0.6 dB 的差距缩小一半以上 → 多出来的顶点确实是在买颜色，轴 4.1 成立，+++ 的第一张图就有了。
两者都可以进当前论文的分析节（"where the bytes go"），不影响截稿。

## 9. 时间线
CVPR 这篇：原理 + codec + 分析，探针 P1/P2 作为分析节材料。+++：截稿后立项，先做 4.1（纹理）和 4.4（定价）在 MeshSplatting 现有光栅器上的版本（最小改动能出结果），再做 4.2/4.3。

## 10. 探针结果（2026-10-30，analysis/commands/planarity-v1b.md、texrefit-v1.md）
**P1 面级（failed）**：面级平面只能找到碎条，内部顶点 ≤ 12.7 %，相邻三角形二面角 < 10° 室内只有 42–52 %：mesh 在三角形尺度上是大角度瓦片。
**P1b 顶点级（passed）**：忽略连接关系，ε 0.5 % / θ 10° 下，≥ 1000 点的平面里的顶点占比 kitchen 75、garden 59（地面一块 42）、room 54、truck 53、train 51、counter 50、bonsai 46、bicycle 42、stump 40、treehill 32、flowers 16 %；λ1.85e-3 和 dense 相同或更高。世界是平的、mesh 是皱的：顶点在编码褶皱，不是形状。
**P2 冻结 mesh 上的颜色重参数化（opaque nvdiffrast，room + kitchen × 2 λ）**，两条可靠的结论：
1. **训练出的 soup 不是 opaque 表面。** 原样 SH3 换成硬三角形 + z-buffer：Room test 25.55 vs soft 27.56（−2.0 dB），kitchen 22.03 vs 25.61（−3.6 dB，平面像素 18.1 dB）。质量有 2–4 dB 来自重叠瓦片的 soft 混合。这是 +++ 轴 4.2/4.3 最硬的动机：引擎里的 opaque mesh 和 MeshSplatting 之间差的不只是精度。
2. **在这个密度上颜色不是顶点的理由。** 每顶点 RGB ≈ 2×2 ≈ 4×4 每面纹理（0.2 dB 内），1.0–2.6 M 面摊在 1.6 Mpx 上每面约 1 px，纹理分辨不出来；粗 mesh + 4×4 纹理 25.27 < 细 mesh + 每顶点 RGB 25.44 < 细 mesh 原样 26.05：多出的 2.7x 顶点买的是像素尺度的几何 / 轮廓 / 混合结构。
**对 §4 的修正**：4.1（纹理）不能脱离 4.2（更粗、更大的面）单独成立，§8 的 P2 判据按原设计无法检验；正确的检验是 P2b：用 P1b 的平面标签把平面区域铺成粗网格（中位边长 2/4/8 倍）再贴纹理，和原 mesh 在同一 opaque 渲染器里比。用户 10/30 决定：先回 CVPR 论文，P2b 留到截稿后。
