# "曲面 + alpha 卡片"介绍（2026-11-05，+++ 线，plan/meshsplat-plus-2026-10-29.md §4.3 的展开）

前提回顾（P2b lego，analysis/diag/results-p2b-lego-v2.md）：soup 的画质来自**多层半透明叠加**，不是顶点密度也不是颜色分辨率。把 soup 压成一张不透明曲面，再怎么细分、再怎么加纹理，train PSNR 都停在 26-27，而 soup 自己是 33。所以"下一代表示"必须保留"层"，但要比 soup 便宜得多、而且是引擎能直接吃的东西。

## 1. 是什么

一个场景 = 两种 primitive：

| | 实体部分 | 薄 / 亚像素 / 半透明部分 |
|---|---|---|
| 表示 | 不透明流形 mesh（闭合、顶点共享、面上纹理或顶点色） | **alpha 卡片**：带 RGBA 纹理的四边形（或少量三角形），alpha 通道决定哪里是空的、哪里半透 |
| 渲染 | 普通 z-buffer | 先画 mesh，再按深度从远到近把卡片 alpha-blend 上去（或 alpha-test：alpha > 0.5 才画，这样不用排序） |
| 例子 | 墙、地、桌面、lego 的大块、车身 | 树叶、草、栏杆、电线、毛发、物体轮廓的软边、lego 的 stud 顶面高光环 |
| 对应 | BakedSDF / BOG 那类 watertight mesh 在室内已经和 MeshSplatting 打平 | 游戏引擎几十年来的植被 billboard / foliage card；MobileNeRF 整个场景都是这种卡片（binary alpha，lego 34.18 dB） |

一句话：soup 里"几万个半透明小三角形叠成一团"的地方，换成"几张带 alpha 纹理的大卡片"；soup 里"几万个三角形铺一个面"的地方，换成"一张流形曲面"。两者都是标准资产（glTF 里就是 alphaMode = OPAQUE / MASK / BLEND 的两种材质）。

## 2. 为什么它能收回 class gap

- 第 3 节基线数字：watertight mesh 室内 ≈ MeshSplatting，室外（植被）差 2-3 dB；允许体积/分层（Radiance Meshes、3DGS）就追到 3DGS。差的那 2-3 dB 是"单一不透明 primitive"的代价，不是"mesh"的代价。
- P2b 把同一件事在 lego 上量出来了：不透明化 −2.9 dB，单层化再 −5 dB。卡片就是"把层留下但只留必要的几层"。
- 一张 64×64 RGBA 卡片 = 16 KB（8 bit，未压缩），能替代 soup 里几百到几千个三角形（每个 ~20 B 几何 + 颜色），而且纹理走图像编码，比顶点 SH 压得更狠。

## 3. 怎么从现在的 soup 走过去（两条路）

**路 A，分解（不训练，用现成 checkpoint，和 P2b 同型）**
1. 曲面：surf_extract_v2 那条链（soup → median depth → TSDF → 流形 mesh）。
2. 残差：渲染"曲面 + soup"时，soup 里对最终像素仍有贡献的三角形（被曲面遮不住、或在曲面前面半透明）就是"残差面"；lego 上预计是轮廓边缘、履带缝隙、stud 顶部。
3. 把残差面聚成平面小块（按法线 + 位置 PCA / RANSAC），每块变一张卡片：平面 = 四边形，RGBA 纹理 = 把这块 soup 在正交投影下烘焙出来（alpha = 累积不透明度）。
4. 联合微调：曲面顶点色/纹理 + 卡片纹理 + 卡片位置，nvdiffrast 不透明画曲面，卡片 alpha-blend，算 test PSNR vs 卡片数。
   最便宜的探针（0.3 GPU-h）是第 2 步之前的"曲面 + 原样 soup 残差面"：只看曲面接管了多少、残差还要多少面才回到 30-33 dB。残差面占比就是卡片要承担的份额，也是这条路值不值的判据。

**路 B，从头训（下一篇论文的正文）**
初始化一个闭合 mesh（4.2：SfM → 粗 SDF / 2DGS→TSDF），只做流形操作；同时允许生成卡片 primitive（位置 + 法线 + 尺寸 + RGBA 纹理），两者一起可微渲染。定价法则（4.4）统一：split 一个面、加一张卡片、提高一张纹理分辨率，都是"一致残差下降 ΔD > λ·ΔR 才做"。这条路的难点在 §6：卡片的可见性梯度只在 alpha 边缘，需要软光栅或 soup 当 finder。

## 4. 代价 / 风险（诚实）

- 不再是"纯 mesh"，是"mesh + 卡片"的引擎资产；审稿人会问"那和 MobileNeRF 的区别"：区别是**大部分像素由真曲面负责**，卡片只放在真需要分层的地方，所以字节是 MobileNeRF 的几十分之一（它 lego 199 MB 全是卡片）。
- alpha-blend 要排序（引擎里按卡片中心深度排，有穿插伪影）；alpha-test 不用排序但没有半透明，软边会锯齿。两种模式都要测。
- lego 这种实体物体，soup 的"层"多半不是物理上的半透明，而是对几何误差 / 亚像素覆盖 / 抗锯齿的补偿。卡片能接住它，但更干净的答案可能是"曲面更准 + 轮廓 alpha 卡"，残差面占比探针会告诉我们是哪种。
- 工程量：烘焙 + 卡片参数化 + 联合微调 + 导出，比 P2b 大一个量级；是下一篇的核心，不是 CVPR 这篇的内容。

## 5. 和已有工作的位置
MobileNeRF（CVPR'23，全卡片，binary alpha）；SNeRG / MERF（体素/三平面 + 小 MLP，不是资产）；BakedSDF / BOG（纯曲面，室外掉）；SuGaR / Mani-GS / GaMeS（高斯绑在 mesh 上，渲染仍要 splat，不是引擎资产）；Triangle Splatting+（半连接不透明三角形）。"曲面 + 少量 alpha 卡片 + 一条定价法则"这个组合没有看到，但卡片本身不是新东西。
