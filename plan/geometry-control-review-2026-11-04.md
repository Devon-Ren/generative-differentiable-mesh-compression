# 能不能在优化里"管住"MeshSplatting 的几何？（审查，2026-11-04）

问题（Wenhao）：能否在 MeshSplatting 生成/连接三角形时加控制，让它们尽量共面、减少 ill-formed / non-manifold，顺便治轮廓锯齿和模糊；大不了复杂处多 split 一点。这不一定非得等 +++。

## 1. 结论先行

- **理论上可以，上游已经在做**：restricted Delaunay 重建（iteration_mesh）、rend/surf normal 一致性（2DGS 式）、单目法线监督（Metric3D）、vertex-depth 正则、sigma 从软退火到硬。lego 上看到的毛边和缝隙碎片，是**在这些先验之下**的 photometric 最优解，不是没加先验。
- **这个项目里已经有三次"加几何约束"的实测，方向一致：约束越强，PSNR 越低**（都是 Room，10/26-10/30）：
  | 约束 | 代价 |
  |---|---|
  | 把拓扑钉在 Delaunay 上（v4/v5，decoder-side topology） | −1.3 … −1.8 dB |
  | 把训练好的 soup 当真正的不透明面渲染（无软混合） | room −2.0 / kitchen −3.6 dB |
  | 只做 snap + freeze（最弱的一种） | +0.19 dB / −1.5 % bytes（单场景） |
  读法：soup 的自由度**就是**它的画质来源；要它更像一张"好网格"，就要还回去画质。所以"加控让它追平 3DGS"方向是反的：加控会拉大差距，不是缩小。
- **"复杂处多 split"不是杠杆**：Room 2.9 M 顶点的 dense 模型 train/test 差 2.5 dB，226 k 顶点只差 1.4 dB；lego 14k→30k 一路涨 train 掉 test。100 张视角下多加三角形是多拟合训练视角，test 不涨。λ sweep 已经覆盖了"多三角形"这个轴。
- **归属**：让 mesh 有界、流形、物理正确 = Yao 说的"only when you make all of it"= +++ 的 4.2（manifold by construction）/ 4.3（轮廓用 alpha cards）。CVPR 这条线 10/30 已定稿为"压缩 MeshSplatting"，改它的训练目标就变成另一篇方法论文，而且 FP32 参照也得跟着改。

## 2. 三类误差分别能不能治（lego 渲染诊断对应）

| 误差 | 机理 | 能加的控制 | 预期 | 归属 |
|---|---|---|---|---|
| 轮廓毛边（底板前沿） | 不透明三角形在轮廓处非 0 即 1，4× supersampling 只能近似覆盖率；碎片为了拟合抗锯齿边缘伸出去 | (a) 更高 eval supersampling（8×）：纯渲染，0.1 GPU-h；(b) mask/alpha loss（NeRF-Synthetic 自带 alpha）：要改训练循环，且 3DGS baseline 没用 mask，审稿人会问；(c) alpha cards | (a) 0-0.2 dB；(b) 未知，边缘明显改善但是新监督 | (a) CVPR 可做；(b) 不建议；(c) +++ |
| 履带下缝隙乱长碎片 | 少视角可见、无法线先验 → 几何无约束 | 恢复作者协议里的 Metric3D 法线先验（Blender 分支只是没加载），0.5 GPU-h | 0.2-0.5 dB，缝隙类误差收敛；这是**恢复协议**不是新方法 | CVPR 可做 |
| 履带齿 / stud 高光糊 | 逐顶点 SH + 重心插值，一个 stud 只摊几个顶点 | 多三角形（λ 更小）或 per-face texture | 多三角形在 100 视角下过拟合；texture 在 ~1 px/face 时无用（+++ P2） | +++（4.1） |
| non-manifold / sliver（他问的） | soup 构造上允许；Delaunay 之后的 split/prune 不保证流形 | 先**量**：mesh_stats_v1.py（CPU）数 non-manifold 边、sliver、连通块；再决定要不要 sliver-aware prune（E4b） | 若 λ 资产比 original 更碎 → 一句话 + E4b；否则不动 | 统计进 CVPR 分析节；约束本身 +++ |

## 3. 建议（按成本，lego，8 场景跑完后）

1. **CPU 现在就能做**：`analysis/diag/mesh_stats_v1.py` 跑 lego 21k（未剪枝）、lego 30k（剪枝后）、Room indoor 30k 对照 → 一张表回答"有多 ill-formed"。粘贴块见 analysis/commands/nerf-synthetic-v1.md §9。
2. **0.1 GPU-h**：lego 21k 用 render_views 在全部 200 test 视角上比 supersample 2 / 4 / 8，看锯齿占多少。
3. **0.5 GPU-h**：Blender 版法线先验（Metric3D 跑 train PNG → normals/，train_ckpt_v3 在 loader 回调里挂上 normal_map），lego 重训一次，看缝隙碎片和 PSNR。
4. 不做：mask loss、平面/流形正则、"多 split"。理由见 §1。

## 4. 给审稿人 / 给 Yao 的一句话

"The remaining 2-3 dB to 3DGS on object scenes is the cost of a single opaque primitive at silhouettes and of per-vertex shading; every geometric constraint we tested on this representation (Delaunay-pinned topology, opaque compositing) lowered PSNR further. Making the mesh bounded and manifold without that loss is the next paper's problem, not a hyperparameter."

## 5. 几何统计结果（mesh_stats_v2，2026-11-04 12:32）

| 模型 | V / F | 边：boundary / manifold / non-manifold | 面触 non-manifold 边 | sliver <5° / <10° | 二面角中位数 / >30° | 连通块（≤10 v 的） |
|---|---|---|---|---|---|---|
| lego 21k 未剪枝 | 508k / 2.98M | 5.9 / 31.6 / **62.6 %** | 97.9 % | 14.7 / 30.1 % | 30.9° / 51 % | 141 (126) |
| lego 30k 剪枝后 | 457k / 1.32M | 38.0 / 33.8 / 28.2 % | 84.5 % | 9.5 / 26.1 % | 24.2° / 43 % | 1441 (1272) |
| Room indoor 30k（作者协议原模型） | 2.55M / 5.49M | 46.1 / 35.6 / 18.4 % | 69.2 % | 7.1 / 21.3 % | 16.8° / 34 % | 7379 (6143) |
| Room λ 6.2e-3（我们的 72× 资产） | 437k / 1.16M | 36.7 / 38.9 / 24.4 % | 78.2 % | 9.5 / 23.4 % | 18.4° / 36 % | 468 (350) |

读法：
- **每个模型（包括作者自己的原模型）只有 1/3 的边是流形边**，其余 2/3 是 boundary 或 non-manifold；70-98 % 的面至少有一条 non-manifold 边。这不是"有一些瑕疵"，而是这个表示的结构：multi-layer shingle soup。正则化改不了 2/3，只能靠构造（+++ 4.2）。
- 21k→30k 的 importance 剪枝删掉 56 % 的面后，non-manifold 从 63 % 降到 28 %，boundary 从 6 % 升到 38 %：被删的是叠在表面下的层，删掉以后留下的是带洞的壳。
- 我们的 λ 资产 vs 原模型（Room）：non-manifold 24 vs 18 %，sliver 23 vs 21 %，二面角 18 vs 17°，连通块 468 vs 7379（少 6× 顶点，碎块也少了 16×）→ **rate control 没有让 soup 更碎**，这是 E2 要的那句话，可以进 CVPR 分析节。
- lego 比 Room 更"尖"：sliver 30 % vs 21 %，二面角中位 24-31° vs 17°，细结构（齿轮、stud）本来就该如此。

CVPR 可用：11 场景 × {original, λ 1.85e-3, λ 6.2e-3} 跑同一脚本（CPU，~30 min）做一张"压缩前后几何统计"表 + limitations 里一句"the asset is a shingled triangle soup (≈1/3 manifold edges), inherited from MeshSplatting"。
