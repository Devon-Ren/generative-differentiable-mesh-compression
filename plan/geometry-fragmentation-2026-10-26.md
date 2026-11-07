# 几何碎片 (triangle-soup) 问题：诊断、与论文的关系、实验清单 (2026-10-26)

触发：用户在 viewer 里离开训练轨迹观察 Room λ6.2e-3 资产，看到钢琴/桌子由悬浮碎片组成，三角形没有贴合 surface，而是以某个角度"卡"在训练视角里正好看起来对；并问能否让 mesh 具有物理属性（bounded）。

## 1. 截图里其实是三件事

| 现象 | 原因 | 是不是我们的压缩造成的 |
|---|---|---|
| 悬浮碎片、斜插的薄片 (view-dependent geometry) | NVS 训练的 opaque triangle soup 的固有性质：只有 photometric loss 约束，三角形只需在训练视角的方向锥里正确。MeshSplatting 上游**已经**带了弱几何先验：2DGS 式 rend_normal/surf_normal 一致性 (λ 5e-5，iteration_mesh 之后)、单目法线监督 (λ 0.01，batch A/B 的 sbatch 都跑了 extract_normals)、vertex depth 正则 (2.5e-4)、sigma 从 1 退火到 1e-4 (soft→hard)。碎片是在这些先验下的 photometric 最优解，不是缺少先验。 | 不是来源，但 73× 点顶点数只有原始的 1/7，每块碎片更大、更显眼。"rate control 是否让每个顶点的碎片化更严重"目前**没有测过**（E2）。 |
| 黑色块 | SH degree 3 在训练中没见过的方向上外推，颜色为负/被 clamp 成黑；是 appearance 问题，不是几何。 | 与压缩无关 (codebook 是对 SH 向量的量化，不改变方向行为)。viewer 右上 "SH degree" 下拉切到 0/1 可以立刻分离这两类现象 (E1)。 |
| 边缘锯齿、细缝 | viewer 是硬光栅 (sigma≈0，与训练末期一致)，关闭 back-face culling，无 anti-aliasing；训练渲染器有 4× supersampling。 | viewer 侧差异，只影响观感，不影响论文指标 (指标来自上游渲染器)。 |

## 2. 这是不是本文要解决的问题

不是，但不能回避：
* 论文定位 = "压缩给定的 differentiable opaque mesh 表示"（story v3）；几何保真是 base method 的责任。审稿人按 test view 指标判断，qualitative figure 也在 test view 上做（标准做法）。
* 必须做的两件事：(a) Limitations 里一句话："the delivered asset inherits MeshSplatting's view-dependent geometry; away from the capture trajectory both the original and the compressed mesh show disconnected fragments"，配一张 supp 图：同一离轨相机下 original vs λ1.85e-3 vs λ6.2e-3 并排（E1 截图即可）；(b) 分析节加一列几何统计，证明 rate control 没有加重碎片化 (E2)。
* "让 mesh 有物理属性 / bounded" 有三种含义，代价完全不同：
  1. **三角形尺寸/形状有界**（无 sliver、最小角 > θ、边长 < L）：便宜，可以放进 growth/prune 的接受准则（E4b）；对碎片感帮助有限（碎片主要是位置/朝向错，不是形状差）。
  2. **顶点位置有界于某个 surface**（SDF 零集附近、或 coarse surface 的薄壳内，如 MiLo/BakedSDF/Neuralangelo→mesh）：这是换表示。按 10/19 的 class-gap 分析，表面先验越强、训练轨迹上的 PSNR 越低；收益只在离轨鲁棒性和几何度量 (Chamfer) 上体现，而现有 benchmark (Mip-360/T&T test view) 不奖励它。需要 DTU/T&T 几何 GT 的新评测、新训练 loop，≥ 10 GPU-h 起。**下一篇的题目**，不是 11 月这篇。
  3. **流形/水密**：restricted Delaunay 已给出近似流形的 surface 子集；水密性对渲染与压缩都无直接收益，不做。

## 3. 实验清单（按性价比排序，GPU-hour 已标）

| # | 内容 | 成本 | 产出 / 判据 |
|---|---|---|---|
| E1 | viewer 同一机位：SH degree 3→0 对比；再导出 Room **original** (ckpt 路线，desktop only) 与 level-1.0 交付资产 (31 MB) 到 .msv，同机位截图 original / λ1.85e-3 / λ6.2e-3 | 0 GPU，CPU 5 min + 手动 10 min | 黑块是否随 SH deg 消失；original 是否同样碎。预期：是。→ supp 图 + limitation 句 |
| E2 | 对已有资产做几何统计 (CPU 脚本，11 场景 × {original level-1.0, λ1.85e-3, λ6.2e-3} + Room collapse 各 level)：sliver 比例 (最小角 < 5°)、相邻面二面角分布、连通分量数/大小、`views_seen==0` 的顶点/面比例及其字节占比 (footprint npz 已有 f_views)、面积分布 | 0 GPU，~1 h CPU | 表格："在相同顶点数下 λ-grown vs collapsed 的碎片统计"。若 λ 资产的 sliver/零可见面比例 ≤ original → "rate control does not increase fragmentation per vertex" 一句话进分析节 |
| E3 | 若 E2 发现 λ 资产仍含不少 `views_seen==0` 的面：在最后一轮 phase-B prune 显式移除零 footprint 面（ΔD=0，纯 rate） | 1 × 26 min (Room λ6.2e-3) | 字节下降、PSNR 不变 → 免费；理论上 phase-B 的 residual-mass 准则已经先删它们，所以先看 E2 再决定 |
| E4a | 上游已有先验的强度消融：lambda_normals ×10、lambda_normals_super ×10，λ 不变 | 2 × 26 min (Room) | test PSNR / MB / E2 统计三者一起看；若 sliver、二面角明显改善且 PSNR 损失 < 0.1 dB → 一句 "compatible with stronger geometric priors"；否则放弃。注意用户 10/13 规则：不为小增益堆方法，这是分析不是新模块 |
| E4b | sliver-aware prune：collapse 候选的代价加入形状项（产生最小角 < θ 的 collapse 拒绝/加价），放在 rd_prune.select_collapses | 1 × 26 min (Room) | 只在 E2 显示 λ 资产 sliver 比例显著高于 original 时才做 |
| E5 | bounded 表示（SDF 锚定顶点 / 薄壳约束 / 2DGS→mesh 再 MeshSplatting 微调）+ 几何评测 (DTU Chamfer) | ≥ 10 GPU-h + 新代码 | Future work；作为下一篇 "geometry-bounded rate-controlled mesh" 的方向写进 conclusion |

建议：E1 今天做（只要机位截图），E2 我来写脚本（CPU，跑在你手里的节点），E3/E4 按 E2 的结果决定，最多 3 次 Room 运行 (~1.5 GPU-h)。E5 不进这篇。

## 4. 与 readiness v2 的顺序关系

不改变 10/25 的顺序：qualitative figure (test view) → related work/intro → Table 1 其它场景 → external baselines。E1/E2 并入 "analysis + limitations"，在 qualitative figure 之后、related work 之前完成即可。
