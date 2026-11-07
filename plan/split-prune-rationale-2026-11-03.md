# Split / prune 规则的来历、ΔD_uv 的含义、B_v / B_f 为什么是常数（2026-11-03）

> **记号更新（11/03 晚）**：论文和 report-latex-v2 只用两个互相独立的常数：B_a = 6 B（一个顶点的属性：位置残差 + DC + SH 索引），B_f = 2 B（一个面的连接字节）。split ΔR_t = 3B_a + 6B_f = 30 B；collapse ΔR_uv = B_a + n_uv·B_f = 10 B（内部边）；ΔD_uv 写成两个 1-ring 并集上的求和。下文里的 B_v（= B_a + 2B_f = 10 B）是旧记号，数值不变。


代码依据：analysis/rdgrowth/rd_growth_v2.py（split）、rd_prune.py（prune）、train_rd_v3.py；所有 v3 run 的 sbatch 都是 BPV=10.0、BPF=2.0、MU=1.0、MAXF=0.15（commands/rd-hull6-v1*.md、rd-indoor-v1.md、batchB-rd3-v1.md）。

## 1. 两条规则从哪里来：一个拉格朗日目标的边际决策

目标：在码率约束下让失真最小，等价于最小化 J = D + λR（Everett 1963；Shoham & Gersho 1988；视频编码的 RDO，Sullivan & Wiegand 1998）。
网格的每一次离散操作（切一个三角形 / 合并一条边）都会让 D 和 R 各变一点。只做让 J 变小的操作：

- split：D 降 ΔD_t，R 涨 ΔR_t。ΔJ = −ΔD_t + λΔR_t < 0 ⇔ ΔD_t / ΔR_t > λ。
- collapse：R 降 ΔR_uv，D 涨 ΔD_uv。ΔJ = ΔD_uv − λΔR_uv < 0 ⇔ ΔD_uv / ΔR_uv < λ。

两条规则是同一个不等式的两个方向，λ 是"每字节值多少失真"的汇率。
最优分配的等斜率条件：被接受的操作斜率都 ≥ λ，被拒绝的都 ≤ λ。所以同一个 λ 在不同场景停在不同的顶点数：场景的边际收益掉到 λ 以下时生长自己停。这就是"预算由场景内容决定、不是手设上限"的依据。

两个都是 proxy（审稿人一定问，要主动写）：
- split 前不知道切了之后误差能降多少（要重新优化才知道）。用"当前归到这个三角形上的残差质量"代替：它是切分最多能消掉的误差的上界，3DGS 的 densification 用累计梯度也是同类代理。
- collapse 后误差会涨多少也不知道。用"受影响的面上现在的残差"代替：这些面已经拟合得很好（残差小）时，合并后剩下的 12k 次优化能把局部修回来。这条 proxy 比 split 那条弱，它的依据是实验：四轮 prune 删掉 Delaunay 后 12–22 % 的顶点，字节 −4…−8 %，PSNR +0.01…+0.02 dB（11 场景）。

## 2. ΔD_uv = D_u + D_v − D_uv 每一项是什么

每个面 f 有一个残差质量 d_f（和 split 用的同一个量：每视角、每 Mpx 的 L1，按 rend_ids 归属）。

| 项 | 定义 | 图里的例子 |
|---|---|---|
| D_u | 以 u 为顶点的所有面的 d_f 之和（u 的 1-ring） | u 周围 6 个面 |
| D_v | 以 v 为顶点的所有面的 d_f 之和（v 的 1-ring） | v 周围 6 个面 |
| D_uv | 同时含 u 和 v 的面，即边 uv 两侧的面 | 2 个红色面 |

边上的 2 个面同时在 u 的环和 v 的环里，D_u + D_v 把它们算了两次，所以减一次 D_uv。结果 = 两个环的并集 = 10 个面，正好是 collapse 会改动的全部面（8 个变形、2 个消失）。为什么用并集：只有这些面的形状会因为 u、v 移到中点而改变，别的面不受影响。

## 3. ΔR 为什么是这个样子

交付文件里，一个顶点的字节 = 属性（位置残差 + DC + SH 索引）+ 拓扑；拓扑按面计（连接性流按面编码）。记每面 B_f，每顶点总价 B_v（含平均 2 个面的拓扑）。

- split：1 个三角形 → 4 个，新增 3 个顶点（边中点）。ΔR_t = 3B_v = 30 B。新增的面已包含在 B_v 里（每个新顶点约带 2 个面），所以不另加 B_f。
- collapse：删 1 个顶点的属性 (B_v − 2B_f)，再删掉边上的面 n_uv × B_f。内部边 n_uv = 2 → ΔR = B_v = 10 B；边界边 1 面 → 8 B；非流形边 3 面 → 12 B。

## 4. B_v = 10 B、B_f = 2 B 怎么定的，为什么不逐场景

来源：10/18 设计 v1 时用 Room（q14）的码流预算：拓扑约 4 B + 位置 2–2.6 B + DC 约 1.5 B + 索引约 2.4 B ≈ 10 B/顶点；拓扑 4 B/顶点 ÷ 约 2 面/顶点 ≈ 2 B/面。之后所有场景、所有 λ 都用这两个数。

事后核对（44 个交付资产，扣掉固定 0.61 MB 码本/头部后的每顶点字节）：

| λ | B/V 中位数 | 范围（11 场景） |
|---|---:|---|
| 6.2e-4 | 10.4 | 9.5–12.6 |
| 1.85e-3 | 10.8 | 9.8–13.1 |
| 6.2e-3 | 11.0 | 10.1–13.3 |
| 1.85e-2 | 11.4 | 10.4–12.9 |

全部 44 个：9.5–13.3 B，中位数 10.8 B；最高是 kitchen（面/顶点 3.1–3.4，比别的场景多拓扑）。面/顶点 1.8–3.4，中位数 2.3。

为什么用常数（给审稿人的三条）：
1. 价格必须在训练前就知道。真实 B_v 要训练完再编码才知道，逐场景测量 = 每个场景先完整训一遍，成本翻倍，而且测出来的 B_v 又依赖 λ（上表：λ 越大每顶点越贵）。
2. 规则是 ΔD > λ·3B_v，换成场景自己的 B_v 只等于把该场景的 λ 乘上 B_v/10。实测这个倍数在 0.95–1.33 之间，而我们的 λ 网格每档差 3 倍：最坏（kitchen）也只是在自己的 RD 曲线上挪不到 1/4 档，曲线本身（扫 λ 得到的 hull）不变。
3. 同一个常数让 λ 在所有场景里是同一个单位："同一个 λ = 同样的每字节 L1 汇率"。逐场景改价格会破坏"同一 λ、场景决定预算"的可比性。

可补的实验（若要堵这个问题）：B_v ∈ {8, 10, 13} × 2 场景（kitchen 最偏、garden 居中）× 1 个 λ，6 run ≈ 6 GPU-h，预期三点落在同一条 RD 曲线上、只在曲线上平移。

## 5. 论文可用的一段（英文）

We treat every mesh edit as a discrete decision in the Lagrangian $J = D + \lambda R$. A split removes distortion $\Delta D_t$ at a cost of $\Delta R_t$ bytes and is accepted iff it lowers $J$, i.e. $\Delta D_t/\Delta R_t > \lambda$; an edge collapse saves $\Delta R_{uv}$ bytes at a distortion cost $\Delta D_{uv}$ and is accepted iff $\Delta D_{uv}/\Delta R_{uv} < \lambda$. Because the true distortion change of an edit is only known after re-optimisation, we use the residual currently attributed to the affected faces as its proxy: the triangle's own residual for a split (an upper bound on what refining it can remove), and the residual of the union of the two vertex rings, $D_u + D_v - D_{uv}$, for a collapse (the faces whose shape the collapse changes; the two faces on the edge are counted once). Byte prices are fixed constants of the codec, $B_v = 10$\,B per vertex including topology and $B_f = 2$\,B per face, so $\Delta R_t = 3B_v$ and $\Delta R_{uv} = (B_v - 2B_f) + n_{uv}B_f$. A per-scene price would only rescale $\lambda$: measured delivered bytes per vertex range from 9.5 to 13.3\,B over all 44 assets (median 10.8), a factor below one third of our $\lambda$ grid step, so a single constant keeps $\lambda$ a common exchange rate across scenes and lets the vertex budget follow scene content.
