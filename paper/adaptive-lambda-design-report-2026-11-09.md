# Adaptive-λ（size controller）整体设计与具体操作 —— 向 Yao 汇报用（2026-11-09）

作者：Claude，为 Devon Ren 整理。依据：`release-adapt-v9d/code/rdgrowth/train_rd_adapt_v9d.py`（v7b→v9d 全部版本头注释在文件顶部）、`rd_prune_v2.py`、`release-adapt-v9b/sbatch/adapt-v9b.sbatch`（实际跑的参数）、v9b sweep 分析表、Deep Blending one-shot 表，以及 Yao 在 `meshsplat_compression_Draft_1.pdf` 上的 28 条批注（11 月 5 日，已用 PyMuPDF 全部提取，见文末附录）。

---

## 0. 一句话

**固定-λ 版本（Yao 看的 Draft 1）回答的是"给一个 λ，网格长到多大"；adaptive 版本回答的是"给一个目标文件大小（MB），训练自己去找 λ 并命中它"。** 做法是把真实编码器放进训练循环当"字节表"，用它测出来的 size model 反推需要多少顶点，再把 λ 定义为"让网格恰好长到那么多顶点所需的价格"（rank price），训练后半段再用同一价格反向做边折叠（collapse）修正。结果：Mip-360 / T&T 11 场景 × {5, 10, 20} MB，一套设置，平均命中 1.007×，32/33 在 ±10 % 内，22/33 在 ±5 % 内，与固定-λ 曲线等字节 PSNR 差 −0.05 dB；没见过的 Deep Blending 数据集不改任何东西，命中 1.06–1.12×，质量在它自己的曲线上。

Yao 在 Draft 1 上最尖锐的三条批注（Ba/Bf 怎么定、会不会随码率变、给目标码率怎么知道 λ）正是这个系统直接回答的，见第 7 节。

---

## 1. 为什么需要它：固定-λ 版本的两个硬伤

Draft 1 的机制：训练中每个 densify 间隔，三角形 t 分裂 iff ΔD_t/ΔR_t > λ；ΔR_t = 3B_a + 6B_f = 30 B 是两个**训练前固定**的常数（B_a = 6 B 顶点属性，B_f = 2 B 每面连接）。

1. **用户拿不到想要的大小。** λ 是"每字节换多少 L1 误差"的汇率，4 个 λ 值在不同场景上停在 15–100 % 的稠密顶点数。给一个 10 MB 的要求，没有办法事先知道 λ。Yao 批注 #4 问的就是这个。
2. **价格常数是在 Room 上读一次然后到处用。** 44 个资产上实测 9.5–13.3 B/顶点（中位 10.8），常数给 10。Draft 1 的辩解是"差 1.33× 只是 λ 网格的四分之一步"，但 Yao 批注 #1 问"这些数会不会随码率变"——答案是**会**：实测 5 MB 时 11–12.8 B/顶点，10 MB 时 9–11，20 MB 时 7.7–10（固定开销被更多顶点分摊，属性流在稠密点集上编得更便宜）。常数先验在大目标上把顶点数低估 10–30 %。

adaptive 版本的设计目标就是消掉这两条：**大小变成输入；价格由编码器在训练中实测。**

---

## 2. 上游时间轴与两阶段结构（为什么是"先长后剪"）

MeshSplatting 上游的训练日程决定了控制器的形状（上游代码一行没改，`train.py` 的循环被复制进 `train_rd_adapt_*.py` 后在两处插入我们的决策）：

| 迭代 | 上游做什么 | 我们能做什么 |
|---|---|---|
| 0–1500 | 初始 triangle soup 训练 | 无 |
| 1500–9500，每 500 一次（共 17 个间隔） | 剪枝（opacity / size / importance）+ densify（分裂） | **Phase A：priced split**。这是唯一能**加**字节的时段 |
| 10000 | 生长结束 | — |
| 11000 | restricted Delaunay 重建：soup → 真正共享顶点的网格 | 无。此前所有拓扑信息都是 soup 的，不是交付网格的 |
| 12500–19500（v9b 每 500，15 轮；v9c/d 每 250，30 轮） | opacity floor 逐步抬高（ramp 到 `final_opacity_iter`） | **Phase B：λ-prune（边折叠）**。只能**减**字节 |
| 20000 / 25000 | 超采样开启（start_upsampling） | 必须在此之前结束 collapse（之后 Adam 状态不再对应） |
| 30000 | 最终 importance 清理（max blending ≤ 0.5 的面丢掉）、保存 | 最终 probe，然后真实 encoder 交付 |

所以结构是**不对称的两阶段**：Phase A 决定上限（之后没有任何机制加字节），Phase B 只能向下修。这是后面所有设计决策（包括 1.25 的 aim margin）的根源。

---

## 3. 字节表：size model + byte probe

### 3.1 size model

```
R(V, F) = b_v · V + b_f · F + overhead          (bytes)
```

- `b_v`：每个**交付**顶点的属性字节（位置残差 + DC 颜色 + 两个 SH 字典索引）
- `b_f`：每个面的连接字节
- `overhead`：两个 SH 字典（K4096 + R256）+ 文件头，约 0.6–0.7 MB，与 V 无关
- `fpv`：交付网格每顶点面数（Delaunay 后 2.2–2.9，Blender 3.4）

反函数给出"预算买得起多少顶点"：

```
V* = (target_bytes − overhead) / (b_v + fpv · b_f)
```

为什么是线性的（Devon 之前问过）：编码器的三条流本身就是逐顶点 / 逐面的熵编码，流大小对 V、F 近似线性；非线性全部收进 b_v、b_f 的**取值**里（它们随密度、场景变），而这两个值不是假设的，是每次 probe 重新测的。模型形式简单是故意的：只有三个可解释的量，每个都能被编码器直接读出来。

### 3.2 byte probe（`probe_bytes`）：真实编码器当测量仪

每次 probe 做的事：

1. 在**副本**上模拟上游的最终清理：把 opacity floor 临时设成最终值（0.9999），超采样设成最终值（4×），渲染全部训练视角，取每个面的最大 blending weight，≤ 0.5 的面丢掉，没人用的顶点丢掉。训练模型本身不动。
2. 把清理后的副本以 `coarse_finetune` best.pt 的格式存盘，**用交付时同一个编码器**（`probe_bytes_v1.py`，K4096/R256，场景的 XYZ 位数规则，DC q7）在子进程里真的编一遍。
3. 读回：总文件字节、每条流的字节（`payload_bytes`）、保留的 V/F、保留掩码（kept_v、kept_f）。
4. 代价：Blender 约 15 s，Mip-360 20–45 s；一次训练 7 + 15–30 次 probe，总共加 8–15 分钟。

probe 对最终交付的精度：probe/delivered = 0.999–1.001（49 次全部）。也就是说**字节表没有误差，误差全在"现在的网格到 30k 会变成什么"这一步**。

### 3.3 两种更新

- **Phase A probe（`update_attributes`）**：此时模型是 soup（每面三个独占顶点），它的连接流说不出 Delaunay 之后网格的拓扑成本，所以**只取** b_v 和 overhead；b_f、fpv 保持先验直到 12k 的第一次网格 probe。保留顶点 < 4096 时跳过（K-means 需要 K 个样本）。
- **Phase B probe（`update`）**：真网格，四个量全部换成实测。

---

## 4. Phase A 的每个 densify 间隔（1500…9500，共 17 次）具体做什么

按代码顺序，每 500 迭代一次：

1. **上游剪枝先跑**（opacity / size / importance；`start_pruning` 之后才生效），得到当前顶点数 V。
2. **测存活率 σ**（`SurvivalMeter`）：上一轮分裂新造的顶点，有多少活过了这次剪枝。前三次的均值，下限 0.2，没观测到之前用先验 0.5。
3. **Phase A probe**（上一次 ≥ 1000 迭代之前、且已过 start_pruning）→ 更新 b_v、overhead。
4. **重算目标顶点数** `V* = vertices_for(target × OVER)`，OVER 是 aim margin（Mip/T&T 1.25，Blender 1.1，见第 6 节）。
5. **剩余间隔数** `remaining = ceil((10000 − it)/500)`（v9 修的 off-by-one：之前多算一个，导致每次都把该长的量摊到多一个间隔上，最后一轮只长到 √(V·V*)）。
6. **几何生长计划**（`plan_splits`）：要在剩余间隔里从 V 等比长到 V*，这一轮应分裂多少三角形：

   ```
   f = (V*/V)^(1/remaining);  desired = V (f − 1) / (3 σ)
   ```
   （一次分裂加 3 个顶点，其中 σ 比例活下来。）再在上游的生长限制器 `add_percentage·V − V` 下**向前模拟**余下间隔；若模拟到底也到不了 0.995 V*，这一轮就要满限制器允许的量（front-load），因为生长是复利，现在分裂比以后分裂值钱。
7. **本轮上限** `max_split = min(上游限制器, ceil(desired))`。
8. **选哪些三角形分裂**（`rd_select`）：每个三角形的比值 `ratio_t = ΔD_t / bytes_per_split`，ΔD_t 是该间隔内所有训练视角中、前景面是 t 的像素 L1 误差之和，按每百万像素归一化、按视角平均（`dd_norm mpx`，分辩率无关）。ratio > λ_now 的三角形通过，再按 max_split 截断，分裂。
9. **记录实际通过数 n_pass、新顶点数**，更新 σ 的计数。
10. **为下一轮定价（λ 的核心定义）**：
    - 预期剪枝后顶点数 V_exp = V + σ·ΔV，算下一轮的 desired_next；
    - **yield** = 上次 n_pass / 上次要求数 k_asked（最近三次均值，夹在 0.2–5）——rank price 实际放行的和要求的差多少；
    - k_asked = ceil(desired_next / yield)；
    - **λ_next = rank_lambda(ratio, k_asked)：当前 ratio 分布中第 k 大的值**，夹在 [1e-4, 0.1]。即"在这个场景此刻的残差/字节分布里，恰好让 k 个三角形值得分裂的价格"。
    - v9b：最后一个间隔（9500）plan 已无可要求（k_asked = 0）时，**保留刚用过的 λ**，而不是回落到 lam_max。否则 Phase B 会从 λ = 0.1 起步，把所有边都判成"值得折叠"，渐进式 λ-prune 退化成一刀砍（batch12 日志里 "663k–7.9M edges below lam 1e-1" 就是这个）。
11. 写日志事件 `densify`（lam_used、lam_next、desired、cap、limiter、σ、V*、size_source …）。

**初始 λ**（`lam0_of`）只是第一个间隔用的猜测：`6.2e-3 · (9/T)^1.67`，从第一个间隔起就被 rank price 替代，结果对它不敏感（固定-λ ablation 显示 ±30 % 的价格扰动只重标 λ、不动 RD hull）。

**关于 bytes_per_split 常数**：Phase A 的分裂价 3(b_v + fpv·b_f) 用的是初始常数（6 + 2.5×2 → 33 B），**没有**随 probe 更新。原因：λ 是 ratio 分布的 rank，ΔR 的单位同时缩放 ratio 和 λ，抵消了。它只决定 λ 的量纲，不影响任何决策。这是 Yao 批注 #1/#6 的直接回答之一：在 adaptive 版本里 B_a、B_f 不再是决定"分裂与否"的物理价格，决策由 rank 决定；真正进入字节预算的 b_v、b_f 是实测的。

---

## 5. Phase B 的每一轮（12500…19500）具体做什么

每轮（v9b 15 轮，v9c/d 30 轮）：

1. **残差归因**：自上一轮以来每个面的 ΔD_f（同一个 accumulator、同一单位）。
2. **probe**：真实编码器编清理后的副本 → r_meas（现在交付会多大）、b_v/b_f/fpv/overhead 全部实测、kept_v/kept_f 掩码。
3. **外推落点**（`extrapolate_probe`）：对"虚拟 probe"（probe + 之前所有轮已折叠掉的字节，即不折叠会有多大）做最小二乘斜率，外推到 opacity ramp 结束（`final_opacity_iter`）。**v9c：斜率钳在 ≥ 0**——保留集在 opacity ramp 下只会长（un-stacking），负斜率只是 probe 的噪声；两三个点的负斜率外推 11k 步会读低 0.8 MB，让 gate 晚开 3–9 轮。
4. **本轮要折叠多少对**：excess = r_est − target；每对折叠省的字节 b_coll 用上一轮实测值（首轮先验 b_v + 2b_f）；excess_pairs = excess / b_coll；desired = excess_pairs / 剩余轮数（平摊）。
5. **给每条边打分** `score_uv = ΔD_uv / ΔR_uv`：ΔD_uv 是 u、v 两个顶点环上所有面的残差之和；**v9d：ΔR_uv = b_v·[u 和 v 都在 kept 集里] + b_f·(边上的 kept 面数)**——只对交付文件真会少的字节计价，ΔR = 0 的边（没人渲染的、最终清理反正会丢的）不是候选。v9b/v9c 时最便宜的边正是这些零残差零字节的边，一轮折 4–8k 对只省 0.002–0.03 MB，"在剪不花钱的几何"。
6. **两个阈值**：λ-阈值 thr_λ = λ_now·μ（μ = 1），数出 n_lam 条边低于它（这就是 Draft 1 的固定-λ prune）；rank 阈值 thr_rank 放行约 4×desired 条最便宜的边。取 max。
7. **Gate 与 cap**：`wanted = min(max(desired, n_lam), excess_pairs)` 若 excess > 0，否则 0。落点在目标以下→这一轮一对都不折（日志 mode `none`）；永远不折过目标。
8. **选独立集**（`select_collapses`）：最便宜优先、一个顶点一轮只参与一次折叠、会翻转幸存面的拒绝（foldover guard）、每轮 ≤ 15 % 顶点。若计数是约束（rank 或 cap 起作用），**λ := shadow price**（被折叠边中最大的 score / μ）——λ 跟着实际执行的价格走。
9. **执行折叠**（`collapse_edges`）：u、v 合到中点，参数、顶点 opacity（opacity 空间平均）、**Adam 一二阶矩**一起平均进 u 的槽位，v 走上游 `_prune_vertices` 路径删除，退化面、重复面丢掉。优化器状态保留，这是"训练中折叠"与"训练后折叠"的本质区别（G0 实验：训练后折叠到同顶点数差 0.3–6 dB）。
10. **精确字节核算**：removed_mb = (两端都 kept 的对数)·b_v + (被删的 kept 面数)·b_f，与 probe 的集合一致；记录 bytes_per_pair 供下一轮用。
11. 写日志事件 `prune`。

12500 之后到 30k：上游照常（超采样、opacity ramp、最终清理），我们只在 22000/26000 做诊断 probe（不动网格），最后在 30k 后做最终 probe 并交付。

---

## 6. 版本演进：每一版修了什么、靠什么证据（这是"理解根因"的部分）

| 版本 | 改动（每次只改一处） | 发现它的证据 |
|---|---|---|
| v7b | 常数先验 size model；Phase B 精确核算、gate、shadow price | 47 次 sweep：Mip/T&T 全部**低于**目标（5/10/20 MB：0.89/0.83/0.77×），只有 8/33 在 ±10 %。原因：先验 11 B/V 在大目标上太高→V* 太低，Phase B 无物可剪 |
| v8 | Phase A 用 soup probe 实测 b_v、overhead | b_v 在 9500 与 12k 网格 probe 差 1–8 %；但命中反而更低（0.71–0.92） |
| v9 | 剩余间隔数 off-by-one（17 不是 18）；限制器单位（三角形不是顶点，之前当成 1/3） | 所有 v4–v8 最后一轮 V 都停在 0.81–0.92 V*，等于 √(V_9500/V*) 预测值 |
| v9b | 最后间隔保留 λ（不回 lam_max）；aim margin OVER 1.25/1.1；prune 轮 12500–19500 | batch12 残差分解：hit = κ_end × (b_v,mesh + b_f·fpv)/(b_v,soup + 5)。κ_end（最终清理保留率）0.92–0.94，交付拓扑 3.7–4.4 B/V 对先验 5，soup 多读 b_v 3–9 %。预测 0.880/0.807/0.829，实测 0.877/0.790/0.829。三项在 11k Delaunay 前都不可观测，而 9500 后无法加字节，所以误差必须放在 Phase B 能修的一侧 |
| v9c | 外推斜率钳 ≥ 0；30 轮 | kitchen 10：probe 9.43 MB，估计 8.51/9.24/9.79，最终 11.34；gate 晚开 3–9 轮 |
| v9d | 折叠价只算 kept 顶点/面 | 一轮 4–8k 对只省 0.002–0.03 MB；both-kept 比例 0.35–0.57；每对 0.6–2.4 B 对先验 6.3 |

v9c、v9d 都按预期工作（估计不再读低；每对省的字节上去了），**但室内行的命中没有实质改善**（v9d：room 5 1.047、counter 5 1.046、kitchen 10 1.130、counter 10 1.070、kitchen 20 1.088；v9b 对应 1.061、1.092、1.135、1.079、1.100）。原因在第 8 节：剩余偏差来自一个 Phase B 开始之前不可观测的量。因此**headline 用 v9b 全表，v9c/v9d 作为机制 ablation 报告**，控制器线在 v9d 停止。

---

## 7. 现在还剩哪些常数，各自是什么性质（向 Yao 如实交代）

| 常数 | 值 | 性质 |
|---|---|---|
| bytes_per_vertex / bytes_per_face / faces_per_vertex / overhead 的初值 | 10 / 2 / 2.5 / 0.6 MB | **先验，第一次 probe 后被实测替换**。只影响前 2–3 个间隔（start_pruning 之前 probe 还不能跑） |
| σ 先验 | 0.5 | 同上，第一次观测到 cohort 剪枝后替换 |
| lam0 | 6.2e-3·(9/T)^1.67 | 第一个间隔的猜测，之后由 rank price 决定 |
| λ 夹断 [1e-4, 0.1] | | 数值安全边界；只在 ficus/mic 这类"没有残差可买"的场景碰到下界，这是模型容量限制，控制器会报告（growth_capped / limiter_bound） |
| μ、max_frac、rank margin | 1 / 0.15 / 4 | Phase B 的速度限制，不决定剪什么、剪多少（gate 和 excess 决定） |
| probe_floor、probe_scaling | 0.9999 / 4 | 等于上游的最终值，不是自由参数 |
| 编码器设置 K/R/DC/XYZ | 4096 / 256 / q7 / 场景规则 | 编解码器设定，由 RD 分析固定（K/R 在 11 场景上距联合 hull ≤ 0.16 dB） |
| **OVER（aim margin）** | **Mip/T&T 1.25，Blender 1.1** | **唯一真正的校准常数**，每数据集类一个值。它补偿的是 κ_end 和 11k 之后可见面集合的继续增长（室内 1.1–1.5×，室外 ~1.05×），这两样在 Phase A 结束时原则上不可观测（soup 的拓扑与 Delaunay 网格无关；可见性要等 opacity ramp）。paper 中如实写：calibrated on earlier runs, one value per capture class, reported |

Devon 问过"两个 fail 的常数、同一个问题"：指的就是 OVER 的两个取值（1.25 和 1.1）。它们是同一个不可观测量（un-stacking 增长）在两类数据上的经验值。试过的替代方案（在副本上提前跑 Delaunay 来观测拓扑）我们测过：fpv 从 12.5k 到最终仍长 +10–50 %（室内）/ +5 %（室外），也就是说即便提前拿到拓扑，可见面的增长仍不可见，所以 Delaunay-on-copy 不能去掉这个类偏差，没有做。

---

## 8. 结果与剩余偏差的根因

**v9b，Mip-360 / T&T，33 行，一套设置**

| 目标 | 命中范围 | 均值 | ±5 % | ±10 % | 等字节 ΔPSNR vs 固定-λ hull |
|---|---|---|---|---|---|
| 5 MB | 0.952–1.092 | 1.006 | 8/11 | 11/11 | −0.110 |
| 10 MB | 0.953–1.135 | 1.020 | 8/11 | 10/11 | −0.051 |
| 20 MB | 0.934–1.100 | 0.995 | 6/11 | 11/11 | +0.005 |
| 全部 | | 1.007 | 22/33 | 32/33 | −0.052 |

v7b 对比：均值 0.829，±10 % 只有 8/33。

**NeRF-Synthetic**：2.5 MB（去 ficus）0.934–1.092，均 1.026；4 MB（去 ficus/mic）0.897–1.034。ficus/mic 是容量限制：λ 从前几个间隔就在下界，上游限制器也绑住，场景没有残差可买。

**Deep Blending（没见过的数据集，一个字节不改）**：drjohnson/playroom × 5/10/20 MB 命中 1.056–1.117，等字节 ΔPSNR +0.04，Phase B 每行都在工作，Phase B 期间可见集增长 1.10–1.39×——与 Mip-360 室内（1.10–1.49×）同号同量级同原因。控制器在新数据上的行为与机制分析的预测一致。

**剩余偏差的两种机制（都从日志读出）**

- 室内偏高 5–13 %：第一次 probe 读 0.76–1.05×，之后保留集在 opacity ramp 下持续增长（最终/首次 probe 1.00–1.39），gate 晚开；即使 30 轮全开，Phase B 每轮只能折约 0.4 % 的 V（独立集 + foldover 限制），追不上。
- 室外 20 MB 偏低 4–7 %：最后一个间隔只放行了要求数的 88–92 %（yield 误差，之后没有间隔可修），加上室外 κ_end·拓扑 的隐含落点本来就只有 0.95–0.97。

两者的共同根：**一个在 Phase A 结束时不可观测、只能在 Phase B 单向修正的量**。彻底解决需要双向控制（Delaunay 之后再开一个生长窗），留作 future work，paper 的 limitation 段落如实写。

---

## 9. Yao 的 28 条批注逐条对应

批注提取自 PDF（作者 Yao Wang，2026-11-05）。分三类：**A** = adaptive 工作直接回答；**B** = 固定-λ 版本的机制/写作问题，与控制器无关，但新稿必须回答；**C** = 编解码器细节 / 排版意见。

### A. 由 adaptive 版本直接回答（5 条）

| # | 页 | Yao 的问题 | 回答 |
|---|---|---|---|
| 1 | 2 | B_a=6、B_f=2 怎么定的？是预训练熵编码器的平均吗？会不会随 rate point 变？ | **会变**（5/10/20 MB：11–12.8 / 9–11 / 7.7–10 B/V），这正是 v7b 全部偏低的原因。新版本不再用常数：b_v、b_f、overhead、fpv 每次 probe 由真实编码器在当前网格上实测（§3）。Draft 1 的"读一次 Room"被废弃 |
| 4 | 2 | 训练时 λ 怎么定？给一个目标码率我怎么知道对应的 λ？ | 这就是整个系统：用户给 MB，λ = 让网格长到预算买得起的顶点数所需的 rank price，每个间隔重算（§4 第 10 步），Phase B 用同一价格反向修正（§5）。用户不再接触 λ |
| 6 | 3 | "Why fixed prices" 没看懂，和前面给的 B_a、B_f 数字矛盾 | Draft 1 的确自相矛盾（常数 10 B/V 对实测 9.5–13.3）。新稿删掉这一段；替换为"价格由编码器在环内测量"。固定-λ 的 ablation 保留一条：价格 ±30 % 只重标 λ、不动 hull（8/8 点），说明旧结果没错，只是不能当目标大小用 |
| 3 | 2 | collapse 只去掉一个顶点？ | 是：u、v 合到中点，少一个顶点和边上 n_uv 个面（内部 2，边界 1）。v9d 进一步只对**交付文件真会少的**顶点/面计价（§5 第 5 步）。新稿需在 Fig 2(b) 旁明确写出 |
| 2 | 2 | per-face 和 per-triangle 有什么区别？ | 没有区别，是用词不一致（soup 阶段叫 triangle，网格阶段叫 face）。新稿统一用 face |

### B. 固定-λ 机制与结果呈现（与控制器无关，新稿要改）

| # | 页 | 问题 | 处理 |
|---|---|---|---|
| 5 | 2 | "创建约两面每顶点" 的 ? | Delaunay 后实测 fpv 2.2–2.9（Blender 3.4）。新稿直接给实测值并说明来源 |
| 16 | 4 | 不要 "unless marked T&T"，表里加一行区分 Mip-NeRF vs T&T | 采纳 |
| 17 | 4 | 你有更多的点？ | 是，固定-λ 4 个点 + 控制器 3 个目标；新稿画平均 RD 曲线 |
| 18 | 4 | "2.9–6.4 MB" 什么意思 | 同一 λ 在不同场景停在不同大小的范围。新版本用目标大小后这列消失 |
| 19 | 4 | 写成 (MB (ratio), PSNR)，PSNR 改 ΔPSNR | 采纳（title-abstract 草稿的表已按此） |
| 20 | 4 | 难场景 smallest-λ 点被 dominated 那句没看懂 | 意思是 flowers/treehill/truck/train 上最大网格不涨 test PSNR（过拟合训练视角）。新稿改写或删 |
| 22 | 4 | 表旁边要放平均 RD 曲线（Fig 4） | 采纳 |
| 23 | 6 | 这些点别人都能达到吗？包括右上那个？ | 需要在新稿明确：右上（FP32 MeshSplatting 复现）是上游本身；MiLo / RaDe-GS / BOG 点是各自论文报告的 mesh 大小与 PSNR。要逐点标注来源 |
| 24, 25 | 6 | (b) 两张子图不必要 | 删 |
| 26 | 6 | 别人不和 GS 比，我们也不必 | 采纳：3DGS codec 对比移到一句话 + 补充材料；正文只在 intro 说一次 3 dB 差距 |
| 27 | 6 | "than upstream" 什么意思 | 指 FP32 上游模型。改写 |

### C. 编解码器细节（Draft 1 的 §1.3；控制器不改变答案，新稿需补）

| # | 页 | 问题 | 现状 |
|---|---|---|---|
| 7 | 3 | 字典 "refit 400 步 against rendering loss" 怎么做 | 需补一段：K-means 初始化后，固定索引，字典向量作为参数对训练视角的渲染 L1 做 400 步梯度下降 |
| 8 | 3 | 1-ring context model 要简述 + 引用 | 需补：索引的上下文是已解码的 1-ring 邻居的索引；补引用 |
| 9 | 3 | zstd 引用 | 补 |
| 10 | 3 | extent 是动态范围吗 | 是：场景包围盒的最大边长；b = ⌈log2(extent/(0.1·中位边长))⌉ |
| 11 | 3 | cached-edge arithmetic coder 要更多细节 | 需补一段 + 引用 |
| 12 | 3 | one opacity scalar 是整个网格的？ | 是，交付时面全不透明，只有一个全局标量进 header。新稿明确 |
| 13 | 3 | 字典如何表示 | 4096×45 + 256×45 的 float16（需核对 encoder 代码后写确切精度） |
| 14 | 3 | 说五条流只看到四条 | 位置、连接、SH 主索引、SH 残差索引、DC。新稿列全 |
| 15 | 3 | 几何部分和 SH 类似吗，能否用更好的 context coding 进一步压 | 连接是无损底板（~35–45 % 字节）；我们的 cache-based coder 对比 Draco/MPEG 已小 3.5–15×。进一步压是 future work，新稿说一句 |
| 21 | 4 | MPEG 只编几何？字节更多？ | 是：只替换几何编码器为 MPEG SC3DMC，其他流不变，几何部分字节更多。新稿改写清楚 |
| 28 | 6 | 试过多个字典大小 × λ 找凸包吗？DC 7 bit 是不是高码率掉 PSNR 的原因，试过 8 bit？ | 字典 × λ 联合 hull 做过：K4096/R256 在 11 场景距 hull ≤ 0.16 dB。DC 位数：当时的结论是 DC bits 不承担损失（"SH sharing carries the loss, DC bits do not"），但**DC q8 的直接对照行没有在表里**；这是一条可以用 CPU 跑的小实验（只是重新交付），建议在剩余时间补上 |

---

## 10. 给 Devon 口头汇报的要点（按 Yao 可能的追问顺序）

1. **"你们的 λ 现在怎么来的？"** —— 不是用户给的。用户给 MB；控制器用编码器实测的 size model 算出要多少顶点，再按几何计划算这一轮要分裂多少，λ 就是让恰好那么多三角形通过的 ratio 分位数。每 500 迭代重算一次。
2. **"B_a、B_f 还是常数吗？"** —— 不是。进预算的 b_v、b_f、overhead、fpv 都是每次 probe 从编码器流大小读出来的；分裂公式里剩下的 33 B 只是 λ 的量纲，rank 定义下抵消。
3. **"编码器在训练环里会不会影响公平性 / 泄漏？"** —— probe 在副本上跑，只读字节数；训练模型和梯度不被触碰；交付用的是同一个编码器，probe/delivered 0.999–1.001。
4. **"命中率多少？"** —— 33 行均值 1.007，32/33 在 ±10 %，22/33 在 ±5 %，等字节 −0.05 dB。Deep Blending 不改一字 1.06–1.12。
5. **"为什么室内偏高、室外 20 MB 偏低？"** —— 一个在 10k 前不可观测的量：Delaunay 之后可见面集在 opacity ramp 下继续增长，室内 1.1–1.5×，室外 1.05×。Phase B 单向、每轮只能动 0.4 % 顶点，追不完。补偿是每类一个 aim margin（1.25 / 1.1），这是系统里**唯一**的校准常数，paper 如实报告。
6. **"为什么不干脆训练后再压？"** —— G0 实验：同顶点数下训练到预算比训练后折叠高 +0.01…+6.02 dB（21/21，预算越小差越大）。原因是不透明网格每像素一个三角形、顶点通过连接耦合，优化器不会重新安置幸存者。这是"必须在训练中决定大小"的动机，也是和 T2-Splat 等"报告面数、不报告字节"工作的区别。
7. **"可训练熵编码器呢？"** —— 没做。现在是固定、可解码的熵编码器 + 实测价格；可训练的 rate model 是 future work。价格错 ±30 % 只重标 λ，不动 hull，所以当前方案的 RD 结论不依赖它。

---

## 附录：Yao 批注原文（PyMuPDF 提取，按页序）

见 `paper/yao-comments-draft1-2026-11-05.txt`（完整转录，含每条批注锚定的原文），正文第 9 节已逐条引用。关键三条原文：

- #1（p2，B_a/B_f 处）："how were these determined? based on the average from the pretrained entropy codecs? would these numbers actually vary depending on the rate point you try to operate?"
- #4（p2，λ 处）："how do you determine lambda during actual training? based on a target rate? but how do i know the corresponding lambda for a target rate?"
- #6（p3，Why fixed prices）："did not understand this part. Seem to contradict from the previously stated bytes for Ba and Bf"
- #28（p6）："did you try multiple dictionary size and different lambda, and find convex full? Also using 7 bits for DC may be the reason that you have reduced PSNR even at the highest rate? did you try to use 8 bits?"
