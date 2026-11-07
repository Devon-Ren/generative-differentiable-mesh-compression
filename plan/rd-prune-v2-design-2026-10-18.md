# v2 设计：phase-B 周期 λ-prune 与"自然停止"（2026-10-18）

## 0. 现状（v1，10/18 七场景）
- λ-split 只作用在 phase A（500–10000 步，Delaunay 之前）。同字节 ≥ cap（10/10 可比点，+0.0～+0.57 dB），场景自适应（同 λ 停在 dense 的 41–100 %）。
- 缺口 1：**没有自然停止**。所有场景 9500 步 pass 数仍 > 0（Room 3e-3：5.5 万；stump 6e-3：1.4 万），顶点数由 `densify_until=10000` 截住。λ 现在决定的是增长速率与场景间分配，不是终点。
- 缺口 2：**phase B 没有任何顶点删除**。Delaunay（11000）之后到 30000 步，upstream 只调 opacity floor，最后按 importance ≤ 0.5 清一次三角形。ΔR_v 随面数变化的"变价剪枝"只能在这里做，这也是 story 里方法含量最重的一块。
- 缺口 3：importance 采样的同路径对照（`--dd_mode importance`）还没跑（1 run）。

## 1. 三个候选实验，按 GPU 成本排

| | 内容 | 代码量 | run 数 / GPU 小时 | 回答的问题 |
|---|---|---|---|---|
| C | `--dd_mode importance`，Room，λ=6.2e-3（v2 单位）| 0（已实现）| 1 / 1.5 h | 把"选谁分裂"和"何时停"的贡献分开：importance 采样 + 同样的限速与 λ 门槛，若与 error 模式打平，则 v1 的增益全来自停止规则 |
| A | 延长 densification 窗口：`--densify_until_iter 15000`（Delaunay 自动移到 16000），Room λ=6.2e-3 和 1.85e-3 | 0（upstream 参数）| 2 / 3 h | λ 能否自己停：pass 是否在 15000 前降到 0；代价是 phase B 缩短 5000 步（混淆项，要如实写）|
| B | phase-B 周期 λ-prune（下节）| 1–2 天 | 2 / 4 h（Room 两个 λ，有/无剪枝）| 同字节是否再 +0.2 dB；ΔR_v 变价是否真的优先删"贵而没用"的顶点 |

建议顺序：C 立刻提交（1.5 h，ablation 表必需）；B 的代码在 C 跑的时候写；A 可选，只在 B 之后仍需要"自然停止"的证据时再跑。

## 2. B 的设计

### 2.1 时机
Delaunay 之后、supersampling 之前：12000 / 14000 / 16000 / 18000 步各剪一轮（4 轮）。20000 步后分辨率 ×2、25000 后 ×4，那段只做修复，不再动拓扑。每轮之后至少 2000 步训练修复，这是和事后 collapse 的本质区别。

### 2.2 ΔD_v（删掉顶点 v 的失真代价）
两个候选，先用第一个：
1. **残差归属（和 split 同一套账）**：phase B 继续累计每三角形的残差质量 err_t（`rend_ids` 仍可用），顶点 v 的 ΔD_v = Σ_{t∋v} err_t × (v 在 t 的面积份额)。含义：这个顶点周围当前承载多少像素误差；误差大说明这一带还在被拟合，删了会更差。单位和 split 端一致，λ 可以共用。
2. **渲染敏感度（C2 的 sens_pos）**：对顶点位置的梯度范数 × 像素尺度。事后 collapse 已验证有效，但要重新渲染全部训练视角（约 5 分钟/轮），单位和 split 不同，需要一个 μ 把它换算到 λ 的尺度。作为 ablation。

### 2.3 ΔR_v（删掉顶点 v 省下的字节）
ΔR_v = B_attr + n_faces_removed(v) × B_face。B_attr ≈ XYZ 2–2.6 + DC 1.5 + index 2.4 ≈ 6–6.5 B，B_face ≈ 2 B（拓扑 ≈ 4 B/顶点、约 2 面/顶点）。一次 edge collapse (u→v) 删掉的面数 = 共享该边的面数（边界 1、内部 2、非流形更多），所以 ΔR 在 8–12 B 之间变化。这就是"同面数 ≠ 同字节"（10/13）在训练环里的用法：价高的顶点先删。

### 2.4 规则与实现
- 候选边：所有边；得分 s_uv = (ΔD_u + ΔD_v) / ΔR_uv；贪心独立集（一轮里每个顶点最多参与一次 collapse），按 s 从小到大，s < λ·μ 的才删；μ 默认 1，扫 {0.5, 2} 作 ablation。
- 执行：`collapse_edges(triangles, pairs)`，放在 `analysis/rdgrowth/rd_prune.py`：
  1. 新位置 = 中点（或按 ΔD 加权），`_features_dc/_rest`、`vertex_weight` 同样平均；
  2. 面重映射 + 删退化面（两个顶点相同）+ 删重复面；
  3. Adam 状态按同一映射平均（`exp_avg`、`exp_avg_sq`），走 `_prune_vertex_optimizer` 的路子重建 Parameter；
  4. `image_size / importance_score / pixel_count` 重置为 0（和 `densification_postfix` 一致）。
- 不改仓库：`train_rd_v3.py` 在 phase B 的 `iteration % 2000 == 0` 分支里调用，其余照抄 v2。
- 日志：每轮记录候选边数、删掉数、ΔR 分布（省 1/2/3+ 面的比例）、顶点数前后。

### 2.5 预测（写进论文前先对照）
- 同字节 ≥ λ-only ≥ cap；差距在低码率端更大。
- 删掉的顶点里"省 ≥ 3 面"的比例显著高于随机（证明 ΔR 变价在起作用）；用 ΔR=常数的对照（μ 相同）应该更差。
- 若 B 对 λ-only 没有增益：说明 phase A 的 λ-split 已经把该删的没长出来，剪枝无事可做——这本身也是一个可写的结论（"控制生长比事后剪更重要"），但 story 第 3.3 节的 v2 段要删。

## 3. 一句话给 PhD
v1 已经证明码率控制生长在七个场景上同字节 ≥ cap、而且顶点数由 λ 场景自适应地决定；v2 把同一个 λ 用到训练中期的剪枝上，让删顶点的价格随它实际省下的字节变化。


## 4. 实现记录（2026-10-19）
- **C 的修正**：v2 的 `--dd_mode importance` 不看 λ（每个 interval 按 add_percentage 长满 = dense 原版），不能作为"选谁分裂"的对照。改用 `analysis/rdgrowth/train_rd_v2m.py`（v2 + `--split_schedule <rd_log.jsonl>`）：每个 interval 的选择数上限 = 参考 λ run 在同一步的 `n_split`，选法 = upstream importance 采样。与 λ run 只差"哪些三角形分裂"。提交块 `analysis/commands/rd-ctrl-v1.md`（Room，参考 `room-rd-error-lam0.01`，1 job tandon，2.5 h）。
- **B 已实现**：`analysis/rdgrowth/rd_prune.py`（`select_collapses`：边表、ring 残差 D_u+D_v−D_uv、ΔR=(BPV−2·B_face)+n_faces·B_face、float64 分数 + 随机 jitter 的局部最小边匹配（独立集，≤64 轮）、每轮 cap、`foldover_filter` 翻面拒绝；`collapse_edges`：参数/opacity/Adam 动量平均到 u、v→u 重映射、删退化与重复面、切片 per-face 跟踪器、走 `_prune_vertices` 删 v）；`train_rd_v3.py`（= v2m + `--prune_iters 12000 14000 16000 18000 --prune_mu 1 --prune_max_frac 0.15 --bytes_per_face 2`；Delaunay 后 accumulator 重置并持续累计到最后一轮剪枝；事件 `prune` 记录候选比例、分数分位/门槛、删面份额 sel/cand/all、顶点面数前后）。CPU 单元测试 `analysis/rdgrowth/test_rd_prune_cpu.py` 通过（独立性、计数、Adam 形状、合并值、foldover、cap、阈值）。提交块 `analysis/commands/rd-prune-v3.md`（Room λ 6.2e-3 / 1.85e-3，1 job general，3 h，≈2 GPU 小时；smoke 在 1250 步强制剪一轮）。
- 尺度提醒：ring 残差 ≈ 12 个面之和、ΔR ≈ 10 B，而 split 端是 1 面 / 30 B，所以 μ=1 下只有残差明显低于典型值的区域会被剪（估计候选比例 10–30 %）；若 cand≈0 则 run 退化为 λ-only，结论改写为"该删的没长出来"。
