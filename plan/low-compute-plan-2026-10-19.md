# 低算力计划（2026-10-19）：FairShare 不会恢复，按"每 GPU 小时的论文价值"排

前提：headline 不再重跑。七场景 λ 数字（v2，无剪枝，10/18）就是主表；剪枝作为 Room（+2 场景）验证过的 add-on，论文里如实写"−8 % 字节、PSNR 不变，在 3 个场景上验证"。

## Tier 0：0 GPU 小时（现在就做）
1. **Ablation 表（Room）**全部用已有行：cap / importance-matched / λ / λ+prune / 事后 collapse，在 ~4–6 MB 和 ~13 MB 两档。配图：room-lambda-vs-cap 加 control 点和 prune 点（v2 图）。
2. **自然停止分析**：从已有 rd_log.jsonl 取每个 interval 的 n_pass 和 ratio 分位数（登录节点 CPU 一条命令），画 pass 数随迭代的曲线。回答 C4b 不需要新训练。
3. **训练时间作为附带收益**：λ 运行比 dense 快（22–35 min vs 51 min），从已有 log 的 elapsed_s 读。
4. **同一个 λ 定价三条流**：几何端两 λ 点之间的实测斜率 → 在精度 / 码本的 CPU 重导出 hull 上选局部斜率匹配的点（已有 re-export 数据）。这是"λ 是唯一旋钮"的系统性说法，纯 CPU。
5. 写作：方法段、ablation 段、limitation（等斜率不成立、变价不成立、无自然停止）。

## Tier 1：≤ 0.5 GPU 小时
6. **训末单次剪枝对照**：对已有的 λ checkpoint（Room 6.2e-3 / 1.85e-3）做一次 11 % 的 collapse + 3000 步 finetune + 交付（每个约 12 min，不训练 30k）。回答"周期剪枝 vs 训末一刀"——这是 C5 剩下的那个对照。
7. **FPS / 三角形数**：渲染 7 个场景的交付 mesh 各 100 帧，~5 min。

## Tier 2：≈ 2 GPU 小时（剪枝的泛化，只要 2 个场景）
8. v3 在 counter 和 stump 各跑 λ=6e-3（约 25 + 35 min 训练 + 协议），和已有的无剪枝行比。3 个场景足够写"add-on"。

## Tier 3：取消（10/19）
9. ~~延长 densification 窗口~~：pass 曲线分析（`rd/pass-curves-7scenes-v1.png`）显示 λ=6e-3 时三角形数 7500→9500 已在 ±8 % 内、候选集按半衰期 ≈3000 步几何衰减；延长到 15000 预计只改 <3 % 顶点。C4b 改写为“λ 决定的稳态”，不需要这 1 小时。

## 以后再有 ablation 的省钱协议
所有 ablation 在 20000 步（supersampling 之前）评估，headline 才跑 30000：每 run 成本降到 1/3（supersampling 阶段占时间大头）。几何在 18000 步后不再变，所以 ablation 的排序在 20k 就定了。

## 不做
- 七场景 v3 重跑、flowers / treehill / T&T、μ 扫描、ΔR=const 对照、Room λ 填缝。
