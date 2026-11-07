# Yao 的三个 RD 问题：现状、证据、还缺什么（2026-11-05）

Yao（经 Wenhao 转述，11/05）：(1) 有没有同时遍历 λ 和字典大小的 RD 曲线？字典大小同时影响 D 和 R，选 K 需要 RD 论证；(2) DC 为什么 7 bit？有没有熵编码？q7 vs FP32 的失真量了吗？是不是它造成最高码率点的失真；(3) B_a / B_f 是常数、熵编码器固定 → 不是 end-to-end；熵编码器应可训练。

## 0. 一句话结论

| 问题 | 现状 | 缺口 | 补上要多少 |
|---|---|---|---|
| Q1 λ × K 联合 RD | 有 **顶点数 × K×R**（Room 4 个顶点层级 × 12 点，bonsai/garden/4 场景各 6 点，10/15-17）+ 贪心分配器复现 Room 全局 hull（9/9 点）；**λ 轴上所有点固定 K4096 R256**，没有 λ × K | 每个 λ 资产再导出 K×R 扇面（不训练，每点 3-5 min GPU） | Tier 1 = room/garden/kitchen × 4 λ × 5 点 ≈ 8 GPU-h；全 11 场景 ≈ 30 GPU-h |
| Q2 DC q7 | q7 来自 Room 精度扫描（q8→q7 −1.04 MB / −0.025 dB = 0.022 dB/MB，当时最便宜的杠杆；q6 0.12 dB/MB 被"减顶点"dominated）。编码 = per-channel 均匀量化 + shuffle+zstd，**无预测、无上下文**；garden 14.0 bits/v（raw 21），占文件 ~21 % | (a) indoor 模型 level-1.0 的 −0.5..−0.75 dB 没有分解到 DC / 字典 / 权重（11/01 提议的 0.5 GPU-h 诊断仍待 Wenhao 决定）；(b) DC 没有预测式熵编码 | (a) 0.5 GPU-h，**先跑**；(b) 复用 XYZ 的"最小索引先前邻居"预测 + 现成算术编码器，改码流（用户 codec，需同意），CPU 可先估节省 |
| Q3 end-to-end | 码率在环内但是**校准过的常数代理**：B_a = 6 B、B_f = 2 B；实测 44 个资产 9.5-13.3 B/v（中位 10.8）→ 代理误差 ≤ ±25 %，相当于 λ 乘 0.95-1.33（< 3× λ 网格步长的 1/4）；熵编码器固定（边缓存连接性、XYZ 预测+算术、索引 1-ring 上下文、DC zstd） | 没有可训练熵模型，没有可微码率 | 截稿前做不到真 E2E（见 §3）；能做：B_a 敏感性 6 GPU-h + 诚实表述 + DC 预测编码 |

## 1. Q1：字典大小的 RD 论证

**已有**（都是 Room 无 indoor 标志、post-hoc 顶点层级，不是 λ 训练）：
- Room v0.8（2.55 M v）12 点 K×R：hull = K1024R1 → K4096R1 → K4096R256 → K4096R1024 → K4096R4096 → K16384R4096；默认 K4096 R256 在 hull 上；从默认出发 R256→R1024 0.033 dB/MB，R1024→R4096 0.021，→K16384R4096 0.011；K16384 的 bank 2.4-3 MB 让它只在最高点上 hull。图 analysis/rd/room-rd-codebook-v2.png。
- 4 个顶点层级（1.0/0.8/0.7/0.6）× K×R 的多 hull 图 analysis/rd/room-rd-codebook-levels-v1.png：每层 backbone 都是 K4096 R1→R256→R1024→R4096；0.7 层整体在 0.8 层之下（等字节"多顶点 + 小字典"赢"少顶点 + 大字典"）；1.0 层在 ~28 MB 以上才超过 0.8。
- 贪心分配器（rd/allocator_greedy.py）：从 v1.0K16384R4096 出发，每步砍最便宜的一档（R、K 或顶点层级），**精确复现** Room 全局 hull 9/9 点 → hull 就是局部斜率的贪心规则，这就是"K 的选择是 RD 最优"的论证形式。
- bonsai / garden / counter/kitchen/bicycle/stump 各 6 点 backbone（rd/*-codebook-v1.csv）：同样的 backbone 顺序。

**缺**：λ 训练出来的资产上没有扫 K×R。注意 K 进训练只通过 B_a 里的索引位数（K4096R256 = 12+8 = 20 bit = 2.5 B；K16384R4096 = 26 bit = 3.25 B → B_a 6 → 6.75，λ 有效值移 12 %，远小于 3× 的网格步长），所以联合优化可以分两步：每个 λ 训练一次，导出时按 K×R 扇面取 hull；再证明 λ 内层的 K 选择与贪心分配器一致。这正好回答"即使当前选择最优也要展示"。

**怎么补（不训练，scene-axisA-v1.sbatch 已能对任意资产跑 6 点 backbone）**：
- Tier 1（≈ 8 GPU-h，general）：room / garden / kitchen × λ {6.2e-4, 1.85e-3, 6.2e-3, 1.85e-2} 的 -best 资产 × {K1024R1, K4096R1, K4096R1024, K4096R4096, K16384R4096}（默认 K4096R256 已有）。产出：每场景一张"λ 扇面 + 每 λ hull + 全局 hull"图，以及贪心分配器在 λ 资产上的复现率。
- Tier 2（节点批下来后，≈ 30 GPU-h）：11 场景全做，进附录；主文只放均值 hull。

## 2. Q2：DC 7 bit

**为什么 7**：Room 精度扫描（10/13，geo-only 1.98 M v，无训练）：XYZ q18→q16 免费，q15 −0.016，q14 −0.056 dB（默认）；DC q8→q7 −1.04 MB / −0.025 dB（0.022 dB/MB，当时所有杠杆里最便宜的之一），q7→q6 0.12 dB/MB（SSIM −0.016，被减顶点 dominated）。DC-QAT（STE）比事后量化差 0.1 dB（负结果）。hull-v2 格式切换实测 q8→q7 在所有点 −0.027 ± 0.003 dB。

**有没有熵编码**：只有容器的 shuffle+zstd（高位零字节被吃掉），**没有空间预测、没有上下文模型**。garden v0.8：DC 4.46 MB = 14.0 bits/v（raw 21），文件 21 %；Room dense q8 15.1 bits/v。XYZ 有"最小索引先前邻居"预测 + 算术编码（45 raw → 16.3 bits），DC 没有同样待遇：这是码流里最明显的空位。1-ring 预测的 DC 残差通常能再省 30-50 %（Room 级别 1-2 MB，约文件的 5-8 %），不改失真。

**q7 vs FP32 的失真**：
- 非 indoor Room：sharing + 量化（q14/q8）在每个顶点层级都是 −0.10..−0.15 dB，其中 DC q8→q7 再 −0.03。
- **indoor 模型（11/01）**：FP32 → level-1.0 交付 room −0.49、kitchen −0.76、bonsai −0.62、counter −0.74 dB，其中 finetune 步 −0.1，"量化 + 字典"−0.38..−0.63。**没有分解**到 DC / K×R / 顶点权重三项。11/01 已提议诊断：对 room + kitchen 的 indoor level-1.0 资产不训练重导出三点（DC q8；K16384 R1024；真实 per-vertex weight q6），≈ 0.5 GPU-h，仍待 Wenhao 决定。这就是 Yao "是不是 DC 造成最高码率点失真"的直接回答，先跑它。
- 机制猜测（待验）：indoor 预设 feature_lr 0.004 vs 0.0016、lambda_weight 0、顶点更少 → 每顶点颜色方差更大，同样的 q7 / K4096 损更多；若诊断确认，indoor 行按**类别默认**改 DC q8 或 K16384（不逐场景调），代价 1-3 MB。

## 3. Q3：end-to-end

**我们是什么**：rate-in-the-loop with a calibrated constant-rate proxy。增长/剪枝按 J = D + λR 判定，ΔR 用 B_a、B_f 常数，ΔD 用当前残差归因。实测交付 bytes/vertex 9.5-13.3 B（44 资产，中位 10.8，随 λ 升 10.4→11.4）→ 常数代理最多把 λ 乘 0.95-1.33，< λ 网格步长（3×）的 1/4；一个常数保持 λ 是跨场景统一的汇率（plan/split-prune-rationale-2026-11-03.md）。

**真 E2E 需要什么、为什么截稿前做不到**：可微码率估计要覆盖三种流：(a) SH 字典索引（离散 k-means 分配，需 soft-assignment / Gumbel 才可微）；(b) XYZ / DC 残差（可用 Ballé 式加噪 + 学习的熵模型，但要先把预测器和上下文放进训练图）；(c) 连接性（30 % 以上字节，面编码，不可学习）。HAC/HAC++ 的可训练熵模型只覆盖 Gaussian 属性，没有连接性；我们码率里最大的两项（连接性 31.7 + XYZ 16.3 bits/v，garden）恰是学习熵模型不覆盖的。加上 λ 网格、11 场景重训（每点 ~0.6 GPU-h）—— 这是下一篇的工作量，不是 10 天。

**能做的三件**（按性价比）：
1. **DC 预测式熵编码**（Q2b）：复用 XYZ 预测器和索引上下文算术编码器，CPU 可先离线估 bits/v，码流改动需 Wenhao 同意并升版本（stage2-v3）。
2. **B_a 敏感性**：B_a ∈ {4, 6, 9}（即旧记号 B_v 8/10/13）× kitchen/garden × 1 λ ≈ 6 GPU-h，展示 hull 对代理常数不敏感（等价于 λ 轻微重标）。
3. **表述**：标题/贡献句写 "rate-controlled growth with a calibrated rate proxy and a fixed, decodable entropy coder"，限制段明说 learned entropy model 是 future work，并给出 §2 的数字说明代理误差的量级。审稿风险真实存在（Yao 的判断我同意）：方法新意靠"rate 在网格生长环内"，不靠熵模型。

## 4. 建议顺序（一次一个）

1. **现在**：跑 11/01 的 indoor 重导出诊断（0.5 GPU-h，room + kitchen × {DC q8, K16384R1024, 真实 weight}）→ 回答 Q2 的"最高码率点失真来源"。
2. 然后 Tier 1 λ × K×R 扇面（8 GPU-h）→ 回答 Q1，顺带给 Q2 的 K 轴。
3. 同时（CPU）：DC 1-ring 预测的 bits/v 离线估计；若节省 ≥ 1 MB/场景，提 stage2-v3 改动请 Wenhao 批。
4. B_a 敏感性（6 GPU-h）排在节点批下来之后。

## 5. 给 Yao 的回复草稿（英文，可直接改）

> Thanks — three points. (1) We swept dictionary size jointly with vertex count (Room: 4 vertex levels × 12 K×R points, plus 6-point backbones on 6 more scenes); the default K=4096/R=256 sits on every hull and a greedy allocator on local dB/MB slopes reproduces the Room global hull exactly. What we have not yet done is the same K×R fan on the λ-trained assets; since K enters training only through the index bits in B_a (a ≤12 % shift of the effective λ), this is a re-export, not a retrain: ≈8 GPU-h for room/garden/kitchen × 4 λ, all 11 scenes once the node is granted. (2) DC is uniformly quantized to 7 bits (chosen from the precision sweep: q8→q7 costs 0.025 dB for 1 MB on Room, q6 is dominated by removing vertices) and only zstd-coded, no spatial prediction — 14 bits/vertex, ~21 % of the file. The quantization+dictionary loss on the --indoor models is 0.4–0.6 dB at the top rate and is not yet decomposed; a 0.5 GPU-h re-export (DC q8 / larger dictionary / true per-vertex weight) will tell which term it is, and a 1-ring predictive coder for DC is a straightforward bitstream change. (3) Agreed that this is rate-in-the-loop with a calibrated constant proxy, not a trainable entropy model: measured bytes/vertex over 44 assets are 9.5–13.3 B (median 10.8), so the proxy rescales λ by < 1/4 of our λ grid step. A differentiable rate model would have to cover the k-means index assignment and the connectivity stream (the largest one, which learned entropy models in the 3DGS literature do not cover either); we will state this as a limitation and future work rather than claim end-to-end.
