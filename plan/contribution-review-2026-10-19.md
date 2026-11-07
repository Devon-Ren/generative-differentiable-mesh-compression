# Contribution 审查：够不够 CVPR main track（2026-10-19）

立场：按 CVPR 审稿人 / AC 的标准，用现有证据打分，不替自己辩护。

## 0. 结论

**现在的材料是 borderline（弱接收到弱拒之间），不是稳的。** 能过的理由是"发现 + 系统"；会被拒的理由是"方法相对 cap 的增量小、PSNR 比 3DGS 压缩低 3–7 dB、场景覆盖不全"。要把它推到"弱接收以上"，靠的是口径重排和三件不花（或少花）GPU 的事，不是再跑 λ。

## 1. 按审稿人打分

| 维度 | 现状 | 审稿人会怎么说 | 分 |
|---|---|---|---|
| 新发现 | 事后 collapse 在 7/7 场景 14/14 点输 0.3–4.3 dB；Gaussian 不是这样 | "清楚、可证伪、有反直觉成分（mesh 的顶点被拓扑耦合）。这是全文最硬的东西。" | 强 |
| 方法新颖性 | λ-split（真实 codec 字节作 ΔR）+ phase-B λ-prune（训练环内 edge collapse + Adam 重映射） | "Taming-3DGS / RDO-Gaussian / Mini-Splatting 已经做过 budget / RD 控制的 densification。搬到 mesh 上、ΔR 用真实字节，是增量。λ 比 cap 只高 0–0.57 dB，而且分解后日程占 0.3、排序只占 0.12。" | 中偏弱 |
| 方法必要性 | cap 要逐场景调；一个 λ 跨场景落在 41–100 % / 15–43 % | "scene-adaptive 是真正的实际优势，但要用'同一设置跨场景'的实验来讲，现在讲成 λ vs cap 曲线反而显得差距小。" | 中 |
| 结果强度 | 七场景均值 8.8 MB / −0.88 dB / 80×；绝对 PSNR 24.9 dB | "HAC++ 在同样 MB 上是 27.5 dB 左右，高 2.5 dB。你们说'不同表示类'，但表里得有 mesh 类对手在同 PSNR 上的 MB 和 FPS，现在只有文献数字。" | 中偏弱 |
| 覆盖 | Mip-360 7/9 场景，无 T&T / DeepBlending | "标准 benchmark 不全，会被点名。" | 弱 |
| 诚实度 | 等斜率证伪、变价证伪、无 pass=0 改稳态 | 加分，但审稿人不会因为诚实给高分 | — |

## 2. 真正的风险（按致命程度）

1. **"这就是 Taming-3DGS 搬到 mesh"**。必须正面回答：(a) 在 Gaussian 上事后剪几乎免费，所以 budget densification 对 Gaussian 只是省训练时间；在 mesh 上它是唯一能到低码率的路径（C1）；(b) ΔR 是交付文件的真实字节，λ 有单位，同一个 λ 给 split 和 collapse 定价；(c) phase-B 的训练环内 collapse 是 mesh 独有的操作。这三点要写进 related work 和 intro，而不是藏在 ablation。
2. **绝对 PSNR**。MeshSplatting 本身比 3DGS 低 2.4 dB（24.78 vs 27.21），我们的 80× 点再低 0.9。审稿人看主表第一眼就是这个。对策：主表按"表示类"分块，mesh 类对手（MobileNeRF / BakedSDF / BOG / MiLo / 作者 RGB-only PLY）同表比 MB + PSNR + FPS，3DGS 压缩单独一块并明说解码产物不是 mesh。**作者 RGB-only PLY（~100 MB，−2 dB）必须进表**，它是最便宜也最诚实的对手，而且我们在它之上 10× 小还高 1 dB。
3. **场景覆盖**。7/9 Mip-360。最低限度补 flowers + treehill（各 original ~50 min + 一个 λ ~30 min + 协议 ≈ 3 GPU 小时）；T&T 不补就在 limitation 里写明。如果只有 2–3 GPU 小时可花，花在这里比花在 Tier 2 剪枝泛化上更值。
4. **λ vs cap 差距小**。口径改成：cap 是我们方法的 budget 特例（在 mesh 上也是我们第一个做的），对手是事后 collapse 和 RGB-only PLY；λ 相对 cap 的卖点是"一个设置跨场景"——用现有数据直接给：cap 固定 1.0M 七场景 9.8 MB / −0.97 vs λ 固定 6e-3 七场景 8.8 MB / −0.88，两个轴同时赢。
5. **没有定性结果**。CVPR 论文没有渲染对比图会被当成不完整。事后 collapse vs λ 在同字节下的放大对比（Room 和一个室外场景）、解码后 mesh 在标准 viewer 里的截图、wireframe 显示顶点分布随 λ 变化。都是分钟级 GPU。

## 3. 把 contribution 列表改成这样

1. We show that, unlike Gaussians, opaque differentiable meshes cannot be compressed after training: removing vertices post hoc costs 0.3–4.3 dB more than never growing them (7 scenes, 14/14 points). The primitive count must be decided during growth.
2. We make the primitive count an output of a rate price λ: a Lagrangian split rule in densification and the same rule applied in reverse as an in-training edge collapse, both priced in the bytes of the delivered file. One λ across scenes replaces per-scene budgets, lands on the RD hull, and reaches a λ-set steady state.
3. A complete mesh codec in which only geometry enters the training loop (appearance is shared post hoc, topology is coded in context), giving 36× / 80× smaller files than the original at −0.26 / −0.88 dB and 10× smaller than the authors' RGB-only export at +1 dB, decoding to a standard mesh at real-time frame rates.

第 2 条不要写 "λ beats the cap by X dB"，写 "one setting across scenes"。第 3 条的 FPS 数字要测。

## 4. 加固清单（按性价比）

| 项 | GPU | 作用 |
|---|---|---|
| 主表重排（mesh 类 / 3DGS 类分块）+ RGB-only PLY 行 + "同一设置跨场景"表 | 0 | 直接回应风险 2、4 |
| Related work 段：Taming-3DGS / RDO-Gaussian / Mini-Splatting / LightGaussian 的差异三点 | 0 | 风险 1 |
| 定性图：同字节 collapse vs λ 放大对比、viewer 截图、wireframe | 分钟 | 风险 5 |
| FPS（7 场景交付 mesh）| 分钟 | 第 3 条 contribution 需要 |
| flowers + treehill（original + 一个 λ）| ≈3 h | 风险 3，9/9 Mip-360 |
| 训末单次剪枝对照（Tier 1）| ≤0.5 h | C5 的最后一个 ablation，锦上添花 |
| 剪枝两场景泛化（Tier 2）| ≈2 h | 优先级低于 flowers/treehill |

## 5. 一句话
够不够取决于口径：以"mesh 不能事后压、码率必须进生成环"为主张、以完整 codec + 标准 mesh 产物为系统、把 λ 讲成"一个设置跨场景"而不是"比 cap 高 0.3 dB"，再补 RGB-only PLY 对手、定性图和 9/9 场景，是弱接收以上的材料；现在这样直接投，是 borderline。

## 6. 用户的定位（2026-10-19，采纳）
- 动机是工程需求：differentiable opaque mesh（MeshSplatting）的可压缩性；贡献是**全链路编解码器 + 对这类表示压缩性的全面分析 + 生成式（码率进生长环）压缩能力，50–100×**。
- 不与 3DGS 压缩比绝对 PSNR：差距来自 MeshSplatting 本身（NVS 质量低 2.4 dB），论文按 RD 口径只报相对原始模型的 dB 损失，不承诺近无损。
- 没有公平的外部对手（从 differentiable mesh 到 NVS 到可压缩再渲染，没有同类工作）；作者 RGB-only PLY 也不公平（无 SH）。
- 场景：必要场景全部跑，多 config，能出 RD 就出 RD。

对这个定位的补充（审稿人仍会要的三件事）：
1. **表里不能空着对手**。"没有同类工作"要用"同一表示上的通用编解码器"来填：FP32 原始、zstd/gzip 的 PLY、Draco（量化 + 拓扑编码，已有）、作者 RGB-only PLY（标注无 SH）、事后 collapse 路线。这组对手是"给表示 X 做 codec"这类论文的标准对照，而且我们全赢。
2. **"无人做过"不是贡献，"为什么 mesh 不一样"才是**：事后删顶点失败（C1）+ 只有几何需要进训练环（C6）是全文的智力核心，必须放在 intro，否则会被归为纯工程。
3. **多场景的算力**：补 flowers/treehill/T&T 两场景 = 4 个 original（≈4 h）+ 每场景 2 个 λ 的 v3（11 场景 × 2 × ~45 min ≈ 16 h）≈ 20 GPU 小时；按现在 FairShare 要分批、每批 2–3 h walltime。
