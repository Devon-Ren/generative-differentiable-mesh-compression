# CVPR 竞争力复评 v2（2026-10-25 晚，deployability + R9 之后）

上一版：cvpr-readiness-2026-10-25.md（当天早上，结论 borderline 偏弱接收）。本版只写变化。立场不变：按审稿人 / AC 标准打分。

## 0. 结论

**从 borderline 偏弱接收 → 弱接收（weak accept）材料，前提是定性图和 related-work 段落写出来。** 今天补上的是 R8（解码 / FPS）和 R9（跨场景 gap 表），都是 0 GPU 的活，但它们把早上"完整性：弱"那一格改成了"中偏强"，并且给了论文一个 3DGS 类 codec 没有的列：standard mesh、CPU 解码、手机 145–350 fps。
没变的仍是方法新颖性（中偏弱）。R9 没有把"dense 训练在过拟合"升级成第二个独立发现，它成了一个更准确、也更小的句子（见 §2）。

## 1. 今天的证据（rd/table2-deployability-v1.md）

| 项 | 数字 | 对论文的作用 |
|---|---|---|
| 训练时间 | λ6.2e-3 为原版 0.57×（0.48–0.66，11 场景），λ1.85e-3 为 0.87×；Room λ sweep 时间随 λ 单调 | "码率进生长环不增加训练成本"，一句话 + Table 2 一列 |
| 解码 | 2.3–6.4 s，单 CPU 核，≈5 µs/vertex，输出标准 indexed mesh | R8 闭合；与 3DGS codec 的 GPU 解码形成硬件等级差 |
| 笔记本 FPS | M2 Max，1.2–1.9 ms/帧，full capture resolution（16 Mpx） | "<2 ms per frame" |
| 手机 FPS | iPhone 17 Pro，Room 2.8 ms（350 fps，6.5 Mpx）；garden / bicycle 6.3–6.9 ms（145–160 fps，16–17 Mpx） | Fig. 1 inset + Table 2 列；这是 MobileNeRF / BakedSDF 那一支的卖点，我们用 1/10–1/30 的字节做到 |
| R9 gap 表 | 7 个 Mip-360 场景：多买的字节 test 涨幅是 train 涨幅的 55–80 %；flowers/treehill/truck/train：train 涨 +0.1..+0.6，test ±0.1；original 与 λ1.85e-3 的 gap 相差 ≤0.35 dB | 见 §2 |

## 2. R9 的准确读法（比早上的预期弱）

早上希望 R9 变成"dense 训练在过拟合，所以必须进生长环"的第二个发现。表出来后，能站住的句子是：
- rate control **不改变泛化 gap**（original vs λ1.85e-3，10/10 场景 ≤0.35 dB）；
- 在覆盖充分的场景（Mip-360 7 个），更低 λ 买到的顶点是泛化细节；在低覆盖场景（flowers / treehill / T&T）它们只拟合训练视角，所以那四个场景的 36–58× 点是更优的 operating point。

这是一个**关于 operating point 选择的观察**，不是独立发现；放在分析章节，不进 intro。Intro 的第二论点仍然只能是 R3（事后 collapse 输 0.3–4.3 dB，14/14）。另外 upstream 的 train PSNR 是 6 视角样本（±0.3 dB），写的时候要标。

## 3. 重新打分

| 维度 | 早上 | 现在 | 变化原因 |
|---|---|---|---|
| 新发现 | 强 | 强 | R3 不变；R9 降级为分析观察 |
| 方法新颖性 | 中偏弱 | 中偏弱 | 没有新证据能改变它 |
| 系统完整度 / 工程量 | 强 | 强 | + 解码器计时、viewer |
| 结果强度 | 中 | 中偏强 | 50–100× + 训练更快 + 手机实时，三者同时成立 |
| 覆盖 | 强 | 强 | |
| 完整性 | 弱 | 中偏强 | 只缺定性图 |
| 部署可用性（新列） | 未测 | 强 | 任何 3DGS 类 codec 表里都是空的一列 |

## 4. 风险清单更新

1. **"Taming-3DGS 搬到 mesh"**：仍是最大风险，回应只有 R3 + 训练时间 + 手机。related-work 段落必须写三点差异（价格是真实字节而非点数；collapse 在训练中保留 optimizer 状态；mesh 上事后剪枝不可行而 Gaussian 可行）。
2. **"绝对 PSNR 低"**：不变。Table 2 分块（mesh 类 / 3DGS 类），加 "standard mesh / CPU decode / phone fps" 三列后，审稿人会看到我们在另一条坐标轴上。
3. **FPS 对比的诚实边界**：我们只测了自己的 viewer。不要写"比 3DGS 快 N 倍"；写"standard rasterisation pipeline, no sorting, 145–350 fps on iPhone 17 Pro at capture resolution"，并注明分辨率。若审稿人要对比，MobileNeRF 296 fps（239 MB，无室内）和 BakedSDF 72 fps 浏览器（434 MB）是文献里可引的数。
4. **定性图缺失**：现在是唯一被直接点名的空缺，分钟级 GPU。

## 5. 到截稿的顺序（更新）

1. 定性图（同字节 collapse vs λ 放大对比 + wireframe，Room + bicycle），一次 GPU 交互，<30 min。
2. related-work 区分段 + intro 第二段（R3），零 GPU，立刻可写。
3. Table 1 其余 9 场景的 generic codec（CPU 为主，每场景约 10 min GPU 评估）。
4. 外部 baseline 最后。

## 6. 一句话

今天之后，论文有了 3DGS 压缩文献里没有的一行：标准 mesh、单核秒级解码、手机 145–350 fps，而且训练比原版快。方法新颖性的短板没变，但"结果强度 + 部署可用性 + 完整性"三格都升了，整体是弱接收材料；补上定性图和 related-work 段落后可以投。
