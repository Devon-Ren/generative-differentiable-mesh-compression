# 全部结果 + 够不够 CVPR main track（2026-10-25）

立场同 10/19 的 contribution review：按审稿人 / AC 标准打分，不替自己辩护。baseline model（外部同类方法）按用户要求稍后考虑，本文只评已有材料。

## 0. 结论

**工程量足够，创新量是中等，整体从 10/19 的 borderline 升到了 borderline 偏弱接收。**
10/19 列出的五个风险里，场景覆盖（11/11）和"表里没有对手"（同一资产上的通用 codec，Table 1）已经解决，codec 细节有了按源码写的 bitstream spec。
还没解决的是方法新颖性（会被说成"Taming-3DGS 搬到 mesh"）和三样不花 GPU 训练的东西：定性图、FPS / 解码时间、related work 的区分段落。
能把它推到稳的弱接收的，是把"码率控制生长"讲成一个**关于 opaque mesh 的发现**（事后删顶点失败 + dense 训练在过拟合），而不是一个比 cap 高零点几 dB 的方法。

## 1. 全部结果一览（证据 → 状态）

| # | 主张 | 证据 | 场景 | 状态 |
|---|---|---|---|---|
| R1 | 完整 codec 本身（不减顶点）就 21–25× | level-1.0：Room 30.9 MB −0.20 dB … truck 20.0 MB −0.08 dB | 11/11 | 完成 |
| R2 | 同一资产上的通用 codec 全输 | 同画质 quant+zstd 159/172 MB、Draco 全 SH 456/544 MB，我们 30.9/32.5 MB（5×、15–17× 更小）；纹理 / RGB-only 路线 137 MB 还掉 3 dB | Room、garden | 完成 2 场景，其余场景是 CPU 活 |
| R3 | **mesh 不能事后压**：事后 collapse 同顶点数输 0.3–4.3 dB | 14/14 点，finetune 补不回 | 7/11 | 完成（新 4 场景只有 level-1.0 点，结论不矛盾） |
| R4 | 码率进生长环：一个 λ 跨场景 | λ 1.85e-3：20.7 MB / 32× / −0.22 dB；λ 6.2e-3：9.4 MB / 73× / −0.59 dB（36–142×） | 11/11 | 完成 |
| R5 | λ ≥ cap（同字节） | 10/10 可比点，+0.0 到 +0.57 dB | 7/11 | 完成，写成"cap 是 λ 的特例 + 一个设置跨场景" |
| R6 | phase-B λ-prune | 同画质少 4–8 % 字节；被翻面检查卡住 | 7/11 | 完成，小而稳，ablation 一行 |
| R7 | 冗余按流不同 | 流分解（拓扑最大）；SH 索引接近不可压（11.2/12、7.7/8 bit）；跨场景码本证伪；QAT 负结果；K×R backbone 7/7 同序；精度规则 7/7 命中或低 1 bit | 3–7 场景 | 完成，分析章节 |
| R8 | 解码即标准 mesh | bitstream spec（容器、edge-cache 连接、XYZ 邻居预测、索引上下文），字节级 round-trip 校验 | — | 完成文字，FPS 未测 |
| R9 | **dense 训练在过拟合**（新） | Room train/test gap dense 2.45 dB vs 440k 顶点 1.70 dB；flowers/truck/train 上小档 test PSNR ≥ 大档，且 ≥ dense level-1.0（字节约一半），val 和 LPIPS 仍随顶点上涨 | Room + 3/4 新场景 | 只有部分证据，推断，需要一张跨场景表 |

## 2. 按审稿人重新打分（对比 10/19）

| 维度 | 10/19 | 现在 | 理由 |
|---|---|---|---|
| 新发现 | 强 | 强 | R3 仍是全文最硬的东西；R9 若能补成跨场景表，会变成第二个发现 |
| 方法新颖性 | 中偏弱 | 中偏弱 | 没变。λ-split 和 RD densification（Taming-3DGS、RDO-Gaussian）同族；codec 部件（meshoptimizer 排序、edge cache、上下文算术编码、k-means 码本）都是已知技术 |
| 系统完整度 / 工程量 | 中 | 强 | 11 场景、全链路、bitstream spec、Table 1 对手、可复现脚本 |
| 结果强度 | 中偏弱 | 中 | 50–100× 区间在 11 场景成立；绝对 PSNR 仍低（MeshSplatting 自身差距，审稿人第一眼还是会看到） |
| 覆盖 | 弱 | 强 | Mip-360 9/9 + T&T 2 |
| 完整性（定性 / FPS / 时间） | 弱 | 弱 | 还没做 |

## 3. 会被拒的理由，以及现在能怎么回应

1. **"Taming-3DGS 搬到 mesh + 一堆已知 codec 部件。"** 这是最大风险。回应只能靠 R3 和 R9：在 Gaussian 上事后剪枝几乎免费，所以 budget densification 对 Gaussian 只是省训练时间；在 opaque mesh 上它是唯一能到低码率的路径，而且 dense 训练多出来的顶点主要在记训练视角。要写进 intro 第二段，不能藏在 ablation。
2. **"绝对 PSNR 低。"** 回应已有：RD 口径、相对原始模型的 dB、why-mesh 段落（标准 mesh、任意光栅器）。这个风险降不掉，只能不让它成为唯一印象：主表把 mesh 类和 3DGS 类分块，FPS 和"是否标准 mesh"作为列。
3. **"没有定性结果 / 没有速度。"** 现在会被直接点名，修起来最便宜。
4. **"prune 只省 4–8 %。"** 不当贡献讲，写成 λ 在 phase B 的反向应用，ablation 一行。

## 4. 缺口清单（按性价比）

| 项 | 算力 | 解决哪个风险 |
|---|---|---|
| R9 跨场景过拟合表：11 场景 × {original, λ 1.85e-3, λ 6.2e-3} 的 train / test PSNR 和 gap，从现有训练日志 grep | 0（CPU，分钟） | 风险 1，可能升成第二个发现 |
| 定性图：同字节事后 collapse vs λ 放大对比（Room + 一个室外场景）、wireframe 显示顶点分布随 λ 变化 | 分钟级 GPU | 风险 3 |
| FPS + 解码时间（标准 viewer 或 nvdiffrast 光栅化交付 mesh；codec 解码秒数） | 分钟级 GPU | 风险 3、R8 |
| Table 1 扩到其余 9 场景（`TEX=""` 跳过纹理），顺带用同一 evaluator 重评原始模型 LPIPS | 0 训练，每场景约 10 min GPU 评估 | R2、LPIPS 口径 |
| Related work 区分段（Taming-3DGS / RDO-Gaussian / Mini-Splatting / LightGaussian 三点差异） | 0 | 风险 1 |
| 外部同类 baseline（稍后，用户决定） | 待定 | 风险 2 |

以上除 baseline 外，总共不到 2 GPU 小时。

## 5. 到截稿的顺序建议

1. 先做 R9 表（零 GPU）。它决定 intro 能不能有第二个发现，越早知道越好。
2. 同时开始写：intro、方法、codec 章节都已经有材料，等不到所有数字齐。
3. 定性图 + FPS 一批提交（分钟级）。
4. Table 1 其余场景（CPU 为主）。
5. 最后放外部 baseline。

## 6. 一句话

工程量够 main track；创新量单看方法不够，要靠"opaque mesh 不能事后压、dense 训练在过拟合、所以码率必须进生成环"这条发现撑起来。补齐 R9 表、定性图和 FPS 后，是稳的弱接收材料；现在直接投，是 borderline 偏弱接收。

## 附：R9 表的取数命令（CPU，任意节点，几秒）

```bash
AN=/scratch/zr2245/analysis
cat > $AN/gap-table-v1.sh <<'SHEOF'
#!/usr/bin/env bash
# gap-table-v1.sh (2026-10-25): train/test PSNR at iter 30000 from existing train logs (original + v3 lambda runs).
for S in room garden bonsai counter kitchen bicycle stump flowers treehill truck train; do
  for G in /scratch/zr2245/$S-vis-collapse /scratch/zr2245/$S-rd3-lam0.00185-mu1.0 /scratch/zr2245/$S-rd3-lam0.0062-mu1.0; do
    L=$G/train.log
    if [ ! -f $L ]; then echo "$S | $(basename $G) | no train.log"; continue; fi
    TR=$(grep -h "\[ITER 30000\] Evaluating train" $L | tail -1 | grep -o "PSNR [0-9.]*")
    TE=$(grep -h "\[ITER 30000\] Evaluating test" $L | tail -1 | grep -o "PSNR [0-9.]*")
    echo "$S | $(basename $G) | train ${TR:-n/a} | test ${TE:-n/a}"
  done
done
SHEOF
bash $AN/gap-table-v1.sh | tee $AN/gap-table-v1.txt
```
