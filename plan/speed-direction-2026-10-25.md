# "速度 / 可部署性"作为第二贡献？（2026-10-25，回应用户）

## 1. 先纠正"Taming-3DGS 搬到 mesh"这句话的来源

这是 10/19 审查里**最刻薄的审稿人口吻**，不是我们自己的定位。事实层面：
- 3DGS 不全是事后剪枝。LightGaussian、Compact3DGS（部分）是事后；但 Taming-3DGS 把 budget 放进 densification，RDO-Gaussian（ECCV 2024）把熵码率放进训练环（训练中剪枝 + 熵约束 VQ），HAC 系列训练时带 rate 项。所以"rate in the loop"在 Gaussian 文献里存在，审稿人会拿它来比。
- 我们真正不同的三点：(a) 价格作用在**生长**（split）上，ΔR 是交付文件的真实字节；(b) phase-B 在训练环内做 edge collapse 并保持 optimizer 状态，这是 mesh 独有的操作；(c) 最重要的，**在 mesh 上这是必需而不是可选**：Gaussian 事后剪 60 % 掉 <0.3 dB，所以 rate-in-loop 对 Gaussian 只是省训练时间；opaque mesh 事后 collapse 输 0.3–4.3 dB（14/14 点）。
- 结论：机制上的新颖性是增量的（这点要认），**必要性 + 发现是新的**。论文把 (c) 放 intro 第二段，related work 正面列 Taming-3DGS / RDO-Gaussian 并说清 (a)(b)(c)，这条质疑就能顶住。"码率进生长环"仍然是贡献，但讲法是"mesh 上唯一能到低码率的路径"，不是"比 cap 高 0.3 dB"。

## 2. 速度方向：哪些有空间，哪些没有

用户设想的产品链路：拍照 → 生成 mesh（训练）→ 压缩 → 解码 → 手机 GPU 实时渲染。按阶段看：

| 阶段 | 现状 | 有无空间 | 代价 |
|---|---|---|---|
| 训练（MeshSplatting 30k 步） | 原始 30–90 min / 场景（L40S，20k/25k 后 2×/4× 超采样是主要成本）；λ 运行 22–46 min | **有，而且已经免费拿到**：顶点少 2–7× → 每步更快。只需从现有 train.log 的时间戳抽 wall-clock，零 GPU | 0 |
| 训练再快 5–10× | 需要改 MeshSplatting 的优化器 / 超采样日程 / Delaunay 实现 | 是另一篇论文（Taming-3DGS 那类），我们算力和时间都不够，且会和"不改上游"冲突 | 高，不建议 |
| 编码（k-means 码本 + 400 步共享 finetune + 导出） | 约 8–11 min | 有一点（码本 rounds、finetune 步数），但不是卖点 | 低 |
| 解码（C++ edge / XYZ / index 解码） | 未测，估计秒级 | 只需测量 | 0 |
| 渲染 FPS | 只有上游 4× 超采样可微光栅器的 38 fps，不代表部署 | **这是最大空间**：交付物是标准 mesh + per-vertex SH，用标准 GPU 管线（Gouraud + 顶点 SH shader）在桌面应为数百 fps，在手机浏览器里 1–3 M 三角形很可能 30–60 fps。需要一个 viewer 来测 | 低：写一个 three.js/WebGL viewer（我来写），用户导出一次 decoded mesh，在桌面和手机上各测一次 |

## 3. 判断

- **不要把"把训练做快"当第二创新点**：它是纯代码优化、算力不够、和 Taming-3DGS 正面撞车，而且不解决审稿人对方法新颖性的质疑。
- **要把"可部署性"做成可测量的表**，这和 engineering-need 的定位完全一致，也是 why-mesh 论点的实证：一张"每阶段时间 + 文件大小 + 桌面 / 手机 FPS"的表（Table 2 的列），加一个手机浏览器里跑 9 MB 场景的截图 / 视频作为 Fig. 1 的一部分。外部 mesh 类方法（MobileNeRF、BakedSDF、BOG、MiLo）都报 FPS 和训练时间，我们不报就是缺。
- 顺带的免费结论："码率进生长环 → 顶点少 → 训练更快"是 rate-in-loop 在 mesh 上的第二个红利（Gaussian 文献里 Taming-3DGS 的卖点就是这个），从日志直接抽。
- 真正能顶住新颖性质疑的仍是发现层面（事后压缩失败 + dense 过拟合，见 cvpr-readiness §1 R9），不是速度。

## 4. 具体动作（算力 < 0.5 GPU-h，大部分零）

1. 训练 wall-clock 表（零 GPU）：从 11 场景 original 和两档 λ 的 train.log 时间戳抽总时长 + 顶点数。命令见附录。
2. 解码时间：对 11 场景的 asset.mslb 跑一次 `topology.read_asset`，计时（CPU，秒级）。
3. Viewer：我写 `analysis/viewer-v1/`（three.js，自定义 shader 做 degree-3 SH per vertex + Gouraud，加载一个紧凑二进制），给用户一个导出脚本把 decoded mesh 写成该二进制；用户在桌面 Chrome 和手机上各开一次，读 FPS。
4. 把 1–3 的数字合成 Table 2 的"训练时间 / 编码 / 解码 / FPS 桌面 / FPS 手机 / 是否标准 mesh"列。

## 附录：训练 wall-clock 抽取（CPU，任意节点）

```bash
AN=/scratch/zr2245/analysis
cat > $AN/walltime-v1.sh <<'SHEOF'
#!/usr/bin/env bash
# walltime-v1.sh (2026-10-25): training wall-clock from the first/last "[dd/mm hh:mm:ss]" stamps in train.log, plus final vertex count from rd_log/ckpt.
for S in room garden bonsai counter kitchen bicycle stump flowers treehill truck train; do
  for G in /scratch/zr2245/$S-vis-collapse /scratch/zr2245/$S-rd3-lam0.00185-mu1.0 /scratch/zr2245/$S-rd3-lam0.0062-mu1.0; do
    L=$G/train.log; [ -f $L ] || { echo "$S | $(basename $G) | no train.log"; continue; }
    T0=$(grep -o "\[[0-9][0-9]/[0-9][0-9] [0-9:]*\]" $L | head -1); T1=$(grep -o "\[[0-9][0-9]/[0-9][0-9] [0-9:]*\]" $L | tail -1)
    MIN=$(python3 -c "
import sys,datetime as d
p=lambda s:d.datetime.strptime(s.strip('[]'),'%d/%m %H:%M:%S')
a,b=p('$T0'),p('$T1'); m=(b-a).total_seconds()/60; print(round(m if m>=0 else m+24*60,1))")
    TE=$(grep -h "\[ITER 30000\] Evaluating test" $L | tail -1 | grep -o "PSNR [0-9.]*")
    echo "$S | $(basename $G) | $T0 -> $T1 | ${MIN} min | test ${TE:-n/a}"
  done
done
SHEOF
bash $AN/walltime-v1.sh | tee $AN/walltime-v1.txt
```
