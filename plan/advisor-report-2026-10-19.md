# 给老板的汇报提纲（2026-10-19）

一句话主线：**Differentiable opaque mesh 的顶点数应该由码率决定，而且必须在生长时决定。** 我们把一个 Lagrangian 价格 λ 放进 densification，顶点数变成 λ 的输出，一次训练直接落在 RD hull 上。七个 Mip-360 场景平均 80× 压缩只掉 0.88 dB，解码后是标准 mesh。

## 1. 核心发现与贡献（按汇报顺序）

### 发现 1：mesh 不能像 Gaussian 那样事后剪（动机，最强的证据）
- 训练时就限制顶点数，比训练完再 collapse 到同样顶点数高 **+0.3 到 +4.3 dB**，七个场景 14 个点无一例外，预算越紧差距越大。
- 对比：3DGS 事后剪 60 % 只掉 <0.3 dB，所以 3DGS 压缩全是 post-hoc；mesh 顶点被拓扑耦合，这条路走不通。这解释了为什么 HAC / RDO-Gaussian 那一套不能直接搬到 mesh 上。
- 图：`analysis/rd/budget-vs-collapse-7scenes-v1.png`

### 贡献 2：方法 —— rate-controlled growth（λ-split）
- 分裂判据：三角形承担的像素残差 / 分裂新增的交付字节 > λ。字节按我们 codec 实测的每场景 B/vertex 计。
- 结果 a：同字节下 λ ≥ 手设 cap，**10/10 个可比点**，+0.0 到 +0.57 dB；对事后 collapse +0.3 到 +1.6 dB。
- 结果 b：场景自适应。同一个 λ 不设 cap，各场景自己停在 dense 顶点数的 41–100 %（λ=2e-3）或 15–43 %（λ=6e-3）。
- 结果 c：选择规则的作用随预算收紧而增大（Room：1.3M 打平，0.5M +0.25 dB，0.23M 明显赢，3 MB / 246×）。
- 图：`analysis/rd/lambda-vs-cap-7scenes-v1.png`，`analysis/rd/room-lambda-vs-cap-v1.png`

### 贡献 3：系统结论 —— 只有几何流需要进训练环
- 外观流（SH → K4096×R256 码本索引）事后共享就够：跨场景码本无用、QAT 负结果、K×R backbone 顺序七场景完全相同。
- 拓扑流是地板：Draco 对照拆出 4.5× 顶点，总字节更大。
- XYZ 精度规则 b = ceil(log2(extent / 0.1·median edge)) 七场景命中或只低 1 bit。
- 这回答审稿人"为什么不端到端把码本也放进训练"。

## 2. Headline 数字（七场景均值，相对原 FP32 模型）

| 路线 | 大小 | PSNR 变化 | 压缩比 |
|---|---|---|---|
| λ = 2e-3 | 19.6 MB | −0.26 dB | 36× |
| λ = 6e-3 | 8.8 MB | −0.88 dB | 80× |
| 旧路线：事后 collapse | 23.6 MB | −0.44 dB | 28× |

逐场景表：`analysis/rd/seven-scene-hulls-v2-table.md`；总图 `analysis/rd/seven-scene-hulls-v2.png`。比 MobileNeRF / BakedSDF / BOG / MiLo 小 4–30×（文献数字，见 memory baseline 表）。

## 3. 可以展示的成品

1. 七场景 budget vs collapse 图（发现 1）。
2. 七场景 λ vs cap 图（贡献 2）。
3. 七场景 RD hull 总图 + headline 表（λ 为 headline）。
4. Room λ sweep 图（6 个 λ，从 2.9M 到 0.23M 顶点）。
5. 完整 codec：742 MB → 交付文件，解码后是引擎可用的标准 mesh；B/vertex 分解（拓扑 ≈4、XYZ 2–2.6、DC ≈1.5、索引 ≈2.4）。
6. 分析图（可选）：码本索引空间相关性 `room-index-locality-v1.png`、Room 码本 RD `room-rd-codebook-v2.png`。

## 4. 诚实的负结果（主动讲，显得扎实）
- **等斜率不成立**：同一 λ 各场景的 dB/MB 斜率差 5×（0.019–0.100）。原因是 λ 定价 L1 质量而 PSNR 是 log-MSE。论文只 claim 场景自适应 + ≥ cap。
- **没有自然停止**：9500 步时仍有三角形通过门槛，顶点数被 upstream 的 densify_until=10000 截住。λ 目前决定的是增长速率与场景间分配。
- Room 1.3M 处 λ 和 cap 打平；bonsai 低码率点没有可比的 cap 点。

## 5. 10/19 新出的三个结果（都已完成，可直接汇报）
- **自然停止（0 GPU，从日志分析）**：有价格时通过门槛的三角形比例几何衰减（λ=6e-3 半衰期约 3000 步），没价格时反而膨胀；λ=6e-3 下 7/7 场景三角形数在 7500→9500 步已稳定在 ±8 %。λ 给出的是由它决定的稳态，不是手设的截止步数。图：`analysis/rd/pass-curves-7scenes-v1.png`。
- **C：选择规则对照（已出，10/19）**。同样的增长日程（18 个 interval 每步选同样多的三角形），只把"按误差选"换成 upstream 的 importance 采样：4.31 MB / 26.81 dB，349k 顶点。λ 曲线在 4.31 MB 处插值 26.93 → **误差排序本身 +0.12 dB（同字节）**；cap 曲线在 4.31 MB 处约 26.5 → **渐进的 λ 式增长日程 +0.3 dB**。结论：λ 对 cap 的增益两部分都有，日程占大头、排序占小头，都要写。附带机制：同样数量的分裂，importance 采样长出来的顶点到 Delaunay 时只剩 370k，误差排序的剩 525k —— importance 采样分裂的三角形随后被剪掉得多 30 %，是"白做的功"。
- **B：phase-B λ 剪枝（已出，10/19）**。训练环内的 edge collapse，Delaunay 之后 4 轮，同一个 λ 反向用（ring 残差 / 省下的字节 < λ 就合并）。Room 两个 λ 都是：顶点 −11 %、交付字节 −8 %、PSNR 不变（+0.01～0.02 dB）；同字节 +0.11～0.16 dB。4 轮收敛，限速器从不生效，说明 λ 自己决定了剪多少。诚实处：**“删顶点的价格随面数变化”这一点没有得到支持**（被选中的边并不偏向省字节多的边，残差项主导），论文里剪枝写成“同一价格的反向应用”，不 claim 变价。
- 两个结果合起来给方法段的说法：λ 在生长期决定“长多少、长哪里”（日程 +0.3、排序 +0.12），在 phase B 再免费收回 8 % 字节。

## 6. 想请老板拍板的问题
1. 标题/定位：*Grow What You Can Afford: Rate-Controlled Mesh Growth* 这条线是否 OK？codec 和压缩比降为次要贡献。
2. 对比范围：主表只比 mesh 类（MobileNeRF / BakedSDF / BOG / MiLo）+ 事后 collapse，3DGS 压缩（HAC++）只在文字里承认不同表示类，可以吗？
3. 补场景：flowers / treehill / T&T（truck、train）要不要进主表？每场景约 1.5–2 GPU 小时，当前 FairShare 很低。
4. Deadline 按 11 月中排期是否准确。
