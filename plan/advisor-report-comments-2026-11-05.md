# Yao 对 technical report（report-latex-v2）的逐条 comment：答复 + 改法（2026-11-05）

标记：**W** = 改措辞/补说明（今晚改进 report-latex-v3）；**F** = 改图表；**E** = 要跑实验（今晚提交：commands/lambda-k-fan-v1.md + commands/indoor-diag-v1.md，≈5 GPU-h，不训练）。
与 email 三问的对应：Q1（λ×字典联合 RD）= #22 + E；Q2（DC 7 bit）= #11 + #22 + E；Q3（end-to-end）= #2 + #6 + §末。每条先给中文答案，再给可直接贴进报告的英文。

## 方法部分

**1. "The same per-face quantity ΔD_f": what is the difference from per triangle?** — W
没有区别，face = triangle；报告里 split 用下标 t、prune 用 f 是同一个量换了字母。改法：全文统一用 "triangle" 和下标 t。
> *The prune uses the same per-triangle quantity ΔD_t, summed over the triangles around the edge.*

**2. B_a = 6 B, B_f = 2 B: how determined? from pretrained entropy codecs? vary with rate point?** — W（+ 可选 E）
来历：不是学出来的熵模型，是我们自己这个无损编码器在**一个**交付资产（Room，q14，K4096 R256）上的**实测流大小**除以元素数：位置残差流 ≈ 2.3 B/顶点、DC ≈ 1.5-1.75 B、两个字典索引 11.2 + 7.7 bit ≈ 2.4 B，合计 ≈ 6 B/顶点；连通性流 ≈ 31.7 bit/顶点 ≈ 4 B/顶点，按每顶点约 2 个面折成 2 B/面。它在训练前定下来，之后所有场景、所有 λ 都不变。
随码率变不变：会，但很小。44 个交付资产实测 9.5-13.3 B/顶点（中位 10.8；常数给的是 6 + 2×2 = 10），λ 越大（mesh 越稀）越贵一点（中位 10.4 → 11.4）：边更长，位置残差更大，边缓存命中更少。对决策的影响：判据是 ΔD/ΔR > λ，ΔR 整体偏 c 倍等价于把 λ 换成 cλ，±30 % 的价格误差 = λ 平移 ±30 %，不到 λ 网格一步（×3）的四分之一，工作点沿同一条曲线挪一点，曲线本身不变。更精确的做法（可做、未做）：先跑一遍读出该 λ 的实测价再训一遍（two-pass），或 B_a ∈ {8, 10, 13} 的敏感性实验（≈6 GPU-h，可选）。
> *B_a and B_f are not learned: they are the measured stream sizes of our own lossless coder on one delivered Room asset, divided by the number of vertices / faces (positions 2.3 B, DC 1.5 B, two dictionary indices 2.4 B per vertex; connectivity 31.7 bit per vertex ≈ 2 B per face at two faces per vertex), fixed before training and reused for every scene and λ. The true cost does drift with the operating point: over the 44 delivered assets it is 9.5–13.3 B per vertex (median 10.8 vs the 10 B the constants imply), rising slightly with λ because sparser meshes have longer edges and fewer cache hits. Since the decision rule compares ΔD/ΔR with λ, a price error of a factor c is identical to using cλ; the observed ±30 % is less than a quarter of one step of our λ grid, so it moves the operating point slightly along the same curve rather than changing the curve.*

**3. How is λ chosen for a target rate?** — W
λ 是 Lagrange 乘子，和 x264/HM 的 λ 模式、HAC/RDO-Gaussian 的 λ 扫描一样：给的是斜率不是码率，码率随 λ 单调下降。没有闭式的 λ→码率映射，但实测规律很稳：每个场景 log MB 对 log λ 近似线性，斜率 ≈ −0.6（room 22.1→2.9 MB、garden 36.4→5.0、kitchen 24.5→3.8，λ 跨 30×）。所以要打到目标大小：用这条经验律插值一次，或者做一次短校准。另一个模式是直接按字节预算停（我们的 G0 "train-at-budget" 消融就是这个：按 ΔD/ΔR 从高到低 split 直到预算），但预算模式让每个场景的边际斜率不同，跨场景不是 RD 最优；λ 模式给所有场景同一个汇率，这是我们选 λ 的理由。
> *λ is the Lagrange multiplier, as in λ-mode rate control of video codecs and in HAC / RDO-Gaussian: it fixes the marginal slope, not the size, and the size decreases monotonically with λ. We do not have a closed form; empirically log(size) is close to linear in log λ with slope ≈ −0.6 on every scene (Table 2), so a target size is reached by one interpolation on that law or one short calibration run. The alternative, stopping growth at a byte budget, is our "train-at-budget" ablation; it hits a size exactly but gives each scene a different marginal slope, whereas one λ is the same exchange rate everywhere.*

**4. ΔR_uv = B_a + n_uv B_f: collapse only removes one vertex?** — W
是：一次 edge collapse 把 u、v 合并成一个顶点（中点），减少 1 个顶点和这条边上的 n_uv 个面（内部边 2、边界边 1）。一轮 prune 做很多次这样的 collapse：按 ΔD_uv/ΔR_uv 从小到大取，要求被选的边互不相邻（独立集，每条边的两个环互不重叠），fold-over guard 拒绝会翻转幸存面的 collapse，一轮最多 15 % 顶点，12k/14k/16k/18k 共四轮。
> *One collapse merges u and v into their midpoint: one vertex and the n_uv faces on the edge disappear. A prune round applies many collapses, in increasing order of ΔD_uv/ΔR_uv, on an independent set of edges (no two share a vertex), skipping collapses that would fold a surviving face, up to 15 % of the vertices per round.*

**5. "which then creates about two faces per vertex"?** — W
Euler 公式：闭合流形三角网格 F ≈ 2V（E ≈ 3V）。我们的 mesh 有边界和非流形部分，实测 1.8-3.4 面/顶点（中位 2.3），所以 split 的价按"每新顶点带 2 个面"取整：3 顶点 + 6 面 = 3×6 + 6×2 = 30 B。
> *For a closed triangle mesh Euler's formula gives F ≈ 2V; our delivered meshes have 1.8–3.4 faces per vertex (median 2.3), so each vertex added by a split is charged two faces.*

**6. "Why fixed prices": did not understand; seems to contradict B_a, B_f.** — W
没有矛盾，是写法造成的：这一段里"read once from the codec's stream sizes"听起来像每次运行都去读，实际是**一次性校准**（#2），然后冻结；9.5-13.3 B 是事后在交付文件上量的，用来说明常数离真实值在 ±30 % 以内，不是输入。整段重写为"校准一次 → 冻结 → 事后验证 → 误差等价于 λ 平移"。
> *Why constants. B_a and B_f were calibrated once on one asset and then frozen; they are inputs to training, not outputs of it. Measured afterwards on all 44 delivered assets the true cost is 9.5–13.3 B per vertex against the 10 B the constants imply, and a price error of factor c is equivalent to using cλ (Section 1.2), i.e. it shifts the operating point along the curve by less than a quarter of a grid step. A per-scene price would buy nothing but would make λ a different exchange rate in every scene.*

**7. b = ⌈log2(extent/(0.1·median edge))⌉: extent is the dynamic range?** — W
是每个坐标轴的动态范围：extent = 顶点包围盒最长轴的 max − min；median edge = 训练好的 mesh 的三角形边长中位数。规则的意思是量化步长 = 1/10 中位边长（量化误差 ≤ 5 % 的典型边长），由此得到每轴 b 位（14-18 bit，按场景）。量化是每轴 [min, max] 上 2^b 级均匀格，不是立方体。Room 上的验证：q18→q14 −0.056 dB，q13 −0.19 dB 且出现翻面。
> *Positions are quantised per axis on a uniform grid of 2^b levels over [min, max]; b is chosen so that the grid step is one tenth of the median edge length of the trained mesh (extent = the largest bounding-box side), giving b = 14–18 per scene. On Room, going from 18 to 14 bits costs 0.06 dB and 13 bits starts flipping faces.*

**8. "cached-edge arithmetic coder": need more details.** — W
写清楚（supp 已有完整 spec：analysis/baselines/stage2-v2-bitstream-spec-v1.md §3）：面按 meshoptimizer 的 strip 顺序排；解码器维护最近 64 条**有向边**的环形缓存（每解一个三角形 (a,b,c) 推入 a→b、b→c、c→a）。编一个三角形：若它的某条边在缓存里（流形处通常是反向出现），写 1 bit "命中" + 缓存槽号（≤127 个符号）+ 第三个顶点；否则写 "未命中" + 三个顶点。顶点 id 的写法：第一次出现的顶点写一个 0（id 隐式 = 下一个空号，所以顶点顺序不用传），否则写与上一个 id 的 zigzag 差。所有符号过自适应二进制算术编码器（5 类 × 256 前缀上下文，不传表）。实测 ≈ 15 bit/面（原始 66），同一网格 Draco 因拆顶点反而更大，MPEG TFAN 几何 19.9 MB vs 我们 ~16 MB。它不是 Edgebreaker 一类（流形网格 ~2 bit/三角形），因为这个 mesh 是 45 % 边界边、几千个连通块的 soup，Edgebreaker/TFAN 不能直接用——这也是 #14 的答案。参考：Deering 1995 的 mesh buffer 思路（几何压缩）；自适应算术编码 Witten–Neal–Cleary 1987。
> *Connectivity. Faces are visited in a cache-friendly strip order. The decoder keeps a ring buffer of the 64 most recently decoded directed edges. For each triangle the encoder writes either hit (the triangle shares an edge with the buffer: one flag, the buffer slot, and the third vertex) or miss (three vertices). A vertex seen for the first time is written as a single zero symbol, its id being implicitly the next unused one, so no vertex order is transmitted; a known vertex is written as the zigzag difference to the last id. All symbols go through an adaptive binary arithmetic coder with 5 × 256 contexts and no transmitted tables. The stream costs ≈ 15 bit per triangle (66 raw). On this near-soup mesh (45 % boundary edges) Edgebreaker-class coders do not apply directly, and MPEG TFAN on the same mesh needs 19.9 MB for geometry against our 16 MB. The design follows the mesh-buffer idea of Deering [1995] with adaptive arithmetic coding [Witten et al. 1987].*

**9. "refit for 400 steps against the rendering loss": how?** — W
流程（deliver_coarse.py 默认，细节以仓库 sharing_after.convert 为准）：(i) 在 65 536 个顶点的 SH 样本上做 k-means（8 轮 Lloyd）得到 K 个主码本中心，再对残差做 R 个；(ii) 在全部顶点上做 4 轮"指派 ↔ 更新中心"；(iii) 冻结几何和指派，只把两个码本（K×45、R×45）和每顶点 DC 当参数，用训练视角的渲染损失（和训练同样的 L1 + D-SSIM）Adam 优化 400 步（bank lr 1e-4，residual 3e-4，DC 2e-3，余弦衰减，按验证损失早停、最少 400 步，带一个 0.005 的 anchor 项把码本拉向 k-means 初值）。效果：小码本 +0.6 dB（K1024 R1），大码本 +0.2 dB。
> *SH dictionaries. The 45 non-DC coefficients are clustered by k-means (8 Lloyd iterations on a 65 k-vertex sample, then 4 full-set assignment / update rounds) into K = 4096 main codes; the residuals are clustered into R = 256 codes. With geometry and assignments frozen, the two code banks and the per-vertex DC term are then optimised for 400 Adam steps against the training-view rendering loss (L1 + D-SSIM, as in training) with validation-based early stopping. The refit is worth +0.2 dB at K = 16384 and +0.6 dB at K = 1024.*

**10. "1-ring context model": brief description + reference.** — W
解码顶点 v 的码时，上下文 = 它已解码的 1 环邻居（面流里 id 更小的邻居）所用的**不同码**（≤32 个）。先编一个 "命中" bit（上下文 = 候选个数 m），命中则编槽号 s（1..m），否则用自适应 0 阶模型编原始码。命中率 0.12-0.22（随机为 1/4096），省 ≈ 0.5 MB/资产：索引本身接近不可压（11.2/12 bit），这就是为什么字典大小主要靠 λ 和 K 管而不是靠熵编码。参考：自适应算术编码 Witten–Neal–Cleary 1987；邻域预测属性编码的思路同 Touma–Gotsman 1998 / Draco（连续属性的平行四边形预测），这里是离散码的"同邻居码"预测，我们自己写的。
> *Indices are coded with a 1-ring context: for vertex v the candidates are the distinct codes of its already-decoded neighbours (at most 32); a hit flag (context = number of candidates) followed by the candidate slot is written when v reuses one of them, otherwise the raw code under an adaptive order-0 model [Witten et al. 1987]. Hit rates are 0.12–0.22 (vs 1/4096 for independent codes), saving about 0.5 MB per asset; the indices are otherwise close to incompressible (11.2 of 12 bits), which is why dictionary size is controlled by K and λ rather than by entropy coding. The neighbour prediction is the discrete analogue of parallelogram attribute prediction [Touma and Gotsman 1998; Draco].*

**11. "DC term is quantised to 7 bits and zstd-coded": representation, reference.** — W + E
表示：每通道在 [min, max] 上 2^7 级均匀标量量化，每顶点每通道 1 字节，字节交织（byte-shuffle）后 zstd（Collet，RFC 8878，级别 9），不做空间预测。实测 14.0 bit/顶点（garden，即 4.7 bit/通道），占文件 21 %。为什么 7 bit：Room 上 q8→q7 省 1.3 MB 掉 0.03 dB（0.022 dB/MB，和字典杠杆同一斜率，在 hull 上），q6 0.12 dB/MB（被丢顶点 dominated）。是否造成最高码率的失真：Room（非 indoor）上 q8 只找回 +0.025 dB；今晚在 12 个 λ 资产 + 2 个 indoor 原模型上都测 q8（E）。真正该做的改进是带预测/上下文的 DC 编码（DC 是 21 % 的文件，邻居预测估计能省其中 30-40 % = 文件 6-8 %），作为 future work 写明。
> *DC. The three degree-0 coefficients are uniformly quantised per channel to 2^7 levels over [min, max], stored as one byte per channel and compressed with byte-shuffle + zstd [Collet, RFC 8878] without spatial prediction (14 bit per vertex, 21 % of the file). Seven bits sit on the RD hull: on Room, 8 bits cost 1.3 MB for 0.03 dB (the same slope as the dictionary lever) and 6 bits lose 0.12 dB per MB saved. Table X reports the 8-bit variant on every λ asset.* ← 表 X 今晚的 E。

**12. "(the two dictionaries, one opacity scalar": how represented? opacity for the whole mesh?** — W
码本：两个 FP32 数组 K'×15×3 和 R'×15×3（只存实际用到的码，每码 180 B），zstd 后 ≈ 0.6-0.8 MB，固定开销（fp16 可再减半，未做）。opacity：**整个 mesh 一个标量**，不是每三角形：MeshSplatting 第二阶段的 per-vertex opacity logit 全部饱和（floor 0.999），我们用它们的均值 logit 替代，最大 opacity 误差 ≤ 1e-3，PSNR 差 < 1e-4 dB（Room 上 27.9019 vs 27.9018），省 ≈ 1.7 MB。
> *Side data: the two banks are stored as FP32 arrays over the codes actually used (180 B per code, 0.6–0.8 MB after zstd). Opacity is one scalar for the whole mesh: in MeshSplatting's opaque stage every per-vertex opacity saturates (≥ 0.999), so we store their mean logit; the largest per-vertex deviation is 10^-3 and the PSNR change is below 10^-4 dB, for 1.7 MB saved.*

**13. "reads the five streams" — only see 4.** — W
五个流 = 连通性、位置、**主字典索引、残差字典索引**（文中合写成"SH indices"一条）、DC；外加头部的 side data（码本、opacity、量化范围）。改列表为五项。

**14. "16 MB geometry": similar to SH? can it be further reduced through better context coding?** — W
Room 全量资产 31 MB ≈ 几何 16（连通性 ≈ 12 + 位置 ≈ 4-7）+ SH 15（索引 + DC + 码本）。能不能靠更好的上下文编码再压：位置流已是邻居预测 + 1536 个上下文的算术编码，平行四边形预测在这个 soup 上更差，余地不大；连通性是最大的一条（每顶点 ~4 B ≈ 15 bit/面），比流形网格的 Edgebreaker（~2 bit/三角形）高 5-7×，差的不是上下文模型，是**网格本身非流形**（45 % 边界边、几千连通块），任何 Edgebreaker/TFAN 类编码器都不能直接用。所以几何那 16 MB 的大头靠"网格更像 mesh"来省（流形 mesh 的连通性可降到 ~0.5 B/顶点），不靠熵编码；这篇论文明写为 limitation。
> *Geometry is 16 of the 31 MB (connectivity ≈ 12, positions ≈ 4). The position stream is already neighbour-predicted and context-coded; the connectivity stream, at ≈ 15 bit per triangle, is 5–7× above what Edgebreaker-class coders reach on manifold meshes, but those coders do not apply to MeshSplatting's output (45 % boundary edges, thousands of components). The remaining geometry redundancy is therefore in the mesh structure, not in the entropy coder; making the mesh manifold is future work.*

## 实验部分

**15. Table 1: show average RD curves (Fig 4 style) with the table; add a Mip-360 vs T&T row; T&T has more points.** — F
改法：Table 1 旁放一张"数据集均值 RD 曲线"图（两个面板：Mip-360 9 场景均值、T&T 2 场景均值；x = MB，y = PSNR，4 个 λ 点连线 + level-1.0 点 + FP32 点标在右侧）；表加 "dataset" 列；T&T 四个 λ 全列（28.7 MB/20.17、22.3/20.43、11.4/20.52、5.4/20.10，Table 2 末行已有）。

**16. Table 2: write (MB (ratio), PSNR); use ΔPSNR instead of PSNR.** — F
改为每格 "MB (ratio), ΔPSNR"，ΔPSNR 相对本场景 FP32；FP32 列保留绝对 PSNR。

**17. "The same λ lands at 2.9–6.4 MB": what does this mean?** — W
意思：同一个 λ 在不同场景停在不同大小，因为停止条件是 ΔD/ΔR 的内容依赖，不是顶点上限——这是"预算由内容决定"的证据。原句把两个 λ 的数字混在一起了。改为：
> *At λ = 6.2×10⁻³ the same price stops growth at 5.4–8.0 MB on the indoor scenes and 8.9–13.4 MB outdoors: the byte budget is an outcome of the scene content, not a preset cap.*

**18. "growth that the price lets through does not buy test PSNR there, which is exactly what §2.5 measures": what does this mean?** — W
意思：flowers/treehill/truck/train 上 λ 最小的两点是 dominated 的（更大的文件、不更高的 test PSNR），因为 ΔD_t 量的是训练视角误差，这些低覆盖场景上多出来的三角形只拟合训练视角（val 升 0.5-1.2 dB、test 不动）。和 §2.5 的关系原句写错了——§2.5 量的是"等顶点数下环内选择 vs 事后选择"，不是过拟合。改为：
> *On flowers, treehill, truck and train the two smallest-λ points are dominated: ΔD_t measures training-view error, and on these sparsely covered scenes the extra triangles reduce it without improving the test views (validation PSNR rises by 0.5–1.2 dB, test by ≤ 0.2). The Pareto hull is drawn through the non-dominated points. Section 2.5 shows the complementary fact: at equal vertex count, choosing the triangles in the loop beats choosing them afterwards.*

**19. "swapping only the geometry encoder for MPEG costs 13 %": MPEG takes more bytes? MPEG coding only geometry?** — W
对：那一行里 MPEG（SC3DMC/TFAN）只编位置 + 连通性（19.9 MB），SH 仍用我们的字典流；总文件从 31 变 35 MB（+13 %）。108 MB 那一行是 MPEG 全编（SH 作为 8 bit 属性 88 MB）。改为：
> *Keeping our SH streams and replacing only our position + connectivity streams by MPEG SC3DMC/TFAN (19.9 MB vs our 16 MB) makes the file 13 % larger; the 108 MB row is MPEG end to end, with the SH coefficients as 8-bit attributes (88 MB).*

**20. Fig 4 (engine-ready meshes): are all these points achievable by others, including the top right one? (b) not necessary.** — F + W
(a) 里每个点是一个已发表方法自己报的 (MB, PSNR)（Table 3 原样）：右上 = Binary Opacity Grids 913 MB / 25.62 dB；MiLo 253 / 24.09；RaDe-GS 1100 / 23.56；GOF 1500 / 20.78；MeshSplatting 作者 PLY 100 / 24.78。它们都没有码率控制，所以每个方法只有一个点；只有我们是一条曲线。别人到不了我们的点（没有编码器），我们也到不了 BOG 的 25.62（不透明 mesh 的 class gap）。(b) 删。图注补一句说明。

**21. Fig 5 (3DGS-class codecs): if others don't compare with GS we don't need it; (b) not wanted.** — F
主文删 Fig 5 和 Table 4，移到 supplement；主文留一句"绝对 PSNR 比 3DGS 类低 ≈3 dB 是不透明 mesh 表示本身的差距，未压缩的 MeshSplatting 已低 2.7 dB"。

**22. "Training is faster than upstream": what does this mean?** — W
> *Training takes 0.57× the upstream time because the priced growth stops at fewer triangles and the rasteriser's cost scales with their number; nothing else in the optimisation changed.*

**23. Indoor assets lose 0.5–0.75 dB more: did you try multiple dictionary sizes and λ → convex hull? 7-bit DC the reason at the highest rate? tried 8 bits?** — E（今晚）+ W
已有：K×R 扫描只在事后顶点层级上做过（Room 4 个层级 × 12 个 K×R、6 场景 backbone、贪心分配器 9/9 复现 Room 全局 hull），没在 λ 资产上做；DC q8 只在 Room 非 indoor 上做过（+0.025 dB / +1.3 MB）。今晚补：room / kitchen（indoor）/ garden × 4 λ × {K4096R1, K1024R256, K4096R1024, K16384R1024} + DC q8（lambda-k-fan-v1，≈4.5 GPU-h）和 indoor 原模型 level-1.0 的 q8 / keep / K 诊断（indoor-diag-v1，0.6 GPU-h）。结果进 Table X（每场景一张 λ×K×R 扇面图 + 联合 hull）。预期（Room 的斜率推）：扇面斜率 0.014-0.04 dB/MB 低于 λ 曲线斜率 → K4096R256 在联合 hull 上；q8 在最高码率点 +0.02-0.05 dB。若 indoor 的 K16384 或 q8 给 > +0.2 dB，顶部点改设置并在表里注明。

## 与 email 三问的衔接（Q3 end-to-end）
Q1、Q2 由上面的 E 直接回答。Q3（B_a/B_f 常数 + 固定上下文熵编码器 ≠ 端到端）：诚实的标签是"rate-in-the-loop with a calibrated constant proxy"。可训练熵模型会把 B_a 从常数变成随局部上下文变化的量（可预测的顶点更便宜），我们的两个常数是它的零阶近似；决策只看 ΔD/ΔR 的比值，代理误差 ≤30 %（#2）。截稿前做不到可微的指派 + 连通性码率；能做的是 (i) #2 的敏感性实验（可选 6 GPU-h），(ii) DC 预测编码的 CPU 估计，(iii) 把这一段写成 limitation + future work。英文草稿见 plan/advisor-rd-questions-2026-11-05.md §5。
