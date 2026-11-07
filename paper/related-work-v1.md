# Related work 草稿 v1（2026-10-30）

用途：论文 §2 Related Work，四段 + 一个"三点区分"清单（readiness 清单里的风险 1：被说成 "Taming-3DGS 搬到 mesh"）。引用用 bibkey 占位，数字只用我们已核实过的（analysis/baselines/baselines-room-garden-v1.md、rd/generic-codecs-v1）。英文为论文正文，每段后的中文是写作意图，不进论文。

---

## 2. Related Work

**Mesh-based radiance representations.**
Several methods reconstruct a scene as a triangle mesh so that it can be rendered by a standard rasterizer. MobileNeRF [chen2023mobilenerf] bakes a NeRF into textured polygons with a small per-pixel MLP; BakedSDF [yariv2023bakedsdf] and Binary Opacity Grids [reiser2024bog] extract a watertight surface from a volumetric model and attach view-dependent appearance to it; MiLo [milo2025] keeps a mesh in the loop of Gaussian-splat training. More recently, *differentiable* opaque meshes are optimised directly from images: Triangle Splatting [held2025triangle] and MeshSplatting [meshsplatting2025] treat triangles as the primitive, grow and prune them during training, and output an ordinary vertex-coloured mesh (per-vertex spherical harmonics) that any engine can load; Radiance Meshes [radiancemeshes2026] reach 3DGS-level quality with volumetric tetrahedra instead of an opaque surface. All of these are concerned with reconstruction and rendering speed; the delivered assets are large (hundreds of MB per Mip-NeRF 360 scene: MobileNeRF 239 MB, BakedSDF 434 MB, BOG 0.85 GiB, MeshSplatting 742 MB for *room*) and none of them studies how the representation compresses. The only released reduction for MeshSplatting is an RGB-only PLY export that drops the spherical harmonics (about 100 MB at about −2 dB). Our work starts from MeshSplatting's representation and asks the compression question.

> 意图：把 differentiable opaque mesh 放进 mesh 系谱，强调"都没做过压缩"，并把 Radiance Meshes 点出来（它说明 2–3 dB 的差距是 opaque 单一 primitive 的类属性，不是 mesh 本身，和我们 why-mesh 段一致）。

**Compressing 3D Gaussian Splatting.**
Compression of 3DGS is a mature line: Compact3D [lee2024compact], LightGaussian [fan2024lightgaussian] and EAGLES [girish2024eagles] prune primitives post hoc and quantise or distil their attributes; Scaffold-GS [lu2024scaffold] and the hash-grid context models of HAC [chen2024hac] and HAC++ [chen2025hacpp] reach 3–8 MB on *room* at 31.8–32.2 dB by replacing free Gaussians with anchors and entropy-coding their features; RDO-Gaussian [wang2024rdo] optimises rate and distortion end to end with entropy-constrained vector quantisation. Two properties of Gaussians make this family possible and make it inapplicable to an opaque mesh. First, Gaussians are independent: removing one changes nothing but its own contribution, so pruning 50–60 % of them costs a few tenths of a dB and almost all compression can happen *after* training. On an opaque mesh the vertices are coupled through the connectivity, and we show (Sec. X) that collapsing a trained mesh to a vertex count that training itself reaches costs 0.3–4.3 dB on all 14 tested points, with no recovery by fine-tuning. Second, Gaussian codecs own their decoder: the learned context model is part of the format. We require the decoded asset to be a plain indexed triangle mesh with per-vertex attributes, so the codec may only use priors that a standard decoder can undo.

> 意图：承认 3DGS 压缩很强（不藏 HAC++ 的数字），然后把"不适用"的理由落在两条可检验的事实上：事后删顶点失败（R3）和解码端是标准 mesh（R8）。

**Densification under a budget.**
Our rate-controlled growth is related to methods that constrain *how many* primitives a model grows. Taming 3DGS [mallick2024taming] densifies toward a user-set primitive budget with a score-based selection; 3DGS-MCMC [kheradmand2024mcmc] fixes a maximum count and relocates Gaussians; Mini-Splatting [fang2024mini] densifies freely and then simplifies to a sparse set; RDO-Gaussian folds a learned rate into the loss. These methods control a *count* (or the entropy of their own latent) and, because post-hoc pruning of Gaussians is nearly free, their budget is mostly a training-time economy. We price each split by the **bytes it adds to the delivered file** (vertex attributes and topology streams measured on the actual codec), compare that price with the view-consistent error it removes, and let the vertex count be an *output* of a single Lagrangian λ that transfers across scenes (Sec. X). On an opaque mesh this is not an economy but the only route to low rates, since the same count cannot be reached after training (Sec. X).

> 意图：这一段就是"三点区分"的正文版本：定价对象（真实交付字节 vs 数量 / 自有 latent 的熵）、控制作用的位置（mesh 上事后不可达 vs Gaussian 事后免费）、输出（顶点数是 λ 的输出，一个 λ 跨场景）。

**Mesh and attribute compression.**
Connectivity coding is a classical problem: Edgebreaker [rossignac1999edgebreaker] and TFAN [mamou2009tfan] reach a few bits per vertex on manifold meshes, Draco [draco] and meshoptimizer [meshoptimizer] are the practical references, and MPEG V-DMC [vdmc] targets dynamic textured meshes. Attribute coding relies on quantisation plus prediction from already-decoded neighbours (parallelogram prediction and its variants). These codecs are agnostic to the rendering loss and to the statistics of a *trained* mesh: on MeshSplatting's non-manifold triangle soup (35 % boundary edges, 23 % edges with three or more faces on *room*) and 48 spherical-harmonic coefficients per vertex, Draco with full attributes is 15–17× larger than our asset at equal quality (Table 1). We reuse the standard pieces where they apply (vertex ordering, an edge cache for connectivity, context-adaptive arithmetic coding) and add what the representation needs: a geometry-precision rule derived from the edge length, a shared spherical-harmonic dictionary with neighbour-context index coding, and the in-training rate control above.

> 意图：说明 codec 部件都是已知技术（接受"不新"的批评），但对手在同一资产上输 15–17×，差的是表示专属的先验，不是编码技巧。

---

## 三点区分（intro 第二段 / rebuttal 用，不逐字进 related work）

| | Taming-3DGS / 3DGS-MCMC / Mini-Splatting | RDO-Gaussian / HAC++ | 本文 |
|---|---|---|---|
| 定价对象 | primitive 数量（或分数） | 自有 latent 的学习熵 | 交付文件的真实字节（顶点属性 + 拓扑流，按 codec 实测） |
| 控制为什么必须在训练环里 | 不必须：Gaussian 事后剪枝几乎免费，预算主要省训练时间 | 不必须：端到端但对象仍是 Gaussian | 必须：opaque mesh 事后 collapse 到同顶点数掉 0.3–4.3 dB（7 场景 14 点），finetune 补不回 |
| 输出 | 用户设的数量 | 带自定义解码器的码流 | 标准 indexed mesh + 每顶点属性，任何光栅器可解；顶点数是一个 λ 的输出，同一 λ 跨 11 场景停在 dense 的 15–100 % |

## 需要核对的引用（写进 bib 前）
- MobileNeRF (Chen et al., CVPR 2023); BakedSDF (Yariv et al., SIGGRAPH 2023); Binary Opacity Grids (Reiser et al., SIGGRAPH 2024); MiLo (2025, 核对会议); Triangle Splatting (Held et al., 2025); MeshSplatting (arXiv 2512.06818); Radiance Meshes (CVPR 2026, 核对作者).
- Compact3D (Lee et al., CVPR 2024); LightGaussian (Fan et al., NeurIPS 2024); EAGLES (Girish et al., ECCV 2024); Scaffold-GS (Lu et al., CVPR 2024); HAC (Chen et al., ECCV 2024); HAC++ (Chen et al., 2025); RDO-Gaussian (Wang et al., ECCV 2024).
- Taming 3DGS (Mallick et al., SIGGRAPH Asia 2024); 3DGS-MCMC (Kheradmand et al., NeurIPS 2024); Mini-Splatting (Fang & Wang, ECCV 2024).
- Edgebreaker (Rossignac, TVCG 1999); TFAN (Mamou et al., 2009); Draco (Google); meshoptimizer (Kapoulkine); MPEG V-DMC.
- 文中数字的出处：MobileNeRF / BakedSDF / BOG 大小 → baselines-room-garden-v1.md；MeshSplatting 742 MB、RGB-only PLY ≈ 100 MB / −2 dB → 同文件；HAC++ room 3.4–7.6 MB / 31.75–32.05 → 同文件；事后 collapse 0.3–4.3 dB / 14 点 → scenes7-budget-vs-collapse；Draco 15–17× → generic-codecs-2026-10-24.md；边的流形统计 → P1 planarity（room λ6.2e-3）；15–100 % → rdgrowth-scenes7-results。
