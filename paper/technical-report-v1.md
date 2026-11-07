# Rate-aware compression of MeshSplatting: what we did, stage by stage
Technical report v1 for the group, 2026-10-30. Wenhao Su (NYU). Every number below is a measured value from the project files named in brackets; nothing is projected.

## 0. One paragraph

MeshSplatting (arXiv 2512.06818) trains a triangle soup with per-vertex spherical harmonics (SH, degree 3) directly from images and outputs an ordinary vertex-coloured mesh. Its assets are huge: 740 MB for a Mip-NeRF 360 scene (3 M vertices x 48 SH floats + faces). We built the first complete encode/decode pipeline for this representation and showed three things. (1) A codec alone, keeping every vertex, brings a scene to 21-25x smaller at -0.2 dB (Room 742 MB -> 30.9 MB, 28.26 -> 28.06 dB). (2) Vertices cannot be removed after training: collapsing a trained mesh to a given vertex count loses 0.3-4.3 dB compared with a mesh that was trained to that count (7 scenes, 14 points, fine-tuning does not recover it). (3) Therefore the vertex count has to be decided inside training: we put a byte price lambda into MeshSplatting's densification and pruning, so the number of vertices becomes an output of one lambda that transfers across scenes. On 11 scenes (Mip-NeRF 360 + truck/train) one lambda gives 20.7 MB / 32x / -0.22 dB and a larger lambda 9.4 MB / 73x / -0.59 dB, with training 0.57-0.87x of the original wall-clock, 2-6 s decode on one CPU core, and the decoded file being a standard indexed mesh that renders at 350 fps on an iPhone 17 Pro.

## 1. What MeshSplatting does (upstream, unchanged parts marked)

Upstream training runs 30 000 iterations:
- **Representation.** Vertices with position, one opacity logit (`vertex_weight`), SH degree 3 (1 DC + 15 rest coefficients x RGB = 48 floats); faces as an index list; one global softness `sigma`. Colour is evaluated per vertex and interpolated across the triangle (Gouraud); the rasteriser blends soft triangles with per-layer alpha, and an opacity schedule makes the model opaque by the end.
- **Growth (densification).** Every interval in the first ~9 500 iterations, upstream splits a fixed fraction of triangles (`add_percentage`) chosen by importance sampling, up to a hand-set cap `--max_points`. This is the step where the vertex count is decided. [upstream `TriangleModel.add_new_gs`]
- **Topology maintenance.** Importance-based face pruning during growth; at `densify_until + 1000` a restricted Delaunay rebuild (effrdel) re-triangulates the soup; a final importance cleaning removes unused faces. [upstream, untouched]
- **Optimisation.** L1 + D-SSIM loss, per-parameter learning rates, 2x/4x supersampling after 20k/25k iterations, opacity schedule. [upstream, untouched]
- **Output.** `point_cloud_state_dict.pt` with positions, faces, weights, SH. Rendered through the same rasteriser for evaluation (test = every 8th image).

**Where we intervene.** Only in two places, both in the *growth* phase: (a) the split decision at every densification interval (Sec. 3), and (b) four extra prune rounds at iterations 12k/14k/16k/18k, after the Delaunay rebuild and before supersampling (Sec. 3). The losses, learning rates, opacity schedule, Delaunay rebuild, supersampling and final cleaning are upstream code imported, not edited. [analysis/rdgrowth/train_rd_v3.py header]

So to the question "did we change the optimisation (step 2)?": **no**. We change *which triangles are split and when growth stops*, and we add a *reverse* step that collapses edges under the same price. The appearance optimisation is the upstream one.

## 2. The codec (what makes 740 MB into 31 MB without touching the vertex count)

Measured on Room, 742 MB FP32: the three streams are SH (48 floats/vertex, the bulk), positions (3 floats), and faces (3 int32 per face). [analysis/baselines/generic-codecs-2026-10-24.md]

**2.1 Appearance: a shared SH dictionary.** The 45 non-DC coefficients of every vertex are replaced by two indices into two k-means banks fitted per scene: a main bank with K = 4096 entries and a residual bank with R = 256 entries (each 15 x 3 floats). The DC colour stays per vertex, quantised to 7 bits per channel. A 400-step refit of the banks against the rendering loss follows the assignment (`deliver_coarse`). Cost of this step alone: ~-0.03 dB beyond quantisation on Room. Why it works: per-vertex SH is spatially redundant, which we tested rather than assumed: a codebook fitted on another scene does not transfer (cross-scene codebook refuted), quantisation-aware training of DC is 0.1 dB *worse* than post-hoc quantisation, index frequencies are uniform (no symbol skew to exploit); the only redundancy a coder can cash in is spatial agreement between neighbours. [memory rate-aware-results, crossbank-index-locality-results]

**2.2 Geometry precision.** Positions are quantised on a per-axis integer grid with b bits where b = ceil(log2(extent / (0.1 * median edge length))); this rule hit the measured knee on 7/7 scenes (q14 Room, q15 garden/bonsai/counter, q16 kitchen/bicycle/stump). Going one bit finer is free in dB and costs ~1.5 MB; one bit coarser costs 0.06-0.2 dB. [memory rate-aware-results, scenes4-results]

**2.3 Entropy coding of the five streams, in decode order** [analysis/baselines/stage2-v2-bitstream-spec-v1.md, written from the source]:
1. *Connectivity*: faces are reordered by meshoptimizer (cache-aware strip order), vertices are renumbered in first-reference order, so the permutation is never transmitted. A directed cache of the 64 most recent edges turns each face into "hit: cache slot + one new vertex" or "miss: three vertices"; a vertex seen for the first time costs one arithmetic-coded zero symbol. Adaptive binary arithmetic coder with 5 x 256 contexts, no tables sent. Result 30.4 bits/vertex on Room (raw 66 bits/face). Draco's Edgebreaker on the same soup splits vertices 4.5x and ends up larger than our whole file.
2. *Positions*: integer residuals against a causal neighbour predictor rebuilt from the faces decoded so far, arithmetic coded.
3. *SH indices*: the two index arrays coded with a context made of the SH codes of already-decoded 1-ring neighbours (saves ~2 bits/vertex over zstd).
4. *DC colour*: 7-bit, zstd with byte shuffle.
5. *Side data*: the two SH banks (4096 x 45 + 256 x 45 floats = ~0.8 MB), one opacity constant (the trained `vertex_weight` logits are all equal, so one scalar replaces 1.3-2 MB), quantisation ranges.

The codec is lossless over its input; all loss is in 2.1-2.2. The encoder verifies a byte-exact round trip before writing, and every block carries a SHA-256.

**2.4 Result of the codec alone (level 1.0, all vertices kept).** Room 30.9 MB / 28.061 (-0.20 dB), garden 32.5 MB / 24.458 (-0.27). Against generic codecs on the *same* model: quantised PLY + zstd-19 158.5 MB at the same -0.17 dB (5.1x larger than ours), Draco with SH as a generic attribute 456 MB (14.8x), the texture route (xatlas UV + baked 8k JPEG) 137 MB at -3.0 dB because view dependence is lost. [generic-codecs-2026-10-24.md, Table 1 draft; figure analysis/rd/generic-codecs-v1.png]

## 3. Why fewer vertices must be decided during training (the key finding) and how we decide them

**3.1 Post-hoc removal fails on meshes.** Take the trained dense model and collapse edges down to N vertices (upstream importance order, then a 3000-step fine-tune and the same codec). Compare with a model that was *trained* with a vertex cap of N. At equal N the collapsed mesh is 0.3-4.3 dB worse on all 14 points over 7 scenes (Room 1.35 M vertices: 26.66 vs 27.98 dB). On Gaussians the same post-hoc pruning of 50-60 % of primitives costs a few tenths of a dB; on an opaque mesh the vertices are coupled through the connectivity, and the optimiser never gets to re-place the survivors. [memory scenes7-budget-vs-collapse; figure analysis/rd/budget-vs-collapse-7scenes-v1.png]

**3.2 The dense model over-fits.** The train/test PSNR gap of the dense model reaches 1.3-2.0 dB on kitchen and Room and shrinks when fewer vertices are grown (Room 2.45 dB dense vs 1.70 dB at 0.44 M vertices); on flowers / truck / train the smaller asset has *higher* test PSNR than the larger one, on treehill it is within 0.1 dB. That is why removing roughly half of the vertices in the loop costs only 0.2-0.3 dB of test PSNR: those vertices were fitting training views. [analysis/rd/table2-deployability-v1.md, R9 table]

**3.3 Rate-controlled growth (what replaces `add_new_gs`).** At every densification interval we accumulate, per triangle t, the residual mass dD_t = sum over pixels whose front-most triangle is t of the per-pixel L1 error, averaged per view and normalised per megapixel. The price of splitting t is dR_t = bytes the split adds to the *delivered file*: three midpoint vertices x bytes per vertex measured on that scene with the codec of Sec. 2 (topology ~4 B + XYZ 2-2.6 B + DC ~1.5 B + SH indices ~2.4 B = 9-11 B). Rule: **split t iff dD_t > lambda * dR_t.** Upstream's per-interval growth limiter stays as a speed cap; lambda decides which triangles split and when growth stops. lambda = 0 degenerates to error-ranked growth with no price (ablation); importance mode reproduces upstream sampling through the same code path (control). [analysis/rdgrowth/rd_growth_v2.py]

**3.4 Phase-B pruning (same price, reverse direction).** At iterations 12k/14k/16k/18k, collapse edge (u,v) iff the residual mass of the faces around u and v, per byte the collapse saves (one vertex and the 1-3 faces that disappear), is below lambda. At most 15 % of vertices per round; a fold-over check rejects collapses that flip faces. It removes 10-11 % of the vertices and 8 % of the bytes at no PSNR cost. The optimiser state survives, which is what post-hoc collapse cannot offer. [analysis/rdgrowth/rd_prune.py; analysis/rd/room-ablation-table-v1.md]

**3.5 What lambda buys versus a hand-set cap.** At equal bytes lambda >= cap on 10/10 comparable points (+0.0 to +0.57 dB); at low rate on Room: cap 26.63 < importance sampling replayed on the lambda schedule 26.81 < lambda 27.37 < lambda + prune 27.38 at 5.2 MB. One lambda stops growth at 15-100 % of the dense vertex count depending on the scene (bicycle keeps most, Room least), i.e. the count is scene-adaptive without per-scene tuning. [analysis/rd/room-ablation-table-v1.md; memory rdgrowth-scenes7-results]

## 4. The delivered pipeline and the no-leakage check

The package `meshsplatting-rd-complete-v1.zip` runs the whole chain as stages (`rd_pipeline.py`, restartable): train with rate control -> precision rule -> optional level selection -> 3000-step fine-tune -> dictionary fit (K4096 R256, 400 steps) -> **encode** (writes exactly one file, `asset.mslb`; all diagnostics stay encoder-side) -> **standalone decode** -> render from the decoded file -> viewer export. [memory rd-complete-package-v1]

The decoder is checked for information leakage mechanically, not by inspection: `verify_isolation.py` launches `decode.py` as a child process in a scrubbed environment, inside a work directory that contains only the asset file; a Python audit hook (`sys.addaudithook`) classifies every file open and subprocess the child makes, and any access outside the one file fails the run (VIOLATION -> exit 1). The decoded tensors are then compared with the encoder's input (up to the vertex permutation, which is not transmitted by design) and must match exactly; the encoder is also checked to be deterministic (same input -> same bytes). `result.json` records `isolation_pass`, `encoder_deterministic`, file MB, ratios and dB. This package has been verified locally; the full HPC run of the one-shot chain is still pending.

Decode cost: 2.3-6.4 s on one CPU core for 0.44-1.14 M vertices (~5 us/vertex); output is an indexed triangle mesh plus per-vertex codebook indices, rendered by an ordinary vertex shader that looks up the two banks as textures. [table2-deployability-v1.md]

## 5. Current results (11 scenes: Mip-NeRF 360 x 9 + Tanks&Temples truck, train)

Operating points of the delivered file, test PSNR of the decoded asset [analysis/rd/rd-11scenes-v1.csv; original MB from rd/table2-deployability-v1.md (bonsai is listed as 680 MB in seven-scene-hulls-v2-table.md; to reconcile); figure analysis/rd/rd-11scenes-v1.png]:

| scene | original MB / PSNR | lambda 1.85e-3: MB / PSNR / x / dB | lambda 6.2e-3: MB / PSNR / x / dB |
|---|---|---|---|
| room | 742 / 28.26 | 12.6 / 28.00 / 59x / -0.26 | 5.2 / 27.38 / 142x / -0.88 |
| garden | 740 / 24.73 | 24.6 / 24.50 / 30x / -0.23 | 10.2 / 23.65 / 73x / -1.08 |
| bonsai | 662-680 / 27.29 | 14.0 / 26.65 / 47x / -0.64 | 6.1 / 26.16 / 109x / -1.13 |
| counter | 575 / 25.94 | 14.8 / 25.79 / 39x / -0.15 | 6.7 / 25.26 / 86x / -0.68 |
| kitchen | 456 / 26.45 | 16.7 / 26.24 / 27x / -0.21 | 7.5 / 25.41 / 61x / -1.05 |
| bicycle | 680 / 23.03 | 27.0 / 23.02 / 25x / -0.02 | 11.9 / 22.69 / 57x / -0.35 |
| stump | 698 / 24.93 | 21.4 / 24.71 / 33x / -0.21 | 8.9 / 24.12 / 79x / -0.81 |
| flowers | 668 / 19.45 | 24.2 / 19.07 / 28x / -0.38 | 12.0 / 19.21 / 56x / -0.24 |
| treehill | 692 / 20.39 | 27.4 / 20.45 / 25x / +0.06 | 11.9 / 20.38 / 58x / -0.01 |
| truck | 432 / 22.39 | 19.3 / 22.20 / 22x / -0.19 | 9.4 / 22.37 / 46x / -0.02 |
| train | 485 / 18.90 | 25.4 / 18.66 / 19x / -0.24 | 13.4 / 18.68 / 36x / -0.22 |
| **mean** | | **20.7 MB / 32x / -0.22 dB** | **9.4 MB / 73x / -0.59 dB** |

Other measured columns: training wall-clock 0.87x (lambda 1.85e-3) and 0.57x (6.2e-3) of the original on an L40S; decode 2.3-6.4 s on one CPU core; rendering of the decoded mesh at full capture resolution: < 2 ms on a MacBook M2 Max, iPhone 17 Pro 2.8 ms (352 fps) on Room and 6.3-6.9 ms (145-160 fps) on garden / bicycle at 16-17 Mpx. [table2-deployability-v1.md; screenshot analysis/figures/viewer-iphone17pro-bicycle-2026-10-25.jpg]

Absolute quality is MeshSplatting's: on Room the original model is 28.26 dB where 3DGS is 31.7 and HAC++ reaches 31.75 at 3.4 MB. The codec loses 0.2-0.9 dB of that; the 3.5 dB gap is the representation (the authors' own table shows the same gap). We report it as is: the asset is a standard mesh, decodes on a CPU in seconds, and renders in any engine at phone frame rates, which none of the 3DGS codecs offer.

## 6. Figures we have (paths under /mnt/project-files/analysis/)

- rd/rd-11scenes-v1.png: RD points per scene (original, codec level 1.0, lambda 1.85e-3, lambda 6.2e-3, post-hoc collapse).
- rd/budget-vs-collapse-7scenes-v1.png: train-at-budget vs post-hoc collapse at equal vertices (the key finding).
- rd/lambda-vs-cap-7scenes-v1.png and rd/room-lambda-vs-cap-v2.png: lambda vs hand-set cap at equal bytes; rd/pass-curves-7scenes-v1.png: growth curves (vertices vs iteration) under one lambda across scenes.
- rd/generic-codecs-v1.png: Table 1 as a plot (ours vs quantised+zstd, Draco, texture route).
- rd/room-rd-codebook-v2.png, rd/room-index-locality-v1.png: K x R dictionary hull and spatial locality of the SH indices (why the context coder works).
- figures/viewer-iphone17pro-bicycle-2026-10-25.jpg: decoded asset on a phone.
- Still to make: a qualitative figure (same bytes, post-hoc collapse vs lambda, with wireframes) and the FPS-vs-triangles plot.

## 7. Open items before submission

Qualitative figure; Table 1 extended from Room/garden to all scenes (CPU work); the external baseline table (mesh class: MobileNeRF, BakedSDF, BOG, MiLo, Triangle Splatting, MeshSplatting RGB-only PLY; 3DGS class: 3DGS, Scaffold-GS, Compact3D, LightGaussian, HAC, HAC++, RDO-Gaussian, reported with their PSNR/SSIM/LPIPS as they are); the HPC run of the one-shot package.
