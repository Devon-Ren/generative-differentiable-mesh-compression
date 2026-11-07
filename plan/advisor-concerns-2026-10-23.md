# Advisor concerns after the 10/22 report — baselines, topology coding, MPEG — 2026-10-23

Advisor (10/22): (a) accepts the generative-compression gain IF it is a real encoder→bytes→decoder result with no
leakage; (b) accepts the mesh-vs-Gaussian argument (no blending, plain rasterization, fewer device limits, real surface).
Concern 1: which baselines (include 3DGS class and get attacked on PSNR, or mesh class only; self-vs-self not accepted).
Concern 2: topology/geometry coding details (index assignment, location residual, connectivity coding, how neighbours
enter the context coder); is our codec optimal; try MPEG on the same mesh with vertex colours baked to a texture map.

## 0. On "real encoder/decoder, no leakage" — yes, and here is what the number is

Every PSNR we report is: train → export_asset → bytes on disk (`asset.mslb`) → decode on CPU → render the decoded mesh
on the held-out test views (every 8th image, never used in training). The export path asserts a byte-exact round trip
of topology and attributes (stage2 `equal(...)` + topology round-trip assert, see stage2-v2 patch), and the ctx index
coder was verified decodable on CPU. λ is a global constant (2e-3 / 6e-3 for every scene, not tuned per scene or on
test views). Train/test gap is reported (rd/error-anatomy-room-2026-10-19.md). The only thing not in the bytes is the
renderer itself. This should be a paragraph in the paper ("Protocol") and one line in Table 1's caption.

## 1. Baselines — answer: both tiers, and the MPEG/Draco runs ARE the external baseline

The advisor's two concerns answer each other: the external, independently built codecs (Draco, MPEG V-DMC, glTF +
meshopt + KTX2 textures) applied to the identical MeshSplatting asset are exactly the "not self-vs-self" baselines
for Table 1. They were built by other people, are the industry standard for the representation we output, and a
reviewer cannot call them straw men if we run them at their best settings and report their RD points.

Table 1 (same representation, same scenes, decoded test PSNR vs MB):
| row | what | status |
|---|---|---|
| FP32 PLY | minimal float asset | known (456–742 MB) |
| zstd / lzma on the PLY | "write bytes, entropy code" | CPU, trivial |
| Draco, full attribute (qp = our precision rule, SH as 8-bit generic attributes) | best general mesh codec | run_draco.sh exists for connectivity; full-attribute run pending (CPU) |
| Draco + baked RGB texture (xatlas) + JPEG/KTX2 | the "MPEG-style" texture-mapped route; **no view dependence** | to build (CPU); flag no-SH |
| MPEG V-DMC test model, if it accepts the mesh | same, with video-coded displacement + texture | feasibility check (see §3) |
| authors' RGB-only PLY | ~100 MB, −2 dB; flag no-SH | download + measure |
| post-hoc collapse route (our old codec) | | done, 7 scenes |
| rate-controlled growth (ours) | | done, 7 scenes; 11 after batch B |

Table 2 (cross-representation, reference only): include 3DGS, Scaffold-GS, HAC++ **greyed**, with columns
`standard mesh? / renderer / FPS on a plain rasterizer / training time / size`, and the mesh family (MobileNeRF,
BakedSDF, BOG, MiLo, MeshSplatting). Including the 3DGS rows is safer than omitting them: the gap is public
(MeshSplatting's own Table 1 shows 24.78 vs 27.21), a reviewer will compute it anyway, and an omitted row reads as
hiding. The defence is one sentence + citations: every opaque-mesh method is 2.5–3.5 dB under 3DGS; the paper's
claim is quality per byte **within the engine-ready mesh class** and a codec that is 36–80× on the same asset.
Precedent: MobileNeRF, BakedSDF, BOG were all accepted with PSNR below their volumetric contemporaries because they
target deployment; the comparison they made was size/FPS, which is ours too.

So: not "only self", not "beat 3DGS", but (i) external generic codecs on the same asset, (ii) external mesh methods on
size + FPS + PSNR, (iii) 3DGS as a labelled reference row.

## 2. What our codec actually does (from the 10/13 code audit; to be re-verified against the stage2 source)

Streams per vertex (Room dense, 41.4 MB, 3.19 M v): topology 30.4 bits, XYZ q18 29.9 (→ less at q14), DC 15.1 (now q7),
primary index 10.8, residual index 10.2, vertex_weight 5.3 (now 0, constant), banks 1.9.

1. **Vertex index assignment**: vertices are renumbered by a *strip-fetch* traversal of the triangle list (a vertex gets
   its new index the first time a triangle references it; triangles are visited in a cache-friendly order, like
   meshoptimizer's vertex-cache optimizer). Result: the three indices of a face are close to recently issued indices.
2. **Connectivity**: *edge-cache arithmetic coding*. A FIFO of recently decoded edges is kept; each new triangle is coded
   as (which cached edge it shares, + its third vertex as either a recently referenced vertex or a brand-new index),
   all symbols through an adaptive arithmetic coder. This is the meshoptimizer-index-codec idea, not "write 0,1,2,
   1,2,3 as bytes". Measured 30.4 bits/vertex ≈ 15 bits/face (raw 3×22 = 66 bits/face; int32+lzma 71 bits/vertex).
3. **Location**: XYZ on the scene-adaptive grid (q14 Room … q16 bicycle, rule step ≤ 0.1 × median edge), coded as the
   residual against a causal prediction from already-decoded neighbours (parallelogram-type), adaptive binary
   arithmetic coding. So yes: residual to nearby decoded points.
4. **SH index context** (stage2-v2 `index_ctx.cpp`): decoding order = topology order, so when vertex i is decoded its
   already-decoded 1-ring neighbours N(i) are known from the faces. The coder first codes "same primary code as one
   of N(i)?" (hit rate 0.12–0.22 vs 1e-3 for independent codes) and which one; on a miss, the raw code through an
   adaptive order-0 model. The neighbour set is the **whole decoded ring**, i.e. the "0–1–2–1–2–3" reading, not a
   one-to-one previous-vertex chain. Residual index: hit 0.02–0.05, coded ≈ raw (noise-like layer).
5. **DC**: quantised q7, no neighbour prediction (zstd). Known gap, ~1–2 bits/vertex at most.

## 3. Is it optimal? What the measurements already say, and what MPEG would add

- Draco connectivity-only on the same face list: 2.65 bits/vertex *but* it must split non-manifold vertices: 3.19 M →
  14.36 M decoded vertices (4.5×), so its qp14 total is 32.8 MB vs our topology+XYZ 24.2 MB. The mesh is a
  near-soup: 45 % boundary edges, 18 % non-manifold edges, 8,277 components; Edgebreaker/TFAN's 1–2 bits/face
  assume a manifold. Our 30 bits/vertex is mostly the information "which corners share a vertex" in a soup, and
  topology cannot be predicted from geometry (only 21 % of edges lie within 16-NN). So the coder is not the
  bottleneck; vertex count is — which is what rate-controlled growth attacks.
- Honest gaps that a reviewer could still raise, all cheap (CPU): DC without prediction; the index coder's ~1.3 bit
  headroom at K1024; no attribute-aware vertex order. Each ≤ 1–2 bits/vertex ≈ 2–5 % of the file. Worth an
  ablation row, not a method.
- **MPEG route (V-DMC: base mesh via Edgebreaker/Draco + displacements + texture atlas via a video codec).** Two
  obstacles make it a *different, weaker representation* rather than a better codec of ours: (i) it needs a UV
  parametrisation; xatlas on a 3.2 M-vertex, 8 k-component non-manifold soup produces thousands of charts with seams
  on almost every triangle (chart count ≈ component count or worse), so the atlas is large and the seams cost bits;
  (ii) a texture stores RGB, so the 45 view-dependent SH coefficients are dropped — that is the authors' RGB-only
  PLY, which loses ~2 dB. A texture of 48 channels is not something a video codec can take. So the MPEG/texture
  route lands in the "no-SH" tier of Table 1, which is still worth one row because it is the question every mesh
  person will ask ("why not just bake a texture?"), and the answer (−2 dB, bigger file) is in our favour.
- What I propose to run (all CPU, zero GPU, Room + garden level-1.0 meshes first):
  1. Draco full-attribute: positions at our qp, SH as 16 generic attributes of 3 × 8 bit, compression level 10;
     report MB and decoded PSNR (needs a decode → our renderer path, same as the PLY baseline).
  2. Texture route: bake per-vertex DC (and, separately, the SH-evaluated colour at the training-view mean direction)
     into an xatlas atlas at 4k/8k, Draco for geometry + UV, JPEG q90 / KTX2 for the texture; decode, render, PSNR.
  3. V-DMC test model feasibility: check whether the public test model accepts a non-manifold mesh without UVs
     (it expects a base mesh + UV atlas); if it does not, say so in the paper and keep row 2 as the MPEG-style
     stand-in. Do not spend more than a day on it.
  4. A one-page bitstream specification of our codec for the supplement (traversal, edge cache symbols, XYZ
     predictor, index context, DC), written from the stage2 source — this is what the advisor is asking for when she
     says "re-examine the details". Needs the stage2 codec directory (topology.py, edge.cpp, native.cpp,
     index_ctx.cpp) copied into the project folder.

## 4. Slide fix
Deck clip 01 said "faces → context coder, ~2 bit/face"; the measured cost is ≈15 bits/face. Fixed in
slides/deck-v2/01-pipeline-v2.mp4 (clips 02–04 unchanged).
