# Bitstream specification of the MeshSplatting delivery codec (stage2-v2) — supplement draft v1

Written 2026-10-24 from the uploaded source `stage2-v2-src-1024.zip` (`STAGE2_VERSION = stage2-v2 2026-10-15`:
`codec/{delivery.py, topology.py, edge.cpp, native.cpp, local_codec.py, index_ctx.cpp, index_context.py,
appearance.py, codec_benchmark.py}`, `runtime.export_asset`). Every statement below is read off that code, not off memory.
Decoder-side facts are marked **D**: the decoder needs nothing but the bitstream.

## 0. One-paragraph summary (for the paper)

The asset is a single file. After an entropy-coded container header, five streams are decoded **in this order**, each
conditioned only on what was decoded before: (1) connectivity, by a directed recent-edge cache with adaptive binary
arithmetic coding; (2) vertex positions, as integer residuals against a causal neighbour predictor rebuilt from the
decoded faces; (3) the two SH dictionary index arrays, with a context formed by the SH codes of already-decoded
1-ring neighbours; (4) DC colour, quantised and zstd-coded without prediction; (5) small FP32 side data (the two SH
banks, one opacity constant, quantisation ranges). No vertex permutation, neighbour table or predictor is transmitted:
vertex order is *defined* as first-reference order of the face stream, so a new vertex costs one arithmetic-coded
zero symbol. The lossy steps are all upstream of the bitstream (SH dictionary assignment, XYZ/DC quantisation, one
opacity constant); the codec itself is lossless and the encoder verifies a byte-exact round trip before writing.

## 1. Container (`codec_benchmark.pack/unpack`)

```
"MSLB001\0" | uint64 LE header_length | JSON header | concatenated block payloads
```
The header lists every field: `name`, `kind` (torch / numpy / scalar), `dtype`, `shape`, scalar `value`, and for arrays
a list of blocks (≤ 4 MiB of rows each) with `offset, size, rows, cols, mode, compressed, sha256`. Per block the encoder
tries `zstd-9(raw)` and `zstd-9(byte-shuffle(raw))` and keeps the smaller (`mode ∈ {raw, shuffle}`); raw is kept when
neither is smaller. **D** every block is SHA-256-checked on unpack. The header is ~2 kB; it is the only uncompressed text.

Fields of a final asset (`export_asset(..., weight='const', index_codec='ctx')`, topology `edge-arithmetic`, XYZ mode 2,
entropy 1):

| group | fields | dtype / size |
|---|---|---|
| connectivity | `top_schema=1, top_codec=3, top_n (V), top_f (F), top_stream` | uint8 AC stream |
| positions | `local_schema=1, local_topology=0, local_xyz=2, local_entropy=1, local_vertex_count, local_xyz_itemsize, local_residual` | uint8 AC stream |
| position dequant | `quant_triangles_points_{bits, kind=2, low[3], scale[3]}` | scalars + 2 float64[3] |
| SH indices | `ctx_schema=1, ctx_mapping, ctx_mapping_k, ctx_mapping_u32, ctx_residual, ctx_residual_k, ctx_residual_u32` | uint8 AC streams + scalars |
| SH banks | `app_schema=2, app_bank (K'×15×3), app_residual_bank (R'×15×3)` | float32 |
| SH bookkeeping | `app_rows (= 0..V-1 when the residual layer covers every vertex), app_gain (0×3), app_correction (0×15×3)` | uint32 / empty |
| DC | `features_dc (V×1×3) + quant_features_dc_{bits, kind=2, low[3], scale[3]}` | uint16 (b ≤ 16), zstd/shuffle |
| opacity | `vertex_weight_const, vertex_weight_cols, opacity_floor` | scalars |
| render meta | `sigma, active_sh_degree, delivery_schema=1` | scalars |

## 2. Vertex and face order (`topology.reorder('strip-fetch')`)

Encoder only. Faces are reordered by meshoptimizer `optimize_vertex_cache_strip` (a cache-aware, strip-like triangle
order), then vertices are renumbered by `optimize_vertex_fetch_remap`: a vertex receives the next free id the first
time a face references it; vertices referenced by no face are appended at the end in source order. All per-vertex
arrays (positions, DC, SH indices) are permuted accordingly, residual rows are re-sorted, faces are rewritten in the new
ids **keeping their orientation**. The permutation is **not** transmitted (`semantic_audit`: same attribute multiset,
same oriented face multiset, `permutation_transmitted=False`). Consequence used by every later stream: when face t is
decoded, every vertex with id < (ids issued so far) is already known, and a vertex that appears for the first time has
exactly id = `next`.

## 3. Connectivity stream (`edge.cpp`, mode `edge-arithmetic`)

State: `Cache` = ring buffer of the **64 most recent directed edges** (each decoded triangle (a,b,c) pushes a→b, b→c,
c→a; a hash map gives the slot of a directed edge if it is still in the buffer); `next` = number of implicitly issued
vertex ids; `last` = last vertex id coded (any role).

Per triangle, the encoder scans its three cyclic edges (x,y), j = 0,1,2, and takes the **first** one found in the
cache, same direction first (`reverse=0`), then opposite direction (`reverse=1`; in a consistently oriented manifold
the shared edge is seen reversed, so this is the common case). Symbols:

```
hit :  tag=1 (cat 0) | token = slot*2 + reverse (cat 1, ≤ 127) | vertex c (cat 4)
miss:  tag=0 (cat 0) | vertex a (cat 2) | vertex b (cat 3) | vertex c (cat 4)
vertex v:  varint 0            if v == next   (new vertex, id implicit; next++)
           varint zigzag(v − last) + 1  otherwise;   last = v
```
On a hit the triangle is emitted rotated to start at the matched edge, so decoded faces equal the encoder faces
**up to cyclic rotation** (orientation preserved; checked at export). Every symbol byte goes through the adaptive
binary arithmetic coder of `native.cpp` (32-bit range coder, counts halved at 32768) as 8 binary decisions down a
prefix tree: context = (category ∈ {tag, token, a, b, c}, 8-bit prefix) → 5 × 256 adaptive contexts, no tables
transmitted. **D** decoder rebuilds the cache from its own output; it range-checks slot < min(count, 64), tag ∈ {0,1},
and 0 ≤ v < V.

Measured: Room dense 30.4 bits/vertex ≈ 15 bits/face; garden v0.8 10.08 MB / 2.54 M v = 31.7 bits/vertex. (Raw:
3 × ⌈log2 V⌉ ≈ 66 bits/face; Draco connectivity on the same soup splits vertices 4.5× and is larger than our whole asset.)

## 4. Position stream (`delivery.quantize`, `native.cpp` build/encode, `local_codec`)

**Quantisation (encoder, lossy).** Per axis k: `low_k = min x_k`, `scale_k = (max x_k − low_k)/(2^b − 1)`,
`q_k = rint((x_k − low_k)/scale_k)` clipped to [0, 2^b−1]; b is the scene rule b = ⌈log2(extent/(0.1·median edge))⌉
(Room q14, garden/bonsai/counter q15, kitchen/bicycle/stump q16); stored as uint16 when b ≤ 16. **D** dequantisation
`x = q·scale + low` in float64, cast to float32. It is a per-axis grid, not a bounding-cube grid.

**Predictor (D, rebuilt from decoded faces, `native build`).** For every vertex i, over all edges (u,i) of the face
list with u < i, `p0[i]` = the **smallest-index earlier neighbour**; −1 if none. (The build also records a
parallelogram triple `p1..p3[i]` — the opposite vertex of a face across an edge (b,c) with b,c < i — but the shipped
export uses mode 2, where it is not read.)

**Residual coding.** Vertices in id order, axes in order; `kind(i)` = 2 if `p0[i] ≥ 0`, else 1 (predict from vertex
i−1), else 0 for vertex 0 (predict 0). `pred = q[p0[i]]` (mode 2). Symbol `r = zigzag(q_i − pred)`, coded as 32 binary
decisions MSB-first with context
```
(kind ∈ {0,1,2,3}, axis ∈ {x,y,z}, bucket(previous r on this axis) ∈ {≤1, ≤15, ≤255, >255}, bit position 0..31)
```
→ 4·3·4·32 = 1536 adaptive contexts. Because of §2, `p0[i]` is almost always a spatial neighbour that appeared in the
same or a recent triangle, so the residual is on the order of one edge vector. **D** decoded q must lie in [0, 2^24).

Measured: Room dense q18 29.9 bits/vertex (raw 54); garden v0.8 q15 5.16 MB / 2.54 M v = 16.3 bits/vertex (raw 45).
`local_codec benchmark` also runs mode 1 (previous vertex) and mode 3 (parallelogram); the export uses mode 2
("neighbor-arithmetic"), the smallest in that benchmark on this soup-like mesh (45 % boundary edges).

## 5. SH dictionary index streams (`index_ctx.cpp`, `index_context.py`)

Appearance (encoder side, `appearance.py`): SH degree-1..3 coefficients (15×3 per vertex) = `bank[mapping[v]]`
(+ `residual_bank[residual_mapping[v]]` on rows `app_rows`; in the final format the residual layer covers every vertex,
so `app_rows = 0..V−1`). Banks are k-means centroids; only codes actually used are stored (`torch.unique` renumbering, so
K' ≤ K, R' ≤ R). `app_mapping` and `app_residual_mapping` are coded by the same context coder with K = K' and K = R'.

**Context (D).** From the decoded faces, undirected edges (lo,hi) are made unique and stored as CSR of **earlier
neighbours** u < v, ascending. For vertex v, `cand` = the distinct codes of those neighbours in that order, at most 32;
m = |cand|.

```
m > 0 : hit bit, context = m (33 contexts)
        hit  → slot s ∈ [0,m) as 5 binary decisions down a prefix tree, contexts (m, prefix) (33 × 64)
        miss → raw code
m = 0 : raw code
raw code: ⌈log2 K⌉ binary decisions down an adaptive prefix tree (2^(nbits+1) contexts) = adaptive order-0 model
```
So the neighbour information enters as "same code as already-decoded neighbour number s", over the *whole decoded
part of the 1-ring* (the "0–1–2–1–2–3" reading), not as a one-to-one previous-vertex chain. The arithmetic coder and
`Count` model are the ones of `native.cpp`. **D** slot < m and code < K are checked.

Measured: Room primary-index hit rate 0.12–0.22 (independent codes would give ≈ 1/K); garden v0.8 K4096: 3.56 MB =
11.2 bits/vertex (raw 12); R256 residual: 2.44 MB = 7.7 bits/vertex (raw 8), hit rate 0.02–0.05. Honest reading: the
indices of an unconstrained k-means assignment are close to incompressible; the saving of the context coder is
≈ 0.5 MB on Room. The larger lever is to make assignments rate-aware upstream (index RD reassignment, growth with λ).

## 6. DC, opacity, banks

* `features_dc`: per-channel min/max uniform quantiser, b = 7 (`q7`; the stage2 lists carry q8/q10/q12, q7 is added at
  runtime in the analysis package, the decoder accepts 1..24 bits), stored as uint16 (V×1×3), compressed by the
  container's shuffle+zstd (the zero high byte vanishes). No spatial prediction. Measured Room q8 15.1 bits/vertex (raw 24).
* `vertex_weight`: replaced by one scalar `vertex_weight_const` = mean logit; **D** opacity = floor + (1−floor)·σ(w) with
  floor 0.999, the encoder reports max |Δopacity| (≤ 1e-3·|Δσ|, i.e. negligible). `vertex_weight_cols` restores the
  (V,1) vs (V,) shape.
* banks: FP32 `app_bank` (K'×15×3) and `app_residual_bank` (R'×15×3), 180 B per code; K4096+R256 ≈ 0.6–0.8 MB after zstd.

## 7. Decode order and guarantees

1. `unpack` (per-block SHA-256). 2. `edge decode` → faces (V, F from header). 3. `native build` on decoded faces →
predictors; `native decode` → q; dequantise XYZ. 4. `index_ctx decode` ×2 on decoded faces → `app_mapping`,
`app_residual_mapping`. 5. `delivery.restore` → float XYZ/DC, constant opacity; `expand_wire` → SH = bank[mapping] +
residual_bank[residual_mapping]. Steps 3 and 4 read only the faces from step 2: neither the encoder's predictor table
nor its neighbour lists are in the file.

Encoder-side checks before the file is accepted (`export_asset`): all delivered arrays byte-exact after decode
(`equal`), faces equal up to cyclic rotation, reorder audit (same attributes under the permutation, same oriented face
multiset), geometry health (no NaN, no new zero-area faces, flip fraction ≤ 1e-4 after quantisation).

## 8. Measured stream budget (garden v0.8, 2.54 M v, K4096 R256 q15/q7; 10.4 B/vertex)

| stream | MB | bits/vertex | raw bits/vertex |
|---|---|---|---|
| connectivity `top_stream` | 10.08 | 31.7 | ≈ 2 faces × 3 × 22 = 132 |
| positions `local_residual` | 5.16 | 16.3 | 45 |
| DC `features_dc` | 4.46 | 14.0 | 21 |
| primary index `ctx_mapping` | 3.56 | 11.2 | 12 |
| residual index `ctx_residual` | 2.44 | 7.7 | 8 |
| banks | 0.61 | — | — |

Room dense (3.19 M v, q18/q8, 41.4 MB): connectivity 30.4, XYZ 29.9, DC 15.1, primary 10.8, residual 10.2 bits/vertex.
Connectivity is the largest stream at every operating point, which is why vertex count (rate-controlled growth) is the
dominant lever and why the paper does not claim an optimal connectivity coder: on this near-soup mesh (45 % boundary
edges, 8 k components) a cached-edge coder at ≈ 15 bits/face is already below Draco's split-vertex output.

## 9. Corrections to the 10/23 memo (`plan/advisor-concerns-2026-10-23.md` §2)

* §2.3 said "parallelogram-type" prediction. The shipped predictor is the **single smallest-index earlier
  neighbour** (mode 2); the parallelogram predictor exists only as mode 3 in the CPU benchmark. Say "nearest
  already-decoded neighbour on the mesh graph".
* §2.3 "scene-adaptive grid": the grid is **per axis** (min/max per coordinate), not a cube.
* §2.2 the third vertex is coded as a **zigzag delta to the last coded vertex id** (+1) or as the implicit new id (0),
  not as "recently referenced vertex" in the cache sense. Cache = 64 directed edges ≈ last 21 triangles.
* §2.4 the neighbour set is the **earlier-index** part of the 1-ring (u < v), distinct codes, ≤ 32, ascending by
  neighbour id; hit-bit context = number of candidates m.
* Face order is preserved (strip order); decoded faces match up to cyclic rotation, not up to permutation.
