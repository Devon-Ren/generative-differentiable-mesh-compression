# Generic codecs on the same MeshSplatting asset — Room + garden, 2026-10-24 (batch D, run_generic_v1.sh)

Raw rows: analysis/rd/generic-codecs-2026-10-24.txt. Figure: analysis/rd/generic-codecs-v1.png/.pdf (plot_generic_codecs_v1.py).
Protocol self-check passed: `original` rows reproduce 28.263 (Room) and 24.732 (garden) through eval_ckpt_v1.py.
All rows: same original FP32 model, decoded on CPU, rendered on the held-out test views, 4x supersample.

## Table 1 draft (decoded test PSNR / SSIM / LPIPS; Δ vs FP32 original)

| row | Room MB | Room PSNR (Δ) | garden MB | garden PSNR (Δ) | note |
|---|---:|---|---:|---|---|
| FP32 minimal PLY | 741.7 | 28.263 | 739.8 | 24.732 | xyz + 48 SH + weight, int32 faces |
| PLY + zstd-19 | 632.6 | = | 626.4 | = | floats do not entropy-code (-15 %) |
| quantised PLY + zstd-19 (xyz q14/q15, SH 8 bit) | 158.5 | 28.092 (−0.17) | 172.0 | 24.594 (−0.14) | the "write bytes, entropy-code" baseline at our precision |
| Draco level 10, SH as 48-ch 8-bit generic attribute | 456.1 | 28.092 (−0.17) | 544.4 | 24.594 (−0.14) | decoded 14.4 M / 14.7 M vertices (4.5× split) |
| Draco, SH 16-bit | 1126.7 | 28.101 (−0.16) | 1228.1 | 24.599 (−0.13) | bigger than FP32 |
| Draco, RGB only (SH0, no view dependence) | 62.2 | 25.287 (−2.98) | 73.0 | 21.062 (−3.67) | the authors' RGB-only PLY tier |
| texture route: xatlas UV + Draco(pos, faces, UV 12 bit) + baked 8k JPEG q90 | 136.8 (PNG 144.5) | 25.287 (−2.98) | 138.3 (PNG 152.7) | 21.062 (−3.67) | MPEG / texture-mapped stand-in; geometry+UV alone 59.9 / 63.5 MB |
| **ours, codec only, all vertices (level 1.0)** | **30.9** | **28.061 (−0.20)** | **32.5** | **24.458 (−0.27)** | shared SH codebook K4096 R256, ctx coder |
| **ours, rate-controlled growth λ = 2e-3** | **13.7** | **27.985 (−0.28)** | **24.8** | **24.445 (−0.29)** | |
| **ours, λ = 6e-3** | **5.7** | **27.370 (−0.89)** | **10.7** | **23.637 (−1.09)** | |
| ours, λ = 1.85e-2 (Room) | 3.0 | 26.509 (−1.75) | | | |

## Reading

1. **Quantisation is cheap, bytes are not.** Quantising xyz to our grid and SH to 8 bit costs only 0.14–0.17 dB, and
   that is the entire quality gap between the generic rows and ours (level 1.0 is at −0.20 / −0.27). So at equal quality
   the comparison is pure bytes: ours 30.9 MB vs 158.5 MB quantised+zstd (5.1×) vs 456 MB Draco (14.8×); garden
   5.3× and 16.7×. With the λ route, 13.7 MB at −0.28 is 11.6× / 33× below the generic rows at the same loss.
2. **Where the generic codecs lose.** The 48-byte-per-vertex SH block is noise-like to a generic entropy coder:
   quantised PLY raw = 19 (xyz) + 77 (faces) + 153 (SH) MB = 249 MB, zstd gets it to 158.5 MB, i.e. the faces and xyz
   compress and the SH bytes do not. Our codebook turns 48 bytes into ~2.5 bytes per vertex (index + residual, context
   coded) at −0.03 dB beyond quantisation. Draco is worse than zstd because Edgebreaker splits the near-soup mesh
   4.5× and every split vertex carries its own 48-byte attribute copy.
3. **Topology.** Draco connectivity+positions alone is 32.8 / 35.6 MB; our whole asset at level 1.0 is 30.9 / 32.5 MB,
   and our topology+XYZ streams at this level are ~16 MB. The 30 bits/vertex we pay is the price of the soup, not
   of the coder; Draco pays more for the same face list (plan/advisor-concerns-2026-10-23.md §3).
4. **The texture / MPEG route is the wrong container for this asset.** Dropping view dependence costs 3.0 dB
   indoors and 3.7 dB outdoors (worse than the authors' "~−2 dB" because nothing is refit after dropping SH; a
   DC refit would recover part of it, but cannot recover view dependence). And it is not even small: 6.4 M triangles
   on an 8k atlas is ~10 texels per triangle, chart seams duplicate vertices, and geometry+UV alone is 60 MB.
   At 137 MB / −3.0 dB it is 24× larger and 2.1 dB worse than our λ = 6e-3 point (5.7 MB / −0.89). V-DMC would
   change the entropy coder, not these two facts, so no need to build the test model.
5. **What the advisor can now say.** Table 1 has four external, independently built codecs on the identical asset.
   None is a straw man: quantised+zstd and Draco are run at our precision with the best settings they offer, and the
   RGB/texture rows are the industry's standard mesh container. Ours wins by 5–17× at equal quality and by 2–3.7 dB
   at equal size against the texture tier.

## Caveats to state in the paper
- Draco quantization_bits is per-axis over the bounding cube, same meaning as our q<b>. SH attributes are
  quantised per channel with min/max (96 floats side info included).
- The texture rows' PSNR equals the Draco-RGB row by construction (the texture carries vertex colours only; JPEG
  loss is not modelled), so they are an upper bound for that route.
- RGB-only rows have no DC refit after dropping the SH rest. If a reviewer asks, a 400-step DC refit (2 GPU-min)
  gives the fair number; expected to land near the authors' −2 dB, still ≥ 1 dB below our λ = 6e-3 point.
- Both scenes only; the other nine scenes cost ~40 min each on CPU+GPU (xatlas dominates) and can be run with
  `RUN=... DATA=... QP=... bash run_generic_v1.sh <scene>` once batch A/B originals exist.
