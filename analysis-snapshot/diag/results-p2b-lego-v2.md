# P2b lego v2 result (2026-11-05, +++ line) — soup as finder → TSDF surface → opaque refits

Job p2b-lego-v2 (exit 0, 20 min wall, ~0.35 GPU-h). Table: /scratch/zr2245/analysis/p2b-lego-v2.txt. Test PSNR on the 200 lego test views, black bg, nvdiffrast opaque z-buffer, 3000 fit steps.

## Surface (surf_extract_v2, median depth alpha ≥ 0.5, 100 train views, 28.1 M observed voxels)
raw 600k V / 1.17 M F, 11 components kept (1713 dropped), 34.7k boundary edges (underside never seen + holes), no non-manifold edges; T50k/100k/200k meshes 0.17–0.20 % diag median distance to the soup (one-sided).

## Appearance fits (test PSNR / ~MB at 8 bit)
| geometry | vsh3_init | vdc | vsh3 | tb2 | tb4 | tb8 | tb4sh1 | sub1 vsh3 | sub2 vsh3 |
|---|---|---|---|---|---|---|---|---|---|
| soup 495k V / 2.51 M F (opaque) | 28.04 / 24 | – | **30.61** / 24 (train 33.35) | – | 28.63 / 75 | – | – | – | – |
| T50k | 20.69 / 1 | 22.19 / 0.1 | 23.13 / 1 | 22.49 / 0.4 | 22.52 / 1.4 | 22.37 / 5 | 23.10 / 2 | 23.79 / 5 | 23.92 / 19 |
| T100k | 21.43 / 2 | 23.21 / 0.2 | 24.34 / 2.5 | 23.45 / 1 | 23.40 / 3 | 23.02 / 10 | 24.00 / 4 | 24.87 / 10 | 24.80 / 38 |
| T200k | 22.01 / 5 | 24.15 / 0.3 | 25.39 / 5 (train 26.36) | 24.19 / 2 | 23.96 / 6 | 23.21 / 21 | 24.54 / 8 | 25.65 / 19 | 25.26 / 77 (train 27.52) |
Reference: delivered soup (soft blending, codec) 9.26 MB / 33.50; val 35.20.

## Reading (against plan/surface-first-thinking-2026-11-05.md §4)
1. **Colour resolution is not the lever** (reading i): per-face textures lose to vertex SH on the same mesh at every T (tb2 < vsh3 by 1.2 dB at T200k), lose to subdivided vertex SH at equal bytes (T50k tb8 5 MB 22.37 vs T50k sub1 5 MB 23.79 vs T200k vsh3 5 MB 25.39), and get WORSE with resolution while train PSNR rises (T200k tb2 24.19 → tb8 23.21; train 24.75 → 25.22) = overfit/aliasing, not missing capacity. Same as P2 on room/kitchen and as tb4 on the soup itself (28.63 < 30.61).
2. **Geometry is the bottleneck, and vertex count does not fix it**: same rasterizer, same vertex SH: soup train 33.35 / test 30.61 with 2.5 M faces; extracted surface 200k faces train 26.36 / test 25.39; subdivided to 3.1 M faces (more than the soup) train 27.52 / test 25.26. Train PSNR is the capacity measure: the single smooth surface cannot even reproduce the training images; the soup can. What the surface lacks is what the soup has: overlapping layers, thin/fuzzy parts (TSDF with alpha ≥ 0.5 median depth drops every semi-transparent layer; 1713 fragments dropped), gaps under the treads.
3. **Opaque compositing itself costs 2.9 dB** on the soup (33.50 soft → 30.61 opaque refit), consistent with P2 (room −2.0, kitchen −3.6).

Verdict: the "coarse smooth surface + per-face texture" premise fails on lego: ceiling of any opaque single surface built from this soup ≤ 30.6 (the opaque soup), realised 25.4–25.7 with a clean 200k-face surface. The detail is multi-layer geometry (reading i) → the only +++ route left is 4.3 (surface + alpha cards / layered), a different and much larger project. Caveats: 3000 steps, no texture mip filtering, surface from one depth threshold; none of these can close a 5 dB train-PSNR gap. No follow-up runs (user's rule).
