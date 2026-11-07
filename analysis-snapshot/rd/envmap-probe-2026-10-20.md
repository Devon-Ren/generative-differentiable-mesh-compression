# env-map probe (2026-10-20, analysis/diag/envmap_fit_v2.py, alpha_thr 0.99, 64x128 fp16 map ~49 KB, mesh frozen, fit on train views)

| scene | test PSNR | alpha<0.99 px (test) | their SSE share (pooled) | upper bound | after env map | gain | val gain |
|---|---|---|---|---|---|---|---|
| room (images_2) | 28.263 | 0.39 % | 36.8 % | 29.006 | 28.385 | +0.121 | +0.003 |
| garden (images_4) | 24.732 | 0.19 % | 5.7 % | 24.810 | 24.745 | +0.013 | +0.003 |
| stump | 24.928 | 0.27 % | 0.9 % | 24.955 | 24.935 | +0.007 | +0.300 |
| bicycle | 23.033 | 0.09 % | 0.4 % | 23.046 | 23.035 | +0.002 | +0.002 |

Reading: the mesh covers the field of view on every scene (uncovered < 0.4 % of pixels, outdoor included: densification fills the sky with far triangles), so "no background model" is not a loss. Room's uncovered pixels are novel-view holes (train views uncovered 0.06 %, test 0.39 %; concentrated in the two worst test views DSCF4939/4947, +0.36/+0.47 there): an env map paints them with a far colour and recovers 0.12 of the 0.74 dB bound; the rest is missing geometry only visible from test viewpoints. Gains below the 0.3 / 0.5 dB gates on 4/4 scenes -> LINE CLOSED. Paper: one sentence in the analysis section ("uncovered pixels < 0.4 %, an env map adds <= 0.12 dB; the opaque mesh needs no background model"). v1 run (thr 0.999) void: rasterizer caps per-layer alpha at 0.999.
