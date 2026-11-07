# error anatomy, Room (2026-10-19, analysis/diag/error_anatomy_v1.py, 4x supersample, test 39 / validation train[::8] 34 views)

Buckets are exclusive per output pixel: background = alpha<0.99; silhouette = id change + depth jump >5 % in the 4x4 block; seam = id change without depth jump; interior_hf = GT |Laplacian| above the 75th percentile; interior_lf = rest.
NOTE: at this tessellation (6.45M faces, triangles mostly <16 px) nearly every 4x4 block crosses a triangle edge, so "seam" = ordinary interior (70 % of pixels, error density 0.83x average), not a seam problem.

## original dense (3,193,592 v / 6,452,971 f)
| split | PSNR | SSIM | affine UB | lowpass2 | lowpass4 |
|---|---|---|---|---|---|
| test | 28.263 | 0.8833 | 28.599 (+0.34) | 30.342 | 31.064 |
| validation | 30.715 | 0.9041 | 31.015 (+0.30) | 33.339 | 34.140 |
train/test gap 2.451 dB

| bucket | pixel share | SSE share | error density (SSE/pix) | PSNR if zero |
|---|---|---|---|---|
| background | 0.4 % | 9.1 % | 23x | +0.74 |
| silhouette | 1.8 % | 5.9 % | 3.3x | +0.27 |
| seam (= interior) | 70.6 % | 58.4 % | 0.83x | +4.19 |
| interior_hf | 6.8 % | 15.6 % | 2.3x | +0.77 |
| interior_lf | 20.4 % | 11.0 % | 0.54x | +0.51 |
footprint (interior): [1,4) 0.00137, [4,16) 0.00118 (2.2 %), [16,64) 0.00122 (9.9 %), [64,inf) 0.00203 (15.1 %)

## rd3 lambda 6.2e-3 (439,781 v / 1,022,252 f)
| split | PSNR | SSIM | affine UB | lowpass2 | lowpass4 |
|---|---|---|---|---|---|
| test | 27.447 | 0.8424 | 27.764 (+0.32) | 29.867 | 30.826 |
| validation | 29.151 | 0.8557 | 29.384 (+0.23) | 32.135 | 33.285 |
train/test gap 1.704 dB

| bucket | pixel share | SSE share | error density | PSNR if zero |
|---|---|---|---|---|
| background | 0.4 % | 6.7 % | 17x | +0.54 |
| silhouette | 1.6 % | 5.2 % | 3.3x | +0.24 |
| seam | 47.7 % | 47.4 % | 1.0x | +2.95 |
| interior_hf | 12.6 % | 22.9 % | 1.8x | +1.15 |
| interior_lf | 37.8 % | 17.8 % | 0.47x | +0.86 |
footprint (interior): [1,4) 0.00324, [4,16) 0.00227 (1.0 %), [16,64) 0.00175 (9.4 %), [64,inf) 0.00151 (40.1 %)

## reading
1. Error is concentrated where an opaque mesh must fail: uncovered pixels (23x density) and object edges (3.3x), then high-frequency texture (2.3x). Flat interior is 0.5x. No single bucket above 0.8 dB on the dense model: the loss is spread, no one "mismatch".
2. Exposure is minor (affine UB +0.3). Low-pass numbers are not interpretable without a 3DGS reference rendered the same way.
3. Dense vs lambda: 2.75M extra vertices buy +1.56 dB on training views but only +0.82 on test: about half of the dense model's extra capacity is memorisation (train/test gap 2.45 -> 1.70). Supports budget/lambda growth as a regulariser (paper: analysis section).
4. Footprint: in the dense model the largest triangles (15 % of pixels) carry 1.7x the MSE of small ones (importance sampling left high-error regions coarse); in the lambda model large triangles (40 % of pixels) have the LOWEST MSE (vertices spent where the residual is). Tentative, one scene.
5. Decision: background bucket upper bound 0.74 dB on an INDOOR scene >= 0.5 threshold -> env-map probe (analysis/commands/envmap-probe-v1.md) on Room + garden (+ stump/bicycle originals if present) before batch A.
