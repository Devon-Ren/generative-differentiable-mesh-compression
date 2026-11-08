# Deep Blending one-shot (node-batch18, 2026-11-08): drjohnson, playroom — nothing changed, 10/10 jobs exit 0

Setup: upstream `--indoor` preset, XYZ q15, full-resolution images, v9d controller with its sbatch defaults (aim 1.25, 30 prune rounds), K4096 R256 DC q7. Raw file: `batch18-deep-blending-2026-11-08-raw.txt`.

## References (this dataset had never been run)
| scene | MeshSplatting FP32 (ours, --indoor) | codec only (level 1.0) | λ 1.85e-3 | λ 6.2e-3 |
|---|---|---|---|---|
| drjohnson | ≈778 MB, 26.737 dB | 39.8 MB, 26.289 (−0.45) | 16.40 MB, 26.44 | 6.89 MB, 25.65 |
| playroom | ≈765 MB, 27.718 dB | 37.5 MB, 27.306 (−0.41) | 14.37 MB, 27.29 | 5.76 MB, 26.42 |

(FP32 MB estimated as V×208 B + F×12 B. 3DGS reports ≈29.4 dB at ≈660 MB on this dataset; per-scene 3DGS / HAC++ numbers to be copied from their tables for the paper.)

## Adaptive controller, as is
| scene | target | delivered | hit | test PSNR | ratio vs FP32 | ΔPSNR vs FP32 | equal-bytes ΔPSNR vs own fixed-λ points | phase B active / 30 | growth during phase B |
|---|---|---|---|---|---|---|---|---|---|
| drjohnson | 5 | 5.30 | 1.059 | 25.37 | 147× | −1.37 | −0.04* | 30 | 1.10× |
| playroom | 5 | 5.28 | 1.056 | 26.44 | 145× | −1.28 | +0.10* | 30 | 1.10× |
| drjohnson | 10 | 10.70 | 1.070 | 26.03 | 73× | −0.71 | −0.02 | 30 | 1.21× |
| playroom | 10 | 11.17 | 1.117 | 27.07 | 68× | −0.65 | +0.02 | 30 | 1.20× |
| drjohnson | 20 | 21.71 | 1.086 | 26.37 | 36× | −0.36 | −0.02 | 27 | 1.32× |
| playroom | 20 | 22.20 | 1.110 | 27.53 | 34× | −0.19 | +0.23 | 23 | 1.39× |

Mean hit 1.083 (1.056–1.117), 4/6 within ±10 %, all within 12 %; mean equal-bytes ΔPSNR +0.04 dB (* = extrapolated below the smallest fixed-λ point). Phase B active on every row (priced collapse, both-kept share 1.00), and the un-stacking growth during phase B is 1.10–1.39×, i.e. exactly the indoor pattern measured on Mip-360 (room / kitchen / counter / bonsai: 1.10–1.49×). The controller behaved on an unseen dataset precisely as its mechanism analysis predicted: same sign, same magnitude, same cause. Quality is on the scene's own fixed-λ curve.

## What this adds to the paper
One row "unseen dataset, no changes": size as an input lands within 6–12 % on Deep Blending with quality on the curve; the residual bias is the documented indoor un-stacking. Plus two more scenes for the headline table (FP32 vs 36×/73×/145× points) on the third standard 3DGS-compression dataset.
