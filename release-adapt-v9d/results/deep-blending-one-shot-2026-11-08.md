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

## Full fixed-λ hull (batch19, 2026-11-09): `deep-blending-fixed-lambda-2026-11-09.txt`
| scene | FP32 | λ 6.2e-4 | λ 1.85e-3 | λ 6.2e-3 | λ 1.85e-2 |
|---|---|---|---|---|---|
| drjohnson | ≈778 MB / 26.74 | 30.7 MB / 26.45 (25×, −0.29) | 16.4 / 26.44 (47×, −0.30) | 6.9 / 25.65 (113×, −1.09) | 3.3 / 24.65 (236×, −2.08) |
| playroom | ≈765 MB / 27.72 | 29.9 / 27.65 (26×, −0.07) | 14.4 / 27.29 (53×, −0.43) | 5.8 / 26.42 (133×, −1.30) | 2.9 / 25.26 (262×, −2.46) |

## 2.5 MB controller rows (batch19, Mip/T&T, v9d as is)
room 1.010, kitchen 1.013, counter 0.999, bonsai 0.979, garden **1.180**, bicycle 0.939, stump 0.942, flowers 0.946, treehill 0.946, truck 0.949, train 0.967 (mean 0.988 without garden; outdoor rows 0.94–0.95 because the landing estimate over-reads the final by 4–5 % on this class and the gate closes early). Garden 2.5 MB is a different limit: the initial soup already exceeds the budget (first phase-A probe 3.19 MB at 4500 it, λ pinned at its start value, no split ever admitted), phase B collapsed 77k pairs in 30/30 rounds and still landed 1.18. Targets below the initial soup size cannot be reached by a controller that can only grow before the Delaunay and only trim after it. Document as the low-end limit.
