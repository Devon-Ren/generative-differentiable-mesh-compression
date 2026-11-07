# Indoor (--indoor) rerun, first look — 2026-11-01 (room + kitchen complete; counter/bonsai originals only, λ jobs pending)

Source: user's paste of originals-indoor.txt and rd-prune3i-*.txt. "original test" = train.log (3DGS [0,1] LPIPS); "delivered" = pipeline evaluator ([-1,1] LPIPS, ≈ +0.03).

## 1. Reproduction is now exact

| scene | ours FP32 --indoor PSNR / SSIM / LPIPS | paper | Δ PSNR | old (no flag) |
|---|---|---|---|---|
| room | 28.480 / .8740 / .2710 | 28.52 / .873 / .271 | −0.04 | 28.263 |
| kitchen | 27.465 / .8592 / .2265 | 27.42 / .858 / .227 | +0.05 | 26.394 |
| bonsai | 28.280 / .8785 / .2937 | 28.19 / .876 / .294 | +0.09 | 27.359 |
| counter | 26.479 / .8462 / .2787 | 26.51 / .846 / .279 | −0.03 | 26.067 |

LPIPS matches the paper to the third decimal: the whole indoor gap was the missing flag. Vertex counts drop (room 3.19 M → 2.55 M, kitchen 2.14 M, bonsai 2.94 M, counter 2.22 M).

## 2. λ hull at equal bytes barely moves (room / kitchen)

| λ | room old MB / PSNR | room indoor MB / PSNR | kitchen old | kitchen indoor |
|---|---|---|---|---|
| 6.2e-4 | 28.38 / 28.272 | 22.07 / 28.255 | – | 24.46 / 27.092 |
| 1.85e-3 | 13.70 / 27.985 | 12.40 / 28.023 | 15.89 / 26.144 | 16.54 / 26.886 |
| 6.2e-3 | 5.70 / 27.370 | 5.40 / 27.305 | 7.60 / 25.312 | 8.02 / 25.971 |
| 1.85e-2 | 3.01 / 26.509 | 2.89 / 26.383 | – | 3.78 / 24.623 |

Room: same curve (±0.1 dB at equal bytes, slightly fewer bytes). Kitchen: +0.65–0.75 dB at every λ (the flag matters most there). Absolute RD vs other methods is therefore unchanged or better.

## 3. But the codec loss relative to the own FP32 base doubled

| scene | FP32 → level 1.0 delivered, old | indoor | λ 6.2e-4 pre-codec test → delivered (indoor) |
|---|---|---|---|
| room | 28.263 → 28.061 (−0.20) | 28.480 → 27.994 (−0.49; finetune step −0.11, quant+codebook −0.38) | 28.567 → 28.255 (−0.31) |
| kitchen | 26.394 → 26.094 (−0.30) | 27.465 → 26.706 (−0.76; −0.14 / −0.62) | 27.706 → 27.092 (−0.61) |
| bonsai | 27.359 → 27.102 (−0.26) | 28.280 → 27.663 (−0.62; −0.08 / −0.54) | pending |
| counter | 26.067 → 25.593 (−0.47) | 26.479 → 25.743 (−0.74; −0.11 / −0.63) | pending |

The indoor models carry more information per vertex (feature_lr 0.004 vs 0.0016, lambda_weight 0, fewer vertices) and the
fixed delivery settings (DC q7, K4096 R256, const vertex weight) take 0.5–0.75 dB off them instead of 0.2–0.3. LPIPS shows the
same: room level 1.0 +0.09 beyond the +0.03 evaluator offset (was +0.02).

Consequence for the paper: "Δ vs own FP32" at λ = 6.2e-4 is now −0.22 (room) / −0.37 (kitchen) instead of ≈ 0 / −0.25, while the
absolute points stand. Either (a) report it as is (lossy RD, the absolute hull is the claim), or (b) find which delivery lever
loses the 0.3–0.5 dB: a no-training re-export of the indoor level-1.0 asset at DC q8, K16384 R1024, and with the true per-vertex
weight (q6) — three points per scene, ≈ 5 min GPU each on room + kitchen (≈ 0.5 GPU-h). If one lever recovers most of it, the
indoor rows get that setting as a class default (not per-scene tuning), costing ≈ 1–3 MB.

## 4. Pending
counter / bonsai λ jobs (4, moved to general 11/01); then the 11-scene table/hull/figures are rebuilt with the indoor rows (room, kitchen, bonsai, counter replaced).
