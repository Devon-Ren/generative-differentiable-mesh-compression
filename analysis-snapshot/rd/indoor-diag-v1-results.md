# indoor-diag-v1 results (2026-11-05 09:06, job exit 0) — decomposition of the level-1.0 delivery loss on the --indoor originals

Baseline = level-1.0 delivery (no collapse), K4096 R256, const/ctx, DC q7, XYZ room q14 / kitchen q16. FP32 / finetune / delivered test PSNR from rd/originals-indoor-2026-11-01.csv.

| scene | FP32 | finetune best | delivered | total | finetune step | quant + dictionary |
|---|---|---|---|---|---|---|
| room | 28.480 | 28.369 | 27.994 | −0.486 | −0.111 | −0.375 |
| kitchen | 27.465 | 27.327 | 26.706 | −0.759 | −0.138 | −0.621 |

Re-export levers (test PSNR, dPSNR vs baseline, dMB, share of the quant+dictionary loss recovered, dB/MB):

| scene | lever | MB | PSNR / SSIM / LPIPS | dPSNR | dMB | share | dB/MB |
|---|---|---|---|---|---|---|---|
| room | baseline K4096R256 q7 | 25.735 | 27.994 / .8509 / .3949 | — | — | — | — |
| room | (a) DC q8 | 26.861 | 28.071 / .8602 / .3822 | +0.077 | +1.126 | 21 % | 0.068 |
| room | (b) true per-vertex weight | 28.110 | 27.994 / .8509 / .3948 | +0.000 | +2.375 | 0 % | 0 |
| room | (c) DC q8 + true weight | 29.236 | 28.071 | +0.077 | +3.501 | 21 % | 0.022 |
| room | (d) K4096 R1024 q7 | 26.443 | 28.029 / .8521 / .3935 | +0.035 | +0.708 | 9 % | 0.049 |
| room | (d) K16384 R1024 q7 | 28.979 | 28.085 / .8542 / .3903 | +0.090 | +3.244 | 24 % | 0.028 |
| kitchen | baseline K4096R256 q7 | 24.325 | 26.706 / .8294 / .3206 | — | — | — | — |
| kitchen | (a) DC q8 | 25.406 | 26.749 / .8339 / .3152 | +0.043 | +1.081 | 7 % | 0.040 |
| kitchen | (b) true per-vertex weight | 26.555 | 26.706 / .8294 / .3206 | +0.000 | +2.229 | 0 % | 0 |
| kitchen | (c) DC q8 + true weight | 27.636 | 26.749 | +0.043 | +3.310 | 7 % | 0.013 |
| kitchen | (d) K4096 R1024 q7 | 24.928 | 26.774 / .8321 / .3175 | +0.068 | +0.602 | 11 % | 0.113 |
| kitchen | (d) K16384 R1024 q7 | 27.349 | 26.887 / .8363 / .3119 | +0.181 | +3.024 | 29 % | 0.060 |

## Reading
1. **DC 7-bit is not the cause of the top-rate loss** (Yao Q2): q8 recovers 0.08 / 0.04 dB of the 0.38 / 0.62 dB quantisation+dictionary loss (21 % / 7 %) for +1.1 MB (+4.4 %). q8 → FP32 DC would add at most a few hundredths more (Room precision sweep: q8 was already the knee). SSIM/LPIPS move more than PSNR (+0.009 / −0.013 room), worth a sentence.
2. **The global opacity scalar costs exactly nothing** (+0.000 dB, −2.4 / −2.2 MB): the "one scalar per mesh" design (Yao comment on opacity) is confirmed on the indoor models too.
3. **Dictionary size is the larger lever at the top rate**, more so on kitchen: K16384 R1024 +0.09 / +0.18 dB for +12.5 % bytes; K4096 R1024 is the cheapest step on kitchen (0.113 dB/MB) and would sit on the hull there. This is exactly what the λ × K fan (lambda-k-fan-v1) measures per λ; the top-rate point should be allowed a larger dictionary by the hull, not by a per-scene choice.
4. **Levers together recover less than half**: q8 + K16384R1024 ≈ +0.17 / +0.22 dB (if additive) for +4.4 / +4.1 MB (+17 %). ≥ 0.2 / 0.4 dB remains and is not DC, not weight, and only partly dictionary size. The two remaining candidates are the SH sharing step itself (coarse → shared_fitted; VQ of per-vertex SH, grows with the indoor feature_lr 0.004) and the XYZ grid (shared_fitted → decoded). summary.json `stages` (validation PSNR per stage) separates them without any GPU: see commands/indoor-diag-v1-stages.md.

Table file: /scratch/zr2245/analysis/indoor-diag-v1.txt (12 rows). Refit outputs: /scratch/zr2245/{room,kitchen}-indoor/diag-v1/k{K}_r{R}/summary.json.

## Stage decomposition (summary.json validation PSNR, 11/05 14:16)

| scene | dictionary | coarse → shared_init → shared_fit → decoded (val) | sharing | quant+coding | test coarse → delivered | MB |
|---|---|---|---|---|---|---|
| room | K4096 R256 (base) | 30.929 → 29.864 → 30.459 → 30.113 | −0.470 | −0.346 | 28.369 → 27.994 (−0.374) | 25.735 |
| room | K4096 R1024 | 30.929 → 29.985 → 30.530 → 30.180 | −0.398 | −0.351 | → 28.029 (−0.340) | 26.443 |
| room | K16384 R1024 | 30.929 → 30.199 → 30.643 → 30.286 | −0.285 | −0.357 | → 28.085 (−0.284) | 28.979 |
| kitchen | K4096 R256 (base) | 28.468 → 27.253 → 27.739 → 27.616 | −0.729 | −0.124 | 27.327 → 26.706 (−0.621) | 24.325 |
| kitchen | K4096 R1024 | 28.468 → 27.375 → 27.825 → 27.697 | −0.644 | −0.127 | → 26.774 (−0.553) | 24.928 |
| kitchen | K16384 R1024 | 28.468 → 27.586 → 27.962 → 27.830 | −0.506 | −0.132 | → 26.887 (−0.440) | 27.349 |

Reading: kitchen = SH sharing (85 % of the val loss; a bigger dictionary cuts it 0.73 → 0.51, the lever is K/R and the fan prices it per λ). room = sharing 58 % + quant+coding 42 % (−0.35), and DC q8 explains only ~0.08 of that, const weight / ctx indices are lossless → ≈0.27 dB is the XYZ q14 grid; the 10/17 seven-scene precision sweep already found the b-rule 1 bit low on 4/7 scenes. The 400-step bank/DC fit recovers 0.4-0.6 dB of the raw k-means loss everywhere (shared_init → shared_fit). Follow-up: commands/indoor-xyz-v1.md (room q15/q16 re-export, ~10-15 min GPU).

## XYZ re-export on room indoor (indoor-xyz-v1, 11/05 14:28)

| xyz | MB | test PSNR / SSIM / LPIPS | dPSNR | dMB | dB/MB |
|---|---|---|---|---|---|
| q14 (base) | 25.735 | 27.994 / .8509 / .3949 | — | — | — |
| q15 | 26.700 | 28.077 / .8543 / .3902 | +0.082 | +0.965 | 0.085 |
| q16 | 27.651 | 28.101 / .8553 / .3888 | +0.106 | +1.916 | 0.025 (q15→q16) |

Reading: on test, room's −0.374 dB is now fully accounted for: SH sharing ≈ −0.2 (val −0.47 scales ~0.5× to test), DC q7 −0.08, XYZ q14 −0.08..−0.11. Nothing is left unexplained; the val-stage numbers are ~2× the test numbers at every step. Lever prices at the room-indoor top point (dB/MB): XYZ q15 0.085 > DC q8 0.068 > K4096R1024 0.049 > K16384R1024 0.028 > XYZ q16 0.025. Kitchen: K4096R1024 0.113 > K16384R1024 0.060 > DC q8 0.040. By the 10/17 RD criterion (argmax PSNR − 0.05·MB) room indoor picks q15, so the b-rule is 1 bit low on 4 of 8 scene-models (counter/kitchen/bonsai 10/17 + room-indoor). Paper statement: the top-rate loss is distributed over sharing (largest), DC and XYZ; each remaining lever buys 0.03–0.18 dB for 4–12 % bytes, i.e. the ceiling is a hull property priced by the λ×K fan, not a single quantiser defect. Operating points are NOT changed now (a +1-bit XYZ margin for all 44 assets would be a ~2–3 GPU-h re-export; deferred unless Wenhao wants it).
