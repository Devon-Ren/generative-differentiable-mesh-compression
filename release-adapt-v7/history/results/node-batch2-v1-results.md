# node-batch2-v1 results (2026-11-06 morning) — all 16 jobs exit 0 (gl030, 11/05 11:05–17:09 EST)

Data: adapt-v1-2026-11-06.txt (12 rows), ba-sens-v1-2026-11-06.txt (8), lambda-k-fan-v2-2026-11-06.txt (264 = 72 of 11/05 + 192 new).
Analyses: adapt_analysis_v1.py -> adapt-v1-analysis.txt / adapt-v1-table.tex / adapt-v1.csv; ba_sens_analysis_v1.py -> ba-sens-v1-analysis.txt / -table.tex / .csv; fan_analysis_v2.py -> fan-analysis-v2.txt / fan-table-v2.tex (44 rows) / fan-summary-v2.tex (11 rows) / lambda-k-fan-v2.csv.
Figures: report/fig-adapt-v1, fig-ba-sens-v1, fig-fan-v2 (11 panels). Report: report/report-v5.pdf (latex-v5, 16 pages).

## 1. Adaptive lambda (train_rd_adapt_v1, targets 5/10/20 MB, room/kitchen --indoor, garden, bicycle)
- Quality: test PSNR vs the fixed-lambda curve at the delivered bytes -0.12..+0.07 dB, mean +0.005 (n=10 inside the range) -> searching lambda online costs nothing.
- Size: hit 0.52-0.77 (mean 0.64), systematic. hit = (R12k/T) 1.12 x (R18k/R12k) 0.87 x (R30k/R18k) 0.85 x (delivered/R30k) 0.77.
  (a) BUG in projected_mb: V*B_v + F*B_f double counts connectivity (B_v = B_a + 2 B_f). Corrected V*(B_v-2B_f) + F*B_f matches the delivered file 0.97-1.035 on all 12 runs.
  (b) post-control shrink 0.78-0.89: upstream opacity pruning + final cleaning after the last prune round (18000); prune rounds cannot move past 20000 (supersampling).
  (c) phase A ends 0.95-1.26 of target (biased units), phase B removes 13 %.
- lam_final / lam_eff (fixed lambda with the same bytes): 1.0-6.2, i.e. the controller ends at a higher lambda than the fixed run with the same bytes (phase-B lambda rises when R_hat > target on the biased model).
- v2 (train_rd_adapt_v2, not written yet): corrected projection + plan against target/0.85 (one constant) -> rerun 4 scenes ~12 GPU-h.

## 2. B_a sensitivity (kitchen --indoor / garden x B_v {8, 13} x lambda {6.2e-3, 6.2e-4})
- All 8 points on the B_v=10 curve: dPSNR -0.013..+0.065 (mean +0.027). lam_eff/lam = 0.78 / 1.31-1.33 at 6.2e-3 vs predicted B_v/10 = 0.80 / 1.30; at 6.2e-4 the curve is flat so the reading is ill-conditioned (0.86-0.88 / 1.2-2.1), direction right.
- Paper: a price error is a lambda rescale; the hull does not move. Answers Yao's "constant B_a = not end-to-end" with a measurement.

## 3. Fan on 11 scenes (264 points)
- Joint hull above the lambda-only hull by at most 0.02-0.15 dB (counter 0.154, room 0.137, bonsai 0.124, kitchen/stump 0.096, garden 0.083, truck 0.067, bicycle 0.041, flowers 0.030, treehill 0.022, train 0.019).
- K1024 R256 on the hull at lambda 1.85e-2 and 6.2e-3 on 11/11 scenes (22/22 assets); K4096 R1024 / K16384 R1024 on the hull at the top of every scene's hull (lambda <= 1.85e-3 on 8 scenes; at 6.2e-3 on flowers/truck/train whose top points are dominated) -> K ∝ V rule holds everywhere.
- K4096 R1 the most expensive cut everywhere (mean 0.21 dB/MB; indoor 0.32-0.55). DC 8 bit: 0.13-0.38 dB/MB indoor vs 0.01-0.04 outdoor, on the hull at room (3 lambda), stump (2), counter/bonsai/treehill (1). K16384 R1024: 0.05-0.08 dB/MB indoor vs 0.003-0.014 outdoor.
- Mean lever prices over 44 assets: q8 0.103, R1 0.210, K1024 0.081, R1024 0.084, K16384R1024 0.026 dB/MB.
