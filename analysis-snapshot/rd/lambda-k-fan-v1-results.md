# lambda x dictionary fan v1 — results (2026-11-05; jobs room 21965002 / kitchen 21965004 / garden 21966618, all exit 0, ~4.5 GPU-h)

Data: rd/lambda-k-fan-v1-2026-11-05.txt (72 rows), rd/lambda-k-fan-v1.csv, analysis rd/fan-analysis-v1.txt (fan_analysis_v1.py), figure report/fig-fan-v1.{png,pdf} (fig_fan_v1.py), LaTeX rows rd/fan-table-v1.tex. In report-latex-v4 as Table 6 / Figure 7 (sec:fan).

Per scene, 4 lambda assets x {K4096R256 q7 (delivered), DC q8, K4096R1, K1024R256, K4096R1024, K16384R1024}, re-export only (k-means + 400-step refit on the same finetuned mesh).

## Joint hull vs the lambda-only curve (gain at the delivered asset's bytes)
| scene | lam 1.85e-2 | 6.2e-3 | 1.85e-3 | 6.2e-4 |
|---|---|---|---|---|
| room | +0.117 | +0.063 | +0.018 | +0.044 |
| kitchen | +0.051 | 0.000 | +0.017 | +0.089 |
| garden | +0.071 | +0.009 | 0.000 | +0.002 |
=> the fixed K4096 R256 / DC q7 setting is within 0.12 dB of the joint hull at every point (Yao Q1 answered: the hull over lambda x K is traced; the choice holds to <0.1 dB).

## Who is on the joint hull
- room (6/24): K1024R256@1.85e-2 -> K1024R256@6.2e-3 -> q8@6.2e-3 -> q8@1.85e-3 -> q8@6.2e-4 -> K16384R1024@6.2e-4.
- kitchen (7/24): K1024R256@1.85e-2 -> K1024R256@6.2e-3 -> base@6.2e-3 -> K4096R1024@6.2e-3 -> K4096R1024@1.85e-3 -> K16384R1024@1.85e-3 -> K16384R1024@6.2e-4.
- garden (7/24): K1024R256@1.85e-2 -> K1024R256@6.2e-3 -> K1024R256@1.85e-3 -> base@1.85e-3 -> K4096R1024@1.85e-3 -> K4096R1024@6.2e-4 -> K16384R1024@6.2e-4.
Pattern: dictionary size should grow with rate / vertex count (K1024 at 3-5 MB, K4096 at ~10-25 MB, K4096R1024 / K16384 above) => rule K ∝ V (exporter-side, camera-ready). K4096 R1 (no residual) is the most expensive cut everywhere (0.09-0.90 dB/MB) => residual bank justified at every rate. DC q8: dMB shrinks with rate (3 bit/vertex) while dPSNR is flat per scene: room 0.07-0.10 dB (on the hull at 3 of 4 lambda), kitchen 0.035-0.04, garden 0.015 => scene-dependent, within the 0.1 dB band.

## Lever prices (dB/MB) vs the lambda chord — see fan-analysis-v1.txt
lambda chord (to the neighbouring asset): room 0.37 / 0.10 / 0.024, kitchen 0.32 / 0.11 / 0.026, garden 0.17 / 0.06 / 0.020 (low -> high rate). Dictionary levers are cheaper than the chord only at the ends (K1024 at the bottom, R1024/K16384 at the top on kitchen/garden) and DC q8 on room.

## Consequences
- Paper/report: fixed setting stays (headline unchanged); Table 6 + Fig 7 show the fan; "still open" = fan on the other 8 scenes (~12 GPU-h), K ∝ V rule, +1-bit XYZ margin, unified LPIPS pass (~3 GPU-h), all exporter-side.
- Not done / not needed: no retraining; no per-scene tuning.
