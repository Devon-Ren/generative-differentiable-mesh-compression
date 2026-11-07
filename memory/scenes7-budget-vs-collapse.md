---
name: scenes7-budget-vs-collapse
description: 7/7 Mip-360 scenes (10/18) train-at-budget beats post-hoc collapse at equal vertices, +0.3 to +4.3 dB, 14/14 points; per-scene numbers, figure, csv
metadata:
  type: project
---

Completed 2026-10-18 (budget-sweep-v1 + budget-control-v1 chained, tandon_advanced account). Budgets --max_points 1.6M / 1.0M, level-1.0 protocol, K4096 R256, scene XYZ precision / q7, stage2-v2 const/ctx. Control = dense model collapsed (geo-only sens_pos) to the budget run's exact vertex count, same finetune + delivery. Delivered MB / test PSNR:

| scene (XYZ) | vertices | budget | collapse | gain | vertices | budget | collapse | gain |
| room q14 | 1.35M | 13.56/27.984 | 13.93/26.660 | +1.32 | 967k | 10.38/27.711 | 10.14/25.665 | +2.05 |
| garden q15 | 1.48M | 16.51/23.996 | 16.25/23.154 | +0.84 | 1.09M | 12.52/23.699 | 12.14/22.299 | +1.40 |
| bonsai q15 | 1.44M | 13.96/26.585 | 13.83/25.028 | +1.56 | 912k | 9.54/26.585 | 8.96/22.287 | +4.30 |
| counter q15 | 1.46M | 15.35/25.403 | 15.81/24.707 | +0.70 | 719k | 8.10/25.102 | 8.15/22.393 | +2.71 |
| kitchen q16 | 1.35M | 15.24/25.722 | 15.96/25.406 | +0.32 | 802k | 9.68/25.211 | 10.10/23.395 | +1.82 |
| bicycle q16 | 1.50M | 14.94/22.424 | 14.87/22.006 | +0.42 | 864k | 9.14/22.025 | 8.63/20.515 | +1.51 |
| stump q16 | 1.14M | 11.50/23.783 | 11.12/22.879 | +0.90 | 879k | 9.31/23.506 | 8.62/22.158 | +1.35 |

Budget cost vs original (dense FP32) at the 1.0M budget: room -0.55, garden -1.03, bonsai -0.71, counter -0.84, kitchen -1.24, bicycle -1.01, stump -1.42 dB at 45-80x. At 1.6M: -0.28 / -0.74 / -0.71 / -0.54 / -0.73 / -0.61 / -1.15. Budget curve vs dense-collapse curve at equal bytes: budget above everywhere below ~20 MB; the two meet near the dense point. Faces/vertex of budget meshes (1.8-2.4) higher than collapsed ones (1.3-2.4): collapse removes faces faster than vertices.

Outdoor (garden/bicycle/stump) gaps at the higher budget are smaller (0.4-0.9) than indoor (0.3-1.6), but at the lower budget everything is >= 1.35 dB. Worst val view at budget stays 3-5 dB above the collapse control on every scene.

Files: rd/budget-sweep-2026-10-17.txt (all raw rows), rd/budget-vs-collapse-7scenes-v1.csv, plot_budget_vs_collapse_7scenes.py, budget-vs-collapse-7scenes-v1.png/.pdf (2x4 panels, legend in the 8th). Supersedes the 3-scene figure. SEVEN-SCENE HULL + TABLE DONE 10/18: rd/seven-scene-points-v1.csv (all delivery points, series collapse/budget), plot_scene_hulls_7.py -> seven-scene-hulls-v1.png/.pdf + seven-scene-hulls-v1-table.md. Budget points sit ON the global hull on 6/7 scenes (bonsai 1.44M dominated by 0.91M-dense line, same PSNR); hull below ~20 MB = budget route only. Headline means over 7 scenes: budget 1.6M 14.4 MB / -0.68 dB / 46x; budget 1.0M 9.8 MB / -0.97 dB / 67x; old collapse default 23.6 MB / -0.44 dB / 28x. Next: story rewrite; train_rd.py only after user go-ahead. See [[g0-train-at-budget-results]], [[scenes4-results]], [[meshsplatting-compression-status]].
