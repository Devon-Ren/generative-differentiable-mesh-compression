---
name: scenes4-results
description: counter / kitchen / bicycle / stump 2026-10-17: axis B (q14), precision re-export q13-q16 (rule exact or 1 bit low, 7 scenes), K x R backbone order 7/7 scenes, scene XYZ q15/q16/q16/q16
metadata:
  type: project
---

Jobs 20131633 counter / 34 kitchen / 35 bicycle / 37 stump, all ~1.1-1.6 h on L40S, general account; outputs /scratch/zr2245/<scene>-vis-collapse, raw rows analysis/rd/scenes4-axisB-2026-10-17.txt. Original FP32 estimate = vertices x 208 B + faces x 12 B (matches Room 742 / garden 740).

| scene | orig MB / test | v1.0 MB test (x, dB) | v0.8 | v0.7 | share+quant cost @1.0 | vertex slopes dB/MB |
| counter (images_2) | 575 / 25.944 | 24.4 25.544 (23.5x -0.40) | 20.3 25.328 (28x -0.62) | 18.0 25.089 (32x -0.86) | -0.40 | 0.052 / 0.103 |
| kitchen (images_2) | 456 / 26.454 | 19.9 25.675 (23x -0.78) | 16.9 25.416 (27x -1.04) | 15.1 25.088 (30x -1.37) | -0.78 | 0.085 / 0.184 |
| bicycle (images_4) | 680 / 23.033 | 26.1 21.875 (26x -1.16) | 21.5 21.793 (32x -1.24) | 18.9 21.737 (36x -1.30) | -1.16 | 0.018 / 0.022 |
| stump (images_4) | 698 / 24.928 | 26.9 23.391 (26x -1.54) | 21.8 23.246 (32x -1.68) | 19.1 23.105 (36.5x -1.82) | -1.54 | 0.028 / 0.052 |

KEY: sharing+quant cost at 100 % vertices is 0.4 / 0.78 / 1.16 / 1.54 dB vs Room 0.17, garden 0.27, bonsai 0.30 -> q14 XYZ is almost certainly too coarse for large-extent scenes (garden already needed q15); the finetuned meshes themselves lose only 0.2-0.5 dB per 20 % vertices. So the q14 points are NOT the operating points for these scenes; vertex slopes on bicycle/stump (0.02-0.03) are confounded by the quantisation floor. Vertex counts 1.94M (kitchen) - 3.04M (stump): budget not constant (kitchen smallest). Bytes/vertex 8.8-10.2; faces/vertex 1.75-2.24. Worst val views on bicycle/stump 13-16 dB (original model issue, not ours).
AXISA DONE 2026-10-17 (jobs 20153357/62/81/82; rd/scenes4-precision-2026-10-17.txt, scenes4-codebook-v1.csv; q17 unsupported by codec_variants). PRECISION explains the loss: @1.0 q14->q16 recovers +0.07 counter / +0.42 kitchen / +0.96 bicycle / +1.26 stump, residual loss at q16 = 0.33/0.36/0.20/0.28 (= Room/garden/bonsai range; counter's loss is appearance sharing). Rule b=ceil(log2(extent/(0.1 median edge))): counter q14, kitchen q15, bicycle q16, stump q16 (extents 38/73/170/202, median edge 0.031-0.046); RD (argmax psnr-0.05 MB at v0.8) picks q15/q16/q16/q16 -> rule exact or 1 bit low on all 7 scenes. q15->q16 still 0.24/0.31 dB/MB on bicycle/stump (q17 may pay). DC q7->q8 0.010-0.024 dB/MB everywhere (q7 universal, 7 scenes). Vertex slopes at proper precision 1.0->0.8 / 0.8->0.7: counter 0.050/0.102, kitchen 0.086/0.188, bicycle 0.028/0.036, stump 0.054/0.095. BACKBONE at v0.8 (scene XYZ): all 6 points convex in Room order on all 4 scenes -> 7/7; slopes counter .221/.126/.095/.053/.023, kitchen .201/.144/.092/.049/.022, bicycle .057/.036/.023/.013/.006, stump .051/.032/.014/.011/.004 (foliage scenes: appearance lever almost flat). Collapse-based operating points v0.8 K4096R256: counter q15 21.02 MB/25.377 (27x -0.57), kitchen q16 18.03/25.808 (25x -0.65), bicycle q16 23.27/22.693 (29x -0.34), stump q16 23.66/24.347 (29.5x -0.58) -- superseded by budget training (G0), see [[meshsplatting-compression-status]].
