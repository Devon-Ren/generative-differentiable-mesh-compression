---
name: rd-hull6-results-nov2026
description: 11/01 six-point RD hull on 11 scenes (v3 lambda 6.2e-4 / 1.85e-3 / 6.2e-3 / 1.85e-2 + level 1.0 + FP32): means, headline (lambda 6.2e-4 matches FP32 on Mip-360 at 24x), dominated points on low-coverage scenes
metadata:
  type: project
---

Files: analysis/rd/rd-prune3-hull2-2026-11-01.csv (22 new rows), rd-hull6-v1-table.md, rd-11scenes-v2.csv/.png/.pdf (build_11scenes_v2.py, plot_11scenes_v2.py). Same codec/XYZ as batch B ([[rdgrowth-v3-scenes7-results]]).

11-scene means (MB / ratio vs FP32 / dB vs FP32): level 1.0 (dense + codec) 26.6 / 23x / -0.25; lam 6.2e-4 28.2 / 22x / -0.11; 1.85e-3 20.7 / 32x / -0.22; 6.2e-3 9.4 / 73x / -0.59; 1.85e-2 4.5 / 149x / -1.25 (ratio 76-264x, dB -0.08..-2.18). Mip-360 (9): 6.2e-4 = 28.1 MB / 24x / -0.03 dB.

Headline: at lam 6.2e-4 the rate-trained mesh reaches the FP32 original on the 7 well-covered Mip-360 scenes (room +0.06, kitchen +0.19, bonsai +0.20, garden 0.00, counter -0.04, bicycle -0.05, stump -0.12) at the bytes where dense + codec loses 0.2-0.36 dB: train-at-rate beats post-hoc coding by ~0.25 dB at the top of the curve too.
Dominated points (smaller asset with >= PSNR): bicycle 6.2e-4; flowers, truck, train 6.2e-4 + 1.85e-3; treehill 6.2e-4 -> on low-coverage scenes the hull tops out at 6.2e-3 (R9 pattern: extra vertices fit train views). Paper: draw the Pareto hull per scene; lam 1.85e-2 = low end 4.5 MB / 150x / -1.25 dB (indoor -1.35..-2.18, outdoor/T&T -0.08..-0.96).
11/01 figure v3 (rd/rd-11scenes-v3.png, plot_11scenes_v3.py, baselines-rd-v1.csv, notes rd-11scenes-v3-notes.md): Pareto hull + hollow dominated points; HAC per-scene 5-lambda curves (HAC appendix, verbatim) on every panel; Mip-360/T&T average panels vs HAC, HAC++, RDO-Gaussian (6-pt curves, summary-extracted: verify vs PDF), ContextGS, LightGaussian, Compact3DGS, 3DGS, Scaffold-GS. Non-monotone scenes: val rises +0.5..+1.2 dB, test 0..+0.2 (train-view fit). Gap at equal bytes: -3.4 dB vs HAC, -2.6 vs RDO (Mip-360) = class gap.
11/01 MAIN FIGURE reframed after user objection to the absolute gap: rd/rd-main-v1.png = (a) dPSNR vs own uncompressed against ratio (ours beats RDO-Gaussian at every ratio; HAC/HAC++/ContextGS referenced to Scaffold-GS base, +0.1..+0.3 at 10-30x) + (b) absolute among engine-ready meshes (MiLo 253 MB/24.09, BakedSDF 434/24.51, BOG 913/25.62, RaDe-GS, GOF); absolute 3DGS comparison only grey/supplement.
Still open: unified LPIPS pass over all 66 assets, indoor-flag decision, qualitative figure. See [[meshsplatting-compression-status]].

Main figure v2 (11/01, user asked: obvious advantage, 'better' arrow per panel, unambiguous reference, meshes as points): analysis/rd/rd-main-v2.png, x = absolute MB everywhere, ratio only as labels vs the named FP32 asset; (a) one Room asset through generic codecs vs ours (5x smaller at PLY+zstd quality); (b) engine-ready mesh points, ours 10-100x smaller; (c) Δ vs each method's own compressed model, reference in legend (droppable). Notes: rd-main-v2-notes.md.

11/01 EVENING, INDOOR RERUN FOLDED IN (room/kitchen/bonsai/counter with --indoor; data rd/rd-prune3-indoor-2026-11-01.csv + originals-indoor-2026-11-01.csv; rd-11scenes-v3.csv, rd-hull6-v2-table.md, figstyle_v2, fig-mesh-class-v2, fig-gauss-class-v2, tables-v2.md, report-v4.md). Mip-360 means: FP32 642 MB/24.81 (= paper 24.78); lam 6.2e-4 26.6 MB/24x/24.56 (-0.24); 1.85e-3 20.0/32x/24.44 (-0.37); 6.2e-3 9.0/72x/23.89 (-0.92); 1.85e-2 4.3/151x/23.12 (-1.68); level 1.0 26.3/23x/-0.38. Absolute hull unchanged (room) or better (kitchen +0.7 dB); Delta vs own FP32 got worse because indoor FP32 rose and the fixed delivery settings lose 0.5-0.75 dB on indoor models (diagnostic pending). The old headline 'lam 6.2e-4 matches FP32 at 24x' no longer holds for Mip-360 (-0.24); mesh-class lollipops: MiLo 22x, RaDe-GS 175x, GOF 352x, authors' PLY 4x at -0.22.
