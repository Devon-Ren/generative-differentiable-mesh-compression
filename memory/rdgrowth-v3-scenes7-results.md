---
name: rdgrowth-v3-scenes7-results
description: batch B v3 (growth + lambda-prune) 11 scenes 10/24: 11-scene means 20.7 MB/32x/-0.22 dB and 9.4 MB/73x/-0.59; 7-scene 18.7/-0.25 and 8.1/-0.85; v3 = v2 at 4-8 % fewer bytes; prune bound by foldover
metadata:
  type: project
---

Rows: analysis/rd/rd-prune3-7scenes-2026-10-24.csv (user paste 10/24 of $AN/rd-prune3-*.txt; rd-prune-v3b.sbatch, LAMS 0.0062 / 0.00185 v2 mpx units, MU 1, MAXF 0.15, K4096 R256, XYZ room q14, garden/bonsai/counter q15, kitchen/bicycle/stump q16, DC q7).

lam 1.85e-3 (MB / x / dB vs orig): bicycle 26.97/25x/-0.02, bonsai 13.99/47x/-0.64, counter 14.80/39x/-0.15, garden 24.60/30x/-0.23, kitchen 16.66/27x/-0.21, room 12.57/59x/-0.26, stump 21.41/33x/-0.21. Mean 18.7 MB / 37x / -0.25 dB.
lam 6.2e-3: bicycle 11.94/57x/-0.35, bonsai 6.08/109x/-1.13, counter 6.65/86x/-0.68, garden 10.18/73x/-1.08, kitchen 7.50/61x/-1.05, room 5.23/142x/-0.88, stump 8.87/79x/-0.81. Mean 8.1 MB / 87x / -0.85 dB.
vs v2 lambda-only ([[rdgrowth-scenes7-results]], scenes at 2e-3/6e-3, Room same lambdas): mean -4.1 % bytes / +0.017 dB (hi), -8.0 % / +0.022 dB (lo); per point -14..+5 % bytes, -0.08..+0.10 dB; same as Room phase-B finding (prune = lambda in reverse, small consistent gain, no variable-price claim). Note scene lambdas differ slightly from v2 (1.85 vs 2, 6.2 vs 6 e-3).
Prune diagnostics: removes 12-22 % of V@delaunay over 4 rounds; per-round cap never binds (pairs << cap); foldover filter rejects MORE candidates than are accepted from round 2 on (e.g. bicycle lo round 18k 28.9k pairs vs 101k flip-rejected) -> prune is limited by midpoint-collapse foldover, not by price. Bits/vertex 83-108.
NEW 4 DONE 10/24 (rd/rd-prune3-new4-2026-10-24.csv; 11-scene table rd/rd-11scenes-v1.csv, figure rd/rd-11scenes-v1.png via plot_11scenes_v1.py). lam 6.2e-3 / 1.85e-3 (MB, x, dB): flowers 12.03 56x -0.24 / 24.15 28x -0.38; treehill 11.89 58x -0.01 / 27.42 25x +0.06; truck 9.41 46x -0.02 / 19.31 22x -0.19; train 13.44 36x -0.22 / 25.37 19x -0.24. On flowers/truck/train the SMALLER asset has equal or higher test PSNR (val still rises with vertices; LPIPS improves .02-.05 with more vertices) -> extra vertices fit training views, not test, on low-coverage scenes (inferred); v3 lo also beats dense level-1.0 at ~half the bytes there. 11-scene v3 means: 20.7 MB / 32x / -0.22 dB and 9.4 MB / 73x / -0.59 dB (ratio range 36-142x at 6.2e-3).
