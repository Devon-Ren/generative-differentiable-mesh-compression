---
name: rdgrowth-scenes7-results
description: six-scene lambda runs 10/18 (train_rd v2, lambda 2e-3/6e-3 per Mpx) + Room: lambda >= cap at equal bytes on all 10 comparable points (+0.0..+0.57), one lambda stops at 41-100 % / 15-43 % of dense vertices per scene; means 19.6 MB/-0.26 dB/36x and 8.8 MB/-0.88/80x
metadata:
  type: project
---

Jobs 20228117-24 (10/18, 6 jobs x 2 lambdas, ~1.2-1.8 h each, train 22-46 min). Raw rows rd/rd-sweep2-2026-10-18.txt; per-point table rd/lambda-7scenes-v1.csv (ratio, dB vs orig, gain vs cap / vs collapse at equal bytes by linear interpolation on the cap/collapse curves); figure rd/lambda-vs-cap-7scenes-v1.png (plot_lambda_vs_cap_7scenes.py, csv lambda-vs-cap-7scenes-v1.csv). Room lambdas expressed in v2 units (v1/1.618).

lam 2e-3 (MB / test, x, dB vs orig, gain vs cap @bytes): garden 24.78/24.445 30x -0.29 +0.21; bonsai 14.80/26.727 46x -0.56 +0.11; counter 14.90/25.761 39x -0.18 +0.38; kitchen 15.89/26.144 29x -0.31 +0.38; bicycle 30.07/23.018 23x -0.02 (above cap range; +0.18 vs collapsed dense); stump 23.32/24.729 30x -0.20 +0.37; Room(1.85e-3) 13.70/27.985 54x -0.28 +0.00.
lam 6e-3: garden 10.69/23.637 69x -1.09 (below lowest cap point; cap 12.52/23.699 -> about equal by slope); bonsai 6.91/26.066 98x -1.23 (no cap point below 9.54 MB; NOT comparable, cap flat segment 912k-1.44M at 26.585 makes it look worse); counter 6.94/25.263 83x -0.68 (dominates cap 719k 8.10/25.102); kitchen 7.60/25.312 60x -1.14 (dominates cap 802k 9.68/25.211); bicycle 13.83/22.703 49x -0.33 +0.36; stump 10.04/24.166 70x -0.76 +0.57; Room(6.2e-3) 5.70/27.37 130x -0.89 +0.25.
=> lambda >= cap at equal bytes on all 10 comparable points, 0.0-0.57 dB, largest on outdoor/foliage (stump, bicycle) and kitchen/counter; vs post-hoc collapse +0.3..+1.6 dB.
Means over 7 scenes: lam 2e-3 19.6 MB / -0.26 dB / 36x; lam 6e-3 8.8 MB / -0.88 dB / 80x (budget-1.6M was 14.4/-0.68/46x, budget-1.0M 9.8/-0.97/67x -> lambda points dominate the cap headline at the low end: 8.8 MB -0.88 vs 9.8 MB -0.97).
Scene-adaptive stop: same lambda leaves 41-100 % (2e-3) / 15-43 % (6e-3) of dense vertices (bicycle keeps most, Room least). Local dB/MB slope between the two lambda points differs by scene (0.019 bicycle .. 0.100 kitchen): equal-slope in dB does NOT hold tightly (lambda prices L1 mass, PSNR is log-MSE and scene error levels differ) -> paper claim = scene-adaptive stop + >= cap, not equal dB slope. Growth never stops naturally by 9500 (pass > 0): count is schedule-bound; a longer densify window or v2 prune would change this.
See [[rdgrowth-room-lambda-results]], [[rdgrowth-v1-design]], [[scenes7-budget-vs-collapse]].
