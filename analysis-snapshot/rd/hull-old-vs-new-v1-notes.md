# hull-old-vs-new-v1 (2026-10-20): talking points for the advisor

Figure: analysis/rd/hull-old-vs-new-v1.png / .pdf (script plot_hull_old_vs_new_v1.py; data room-lambda-vs-cap-v2.csv, seven-scene-points-v2.csv, lambda-7scenes-v1.csv).

Old route (10/13-10/16) = dense original -> shared-SH dictionary K x R (+ residual) -> post-hoc vertex collapse 100..50 % -> stage2-v2 codec. 49 Room configs, hull from 16.6 MB / 27.22 to 35.8 MB / 28.14; default 25.5 MB / 27.88 (29x, -0.38 dB).
New route (10/18-10/19) = same codec, but the vertex set is decided during training by a Lagrangian price lambda (split if dD/dR > lambda) plus a phase-B lambda-prune; no cap, no post-hoc collapse.

1. Same PSNR at half the bytes (Room): 13.7 MB / 27.99 vs old default 25.5 MB / 27.88.
2. The old route stops at ~17 MB (below that it loses > 1 dB); the new route continues to 5.2 MB / 27.38 (142x) and 3.0 MB / 26.51 (246x). Below 20 MB the whole frontier is new-route only.
3. 7 scenes: one lambda setting, no per-scene tuning: lambda 2e-3 = 19.6 MB / -0.26 dB / 36x vs old default 23.6 MB / -0.44 dB / 28x (fewer bytes AND higher PSNR); lambda 6e-3 = 8.8 MB / -0.88 dB / 80x, a region the old route cannot reach on 5/7 scenes.
4. Why (one sentence): on an opaque mesh vertices are topologically coupled, so removing them afterwards costs 0.3-4.3 dB at equal vertex count (7/7 scenes); deciding them while the mesh grows costs 0.1-0.7 dB. The appearance stream (SH dictionary) is unchanged between the two routes: post-hoc sharing works for appearance, not for geometry.
