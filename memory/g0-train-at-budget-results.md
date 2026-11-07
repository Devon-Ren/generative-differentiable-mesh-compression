---
name: g0-train-at-budget-results
description: G0 Room train-at-budget vs post-hoc collapse (10/17): +1.3 dB at 1.35M v, +2.05 dB at 0.97M v; dense->1.35M flat (0.006 dB/MB); reframes axis B
metadata:
  type: project
---

On 2026-10-17, job 20133555 (L40S): Room trained with --max_points 1600000 ended at 1,348,124 vertices (37 min training). Level-1.0 protocol, K4096 R256 q14/q7:

- TRAIN-AT-BUDGET: finetune test 28.165, delivered 13.558 MB test 27.984 SSIM 0.862 LPIPS 0.3963.
- Control POST-HOC-COLLAPSE of dense Room to same 1,348,124 vertices: finetune test 26.772, delivered 13.926 MB test 26.66 SSIM 0.8372 LPIPS 0.4239.

So +1.3 dB at same bytes; 13.6 MB point (~55x vs 742 MB) beats the old v0.8 default (25.5 MB / 27.883) and is only 0.28 dB below dense original 28.263.

Interpretation: dense densification overgrows and post-hoc collapse cannot recover; axis B curve measured collapse cost, not vertex-count cost. Method chapter candidate: budget/lambda-controlled growth, post-hoc collapse becomes the control.

1.0M BUDGET (same job, done 03:42 EDT): 966,571 v (30 %), finetune test 27.843, delivered 10.378 MB / 27.711 / SSIM .855 / LPIPS .413 (72x, -0.55 vs original); control post-hoc collapse to 966,571 v: 10.139 MB / 25.665 / .813 / .456 -> +2.05 dB. Dense 31.87 MB/28.091 -> 1.35M 13.56/27.984 costs 0.006 dB/MB (flat); 1.35M -> 0.97M 0.086 dB/MB: the knee is at or below 1.35M. Worst val view stays 26.3-26.7 at budget vs 21.9-23.7 after collapse. Raw rows rd/g0-budget-2026-10-17.txt.

BUDGET SWEEP DONE 2026-10-17 (rd/budget-sweep-2026-10-17.txt): Room 700k -> 678k v 7.59 MB / 27.461 (98x, -0.80); 500k -> 412k v 4.99 / 26.986 (149x, -1.28); 350k -> 362k v 4.48 / 26.628. Room budget slopes 1.35M->0.97M 0.086, ->678k 0.090, ->412k 0.18, ->362k 0.7 dB/MB: knee ~0.7M. Garden (q15): 1.6M -> 1.48M v 16.51 MB / 23.996 (-0.74 vs 24.732), 1.0M -> 1.09M v 12.52 / 23.699; collapse v0.7 q14 was 23.18 MB / 23.988 -> same PSNR at 71 % bytes (gain ~+0.5 dB at equal bytes, smaller than Room). Bonsai (q15): 1.6M -> 1.44M v 13.96 / 26.585, 1.0M -> 912k v 9.54 / 26.585 (identical PSNR: flat below 1.44M; but -0.52 vs dense delivered 26.61 MB/27.102, the drop happens between dense and 1.44M); collapse v0.7 q15 19.33 / 26.428 -> half the bytes +0.16 dB. Budget cost vs dense is scene-dependent (Room -0.1, garden -0.69, bonsai -0.60 at ~1.4M). Figure rd/room-budget-vs-collapse-v1.png (plot_budget_vs_collapse.py, csv budget-vs-collapse-room-v1.csv). CONTROLS DONE 2026-10-17 (budget-control-v1): garden collapse to 1.48M 16.25 MB / 23.154, to 1.09M 12.14 / 22.299; bonsai to 1.44M 13.83 / 25.028, to 912k 8.96 / 22.287. Gains at equal vertices: room +1.32/+2.05, garden +0.84/+1.40, bonsai +1.56/+4.30 dB -> 3/3 scenes, gap grows as vertices shrink. Figure rd/budget-vs-collapse-3scenes-v1.png (plot_budget_vs_collapse_scenes.py, csv budget-vs-collapse-3scenes-v1.csv). 4 other scenes DONE 10/18: 7/7 scenes win, see [[scenes7-budget-vs-collapse]].

Pass/fail bar set beforehand: good if >=0.3 dB at 1.6M and 1.0M within -0.6 dB of 28.263; flat if <0.1 dB.

Link [[meshsplatting-compression-status]] and [[scenes4-results]].
METHOD DESIGN 2026-10-17 (plan/rd-growth-method-design-2026-10-17.md): cap = input, not output. CVPR version = lambda-split (dD_t > lambda*dR_t, dR ~10 B/vertex by stream) + periodic lambda-prune in training (C2 sens_pos, dR_v varies with faces removed) + allocator coupling (lambda prices precision/codebook); mesh-specific why = vertices topologically coupled, post-hoc removal loses 1-4 dB whereas Gaussians prune post-hoc for free. Implement as analysis/rdgrowth/ subclass overriding add_new_gs/prune, no repo edits. Honest caveat: with constant dR the split rule is a grad threshold; method content lives in prune + allocator.
