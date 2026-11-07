---
name: rdgrowth-room-lambda-results
description: Room lambda sweep 10/18 (train_rd v1, 6 lambdas): tie with cap at 1.3M, +0.25 dB at 0.5M, clearly better at 0.23M (3 MB / 246x); v2 mpx units + 9-job follow-up layout
metadata:
  type: project
---

RESULTS 10/18 (rd/rd-sweep-room-2026-10-18.txt): lam 0 -> 2.90M v 28.02 MB/28.170; 3e-4 -> 2.91M 28.15/28.201; 1e-3 -> 2.89M 28.38/28.272 (+0.21 dB vs dense original 30.93/28.061, finetune test 28.47 vs 28.263); 3e-3 -> 1.32M 13.70/27.985 = TIE with cap-1.6M (1.35M, 13.56/27.984). Reading: lam monotone but steep (knee between 1e-3 and 3e-3; 0..1e-3 limiter-bound at 2.9M); ranking does not matter at 1.3M; at the dense end lam=1e-3 damps the split/prune oscillation (lam 0 splits 0.5-0.7M then prunes ~1M per interval and hits the 4M cap) -> +0.1-0.2 dB. Pass count at 3e-3 still 55k at 9500: stop is schedule-bound, not natural. ALL 6 DONE: 1e-2 -> 494k v 5.70 MB/27.37 (cap interp at 494k ~27.13: +0.25 dB; 130x, -0.89); 3e-2 -> 226k v 3.01 MB/26.509 (246x, -1.75; cap collapses below 400k: 362k cap = 4.48/26.628, so lambda holds PSNR with 38 % fewer vertices). Pattern: tie at 1.3M, +0.25 at 0.5M, clearly better at 0.23M -> error-driven selection pays when the budget is scarce. Figure rd/room-lambda-vs-cap-v1.png (csv room-lambda-vs-cap-v1.csv, plot_lambda_vs_cap.py). V2 CODE 10/18: rd_growth_v2.py / train_rd_v2.py add --dd_norm mpx (dD per megapixel of loss image; lambda_v2 = lambda_v1 / Mpx; Room images_2 1.62 Mpx, outdoor images_4 ~1.0-1.09 Mpx) so one lambda means the same across resolutions. Block commands/rd-sweep-v2.md: 9 jobs = Room fill {1e-3,3e-3,1e-2} + {1.3e-3,4.5e-3} (v2 units), six scenes x {2e-3, 6e-3} (~Room 46x/130x), Room cap controls 560k/260k; tables rd-sweep2-<scene>.txt. User had NOT pasted v2 (10/18 08:00) and asked to run only what is necessary -> commands/rd-sweep-v2-min.md: six scenes x {2e-3, 6e-3} only (12 runs, ~10 GPU-h, 3 h walltime), no Room fill / cap controls. Pending user paste of rd-sweep2-<scene>.txt (2 rows each).


See [[rdgrowth-v1-design]], [[scenes7-budget-vs-collapse]].
